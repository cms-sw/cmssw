#include "../interface/PedestalTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm {

  PedestalTask::PedestalTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "PedestalTask"),
    gainToME_(),
    pnGainToME_()
  {
    using namespace std;

    collectionMask_ = 
      (0x1 << kEBDigi) |
      (0x1 << kEEDigi) |
      (0x1 << kPnDiodeDigi);

    for(unsigned iD(0); iD < BinService::nDCC; ++iD)
      enable_[iD] = false;

    vector<int> MGPAGains(_commonParams.getUntrackedParameter<vector<int> >("MGPAGains"));
    vector<int> MGPAGainsPN(_commonParams.getUntrackedParameter<vector<int> >("MGPAGainsPN"));

    unsigned iMEGain(0);
    for(vector<int>::iterator gainItr(MGPAGains.begin()); gainItr != MGPAGains.end(); ++gainItr){
      if(*gainItr != 1 && *gainItr != 6 && *gainItr != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain" << endl;
      gainToME_[*gainItr] = iMEGain++;
    }

    unsigned iMEPNGain(0);
    for(vector<int>::iterator gainItr(MGPAGainsPN.begin()); gainItr != MGPAGainsPN.end(); ++gainItr){
      if(*gainItr != 1 && *gainItr != 16) throw cms::Exception("InvalidConfiguration") << "PN diode gain" << endl;	
      pnGainToME_[*gainItr] = iMEPNGain++;
    }

    map<string, string> replacements;
    stringstream ss;

    unsigned apdPlots[] = {kOccupancy, kPedestal};
    for(unsigned iS(0); iS < sizeof(apdPlots) / sizeof(unsigned); ++iS){
      unsigned plot(apdPlots[iS]);
      MESet* temp(MEs_[plot]);
      MESetMulti* meSet(new MESetMulti(*temp, iMEGain));

      for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
        meSet->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["gain"] = ss.str();

        meSet->formPath(replacements);
      }

      MEs_[plot] = meSet;
      delete temp;
    }

    unsigned pnPlots[] = {kPNPedestal};
    for(unsigned iS(0); iS < sizeof(pnPlots) / sizeof(unsigned); ++iS){
      unsigned plot(pnPlots[iS]);
      MESet* temp(MEs_[plot]);
      MESetMulti* meSet(new MESetMulti(*temp, iMEPNGain));

      for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
        meSet->use(gainItr->second);

        ss.str("");
        ss << gainItr->first;
        replacements["pngain"] = ss.str();

        meSet->formPath(replacements);
      }

      MEs_[plot] = meSet;
      delete temp;
    }
  }

  bool
  PedestalTask::filterRunType(const std::vector<short>& _runType)
  {
    bool enable(false);

    for(int iFED(0); iFED < 54; iFED++){
      if(_runType[iFED] == EcalDCCHeaderBlock::PEDESTAL_STD ||
	 _runType[iFED] == EcalDCCHeaderBlock::PEDESTAL_GAP){
	enable = true;
	enable_[iFED] = true;
      }
      else
        enable_[iFED] = false;
    }

    return enable;
  }

  void
  PedestalTask::runOnDigis(const EcalDigiCollection &_digis)
  {
    unsigned iME(-1);

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      DetId id(digiItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      int gain(0);
      switch(dataFrame.sample(0).gainId()){
      case 1: gain = 12; break;
      case 2: gain = 6; break;
      case 3: gain = 1; break;
      default: continue;
      }

      if(gainToME_.find(gain) == gainToME_.end()) continue;

      if(iME != gainToME_[gain]){
        iME = gainToME_[gain];
        static_cast<MESetMulti*>(MEs_[kOccupancy])->use(iME);
        static_cast<MESetMulti*>(MEs_[kPedestal])->use(iME);
      }

      MEs_[kOccupancy]->fill(id);

      float mean(0.);
      for(int iSample(0); iSample < 10; iSample++)
	mean += dataFrame.sample(iSample).adc();
      mean /= 10.;

      MEs_[kPedestal]->fill(id, mean);
    }
  }

  void
  PedestalTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    unsigned iME(-1);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      EcalPnDiodeDetId id(digiItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      int gain(0);
      switch(digiItr->sample(0).gainId()){
      case 0: gain = 1; break;
      case 1: gain = 16; break;
      default: continue;
      }

      if(pnGainToME_.find(gain) == pnGainToME_.end()) continue;

      if(iME != pnGainToME_[gain]){
        iME = pnGainToME_[gain];
        static_cast<MESetMulti*>(MEs_[kPNPedestal])->use(iME);
      }

      float mean(0.);
      for(int iSample(0); iSample < 50; iSample++)
	mean += digiItr->sample(iSample).adc();
      mean /= 50.;

      MEs_[kPNPedestal]->fill(id, mean);
    }
  }

  /*static*/
  void
  PedestalTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Occupancy"] = kOccupancy;
    _nameToIndex["Pedestal"] = kPedestal;
    _nameToIndex["PNPedestal"] = kPNPedestal;
  }

  DEFINE_ECALDQM_WORKER(PedestalTask);
}
