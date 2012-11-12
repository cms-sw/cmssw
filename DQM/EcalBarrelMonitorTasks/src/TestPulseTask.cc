#include "../interface/TestPulseTask.h"

#include <algorithm>
#include <iomanip>

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

namespace ecaldqm {

  TestPulseTask::TestPulseTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "TestPulseTask"),
    gainToME_(),
    pnGainToME_()
  {
    using namespace std;

    collectionMask_[kEBDigi] = true;
    collectionMask_[kEEDigi] = true;
    collectionMask_[kPnDiodeDigi] = true;
    collectionMask_[kEBTestPulseUncalibRecHit] = true;
    collectionMask_[kEETestPulseUncalibRecHit] = true;

    for(unsigned iD(0); iD < BinService::nDCC; ++iD){
      enable_[iD] = false;
      gain_[iD] = 0;
    }

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

    std::string apdPlots[] = {"Shape", "Amplitude"};
    for(unsigned iS(0); iS < sizeof(apdPlots) / sizeof(std::string); ++iS){
      std::string& plot(apdPlots[iS]);
      MESetMulti* multi(static_cast<MESetMulti*>(MEs_[plot]));

      for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << std::setfill('0') << std::setw(2) << gainItr->first;
        replacements["gain"] = ss.str();

        multi->formPath(replacements);
      }
    }

    std::string pnPlots[] = {"PNAmplitude"};
    for(unsigned iS(0); iS < sizeof(pnPlots) / sizeof(std::string); ++iS){
      std::string& plot(pnPlots[iS]);
      MESetMulti* multi(static_cast<MESetMulti*>(MEs_[plot]));

      for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
        multi->use(gainItr->second);

        ss.str("");
        ss << std::setfill('0') << std::setw(2) << gainItr->first;
        replacements["pngain"] = ss.str();

        multi->formPath(replacements);
      }
    }
  }

  bool
  TestPulseTask::filterRunType(const std::vector<short>& _runType)
  {
    bool enable(false);

    for(int iFED(0); iFED < 54; iFED++){
      if(_runType[iFED] == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
	 _runType[iFED] == EcalDCCHeaderBlock::TESTPULSE_GAP){
	enable = true;
	enable_[iFED] = true;
      }
      else
        enable_[iFED] = false;
    }

    return enable;
  }

  void
  TestPulseTask::runOnRawData(EcalRawDataCollection const& _rawData)
  {
    for(EcalRawDataCollection::const_iterator rItr(_rawData.begin()); rItr != _rawData.end(); ++rItr){
      unsigned iDCC(rItr->id() - 1);

      if(!enable_[iDCC]){
        gain_[iDCC] = 0;
        continue;
      }
      gain_[iDCC] = rItr->getMgpaGain();

      if(gainToME_.find(gain_[iDCC]) == gainToME_.end())
        enable_[iDCC] = false;
    }
  }

  void
  TestPulseTask::runOnDigis(const EcalDigiCollection &_digis)
  {
    MESet* meOccupancy(MEs_["Occupancy"]);
    MESet* meShape(MEs_["Shape"]);

    unsigned iME(-1);

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      DetId id(digiItr->id());

      meOccupancy->fill(id);

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      if(iME != gainToME_[gain_[iDCC]]){
        iME = gainToME_[gain_[iDCC]];
        static_cast<MESetMulti*>(meShape)->use(iME);
      }

      for(int iSample(0); iSample < 10; iSample++)
	meShape->fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));
    }
  }

  void
  TestPulseTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
  {
    MESet* mePNAmplitude(MEs_["PNAmplitude"]);

    unsigned iME(-1);

    for(EcalPnDiodeDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      EcalPnDiodeDetId const& id(digiItr->id());

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
        static_cast<MESetMulti*>(mePNAmplitude)->use(iME);
      }

      float pedestal(0.);
      for(int iSample(0); iSample < 4; iSample++)
	pedestal += digiItr->sample(iSample).adc();
      pedestal /= 4.;

      float max(0.);
      for(int iSample(0); iSample < 50; iSample++)
	if(digiItr->sample(iSample).adc() > max) max = digiItr->sample(iSample).adc();

      double amplitude(max - pedestal);

      mePNAmplitude->fill(id, amplitude);
    }
  }

  void
  TestPulseTask::runOnUncalibRecHits(const EcalUncalibratedRecHitCollection &_uhits)
  {
    MESet* meAmplitude(MEs_["Amplitude"]);

    unsigned iME(-1);

    for(EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr){
      DetId id(uhitItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      if(iME != gainToME_[gain_[iDCC]]){
        iME = gainToME_[gain_[iDCC]];
        static_cast<MESetMulti*>(meAmplitude)->use(iME);
      }

      meAmplitude->fill(id, uhitItr->amplitude());
    }
  }

  DEFINE_ECALDQM_WORKER(TestPulseTask);
}
