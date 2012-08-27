#include "../interface/TestPulseTask.h"

#include <algorithm>

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

    collectionMask_ = 
      (0x1 << kEBDigi) |
      (0x1 << kEEDigi) |
      (0x1 << kPnDiodeDigi) |
      (0x1 << kEBUncalibRecHit) |
      (0x1 << kEEUncalibRecHit);

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

    unsigned apdPlots[] = {kOccupancy, kShape, kAmplitude};
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

    unsigned pnPlots[] = {kPNOccupancy, kPNAmplitude};
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

  TestPulseTask::~TestPulseTask()
  {
  }

  void
  TestPulseTask::beginRun(const edm::Run&, const edm::EventSetup&)
  {
    for(int idcc(0); idcc < 54; idcc++){
      enable_[idcc] = false;
      gain_[idcc] = 0;
    }
  }

  void
  TestPulseTask::endEvent(const edm::Event&, const edm::EventSetup&)
  {
    for(int idcc(0); idcc < 54; idcc++){
      enable_[idcc] = false;
      gain_[idcc] = 0;
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
    }

    return enable;
  }

  void
  TestPulseTask::runOnDigis(const EcalDigiCollection &_digis)
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
        static_cast<MESetMulti*>(MEs_[kShape])->use(iME);
      }

      MEs_[kOccupancy]->fill(id);

      for(int iSample(0); iSample < 10; iSample++)
	MEs_[kShape]->fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));

      gain_[iDCC] = gain;
    }
  }

  void
  TestPulseTask::runOnPnDigis(const EcalPnDiodeDigiCollection &_digis)
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
        static_cast<MESetMulti*>(MEs_[kPNOccupancy])->use(iME);
        static_cast<MESetMulti*>(MEs_[kPNAmplitude])->use(iME);
      }

      MEs_[kPNOccupancy]->fill(id);

      float pedestal(0.);
      for(int iSample(0); iSample < 4; iSample++)
	pedestal += digiItr->sample(iSample).adc();
      pedestal /= 4.;

      float max(0.);
      for(int iSample(0); iSample < 50; iSample++)
	if(digiItr->sample(iSample).adc() > max) max = digiItr->sample(iSample).adc();

      float amplitude(max - pedestal);

      MEs_[kPNAmplitude]->fill(id, amplitude);
    }
  }

  void
  TestPulseTask::runOnUncalibRecHits(const EcalUncalibratedRecHitCollection &_uhits)
  {
    unsigned iME(-1);

    for(EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr){
      DetId id(uhitItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      if(iME != gainToME_[gain_[iDCC]]){
        iME = gainToME_[gain_[iDCC]];
        static_cast<MESetMulti*>(MEs_[kAmplitude])->use(iME);
      }

      MEs_[kAmplitude]->fill(id, uhitItr->amplitude());
    }
  }

  /*static*/
  void
  TestPulseTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Occupancy"] = kOccupancy;
    _nameToIndex["Shape"] = kShape;
    _nameToIndex["Amplitude"] = kAmplitude;
    _nameToIndex["PNOccupancy"] = kPNOccupancy;
    _nameToIndex["PNAmplitude"] = kPNAmplitude;
  }

  DEFINE_ECALDQM_WORKER(TestPulseTask);
}
