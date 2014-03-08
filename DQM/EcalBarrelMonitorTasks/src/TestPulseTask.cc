#include "../interface/TestPulseTask.h"

#include <algorithm>
#include <iomanip>

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  TestPulseTask::TestPulseTask() :
    DQWorkerTask(),
    gainToME_(),
    pnGainToME_()
  {
    std::fill_n(enable_, nDCC, false);
    std::fill_n(gain_, nDCC, 0);
  }

  void
  TestPulseTask::setParams(edm::ParameterSet const& _params)
  {
    std::vector<int> MGPAGains(_params.getUntrackedParameter<std::vector<int> >("MGPAGains"));
    std::vector<int> MGPAGainsPN(_params.getUntrackedParameter<std::vector<int> >("MGPAGainsPN"));

    MESet::PathReplacements repl;

    MESetMulti& amplitude(static_cast<MESetMulti&>(MEs_.at("Amplitude")));
    unsigned nG(MGPAGains.size());
    for(unsigned iG(0); iG != nG; ++iG){
      int gain(MGPAGains[iG]);
      if(gain != 1 && gain != 6 && gain != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain";
      repl["gain"] = std::to_string(gain);
      gainToME_[gain] = amplitude.getIndex(repl);
    }

    repl.clear();

    MESetMulti& pnAmplitude(static_cast<MESetMulti&>(MEs_.at("PNAmplitude")));
    unsigned nGPN(MGPAGainsPN.size());
    for(unsigned iG(0); iG != nGPN; ++iG){
      int gain(MGPAGainsPN[iG]);
      if(gain != 1 && gain != 16) throw cms::Exception("InvalidConfiguration") << "PN MGPA gain";
      repl["pngain"] = std::to_string(gain);
      pnGainToME_[gain] = pnAmplitude.getIndex(repl);
    }
  }

  void
  TestPulseTask::addDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kEBTestPulseUncalibRecHit, kEcalRawData));
    _dependencies.push_back(Dependency(kEETestPulseUncalibRecHit, kEcalRawData));
  }

  bool
  TestPulseTask::filterRunType(short const* _runType)
  {
    bool enable(false);

    for(int iFED(0); iFED < nDCC; iFED++){
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
      switch(rItr->getMgpaGain()){
      case 1:
	gain_[iDCC] = 12;
	break;
      case 2:
	gain_[iDCC] = 6;
	break;
      case 3:
	gain_[iDCC] = 1;
	break;
      default:
	break;
      }

      if(gainToME_.find(gain_[iDCC]) == gainToME_.end())
        enable_[iDCC] = false;
    }
  }

  template<typename DigiCollection>
  void
  TestPulseTask::runOnDigis(DigiCollection const& _digis)
  {
    MESet& meOccupancy(MEs_.at("Occupancy"));
    MESet& meShape(MEs_.at("Shape"));

    unsigned iME(-1);

    for(typename DigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      DetId id(digiItr->id());

      meOccupancy.fill(id);

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      if(iME != gainToME_[gain_[iDCC]]){
        iME = gainToME_[gain_[iDCC]];
        static_cast<MESetMulti&>(meShape).use(iME);
      }

      for(int iSample(0); iSample < 10; iSample++)
	meShape.fill(id, iSample + 0.5, float(dataFrame.sample(iSample).adc()));
    }
  }

  void
  TestPulseTask::runOnPnDigis(EcalPnDiodeDigiCollection const& _digis)
  {
    MESet& mePNAmplitude(MEs_.at("PNAmplitude"));

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
        static_cast<MESetMulti&>(mePNAmplitude).use(iME);
      }

      float pedestal(0.);
      for(int iSample(0); iSample < 4; iSample++)
	pedestal += digiItr->sample(iSample).adc();
      pedestal /= 4.;

      float max(0.);
      for(int iSample(0); iSample < 50; iSample++)
	if(digiItr->sample(iSample).adc() > max) max = digiItr->sample(iSample).adc();

      double amplitude(max - pedestal);

      mePNAmplitude.fill(id, amplitude);
    }
  }

  void
  TestPulseTask::runOnUncalibRecHits(EcalUncalibratedRecHitCollection const& _uhits)
  {
    MESet& meAmplitude(MEs_.at("Amplitude"));

    unsigned iME(-1);

    for(EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr){
      DetId id(uhitItr->id());

      int iDCC(dccId(id) - 1);

      if(!enable_[iDCC]) continue;

      if(iME != gainToME_[gain_[iDCC]]){
        iME = gainToME_[gain_[iDCC]];
        static_cast<MESetMulti&>(meAmplitude).use(iME);
      }

      meAmplitude.fill(id, uhitItr->amplitude());
    }
  }

  DEFINE_ECALDQM_WORKER(TestPulseTask);
}
