#include "../interface/PresampleTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  PresampleTask::PresampleTask() :
    DQWorkerTask(),
    doPulseMaxCheck_(true),
    pulseMaxPosition_(0),
    nSamples_(0),
    mePedestalByLS(0)
  {
  }

  void
  PresampleTask::setParams(edm::ParameterSet const& _params)
  {
    doPulseMaxCheck_ = _params.getUntrackedParameter<bool>("doPulseMaxCheck");
    pulseMaxPosition_ = _params.getUntrackedParameter<int>("pulseMaxPosition");
    nSamples_ = _params.getUntrackedParameter<int>("nSamples");
  }

  bool
  PresampleTask::filterRunType(short const* _runType)
  {
    for(int iFED(0); iFED < nDCC; iFED++){
      if(_runType[iFED] == EcalDCCHeaderBlock::COSMIC ||
         _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
         _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
         _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
         _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
         _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL) return true;
    }

    return false;
  }

  void
  PresampleTask::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
    // Fill separate MEs with only 10 LSs worth of stats
    // Used to correctly fill Presample Trend plots:
    // 1 pt:10 LS in Trend plots
    mePedestalByLS = &MEs_.at("PedestalByLS");
    if ( timestamp_.iLumi % 10 == 0 )
      mePedestalByLS->reset();
  }

  template<typename DigiCollection>
  void
  PresampleTask::runOnDigis(DigiCollection const& _digis)
  {
    MESet& mePedestal(MEs_.at("Pedestal")); // contains cumulative run stats => not suitable for Trend plots

    for(typename DigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      DetId id(digiItr->id());

      // EcalDataFrame is not a derived class of edm::DataFrame, but can take edm::DataFrame in the constructor
      EcalDataFrame dataFrame(*digiItr);

      // Check that the digi pulse maximum occurs on the 6th sample
      // For cosmics: disable this check to preserve statistics
      if ( doPulseMaxCheck_ ) {
        bool gainSwitch(false);
        int iMax(-1);
        int maxADC(0);
        for(int iSample(0); iSample < EcalDataFrame::MAXSAMPLES; ++iSample){
          int adc(dataFrame.sample(iSample).adc());
          if(adc > maxADC){
            iMax = iSample;
            maxADC = adc;
          }
          if(iSample < nSamples_ && dataFrame.sample(iSample).gainId() != 1){
            gainSwitch = true;
            break;
          }
        } // iSample
        if(iMax != pulseMaxPosition_ || gainSwitch) continue;
      } // PulseMaxCheck

      for(int iSample(0); iSample < nSamples_; ++iSample) {
        mePedestal.fill(id, double(dataFrame.sample(iSample).adc()));
        mePedestalByLS->fill(id, double(dataFrame.sample(iSample).adc()));
      }

    } // _digis loop
  } // runOnDigis

  DEFINE_ECALDQM_WORKER(PresampleTask);
}

