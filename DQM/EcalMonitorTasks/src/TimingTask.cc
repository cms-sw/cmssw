#include "../interface/TimingTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  TimingTask::TimingTask() :
    DQWorkerTask(),
    chi2ThresholdEB_(0.),
    chi2ThresholdEE_(0.),
    energyThresholdEB_(0.),
    energyThresholdEE_(0.)
  {
  }

  void
  TimingTask::setParams(edm::ParameterSet const& _params)
  {
    chi2ThresholdEB_   = _params.getUntrackedParameter<double>("chi2ThresholdEB");
    chi2ThresholdEE_   = _params.getUntrackedParameter<double>("chi2ThresholdEE");
    energyThresholdEB_ = _params.getUntrackedParameter<double>("energyThresholdEB");
    energyThresholdEE_ = _params.getUntrackedParameter<double>("energyThresholdEE");
  }

  bool
  TimingTask::filterRunType(short const* _runType)
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
  TimingTask::runOnRecHits(EcalRecHitCollection const& _hits, Collections _collection)
  {
    MESet& meTimeAmp(MEs_.at("TimeAmp"));
    MESet& meTimeAmpAll(MEs_.at("TimeAmpAll"));
    MESet& meTimeAll(MEs_.at("TimeAll"));
    MESet& meTimeAllMap(MEs_.at("TimeAllMap"));
    MESet& meTimeMap(MEs_.at("TimeMap"));
    MESet& meTime1D(MEs_.at("Time1D"));
    MESet& meChi2(MEs_.at("Chi2"));

    uint32_t mask(~((0x1 << EcalRecHit::kGood) | (0x1 << EcalRecHit::kOutOfTime)));
    float threshold(_collection == kEBRecHit ? energyThresholdEB_ : energyThresholdEE_);
    int signedSubdet;

    std::for_each(_hits.begin(), _hits.end(), [&](EcalRecHitCollection::value_type const& hit){
                    if(hit.checkFlagMask(mask)) return;

                    DetId id(hit.id());

                    float time(hit.time());
                    float energy(hit.energy());

                    float chi2Threshold = ( id.subdetId() == EcalBarrel ) ? chi2ThresholdEB_ : chi2ThresholdEE_;
                    if (id.subdetId() == EcalBarrel) {
                      signedSubdet=EcalBarrel;
                    }
                    else {
                      EEDetId eeId(hit.id());
                      if(eeId.zside() < 0){
                        signedSubdet = -EcalEndcap;
                      }
                      else{
                        signedSubdet = EcalEndcap;
                      }
                    }

                    if(energy > threshold){
                      meChi2.fill(signedSubdet, hit.chi2());
                    }

                    if( hit.chi2() > chi2Threshold ) return;

                    meTimeAmp.fill(id, energy, time);
                    meTimeAmpAll.fill(id, energy, time);

                    if(energy > threshold){
                      meTimeAll.fill(id, time);
                      meTimeMap.fill(id, time);
                      meTime1D.fill(id, time);
                      meTimeAllMap.fill(id, time);
                    }
                  });
  }

  // For In-time vs Out-of-Time amplitude correlation MEs:
  // Only UncalibRecHits carry information about OOT amplitude
  // But still need to make sure we apply similar cuts as on RecHits
  void TimingTask::runOnUncalibRecHits( EcalUncalibratedRecHitCollection const& _uhits )
  {
    MESet& meTimeAmpBXm( MEs_.at("TimeAmpBXm") );
    MESet& meTimeAmpBXp( MEs_.at("TimeAmpBXp") );

    for( EcalUncalibratedRecHitCollection::const_iterator uhitItr(_uhits.begin()); uhitItr != _uhits.end(); ++uhitItr ) {

      // Apply reconstruction quality cuts
      if( !uhitItr->checkFlag(EcalUncalibratedRecHit::kGood) ) continue;
      DetId id( uhitItr->id() );
      float chi2Threshold = ( id.subdetId() == EcalBarrel ) ? chi2ThresholdEB_ : chi2ThresholdEE_;
      if( uhitItr->chi2() > chi2Threshold ) continue;

      // Apply amplitude cut based on approx rechit energy
      float amp( uhitItr->amplitude() );
      float ampThreshold( id.subdetId() == EcalBarrel ? energyThresholdEB_*20. : energyThresholdEE_*5. ); // 1 GeV ~ ( EB:20, EE:5 ) ADC
      if( amp < ampThreshold ) continue;

      // Apply jitter timing cut based on approx rechit timing
      float timeOff( id.subdetId() == EcalBarrel ? 0.4 : 1.8 );
      float hitTime( uhitItr->jitter()*25. + timeOff ); // 1 jitter ~ 25 ns
      if( abs(hitTime) >= 5. ) continue;

      // Fill MEs
      meTimeAmpBXm.fill( id,amp,uhitItr->outOfTimeAmplitude(4) ); // BX-1
      meTimeAmpBXp.fill( id,amp,uhitItr->outOfTimeAmplitude(6) ); // BX+1

    }
  }

  DEFINE_ECALDQM_WORKER(TimingTask);
}
