#include "../interface/TimingTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  TimingTask::TimingTask() :
    DQWorkerTask(),
    bxBinEdges_(),
    bxBin_(0.),
    chi2ThresholdEB_(0.),
    chi2ThresholdEE_(0.),
    energyThresholdEB_(0.),
    energyThresholdEE_(0.),
    timingVsBXThreshold_(0.),
    meTimeMapByLS(nullptr)
  {
  }

  void
  TimingTask::setParams(edm::ParameterSet const& _params)
  {
    bxBinEdges_ = onlineMode_? _params.getUntrackedParameter<std::vector<int> >("bxBins"): _params.getUntrackedParameter<std::vector<int> >("bxBinsFine");
    chi2ThresholdEB_   = _params.getUntrackedParameter<double>("chi2ThresholdEB");
    chi2ThresholdEE_   = _params.getUntrackedParameter<double>("chi2ThresholdEE");
    energyThresholdEB_ = _params.getUntrackedParameter<double>("energyThresholdEB");
    energyThresholdEE_ = _params.getUntrackedParameter<double>("energyThresholdEE");
    timingVsBXThreshold_ = _params.getUntrackedParameter<double>("timingVsBXThreshold");
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
  TimingTask::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
  {
    // Fill separate MEs with only 10 LSs worth of stats
    // Used to correctly fill Presample Trend plots:
    // 1 pt:10 LS in Trend plots
    meTimeMapByLS = &MEs_.at("TimeMapByLS");
    if ( timestamp_.iLumi % 10 == 0 )
      meTimeMapByLS->reset();
  }

  void
  TimingTask::beginEvent(edm::Event const& _evt, edm::EventSetup const&  _es)
  {
    using namespace std;
    std::vector<int>::iterator pBin = std::upper_bound(bxBinEdges_.begin(), bxBinEdges_.end(), _evt.bunchCrossing());
    bxBin_ = static_cast<int>(pBin - bxBinEdges_.begin()) - 0.5;
  }

  void 
  TimingTask::runOnRecHits(EcalRecHitCollection const& _hits, Collections _collection)
  {
    MESet& meTimeAmp(MEs_.at("TimeAmp"));
    MESet& meTimeAmpAll(MEs_.at("TimeAmpAll"));
    MESet& meTimingVsBX(onlineMode_? MEs_.at("BarrelTimingVsBX"): MEs_.at("BarrelTimingVsBXFineBinned"));
    MESet& meTimeAll(MEs_.at("TimeAll"));
    MESet& meTimeAllMap(MEs_.at("TimeAllMap"));
    MESet& meTimeMap(MEs_.at("TimeMap")); // contains cumulative run stats => not suitable for Trend plots
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

                    // Apply cut on chi2 of pulse shape fit
                    float chi2Threshold = ( id.subdetId() == EcalBarrel ) ? chi2ThresholdEB_ : chi2ThresholdEE_;
                    if ( id.subdetId() == EcalBarrel )
                      signedSubdet = EcalBarrel;
                    else {
                      EEDetId eeId( hit.id() );
                      if ( eeId.zside() < 0 )
                        signedSubdet = -EcalEndcap;
                      else
                        signedSubdet =  EcalEndcap;
                    }
                    if ( energy > threshold )
                      meChi2.fill(signedSubdet, hit.chi2());
                    if ( hit.chi2() > chi2Threshold ) return;

                    meTimeAmp.fill(id, energy, time);
                    meTimeAmpAll.fill(id, energy, time);

                    if (energy > timingVsBXThreshold_ && signedSubdet == EcalBarrel) meTimingVsBX.fill(bxBin_, time);

                    if(energy > threshold){
                      meTimeAll.fill(id, time);
                      meTimeMap.fill(id, time);
                      meTimeMapByLS->fill(id, time); 
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
      if( std::abs(hitTime) >= 5. ) continue;

      // Fill MEs
      meTimeAmpBXm.fill( id,amp,uhitItr->outOfTimeAmplitude(4) ); // BX-1
      meTimeAmpBXp.fill( id,amp,uhitItr->outOfTimeAmplitude(6) ); // BX+1

    }
  }

  DEFINE_ECALDQM_WORKER(TimingTask);
}
