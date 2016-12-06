#include "RecoLocalFastTime/FTLCommonAlgos/interface/FTLRecHitAlgoBase.h"

class FTLSimpleRecHitAlgo : public FTLRecHitAlgoBase {
 public:
  /// Constructor
  FTLSimpleRecHitAlgo( const edm::ParameterSet& conf,
                       edm::ConsumesCollector& sumes ) : 
  FTLRecHitAlgoBase( conf, sumes ) { }

  /// Destructor
  virtual ~FTLSimpleRecHitAlgo() { }

  /// get event and eventsetup information
  virtual void getEvent(const edm::Event&) override final {}
  virtual void getEventSetup(const edm::EventSetup&) override final {}

  /// make the rec hit
  virtual FTLRecHit makeRecHit(const FTLUncalibratedRecHit& dataFrame, uint32_t& flags ) const override final;

 private:  

};

FTLRecHit 
FTLSimpleRecHitAlgo::makeRecHit(const FTLUncalibratedRecHit& dataFrame, uint32_t& flags ) const { 
  return FTLRecHit(); 
}
