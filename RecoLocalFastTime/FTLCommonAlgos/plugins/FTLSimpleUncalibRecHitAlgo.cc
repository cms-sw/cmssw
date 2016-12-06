#include "RecoLocalFastTime/FTLCommonAlgos/interface/FTLUncalibratedRecHitAlgoBase.h"

class FTLSimpleUncalibRecHitAlgo : public FTLUncalibratedRecHitAlgoBase {
 public:
  /// Constructor
  FTLSimpleUncalibRecHitAlgo( const edm::ParameterSet& conf,
                              edm::ConsumesCollector& sumes ) : 
  FTLUncalibratedRecHitAlgoBase( conf, sumes ) { }

  /// Destructor
  virtual ~FTLSimpleUncalibRecHitAlgo() { }

  /// get event and eventsetup information
  virtual void getEvent(const edm::Event&) override final {}
  virtual void getEventSetup(const edm::EventSetup&) override final {}

  /// make the rec hit
  virtual FTLUncalibratedRecHit makeRecHit(const FTLDataFrame& dataFrame ) const override final;

 private:  

};

FTLUncalibratedRecHit 
FTLSimpleUncalibRecHitAlgo::makeRecHit(const FTLDataFrame& dataFrame ) const { 
  return FTLUncalibratedRecHit(); 
}
