#ifndef CaloStage1SingleTrackHI_h
#define CaloStage1SingleTrackHI_h

#include "L1Trigger/L1TCalorimeter/interface/CaloStage1TauAlgorithm.h"

//	This is the implementation of the CaloStage1TauAlgorithm abstract base class.
//	This class will be used to find sngle high pt tracks in heavy ion collisions.

namespace l1t {
  
  class CaloStage1SingleTrackHI : CaloStage1TauAlgorithm { 
  public:
    CaloStage1SingleTrackHI();
    virtual ~CaloStage1SingleTrackHI();
    virtual void processEvent(/*const std::vector<l1t::CaloStage1Cluster> & clusters,*/
			      const std::vector<l1t::CaloEmCand> & clusters,	
                              const std::vector<l1t::CaloRegion> & regions,
                              std::vector<l1t::Tau> & taus);    
  
  private:
    double regionLSB_;
 }; 
   
}
#endif 
