///
/// \class l1t::Stage2Layer2DemuxTauAlgo
///
/// Description: demux tau algorithm
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2Layer2DemuxTauAlgo_h
#define Stage2Layer2DemuxTauAlgo_h

#include "DataFormats/L1Trigger/interface/Tau.h"

#include <vector>


namespace l1t {
    
  class Stage2Layer2DemuxTauAlgo { 
  public:
    virtual void processEvent(const std::vector<l1t::Tau> & inputTaus,
			      std::vector<l1t::Tau> & outputTaus) = 0;    

    virtual ~Stage2Layer2DemuxTauAlgo(){};
  }; 
  
} 

#endif
