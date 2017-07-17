///
/// \class l1t::Stage2Layer2DemuxSumsAlgo
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2Layer2DemuxSumsAlgo_h
#define Stage2Layer2DemuxSumsAlgo_h

#include "DataFormats/L1Trigger/interface/EtSum.h"

#include <vector>

namespace l1t {
    
  class Stage2Layer2DemuxSumsAlgo { 
  public:
    virtual void processEvent(const std::vector<l1t::EtSum> & inputSums,
			      std::vector<l1t::EtSum> & outputSums) = 0;    

    virtual ~Stage2Layer2DemuxSumsAlgo(){};

  }; 
  
} 

#endif
