///
/// \class l1t::Stage2Layer2DemuxEGAlgo
///
/// Description: demux EG algorithm 
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2Layer2DemuxEGAlgo_h
#define Stage2Layer2DemuxEGAlgo_h

#include "DataFormats/L1Trigger/interface/EGamma.h"


#include <vector>


namespace l1t {
    
  class Stage2Layer2DemuxEGAlgo { 
  public:
    virtual void processEvent(const std::vector<l1t::EGamma> & inputEgammas, 
			      std::vector<l1t::EGamma> & outputEgammas) = 0;    

    virtual ~Stage2Layer2DemuxEGAlgo(){};
  }; 
  
} 

#endif
