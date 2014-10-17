///
/// \class l1t::Stage2Layer2DemuxJetAlgo
///
/// Description: interface for demux jet algorithm
///
/// Implementation:
///
/// \author: Jim Brooke - University of Bristol
///

//

#ifndef Stage2Layer2DemuxJetAlgo_h
#define Stage2Layer2DemuxJetAlgo_h

#include "DataFormats/L1Trigger/interface/Jet.h"

#include <vector>


namespace l1t {
    
  class Stage2Layer2DemuxJetAlgo { 
  public:
    virtual void processEvent(const std::vector<l1t::Jet> & inputJets,
			      std::vector<l1t::Jet> & outputJets) = 0;    

    virtual ~Stage2Layer2DemuxJetAlgo(){};

  }; 
  
} 

#endif
