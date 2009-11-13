#ifndef DATAFORMATS_METRECO_BEAMHALOSUMMARY_H
#define DATAFORMATS_METRECO_BEAMHALOSUMMARY_H

/*
  [class]:  BeamHaloSummary
  [authors]: R. Remington, The University of Florida
  [description]: Container class that stores Ecal,CSC,Hcal, and Global BeamHalo data
  [date]: October 15, 2009
*/

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/METReco/interface/CSCHaloData.h"
#include "DataFormats/METReco/interface/EcalHaloData.h"
#include "DataFormats/METReco/interface/HcalHaloData.h" 
#include "DataFormats/METReco/interface/GlobalHaloData.h" 

namespace reco {
  class BeamHaloInfoProducer;
  
  class BeamHaloSummary{
    friend class reco::BeamHaloInfoProducer; 
  public:
    
    //constructors
    BeamHaloSummary();
    BeamHaloSummary(CSCHaloData& csc, EcalHaloData& ecal, HcalHaloData& hcal, GlobalHaloData& global);
    
    //destructor
    virtual ~BeamHaloSummary() {}
  
  private: 
    
  };
  
}


#endif
