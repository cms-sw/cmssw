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
  
    const bool HcalLooseHaloId()const { return  HcalHaloReport.size() ?  HcalHaloReport[0] : false ; }
    const bool HcalTightHaloId()const { return  HcalHaloReport.size() > 1 ? HcalHaloReport[1] : false ; }
    
    const bool EcalLooseHaloId()const { return  EcalHaloReport.size() ?  EcalHaloReport[0] : false ; }
    const bool EcalTightHaloId()const { return  EcalHaloReport.size() > 1 ? EcalHaloReport[1] : false ; }

    const bool CSCLooseHaloId() const { return CSCHaloReport.size() ? CSCHaloReport[0] : false ; }
    const bool CSCTightHaloId() const { return CSCHaloReport.size() > 1 ? CSCHaloReport[1] : false ; }
    
    const bool GlobalLooseHaloId() const { return GlobalHaloReport.size() ? GlobalHaloReport[0] : false ; }
    const bool GlobalTightHaloId() const { return GlobalHaloReport.size() > 1 ? GlobalHaloReport[1] : false ; }

    const bool EventSmellsLikeHalo() const { return HcalLooseHaloId() || EcalLooseHaloId() || CSCLooseHaloId() || GlobalLooseHaloId() ; }
    const bool LooseId() const { return EventSmellsLikeHalo(); }
    const bool TightId() const { return HcalTightHaloId() || EcalTightHaloId() || CSCTightHaloId() || GlobalTightHaloId() ; }
    const bool ExtremeTightId ()const { return  GlobalTightHaloId() ; }

    //  Getters
    std::vector<char>& GetHcalHaloReport(){return HcalHaloReport;}
    const std::vector<char>& GetHcalHaloReport() const { return HcalHaloReport;}
    
    std::vector<char>& GetEcalHaloReport(){return EcalHaloReport;}
    const std::vector<char>& GetEcalHaloReport() const {return EcalHaloReport;}

    std::vector<char>& GetCSCHaloReport() { return CSCHaloReport ;}
    const std::vector<char>& GetCSCHaloReport() const { return CSCHaloReport ; }

    std::vector<char>& GetGlobalHaloReport() { return GlobalHaloReport ;} 
    const std::vector<char>& GetGlobalHaloReport() const {return GlobalHaloReport ; }

    std::vector<int>& GetHcaliPhiSuspects() { return HcaliPhiSuspects ; }
    const std::vector<int>& GetHcaliPhiSuspects() const { return HcaliPhiSuspects ; }

    std::vector<int>& GetEcaliPhiSuspects() { return EcaliPhiSuspects ; }
    const std::vector<int>& GetEcaliPhiSuspects() const { return EcaliPhiSuspects ; }

    std::vector<int>& GetGlobaliPhiSuspects() { return GlobaliPhiSuspects ;}
    const std::vector<int>& GetGlobaliPhiSuspects() const { return GlobaliPhiSuspects ;}

  private: 
    std::vector<char>  HcalHaloReport;
    std::vector<char>  EcalHaloReport;
    std::vector<char>  CSCHaloReport;
    std::vector<char>  GlobalHaloReport;

    std::vector<int>  HcaliPhiSuspects;
    std::vector<int>  EcaliPhiSuspects;
    std::vector<int>  GlobaliPhiSuspects;

  };
  
}


#endif
