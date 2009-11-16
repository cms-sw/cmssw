#ifndef DATAFORMATS_METRECO_GLOBALHALODATA_H
#define DATAFORMATS_METRECO_GLOBALHALODATA_H
/*
  [class]:  GlobalHaloData
  [authors]: R. Remington, The University of Florida
  [description]: Container class to store global beam halo data synthesized from EcalHaloData, HcalHaloData, and CSCHaloData. Also stores some variables relevant to MET for possible corrections.
  [date]: October 15, 2009
*/
#include "DataFormats/METReco/interface/EcalHaloData.h"
#include "DataFormats/METReco/interface/HcalHaloData.h"
#include "DataFormats/METReco/interface/CSCHaloData.h"
#include "DataFormats/METReco/interface/PhiWedge.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/MET.h"

namespace reco {
  class GlobalHaloData {

  public:
    // Constructor
    GlobalHaloData();
    // Destructor
    ~GlobalHaloData(){}
    
    //A good cut-variable to isolate halo events with no overlapping physics from collisions 
    float METOverSumEt() const {return METOverSumEt_;}
    
    //Correction to CaloMET x-component 
    float DeltaMEx() const {return dMEx_;}

    //Correction to CaloMET y-component
    float DeltaMEy() const {return dMEy_;}

    //Correction to SumEt
    float DeltaSumEt() const { return dSumEt_;}

    //Get CaloMET Object corrected for BeamHalo
    reco::CaloMET GetCorrectedCaloMET(const reco::CaloMET& RawMET) const ;

    std::vector<PhiWedge>& GetMatchedHcalPhiWedges(){return HcalPhiWedges;}
    const std::vector<PhiWedge>& GetMatchedHcalPhiWedges() const {return HcalPhiWedges;}

    std::vector<PhiWedge>& GetMatchedEcalPhiWedges(){return EcalPhiWedges;}
    const std::vector<PhiWedge>& GetMatchedEcalPhiWedges() const {return EcalPhiWedges;}
    
    //Setters
    void SetMETOverSumEt(float x){METOverSumEt_=x;}
    void SetMETCorrections(float x, float y) { dMEx_ =x ; dMEy_ = y;}
  private:
    
    float METOverSumEt_;
    float dMEx_;
    float dMEy_;
    float dSumEt_;
     
    std::vector<PhiWedge> HcalPhiWedges;
    std::vector<PhiWedge> EcalPhiWedges;
  };
}
#endif
