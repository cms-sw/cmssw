#ifndef __RecoMuon_MuonIdentification_MuonMvaEstimator__
#define __RecoMuon_MuonIdentification_MuonMvaEstimator__
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "TMVA/Reader.h"

namespace reco {
  class JetCorrector;
}
namespace pat {
  class MuonMvaEstimator{
  public:
    MuonMvaEstimator();
    void initialize(std::string weightsfile, 
		    float dRmax);
    void computeMva(const pat::Muon& imuon,
		    const reco::Vertex& vertex,
		    const reco::JetTagCollection& bTags,
		    const reco::JetCorrector* correctorL1=0,
		    const reco::JetCorrector* correctorL1L2L3Res=0);
    float mva() const {return mva_;}
    float jetPtRatio() const {return jetPtRatio_;}
    float jetPtRel() const {return jetPtRel_;}
  private:
    TMVA::Reader tmvaReader_;
    bool initialized_;
    float mva_;
    float dRmax_;
    
    /// MVA VAriables
    Float_t pt_;
    Float_t eta_;
    Float_t jetNDauCharged_;
    Float_t miniRelIsoCharged_;
    Float_t miniRelIsoNeutral_;
    Float_t jetPtRel_;
    Float_t jetPtRatio_;
    Float_t jetBTagCSV_;
    Float_t sip_;
    Float_t log_abs_dxyBS_; 
    Float_t log_abs_dzPV_;            
    Float_t segmentCompatibility_;
  };
}
#endif
