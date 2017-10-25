#ifndef __PhysicsTools_PatAlgos_MuonMvaEstimator__
#define __PhysicsTools_PatAlgos_MuonMvaEstimator__
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
		    const reco::JetCorrector* correctorL1=nullptr,
		    const reco::JetCorrector* correctorL1L2L3Res=nullptr);
    float mva() const {return mva_;}
    float jetPtRatio() const {return jetPtRatio_;}
    float jetPtRel() const {return jetPtRel_;}
  private:
    TMVA::Reader tmvaReader_;
    bool initialized_;
    float mva_;
    float dRmax_;
    
    /// MVA VAriables
    float pt_;
    float eta_;
    float jetNDauCharged_;
    float miniRelIsoCharged_;
    float miniRelIsoNeutral_;
    float jetPtRel_;
    float jetPtRatio_;
    float jetBTagCSV_;
    float sip_;
    float log_abs_dxyBS_; 
    float log_abs_dzPV_;            
    float segmentCompatibility_;
  };
}
#endif
