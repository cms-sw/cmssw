#ifndef DataFormats_DeepFormats_ChargedCandidateFeatures_h
#define DataFormats_DeepFormats_ChargedCandidateFeatures_h

namespace deep {

class ChargedCandidateFeatures {

  public:

    float pt;
    float eta;
    float phi;
    float ptrel;
    float erel;
    float phirel;
    float etarel;
    float deltaR;
    float puppiw;
    float VTX_ass;

    float fromPV;

    float vertexChi2; 
    float vertexNdof;
    float vertexNormalizedChi2;
    float vertex_rho;
    float vertex_phirel;
    float vertex_etarel;
    float vertexRef_mass;

    // covariance
    float  dz;
    float  dxy;

    float  dxyerrinv;
    float  dxysig;

    float  dptdpt;
    float  detadeta;
    float  dphidphi;
    float  dxydxy;
    float  dzdz;
    float  dxydz;
    float  dphidxy;
    float  dlambdadz;

    float BtagPf_trackMomentum;
    float BtagPf_trackEta;
    float BtagPf_trackEtaRel;
    float BtagPf_trackPtRel;
    float BtagPf_trackPPar;
    float BtagPf_trackDeltaR;
    float BtagPf_trackPtRatio;
    float BtagPf_trackPParRatio;
    float BtagPf_trackSip3dVal;
    float BtagPf_trackSip3dSig;
    float BtagPf_trackSip2dVal;
    float BtagPf_trackSip2dSig;

    float BtagPf_trackDecayLen;

    float BtagPf_trackJetDistVal;
    float BtagPf_trackJetDistSig;

    // ID, skipped "charged hadron" as that is true if now the other
    // TODO (comment of Markus Stoye) add reco information
    float isMu; // pitty that the quality is missing
    float isEl; // pitty that the quality is missing
    float charge;

    // track quality
    float lostInnerHits;
    float chi2;
    float quality;

    float drminsv;

};

}

#endif //DataFormats_DeepFormats_ChargedCandidateFeatures_h
