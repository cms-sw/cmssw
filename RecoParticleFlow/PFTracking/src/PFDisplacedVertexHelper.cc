#include "RecoParticleFlow/PFTracking/interface/PFDisplacedVertexHelper.h"

#include "TVector3.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "TMath.h"

using namespace std;
using namespace reco;


const double PFDisplacedVertexHelper::pion_mass2 = 0.0194;
const double PFDisplacedVertexHelper::muon_mass2 = 0.106*0.106;
const double PFDisplacedVertexHelper::proton_mass2 = 0.938*0.938;

//for debug only 
//#define PFLOW_DEBUG

PFDisplacedVertexHelper::PFDisplacedVertexHelper() : 
  tracksSelector_(),
  vertexIdentifier_(),      
  pvtx_(math::XYZPoint(0,0,0)) {}

PFDisplacedVertexHelper::~PFDisplacedVertexHelper() {}

void PFDisplacedVertexHelper::setPrimaryVertex(
					       edm::Handle< reco::VertexCollection > mainVertexHandle, 
					       edm::Handle< reco::BeamSpot > beamSpotHandle){

  const math::XYZPoint beamSpot = beamSpotHandle.isValid() ? 
    math::XYZPoint(beamSpotHandle->x0(), beamSpotHandle->y0(), beamSpotHandle->z0()) : 
    math::XYZPoint(0, 0, 0);

  // The primary vertex is taken from the refitted list, 
  // if does not exist from the average offline beam spot position  
  // if does not exist (0,0,0) is used
  pvtx_ = mainVertexHandle.isValid() ? 
    math::XYZPoint(mainVertexHandle->begin()->x(), 
		   mainVertexHandle->begin()->y(), 
		   mainVertexHandle->begin()->z()) :
    beamSpot;
}

bool 
PFDisplacedVertexHelper::isTrackSelected(const reco::Track& trk, 
					 const reco::PFDisplacedVertex::VertexTrackType vertexTrackType) const {
    

  if (!tracksSelector_.selectTracks()) return true;

  bool isGoodTrack = false;
    
  bool isHighPurity = trk.quality( trk.qualityByName(tracksSelector_.quality().data()) );
        
  double nChi2  = trk.normalizedChi2(); 
  double pt = trk.pt();
  int nHits = trk.numberOfValidHits();

  bool bIsPrimary = 
    (
     (vertexTrackType == reco::PFDisplacedVertex::T_TO_VERTEX)
     ||
     (vertexTrackType == reco::PFDisplacedVertex::T_MERGED)
     );  
    
  
  if (bIsPrimary) {
    // Primary or merged track selection
    isGoodTrack = 
      ( ( nChi2 > tracksSelector_.nChi2_min()
          && nChi2 <  tracksSelector_.nChi2_max())
	|| isHighPurity )
      && pt >  tracksSelector_.pt_min();
  } else {
    // Secondary tracks selection
    int nOuterHits = trk.trackerExpectedHitsOuter().numberOfHits();

    double dxy = trk.dxy(pvtx_);
      
    isGoodTrack =   
      nChi2 <  tracksSelector_.nChi2_max() 
      && pt >  tracksSelector_.pt_min() 
      && fabs(dxy) >  tracksSelector_.dxy_min() 
      && nHits >=  tracksSelector_.nHits_min() 
      && nOuterHits <=  tracksSelector_.nOuterHits_max();
      
  }
    
  return isGoodTrack;
      
}

reco::PFDisplacedVertex::VertexType 
PFDisplacedVertexHelper::identifyVertex(const reco::PFDisplacedVertex& v) const{

  if (!vertexIdentifier_.identifyVertices()) return PFDisplacedVertex::ANY;

  PFDisplacedVertex::M_Hypo massElec = PFDisplacedVertex::M_MASSLESS;
  PFDisplacedVertex::M_Hypo massPion = PFDisplacedVertex::M_PION;

  math::XYZTLorentzVector mom_ee = v.secondaryMomentum(massElec, true);
  math::XYZTLorentzVector mom_pipi = v.secondaryMomentum(massPion, true);

  // ===== (1) Identify fake and looper vertices ===== //

  double ang = v.angle_io();
  double pt_ee = mom_ee.Pt();
  double eta_vtx = v.position().eta();

  //cout << "Angle = " << ang << endl;

  bool bDirectionFake = ang > vertexIdentifier_.angle_max();
  bool bLowPt = pt_ee < vertexIdentifier_.pt_min();
  bool bLooperEta = fabs(eta_vtx) < vertexIdentifier_.looper_eta_max();

  bool isFake = bDirectionFake ||(bLowPt && !bLooperEta); 
  bool isLooper = !bDirectionFake && bLowPt && bLooperEta;

  if (isFake) return  PFDisplacedVertex::FAKE;
  if (isLooper) return PFDisplacedVertex::LOOPER;

  // ===== (2) Identify Decays and Conversions ===== //

  int c1 = v.originalTrack(v.refittedTracks()[0])->charge();
  int c2 = v.originalTrack(v.refittedTracks()[1])->charge();
  double mass_ee = mom_ee.M();

  int nTracks = v.nTracks();
  int nSecondaryTracks = v.nSecondaryTracks();
  bool bPrimaryTracks = v.isTherePrimaryTracks();
  bool bMergedTracks = v.isThereMergedTracks();

  bool bPair = (nTracks == nSecondaryTracks) && (nTracks == 2);
  bool bOpposite = (c1*c2 < -0.1);
  bool bDirection = ang < vertexIdentifier_.angle_V0Conv_max();
  bool bConvMass = mass_ee < vertexIdentifier_.mConv_max();

  bool bV0Conv = bPair && bOpposite && bDirection;

  // If the basic configuration of conversions and V0 decays is respected
  // pair of secondary track with opposite charge and going in the right direction
  // the selection is then based on mass limits
  if (bV0Conv){

    // == (2.1) Identify Conversions == //

    bool isConv = bConvMass;

    if (isConv) return PFDisplacedVertex::CONVERSION_LOOSE;
 
    // == (2.2) Identify K0 == //

    double mass_pipi = mom_pipi.M();

    bool bK0Mass = 
      mass_pipi < vertexIdentifier_.mK0_max() 
      &&  mass_pipi > vertexIdentifier_.mK0_min();

    bool isK0 =  bK0Mass;

    if (isK0) return PFDisplacedVertex::K0_DECAY;

    // == (2.3) Identify Lambda == //

    int lambdaKind = lambdaCP(v); 
    

    bool isLambda = (lambdaKind == 1);
    bool isLambdaBar = (lambdaKind == -1);

    if (isLambda) return PFDisplacedVertex::LAMBDA_DECAY;
    if (isLambdaBar) return PFDisplacedVertex::LAMBDABAR_DECAY;

  }

  // == (2.4) Identify K- and K+ ==
  bool bK = 
    (nSecondaryTracks == 1) && bPrimaryTracks && !bMergedTracks
    && !bOpposite;

  if(bK){

    bool bKMass = isKaonMass(v);
 
    bool isKPlus = bKMass && c1 > 0;
    bool isKMinus = bKMass && c1 < 0;

    if(isKMinus) return PFDisplacedVertex::KMINUS_DECAY_LOOSE;
    if(isKPlus) return PFDisplacedVertex::KPLUS_DECAY_LOOSE;

  }

  // ===== (3) Identify Nuclears, Kinks and Remaining Fakes ===== //

  math::XYZTLorentzVector mom_prim = v.primaryMomentum(massPion, true);
  
  double p_prim = mom_prim.P();
  double p_sec = mom_pipi.P();
  double pt_prim = mom_prim.Pt();
  
  bool bLog = log10(p_prim/p_sec) > vertexIdentifier_.logPrimSec_min();
  bool bPtMin = pt_prim >  vertexIdentifier_.pt_kink_min();

  // A vertex with at least 3 tracks is considered as high purity nuclear interaction
  // the only exception is K- decay into 3 prongs. To be studied.
  bool isNuclearHighPurity = nTracks > 2 && mass_ee > vertexIdentifier_.mNucl_min(); 
  bool isFakeHighPurity = nTracks > 2 && mass_ee < vertexIdentifier_.mNucl_min();
  // Two secondary tracks with some minimal tracks angular opening are still accepted
  // as nuclear interactions
  bool isNuclearLowPurity = 
    (nTracks == nSecondaryTracks)
    && (nTracks == 2)
    && mass_ee > vertexIdentifier_.mNucl_min();

  bool isFakeNucl = 
    (nTracks == nSecondaryTracks)
    && (nTracks == 2)
    && mass_ee < vertexIdentifier_.mNucl_min();

  // Kinks: 1 primary + 1 secondary is accepted only if the primary tracks 
  // has more energy than the secondary and primary have some minimal pT 
  // to produce a nuclear interaction
  bool isNuclearKink = 
    (nSecondaryTracks == 1) && bPrimaryTracks && !bMergedTracks 
    && bLog && bPtMin;

  // Here some loopers may hide appearing in Particle Isolation plots. To be studied...
  bool isFakeKink = 
    ( (nSecondaryTracks == 1) && bMergedTracks && !bPrimaryTracks )
    ||
    ( (nSecondaryTracks == 1) && bPrimaryTracks && !bMergedTracks 
    && (!bLog || !bPtMin) );

  if (isNuclearHighPurity) return PFDisplacedVertex::NUCL;
  if (isNuclearLowPurity) return PFDisplacedVertex::NUCL_LOOSE;
  if (isFakeKink || isFakeNucl || isFakeHighPurity) return  PFDisplacedVertex::FAKE;
  if (isNuclearKink)  return  PFDisplacedVertex::NUCL_KINK;
  

  return PFDisplacedVertex::ANY;

}


int PFDisplacedVertexHelper::lambdaCP(const PFDisplacedVertex& v) const {

  int lambdaCP = 0;

  vector <Track> refittedTracks = v.refittedTracks();

  math::XYZTLorentzVector totalMomentumDcaRefit_lambda;
  math::XYZTLorentzVector totalMomentumDcaRefit_lambdabar;


  reco::Track tMomentumDcaRefit_0 = refittedTracks[0];
  reco::Track tMomentumDcaRefit_1 = refittedTracks[1];

  double mass2_0 = 0, mass2_1 = 0;

  int c1 = v.originalTrack(v.refittedTracks()[0])->charge();

  // --------------------------- lambda --------------------


  if (c1 > 0.1) mass2_0 = pion_mass2, mass2_1 = proton_mass2;
  else mass2_0 = proton_mass2, mass2_1 = pion_mass2;
      
  double E = sqrt(tMomentumDcaRefit_0.p()*tMomentumDcaRefit_0.p() + mass2_0);

  math::XYZTLorentzVector momentumDcaRefit_0(tMomentumDcaRefit_0.px(), tMomentumDcaRefit_0.py(), 
				    tMomentumDcaRefit_0.pz(), E); 



  E = sqrt(tMomentumDcaRefit_1.p()*tMomentumDcaRefit_1.p() + mass2_1);

  math::XYZTLorentzVector momentumDcaRefit_1(tMomentumDcaRefit_1.px(), tMomentumDcaRefit_1.py(), 
				    tMomentumDcaRefit_1.pz(), E); 


  totalMomentumDcaRefit_lambda = momentumDcaRefit_0 + momentumDcaRefit_1;



  // --------------------------- anti - lambda --------------------

  if (c1 > 0.1) mass2_1 = pion_mass2, mass2_0 = proton_mass2;
  else mass2_1 = proton_mass2, mass2_0 = pion_mass2;
  
      
  E = sqrt(tMomentumDcaRefit_0.p()*tMomentumDcaRefit_0.p() + mass2_0);

  math::XYZTLorentzVector momentumDcaRefit_01(tMomentumDcaRefit_0.px(), tMomentumDcaRefit_0.py(), 
				    tMomentumDcaRefit_0.pz(), E); 



  E = sqrt(tMomentumDcaRefit_1.p()*tMomentumDcaRefit_1.p() + mass2_1);

  math::XYZTLorentzVector momentumDcaRefit_11(tMomentumDcaRefit_1.px(), tMomentumDcaRefit_1.py(), 
				    tMomentumDcaRefit_1.pz(), E); 


  totalMomentumDcaRefit_lambdabar = momentumDcaRefit_0 + momentumDcaRefit_1;

  double mass_l    = totalMomentumDcaRefit_lambda.M();
  double mass_lbar = totalMomentumDcaRefit_lambdabar.M();

  if (
      mass_l < mass_lbar 
      && mass_l > vertexIdentifier_.mLambda_min()
      && mass_l < vertexIdentifier_.mLambda_max()) 
    lambdaCP = 1;
  else if (mass_lbar < mass_l 
	   && mass_lbar > vertexIdentifier_.mLambda_min()
	   && mass_lbar < vertexIdentifier_.mLambda_max()) 
    lambdaCP = -1;
  else 
    lambdaCP = 0;

  return lambdaCP;

}



bool PFDisplacedVertexHelper::isKaonMass(const PFDisplacedVertex& v) const {

  math::XYZVector  trkInit = v.refittedTracks()[1].momentum(), 
    trkFinal = v.refittedTracks()[0].momentum();

  if (v.trackTypes()[0] == PFDisplacedVertex::T_TO_VERTEX)
    trkInit = v.refittedTracks()[0].momentum(),
      trkFinal =  v.refittedTracks()[1].momentum();


    math::XYZVector trkNeutre(trkInit.x()-trkFinal.x(),  trkInit.y()-trkFinal.y(),
			  trkInit.z()-trkFinal.z());

    double Ec = sqrt(muon_mass2 + trkFinal.Mag2());
    double En = sqrt(0*0        + trkNeutre.Mag2());



    math::XYZTLorentzVectorD trkMuNu(trkInit.x(), trkInit.y(), trkInit.z(), Ec+En);
    double massMuNu = trkMuNu.M();

    bool bKMass = 
      massMuNu > vertexIdentifier_.mK_min() 
      && massMuNu < vertexIdentifier_.mK_max(); 

    return bKMass; 

}



void PFDisplacedVertexHelper::Dump(std::ostream& out) const {
    tracksSelector_.Dump();
    vertexIdentifier_.Dump();
    out << " pvtx_ = " << pvtx_ << std::endl;
  }
