// -*- C++ -*-
//
// input: L1 TkTracks and  L1MuonParticleExtended (standalone with component details)
// match the two and produce a collection of L1TkGlbMuonParticle
// eventually, this should be made modular and allow to swap out different algorithms


#include "DataFormats/L1TrackTrigger/interface/L1TkGlbMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkGlbMuonParticleFwd.h"


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "TMath.h"

// system include files
#include <memory>
#include <string>


using namespace l1t;

//
// class declaration
//

class L1TkGlbMuonProducer : public edm::EDProducer {
public:
  
  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef std::vector< L1TTTrackType > L1TTTrackCollectionType;
  
  struct PropState { //something simple, imagine it's hardware emulation
    PropState() : 
      pt(-99),  eta(-99), phi(-99),
      sigmaPt(-99),  sigmaEta(-99), sigmaPhi(-99),
      valid(false) {}
    float pt;
    float eta;
    float phi;
    float sigmaPt;
    float sigmaEta;
    float sigmaPhi;
    bool valid;

  };

  explicit L1TkGlbMuonProducer(const edm::ParameterSet&);
  ~L1TkGlbMuonProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  PropState propagateToGMT(const L1TTTrackType& l1tk) const;
  
  float ETAMIN_;
  float ETAMAX_;
  float ZMAX_;             // |z_track| < ZMAX in cm
  float CHI2MAX_;
  float PTMINTRA_;
  //  float DRmax_;
  int nStubsmin_ ;         // minimum number of stubs   
  //  bool closest_ ;
  bool correctGMTPropForTkZ_;

  bool use5ParameterFit_;

  const edm::EDGetTokenT< MuonBxCollection > muToken;
  const edm::EDGetTokenT< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
} ;


//
// constructors and destructor
//
L1TkGlbMuonProducer::L1TkGlbMuonProducer(const edm::ParameterSet& iConfig) :
  muToken(consumes< MuonBxCollection >(iConfig.getParameter<edm::InputTag>("L1MuonInputTag"))),
  trackToken(consumes< std::vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<edm::InputTag>("L1TrackInputTag")))
{

   ETAMIN_ = (float)iConfig.getParameter<double>("ETAMIN");
   ETAMAX_ = (float)iConfig.getParameter<double>("ETAMAX");
   ZMAX_ = (float)iConfig.getParameter<double>("ZMAX");
   CHI2MAX_ = (float)iConfig.getParameter<double>("CHI2MAX");
   PTMINTRA_ = (float)iConfig.getParameter<double>("PTMINTRA");
   //   DRmax_ = (float)iConfig.getParameter<double>("DRmax");
   nStubsmin_ = iConfig.getParameter<int>("nStubsmin");
   //   closest_ = iConfig.getParameter<bool>("closest");

   correctGMTPropForTkZ_ = iConfig.getParameter<bool>("correctGMTPropForTkZ");

   use5ParameterFit_     = iConfig.getParameter<bool>("use5ParameterFit");
   produces<L1TkGlbMuonParticleCollection>();
}

L1TkGlbMuonProducer::~L1TkGlbMuonProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkGlbMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{  
  
  std::unique_ptr<L1TkGlbMuonParticleCollection> tkMuons(new L1TkGlbMuonParticleCollection);
  
  // the L1EGamma objects
  edm::Handle<MuonBxCollection> l1musH;
  iEvent.getByToken(muToken, l1musH);  
  const MuonBxCollection l1mus = (*l1musH.product());
  MuonBxCollection::const_iterator l1mu;


  // the L1Tracks
  edm::Handle<L1TTTrackCollectionType> l1tksH;
  iEvent.getByToken(trackToken, l1tksH);
  const L1TTTrackCollectionType& l1tks = (*l1tksH.product());

  int imu = 0;
  for (l1mu = l1mus.begin(0); l1mu != l1mus.end(0);  ++l1mu){ // considering BX = only
    edm::Ref< MuonBxCollection > l1muRef( l1musH, imu );
    imu++;

    float l1mu_eta = l1mu->eta();
    float l1mu_phi = l1mu->phi();
    
    float l1mu_feta = fabs( l1mu_eta );
    if (l1mu_feta < ETAMIN_) continue;
    if (l1mu_feta > ETAMAX_) continue;


    float drmin = 999;
    float ptmax = -1;
    if (ptmax < 0) ptmax = -1;	// dummy
    
    PropState matchProp;
    int match_idx = -1;
    int il1tk = -1;
   
    int nTracksMatch=0;

    for (auto l1tk : l1tks ){
      il1tk++;

      unsigned int nPars = 4;
      if (use5ParameterFit_) nPars = 5;
      float l1tk_pt = l1tk.getMomentum(nPars).perp();
      if (l1tk_pt < PTMINTRA_) continue;

      float l1tk_z  = l1tk.getPOCA(nPars).z();
      if (fabs(l1tk_z) > ZMAX_) continue;

      float l1tk_chi2 = l1tk.getChi2(nPars);
      if (l1tk_chi2 > CHI2MAX_) continue;
      
      int l1tk_nstubs = l1tk.getStubRefs().size();
      if ( l1tk_nstubs < nStubsmin_) continue;

      float l1tk_eta = l1tk.getMomentum(nPars).eta();
      float l1tk_phi = l1tk.getMomentum(nPars).phi();

      float dr2 = deltaR2(l1mu_eta, l1mu_phi, l1tk_eta, l1tk_phi);
      
      if (dr2 > 0.3) continue;
      nTracksMatch++;

      PropState pstate = propagateToGMT(l1tk);
      if (!pstate.valid) continue;

      float dr2prop = deltaR2(l1mu_eta, l1mu_phi, pstate.eta, pstate.phi);
      
      if (dr2prop < drmin){
	drmin = dr2prop;
	match_idx = il1tk;
	matchProp = pstate;
      }
    }// over l1tks

    
    LogDebug("MYDEBUG")<<"matching index is "<<match_idx;
    if (match_idx >= 0){
      const L1TTTrackType& matchTk = l1tks[match_idx];
      
      float etaCut = 3.*sqrt(l1mu->hwDEtaExtra()*l1mu->hwDEtaExtra() + matchProp.sigmaEta*matchProp.sigmaEta);
      float phiCut = 4.*sqrt(l1mu->hwDPhiExtra()*l1mu->hwDPhiExtra() + matchProp.sigmaPhi*matchProp.sigmaPhi);

      float dEta = std::abs(matchProp.eta - l1mu->eta());
      float dPhi = std::abs(deltaPhi(matchProp.phi, l1mu->phi()));

      LogDebug("MYDEBUG")<<"match details: prop "<<matchProp.pt<<" "<<matchProp.eta<<" "<<matchProp.phi
			 <<" mutk "<<l1mu->pt()<<" "<<l1mu->eta()<<" "<<l1mu->phi()<<" delta "<<dEta<<" "<<dPhi<<" cut "<<etaCut<<" "<<phiCut;

      if (dEta < etaCut && dPhi < phiCut){
	edm::Ptr< L1TTTrackType > l1tkPtr(l1tksH, match_idx);

	unsigned int nPars = 4;
	if (use5ParameterFit_) nPars = 5;
	auto p3 = matchTk.getMomentum(nPars);
	float p4e = sqrt(0.105658369*0.105658369 + p3.mag2() );

	math::XYZTLorentzVector l1tkp4(p3.x(), p3.y(), p3.z(), p4e);

	auto tkv3=matchTk.getPOCA(nPars);
	math::XYZPoint v3(tkv3.x(), tkv3.y(), tkv3.z());
	float trkisol = -999;

	L1TkGlbMuonParticle l1tkmu(l1tkp4, l1muRef, l1tkPtr, trkisol);

	// EP: add the zvtx information
	l1tkmu.setTrkzVtx( (float)tkv3.z() );
      l1tkmu.setdR(drmin);
      l1tkmu.setNTracksMatched(nTracksMatch);

	tkMuons->push_back(l1tkmu);
      }
    }

  }//over l1mus
  
  
  iEvent.put( std::move(tkMuons));
  
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkGlbMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


L1TkGlbMuonProducer::PropState L1TkGlbMuonProducer::propagateToGMT(const L1TkGlbMuonProducer::L1TTTrackType& tk) const {
  auto p3 = tk.getMomentum();
  float tk_pt = p3.perp();
  float tk_p = p3.mag();
  float tk_eta = p3.eta();
  float tk_aeta = std::abs(tk_eta);
  float tk_phi = p3.phi();
  float tk_q = tk.getRInv()>0? 1.: -1.;
  float tk_z  = tk.getPOCA().z();
  if (!correctGMTPropForTkZ_) tk_z = 0;

  L1TkGlbMuonProducer::PropState dest;
  if (tk_p<3.5 ) return dest;
  if (tk_aeta <1.1 && tk_pt < 3.5) return dest;
  if (tk_aeta > 2.5) return dest;

  //0th order:
  dest.valid = true;

  float dzCorrPhi = 1.;
  float deta = 0;
  float etaProp = tk_aeta;

  if (tk_aeta < 1.1){
    etaProp = 1.1;
    deta = tk_z/550./cosh(tk_aeta);
  } else {
    float delta = tk_z/850.; //roughly scales as distance to 2nd station
    if (tk_eta > 0) delta *=-1;
    dzCorrPhi = 1. + delta;

    float zOzs = tk_z/850.;
    if (tk_eta > 0) deta = zOzs/(1. - zOzs);
    else deta = zOzs/(1.+zOzs);
    deta = deta*tanh(tk_eta);
  }
  float resPhi = tk_phi - 1.464*tk_q*cosh(1.7)/cosh(etaProp)/tk_pt*dzCorrPhi - M_PI/144.;
  if (resPhi > M_PI) resPhi -= 2.*M_PI;
  if (resPhi < -M_PI) resPhi += 2.*M_PI;

  dest.eta = tk_eta + deta;
  dest.phi = resPhi;
  dest.pt = tk_pt; //not corrected for eloss

  dest.sigmaEta = 0.100/tk_pt; //multiple scattering term
  dest.sigmaPhi = 0.106/tk_pt; //need a better estimate for these
  return dest;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkGlbMuonProducer);



