/* class PFTau3ProngReco
 * EDProducer of the 
 * author: Ian M. Nugent
 * The idea of the fully reconstructing the tau using a kinematic fit comes from
 * Lars Perchalla and Philip Sauerland Theses under Achim Stahl supervision. This
 * code is a result of the continuation of this work by Ian M. Nugent and Vladimir Cherepanov.
 */


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h" 
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TauReco/interface/PFTau3ProngSummary.h"
#include "DataFormats/TauReco/interface/PFTau3ProngSummaryFwd.h"

#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include <memory>
#include <boost/foreach.hpp>
#include <TFormula.h>

#include <memory>

#include "RecoTauTag/ImpactParameter/interface/Particle.h"
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "RecoTauTag/ImpactParameter/interface/TrackParticle.h"
#include "RecoTauTag/ImpactParameter/interface/ParticleBuilder.h"
#include "RecoTauTag/ImpactParameter/interface/TauA1NuConstrainedFitter.h"
#include "RecoTauTag/ImpactParameter/interface/Chi2VertexFitter.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoTauTag/ImpactParameter/interface/PDGInfo.h"
#include "TLorentzVector.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace reco;
using namespace edm;
using namespace std;
using namespace tauImpactParameter;

class PFTau3ProngReco : public EDProducer {
 public:
  enum Alg{useKalmanFit=0, useTrackHelix};

  struct DiscCutPair{
    DiscCutPair():cutFormula_(0){}
    ~DiscCutPair(){delete cutFormula_;}
    edm::Handle<reco::PFTauDiscriminator> handle_;
    edm::InputTag inputTag_;
    double cut_;
    TFormula* cutFormula_;
  };
  typedef std::vector<DiscCutPair*> DiscCutPairVec;

  explicit PFTau3ProngReco(const edm::ParameterSet& iConfig);
  ~PFTau3ProngReco();
  virtual void produce(edm::Event&,const edm::EventSetup&);
 private:
  edm::InputTag PFTauTag_;
  edm::InputTag PFTauTIPTag_;
  int Algorithm_;
  DiscCutPairVec discriminators_;
  std::auto_ptr<StringCutObjectSelector<reco::PFTau> > cut_;
  int ndfPVT_;
  KinematicParticleVertexFitter kpvFitter_;
};

PFTau3ProngReco::PFTau3ProngReco(const edm::ParameterSet& iConfig):
  PFTauTag_(iConfig.getParameter<edm::InputTag>("PFTauTag")),
  PFTauTIPTag_(iConfig.getParameter<edm::InputTag>("PFTauTIPTag")),
  Algorithm_(iConfig.getParameter<int>("Algorithm")),
  ndfPVT_(iConfig.getUntrackedParameter("ndfPVT",(int)5))
{
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<edm::ParameterSet> discriminators =iConfig.getParameter<std::vector<edm::ParameterSet> >("discriminators");
  // Build each of our cuts
  BOOST_FOREACH(const edm::ParameterSet &pset, discriminators) {
    DiscCutPair* newCut = new DiscCutPair();
    newCut->inputTag_ = pset.getParameter<edm::InputTag>("discriminator");
    if ( pset.existsAs<std::string>("selectionCut") ) newCut->cutFormula_ = new TFormula("selectionCut", pset.getParameter<std::string>("selectionCut").data());
    else newCut->cut_ = pset.getParameter<double>("selectionCut");
    discriminators_.push_back(newCut);
  }
  // Build a string cut if desired
  if (iConfig.exists("cut")) cut_.reset(new StringCutObjectSelector<reco::PFTau>(iConfig.getParameter<std::string>( "cut" )));
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  produces<edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTau3ProngSummaryRef> > >();
  produces<PFTau3ProngSummaryCollection>("PFTau3ProngSummary");
}

PFTau3ProngReco::~PFTau3ProngReco(){

}

void PFTau3ProngReco::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
  // Obtain Collections
  edm::ESHandle<TransientTrackBuilder> transTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",transTrackBuilder);
  
  edm::Handle<std::vector<reco::PFTau> > Tau;
  iEvent.getByLabel(PFTauTag_,Tau);

  edm::Handle<edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef> > > TIPAV;
  iEvent.getByLabel(PFTauTIPTag_,TIPAV);

  auto_ptr<edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTau3ProngSummaryRef> > > AVPFTau3PS(new edm::AssociationVector<PFTauRefProd, std::vector<reco::PFTau3ProngSummaryRef> >(PFTauRefProd(Tau)));
  std::auto_ptr<PFTau3ProngSummaryCollection>  PFTau3PSCollection_out= std::auto_ptr<PFTau3ProngSummaryCollection>(new PFTau3ProngSummaryCollection());
  reco::PFTau3ProngSummaryRefProd PFTau3RefProd_out = iEvent.getRefBeforePut<reco::PFTau3ProngSummaryCollection>("PFTau3ProngSummary");

  // Load each discriminator
  BOOST_FOREACH(DiscCutPair *disc, discriminators_) {iEvent.getByLabel(disc->inputTag_, disc->handle_);}

  // For each Tau Run Algorithim 
  if(Tau.isValid()){
    for(reco::PFTauCollection::size_type iPFTau = 0; iPFTau < Tau->size(); iPFTau++) {
      reco::PFTauRef tau(Tau, iPFTau);
      reco::PFTau3ProngSummary PFTau3PS;
      ///////////////////////
      // Check if it passed all the discrimiantors
      bool passed(true); 
      BOOST_FOREACH(const DiscCutPair* disc, discriminators_) {
        // Check this discriminator passes
	bool passedDisc = true;
	if ( disc->cutFormula_ )passedDisc = (disc->cutFormula_->Eval((*disc->handle_)[tau]) > 0.5);
	else passedDisc = ((*disc->handle_)[tau] > disc->cut_);
        if ( !passedDisc ){passed = false; break;}
      }
      if (passed && cut_.get()){passed = (*cut_)(*tau);}
      if (passed){
	PDGInfo pdgInfo;
	const reco::PFTauTransverseImpactParameterRef theTIP=TIPAV->value(tau.key());
	const reco::VertexRef primaryVertex=theTIP->primaryVertex();
	/////////////////////////////////
	// Now compute the 3 prong Tau 
	bool SecondaryVtxOK(false);
	LorentzVectorParticle a1;
	if(theTIP->hasSecondaryVertex() && primaryVertex->ndof()>ndfPVT_){
	  const VertexRef secVtx=theTIP->secondaryVertex();
          GlobalPoint sv(secVtx->position().x(),secVtx->position().y(),secVtx->position().z());
	  double vtxchi2(0), vtxndf(1);
	  if(useKalmanFit==Algorithm_){
	    vtxchi2=secVtx->chi2();
	    vtxndf=secVtx->ndof();
	    const std::vector<reco::Track>& selectedTracks=secVtx->refittedTracks();
	    std::vector<reco::TransientTrack> transTrkVect;
	    for(unsigned int i = 0; i!=selectedTracks.size();i++) transTrkVect.push_back(transTrackBuilder->build(selectedTracks[i]));
	    KinematicParticleFactoryFromTransientTrack kinFactory;
	    float piMassSigma(1.e-6), piChi(0.0), piNdf(0.0);
	    std::vector<RefCountedKinematicParticle> pions;
	    for(unsigned int i = 0; i<transTrkVect.size();i++) pions.push_back(kinFactory.particle(transTrkVect[i],pdgInfo.pi_mass(),piChi,piNdf,sv,piMassSigma));	   
	    RefCountedKinematicTree jpTree = kpvFitter_.fit(pions);
	    jpTree->movePointerToTheTop();
	    const KinematicParameters parameters = jpTree->currentParticle()->currentState().kinematicParameters();
	    AlgebraicSymMatrix77 cov=jpTree->currentParticle()->currentState().kinematicParametersError().matrix();
	    // get pions
	    double c(0);
	    std::vector<reco::Track> Tracks;
	    std::vector<LorentzVectorParticle> ReFitPions;
	    for(unsigned int i=0;i<transTrkVect.size();i++){
	      c+=transTrkVect[i].charge();
	      ReFitPions.push_back(ParticleBuilder::createLorentzVectorParticle(transTrkVect[i],*secVtx,true,true));
	    }
	    // now covert a1 into LorentzVectorParticle
	    TVectorT<double>    a1_par(LorentzVectorParticle::NLorentzandVertexPar);
	    TMatrixTSym<double> a1_cov(LorentzVectorParticle::NLorentzandVertexPar);
	    for(int i = 0; i<LorentzVectorParticle::NLorentzandVertexPar; i++){
	      a1_par(i)=parameters(i);
	      for(int j = 0; j<LorentzVectorParticle::NLorentzandVertexPar; j++){
		a1_cov(i,j)=cov(i,j);
	      } 
	    }
	    a1=LorentzVectorParticle(a1_par,a1_cov,abs(PdtPdgMini::a_1_plus)*c,c,transTrackBuilder->field()->inInverseGeV(sv).z());
	    SecondaryVtxOK=true;
	    PFTau3PS=reco::PFTau3ProngSummary(theTIP,a1.p4(),vtxchi2,vtxndf);
	  }
	  else if(useTrackHelix==Algorithm_){
	    // use Track Helix
	    std::vector<TrackParticle> pions;
	    GlobalPoint pvpoint(primaryVertex->position().x(),primaryVertex->position().y(),primaryVertex->position().z());
	    const std::vector<edm::Ptr<reco::PFCandidate> > cands = tau->signalPFChargedHadrCands();
	    for (std::vector<edm::Ptr<reco::PFCandidate> >::const_iterator iter = cands.begin(); iter!=cands.end(); ++iter) {
	      if(iter->get()->trackRef().isNonnull()){
		reco::TransientTrack transTrk=transTrackBuilder->build(iter->get()->trackRef());
		pions.push_back(ParticleBuilder::createTrackParticle(transTrk,pvpoint,true,true));
	      }
	      else if(iter->get()->gsfTrackRef().isNonnull()){
		//reco::TransientTrack transTrk=transTrackBuilder->build(iter->get()->gsfTrackRef());
		//pions.push_back(ParticleBuilder::CreateTrackParticle(transTrk,pvpoint,true,true));
	      }
	    }
	    TVector3 pv(secVtx->position().x(),secVtx->position().y(),secVtx->position().z());
	    Chi2VertexFitter chi2v(pions,pv);
	    SecondaryVtxOK=chi2v.fit();
	    double c(0); for(unsigned int i=0;i<pions.size();i++){c+=pions[i].charge();}
	    int pdgid=abs(PdtPdgMini::a_1_plus)*c;
	    a1=chi2v.getMother(pdgid);
	    PFTau3PS =reco::PFTau3ProngSummary(theTIP,a1.p4(),vtxchi2,vtxndf);
	  }
	}
	if(SecondaryVtxOK){
	  // Tau Solver
	  TVector3 pv(primaryVertex->position().x(),primaryVertex->position().y(),primaryVertex->position().z());
	  TMatrixTSym<double> pvcov(LorentzVectorParticle::NVertex);
	  math::Error<LorentzVectorParticle::NVertex>::type pvCov;
	  primaryVertex->fill(pvCov);
	  for(int i = 0; i<LorentzVectorParticle::NVertex; i++){
	    for(int j = 0; j<LorentzVectorParticle::NVertex; j++){
	      pvcov(i,j)=pvCov(i,j);
	    }
	  }
	  for(unsigned int i=0; i<PFTau3ProngSummary::nsolutions;i++){
	    TauA1NuConstrainedFitter TauA1NU(i,a1,pv,pvcov);
	    bool isFitOK=TauA1NU.fit();
	    if(isFitOK){
	      LorentzVectorParticle theTau=TauA1NU.getMother();
	      std::vector<LorentzVectorParticle> daughter=TauA1NU.getRefitDaughters();
	      std::vector<TLorentzVector> daughter_p4;
	      std::vector<int> daughter_charge,daughter_PDGID;
	      for(unsigned int d=0;d<daughter.size();d++){
		daughter_p4.push_back(daughter[d].p4());
		daughter_charge.push_back((int)daughter[d].charge());
		daughter_PDGID.push_back(daughter[d].pdgId());
	      }
	      PFTau3PS.AddSolution(i,theTau.p4(),daughter_p4,daughter_charge,daughter_PDGID,(isFitOK),0.0,-999);
	    }
	  }
	}
      }
      reco::PFTau3ProngSummaryRef PFTau3PSRef=reco::PFTau3ProngSummaryRef(PFTau3RefProd_out,PFTau3PSCollection_out->size());
      PFTau3PSCollection_out->push_back(PFTau3PS);
      AVPFTau3PS->setValue(iPFTau,PFTau3PSRef);
    }
  }
  iEvent.put(PFTau3PSCollection_out,"PFTau3ProngSummary");
  iEvent.put(AVPFTau3PS);
}

DEFINE_FWK_MODULE(PFTau3ProngReco);
