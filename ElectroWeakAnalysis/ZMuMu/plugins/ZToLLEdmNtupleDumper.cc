/* \class ZToLLEdmNtupleDumper
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include <vector>

using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;
//using namespace pat;

class ZToLLEdmNtupleDumper : public edm::EDProducer {
public:
  ZToLLEdmNtupleDumper( const edm::ParameterSet & );
   
private:
  void produce( edm::Event &, const edm::EventSetup & );
  
  std::vector<std::string> zName_;
  std::vector<edm::InputTag> z_, zGenParticlesMatch_ ;
  edm::InputTag beamSpot_,  primaryVertices_;

  std::vector<double> ptThreshold_, etEcalThreshold_, etHcalThreshold_ ,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_; 
  std::vector<double> relativeIsolation_;
};

template<typename T>
  double isolation(const T * t, double ptThreshold, double etEcalThreshold, double etHcalThreshold , double dRVetoTrk, double dRTrk, double dREcal , double dRHcal,  double alpha, double beta, bool relativeIsolation) {
   // on 34X: const pat::IsoDeposit * trkIso = t->isoDeposit(pat::TrackIso);
   const pat::IsoDeposit * trkIso = t->trackerIsoDeposit();
   // on 34X const pat::IsoDeposit * ecalIso = t->isoDeposit(pat::EcalIso);
   const pat::IsoDeposit * ecalIso = t->ecalIsoDeposit();
//    on 34X const pat::IsoDeposit * hcalIso = t->isoDeposit(pat::HcalIso);   
    const pat::IsoDeposit * hcalIso = t->hcalIsoDeposit();

    Direction dir = Direction(t->eta(), t->phi());
    
    pat::IsoDeposit::AbsVetos vetosTrk;
    vetosTrk.push_back(new ConeVeto( dir, dRVetoTrk ));
    vetosTrk.push_back(new ThresholdVeto( ptThreshold ));
    
    pat::IsoDeposit::AbsVetos vetosEcal;
    vetosEcal.push_back(new ConeVeto( dir, 0.));
    vetosEcal.push_back(new ThresholdVeto( etEcalThreshold ));
    
    pat::IsoDeposit::AbsVetos vetosHcal;
    vetosHcal.push_back(new ConeVeto( dir, 0. ));
    vetosHcal.push_back(new ThresholdVeto( etHcalThreshold ));

    double isovalueTrk = (trkIso->sumWithin(dRTrk,vetosTrk));
    double isovalueEcal = (ecalIso->sumWithin(dREcal,vetosEcal));
    double isovalueHcal = (hcalIso->sumWithin(dRHcal,vetosHcal));
    

    double iso = alpha*( ((1+beta)/2*isovalueEcal) + ((1-beta)/2*isovalueHcal) ) + ((1-alpha)*isovalueTrk) ;
    if(relativeIsolation) iso /= t->pt();
    return iso;
  }


double candIsolation( const reco::Candidate* c, double ptThreshold, double etEcalThreshold, double etHcalThreshold , double dRVetoTrk, double dRTrk, double dREcal , double dRHcal,  double alpha, double beta, bool relativeIsolation) {
    const pat::Muon * mu = dynamic_cast<const pat::Muon *>(c);
    if(mu != 0) return isolation(mu, ptThreshold, etEcalThreshold, etHcalThreshold ,dRVetoTrk, dRTrk, dREcal , dRHcal,  alpha, beta, relativeIsolation);
    const pat::GenericParticle * trk = dynamic_cast<const pat::GenericParticle*>(c);
    if(trk != 0) return isolation(trk,  ptThreshold, etEcalThreshold, etHcalThreshold ,dRVetoTrk, dRTrk, dREcal , dRHcal,  alpha, beta, relativeIsolation);
    throw edm::Exception(edm::errors::InvalidReference) 
      << "Candidate daughter #0 is neither pat::Muons nor pat::GenericParticle\n";      
    return -1;
  }



ZToLLEdmNtupleDumper::ZToLLEdmNtupleDumper( const ParameterSet & cfg ) {
  string alias;
  vector<ParameterSet> psets = cfg.getParameter<vector<ParameterSet> > ( "zBlocks" );
  for( std::vector<edm::ParameterSet>::const_iterator i = psets.begin(); i != psets.end(); ++ i ) {
    string zName = i->getParameter<string>( "zName" );
    InputTag z =  i->getParameter<InputTag>( "z" );
    InputTag zGenParticlesMatch = i->getParameter<InputTag>( "zGenParticlesMatch" );
    beamSpot_ =  i->getParameter<InputTag>( "beamSpot" );
    primaryVertices_= i->getParameter<InputTag>( "primaryVertices" ) ;
    // InputTag isolations1 = i->getParameter<InputTag>( "isolations1" );
    //InputTag isolations2 = i->getParameter<InputTag>( "isolations2" );
    double ptThreshold = i->getParameter<double>("ptThreshold");
    double etEcalThreshold = i->getParameter<double>("etEcalThreshold");
    double etHcalThreshold= i->getParameter<double>("etHcalThreshold");
    double dRVetoTrk=i->getParameter<double>("deltaRVetoTrk");
    double dRTrk=i->getParameter<double>("deltaRTrk");
    double dREcal=i->getParameter<double>("deltaREcal");
    double dRHcal=i->getParameter<double>("deltaRHcal");
    double alpha=i->getParameter<double>("alpha");
    double beta=i->getParameter<double>("beta");
    bool relativeIsolation = i->getParameter<bool>("relativeIsolation");
    zName_.push_back( zName );
    z_.push_back( z );
    zGenParticlesMatch_.push_back( zGenParticlesMatch );
    //  isolations1_.push_back( isolations1 );
    //isolations2_.push_back( isolations2 );
    ptThreshold_.push_back( ptThreshold );
    etEcalThreshold_.push_back(etEcalThreshold);  
    etHcalThreshold_.push_back(etHcalThreshold);
    dRVetoTrk_.push_back(dRVetoTrk);
    dRTrk_.push_back(dRTrk);
    dREcal_.push_back(dREcal);
    dRHcal_.push_back(dRHcal);
    alpha_.push_back(alpha);
    beta_.push_back(beta);
    relativeIsolation_.push_back(relativeIsolation);
    produces<vector<unsigned int> >( alias = "EventNumber" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = "RunNumber" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = "LumiBlock" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Mass" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Pt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Eta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Phi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Y" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Pt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Pt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Eta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Eta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Phi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Phi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Iso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Iso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1TrkIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2TrkIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1EcalIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2EcalIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1HcalIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2HcalIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "VtxNormChi2" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau1NofHit" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau2NofHit" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau1NofHitTk" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau1NofHitSta" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau2NofHitTk" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau2NofHitSta" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau1NofMuChambers" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau2NofMuChambers" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau1NofMuMatches" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau2NofMuMatches" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Chi2" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Chi2" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1dB" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2dB" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1dxyFromBS" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2dxyFromBS" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1dzFromBS" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2dzFromBS" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1dxyFromPV" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2dxyFromPV" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1dzFromPV" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2dzFromPV" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TrueMass" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TruePt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TrueEta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TruePhi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "TrueY" ).setBranchAlias( alias );
  }
}



void ZToLLEdmNtupleDumper::produce( Event & evt, const EventSetup & ) {
  Handle<reco::BeamSpot> beamSpotHandle;
  if (!evt.getByLabel(beamSpot_, beamSpotHandle)) {
    std::cout << ">>> No beam spot found !!!"<<std::endl;
  }
  Handle<reco::VertexCollection> primaryVertices;  // Collection of primary Vertices
  if (!evt.getByLabel(primaryVertices_, primaryVertices)){
    std::cout << ">>> No primary verteces  found !!!"<<std::endl;
  }
  unsigned int size = z_.size();
  for( unsigned int c = 0; c < size; ++ c ) {
    Handle<CandidateView> zColl;
    evt.getByLabel( z_[c], zColl );
    bool isMCMatchTrue = false;  
    //if (zGenParticlesMatch_[c] != "")  isMCMatchTrue = true;     
    Handle<GenParticleMatch> zGenParticlesMatch;
    if (evt.getByLabel( zGenParticlesMatch_[c], zGenParticlesMatch )) {
      isMCMatchTrue=true;
    }
    unsigned int zSize = zColl->size();
    auto_ptr<vector<unsigned int> > event( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > run( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > lumi( new vector<unsigned int > );
    auto_ptr<vector<float> > zMass( new vector<float> );
    auto_ptr<vector<float> > zPt( new vector<float> );
    auto_ptr<vector<float> > zEta( new vector<float> );
    auto_ptr<vector<float> > zPhi( new vector<float> );
    auto_ptr<vector<float> > zY( new vector<float> );
    auto_ptr<vector<float> > zDau1Pt( new vector<float> );
    auto_ptr<vector<float> > zDau2Pt( new vector<float> );
    auto_ptr<vector<float> > zDau1Eta( new vector<float> );
    auto_ptr<vector<float> > zDau2Eta( new vector<float> );
    auto_ptr<vector<float> > zDau1Phi( new vector<float> );
    auto_ptr<vector<float> > zDau2Phi( new vector<float> );
    auto_ptr<vector<float> > zDau1Iso( new vector<float> );
    auto_ptr<vector<float> > zDau2Iso( new vector<float> );
    auto_ptr<vector<float> > zDau1TrkIso( new vector<float> );
    auto_ptr<vector<float> > zDau2TrkIso( new vector<float> );
    auto_ptr<vector<float> > zDau1EcalIso( new vector<float> );
    auto_ptr<vector<float> > zDau2EcalIso( new vector<float> );
    auto_ptr<vector<float> > zDau1HcalIso( new vector<float> );
    auto_ptr<vector<float> > zDau2HcalIso( new vector<float> );
    auto_ptr<vector<float> > vtxNormChi2( new vector<float> );
    auto_ptr<vector<unsigned int> > zDau1NofHit( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > zDau2NofHit( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > zDau1NofHitTk( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > zDau2NofHitTk( new vector<unsigned int> ); 
    auto_ptr<vector<unsigned int> > zDau1NofHitSta( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > zDau2NofHitSta( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > zDau1NofMuChambers( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > zDau2NofMuChambers( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > zDau1NofMuMatches( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > zDau2NofMuMatches( new vector<unsigned int> );
    auto_ptr<vector<float> > zDau1Chi2( new vector<float> );
    auto_ptr<vector<float> > zDau2Chi2( new vector<float> );
    auto_ptr<vector<float> > zDau1dB( new vector<float> );
    auto_ptr<vector<float> > zDau2dB( new vector<float> );
    auto_ptr<vector<float> > zDau1dxyFromBS( new vector<float> );
    auto_ptr<vector<float> > zDau2dxyFromBS( new vector<float> );
    auto_ptr<vector<float> > zDau1dzFromBS( new vector<float> );
    auto_ptr<vector<float> > zDau2dzFromBS( new vector<float> );
    auto_ptr<vector<float> > zDau1dxyFromPV( new vector<float> );
    auto_ptr<vector<float> > zDau2dxyFromPV( new vector<float> );
    auto_ptr<vector<float> > zDau1dzFromPV( new vector<float> );
    auto_ptr<vector<float> > zDau2dzFromPV( new vector<float> );
    auto_ptr<vector<float> > trueZMass( new vector<float> );
    auto_ptr<vector<float> > trueZPt( new vector<float> );
    auto_ptr<vector<float> > trueZEta( new vector<float> );
    auto_ptr<vector<float> > trueZPhi( new vector<float> );
    auto_ptr<vector<float> > trueZY( new vector<float> );
    event -> push_back(evt.id().event());
    run -> push_back(evt.id().run());
    lumi -> push_back(evt.luminosityBlock());
    for( unsigned int i = 0; i < zSize; ++ i ) {
      const Candidate & z = (*zColl)[ i ];
      CandidateBaseRef zRef = zColl->refAt(i);
      zMass->push_back( z.mass() );
      zPt->push_back( z.pt() );
      zEta->push_back( z.eta() );
      zPhi->push_back( z.phi() );
      zY->push_back( z.rapidity() );
      vtxNormChi2->push_back(z.vertexNormalizedChi2() );
      const Candidate * dau1 = z.daughter(0); 
      const Candidate * dau2 = z.daughter(1); 
      zDau1Pt->push_back( dau1->pt() );
      zDau2Pt->push_back( dau2->pt() );
      zDau1Eta->push_back( dau1->eta() );
      zDau2Eta->push_back( dau2->eta() );
      zDau1Phi->push_back( dau1->phi() );
      zDau2Phi->push_back( dau2->phi() );      
      if(!(dau1->hasMasterClone()&&dau2->hasMasterClone()))
	throw edm::Exception(edm::errors::InvalidReference) 
	  << "Candidate daughters have no master clone\n"; 
      const Candidate * m1 = &*dau1->masterClone(), * m2 = &*dau2->masterClone();
      // isolation as defined by us into the analyzer
      double iso1 = candIsolation(m1,ptThreshold_[c], etEcalThreshold_[c], etHcalThreshold_[c] ,dRVetoTrk_[c], dRTrk_[c], dREcal_[c] , dRHcal_[c],  alpha_[c], beta_[c], relativeIsolation_[c]);
      double iso2 = candIsolation(m2,ptThreshold_[c], etEcalThreshold_[c], etHcalThreshold_[c] ,dRVetoTrk_[c], dRTrk_[c], dREcal_[c] , dRHcal_[c],  alpha_[c], beta_[c], relativeIsolation_[c] );
      // tracker isolation : alpha =0
      double trkIso1 = candIsolation(m1,ptThreshold_[c], etEcalThreshold_[c], etHcalThreshold_[c] ,dRVetoTrk_[c], dRTrk_[c], dREcal_[c] , dRHcal_[c], 0.0, beta_[c], relativeIsolation_[c]);
      double trkIso2 = candIsolation(m2,ptThreshold_[c], etEcalThreshold_[c], etHcalThreshold_[c] ,dRVetoTrk_[c], dRTrk_[c], dREcal_[c] , dRHcal_[c],  0.0, beta_[c], relativeIsolation_[c] );
      // ecal isolation : alpha =1, beta =1
      double ecalIso1 = candIsolation(m1,ptThreshold_[c], etEcalThreshold_[c], etHcalThreshold_[c] ,dRVetoTrk_[c], dRTrk_[c], dREcal_[c] , dRHcal_[c], 1.0, 1.0, relativeIsolation_[c]);
      double ecalIso2 = candIsolation(m2,ptThreshold_[c], etEcalThreshold_[c], etHcalThreshold_[c] ,dRVetoTrk_[c], dRTrk_[c], dREcal_[c] , dRHcal_[c],  1.0, 1.0, relativeIsolation_[c] );
      // hcal isolation : alpha =1, beta =-1
      double hcalIso1 = candIsolation(m1,ptThreshold_[c], etEcalThreshold_[c], etHcalThreshold_[c] ,dRVetoTrk_[c], dRTrk_[c], dREcal_[c] , dRHcal_[c], 1.0, -1.0, relativeIsolation_[c]);
      double hcalIso2 = candIsolation(m2,ptThreshold_[c], etEcalThreshold_[c], etHcalThreshold_[c] ,dRVetoTrk_[c], dRTrk_[c], dREcal_[c] , dRHcal_[c],  1.0, -1.0, relativeIsolation_[c] );
      zDau1Iso->push_back( iso1 );
      zDau2Iso->push_back( iso2 );
      zDau1TrkIso->push_back( trkIso1 );
      zDau2TrkIso->push_back( trkIso2 );
      zDau1EcalIso->push_back( ecalIso1 );
      zDau2EcalIso->push_back( ecalIso2 );
      zDau1HcalIso->push_back( hcalIso1 );
      zDau2HcalIso->push_back( hcalIso2 );
      if (isMCMatchTrue){
	GenParticleRef trueZRef  = (*zGenParticlesMatch)[zRef];	
	//CandidateRef trueZRef = trueZIter->val;
	if( trueZRef.isNonnull() ) {
	  const Candidate & z = * trueZRef;
	  trueZMass->push_back( z.mass() );
	  trueZPt->push_back( z.pt() );
	  trueZEta->push_back( z.eta() );
	  trueZPhi->push_back( z.phi() );
	  trueZY->push_back( z.rapidity() );
	} else {
	  trueZMass->push_back( -100 ); 
	  trueZPt->push_back( -100 );
	  trueZEta->push_back( -100 );
	  trueZPhi->push_back( -100 );
	  trueZY->push_back( -100 );  
	}
      }      
      // quality variables 
      const pat::Muon * mu1 = dynamic_cast<const pat::Muon*>(m1);    
      zDau1NofHit->push_back(mu1->numberOfValidHits());
      zDau1NofHitTk->push_back(mu1->innerTrack()->numberOfValidHits());
      zDau1NofHitSta->push_back(mu1->outerTrack()->numberOfValidHits());
      zDau1NofMuChambers->push_back(mu1->numberOfChambers());
      zDau1NofMuMatches->push_back(mu1->numberOfMatches());
      zDau1Chi2->push_back(mu1->normChi2());
      zDau1dB->push_back(mu1->dB());
      TrackRef mu1TrkRef = mu1->innerTrack();
      zDau1dxyFromBS->push_back(mu1TrkRef->dxy(beamSpotHandle->position()));
      zDau1dzFromBS->push_back(mu1TrkRef->dz(beamSpotHandle->position()));
      zDau1dxyFromPV->push_back(mu1TrkRef->dxy(primaryVertices->begin()->position() ));
      zDau1dzFromPV->push_back(mu1TrkRef->dz(primaryVertices->begin()->position() ));     
      // would we like to add another variables???, such as 
      // double nChi2_tk_1= mu1->innerTrack()->normalizedChi2().....
      const pat::Muon * mu2 = dynamic_cast<const pat::Muon*>(m2);
      // for ZMuTk case...
      if (mu2!=0 ) {
	zDau2NofHit->push_back(mu2->numberOfValidHits());
	zDau2NofHitTk->push_back(mu2->innerTrack()->numberOfValidHits());
	zDau2NofHitSta->push_back(mu2->outerTrack()->numberOfValidHits());
	zDau2NofMuChambers->push_back(mu2->numberOfChambers());
	zDau2NofMuMatches->push_back(mu2->numberOfMatches());
	zDau2Chi2->push_back(mu2->normChi2());
	zDau2dB->push_back(mu2->dB());
	TrackRef mu2TrkRef = mu2->innerTrack();
	zDau2dxyFromBS->push_back(mu2TrkRef->dxy(beamSpotHandle->position()));
	zDau2dzFromBS->push_back(mu2TrkRef->dz(beamSpotHandle->position()));
	zDau2dxyFromPV->push_back(mu2TrkRef->dxy(primaryVertices->begin()->position() ));
	zDau2dzFromPV->push_back(mu2TrkRef->dz(primaryVertices->begin()->position() ));
      } else{
	// it's a track......
	const pat::GenericParticle * trk2 = dynamic_cast<const pat::GenericParticle*>(m2);
	TrackRef mu2TrkRef = trk2->track(); 
	zDau2NofHit->push_back(mu2TrkRef->numberOfValidHits());
	zDau2NofHitTk->push_back( mu2TrkRef->numberOfValidHits());
	zDau2NofHitSta->push_back( 0);
	zDau2NofMuChambers->push_back(0);
	zDau2NofMuMatches->push_back(0);
	zDau2Chi2->push_back(mu2TrkRef->normalizedChi2());
	zDau2dB->push_back( mu2TrkRef->dxy(beamSpotHandle->position())); // for now without beam spot.... 
	zDau2dxyFromBS->push_back(mu2TrkRef->dxy(beamSpotHandle->position()));
	zDau2dzFromBS->push_back(mu2TrkRef->dz(beamSpotHandle->position()));
	zDau2dxyFromPV->push_back(mu2TrkRef->dxy(primaryVertices->begin()->position() ));
	zDau2dzFromPV->push_back(mu2TrkRef->dz(primaryVertices->begin()->position() ));	
      }
    }
    const string & zName = zName_[c];
    evt.put( event,"EventNumber" );
    evt.put( run,"RunNumber" );
    evt.put( lumi,"LumiBlock" );
    evt.put( zMass, zName +  "Mass" );
    evt.put( zPt, zName + "Pt" );
    evt.put( zEta, zName + "Eta" );
    evt.put( zPhi, zName + "Phi" );
    evt.put( zY, zName + "Y" );
    evt.put( zDau1Pt, zName + "Dau1Pt" );
    evt.put( zDau2Pt, zName + "Dau2Pt" );
    evt.put( zDau1Eta, zName + "Dau1Eta" );
    evt.put( zDau2Eta, zName + "Dau2Eta" );
    evt.put( zDau1Phi, zName + "Dau1Phi" );
    evt.put( zDau2Phi, zName + "Dau2Phi" );
    evt.put( zDau1Iso, zName + "Dau1Iso" );
    evt.put( zDau2Iso, zName + "Dau2Iso" );
    evt.put( zDau1TrkIso, zName + "Dau1TrkIso" );
    evt.put( zDau2TrkIso, zName + "Dau2TrkIso" );
    evt.put( zDau1EcalIso, zName + "Dau1EcalIso" );
    evt.put( zDau2EcalIso, zName + "Dau2EcalIso" );
    evt.put( zDau1HcalIso, zName + "Dau1HcalIso" );
    evt.put( zDau2HcalIso, zName + "Dau2HcalIso" );
    evt.put( vtxNormChi2, zName + "VtxNormChi2" );
    evt.put( zDau1NofHit, zName + "Dau1NofHit" );
    evt.put( zDau2NofHit, zName + "Dau2NofHit" );
    evt.put( zDau1NofHitTk, zName + "Dau1NofHitTk" );
    evt.put( zDau2NofHitTk, zName + "Dau2NofHitTk" );
    evt.put( zDau1NofHitSta, zName + "Dau1NofHitSta" );
    evt.put( zDau2NofHitSta, zName + "Dau2NofHitSta" );
    evt.put( zDau1NofMuChambers, zName + "Dau1NofMuChambers" );
    evt.put( zDau1NofMuMatches, zName + "Dau1NofMuMatches" );
    evt.put( zDau2NofMuChambers, zName + "Dau2NofMuChambers" );
    evt.put( zDau2NofMuMatches, zName + "Dau2NofMuMatches" );
    evt.put( zDau1Chi2, zName + "Dau1Chi2" );
    evt.put( zDau2Chi2, zName + "Dau2Chi2" );
    evt.put( zDau1dB, zName + "Dau1dB" );
    evt.put( zDau2dB, zName + "Dau2dB" );
    evt.put( zDau1dxyFromBS, zName + "Dau1dxyFromBS" );
    evt.put( zDau2dxyFromBS, zName + "Dau2dxyFromBS" );
    evt.put( zDau1dxyFromPV, zName + "Dau1dxyFromPV" );
    evt.put( zDau2dxyFromPV, zName + "Dau2dxyFromPV" );
    evt.put( zDau1dzFromBS, zName + "Dau1dzFromBS" );
    evt.put( zDau2dzFromBS, zName + "Dau2dzFromBS" );
    evt.put( zDau1dzFromPV, zName + "Dau1dzFromPV" );
    evt.put( zDau2dzFromPV, zName + "Dau2dzFromPV" );
    evt.put( trueZMass, zName +  "TrueMass" );
    evt.put( trueZPt, zName + "TruePt" );
    evt.put( trueZEta, zName + "TrueEta" );
    evt.put( trueZPhi, zName + "TruePhi" );
    evt.put( trueZY, zName + "TrueY" );
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZToLLEdmNtupleDumper );

