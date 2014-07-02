/* \class ZToLLEdmNtupleDumper
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
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
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"

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
  typedef math::XYZVector Vector;
  ZToLLEdmNtupleDumper( const edm::ParameterSet & );

private:
  void produce( edm::Event &, const edm::EventSetup & ) override;
  std::vector<std::string> zName_;
  std::vector<edm::EDGetTokenT<CandidateView> > zTokens_;
  std::vector<edm::EDGetTokenT<GenParticleMatch> > zGenParticlesMatchTokens_ ;
  edm::EDGetTokenT<BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<VertexCollection> primaryVerticesToken_;

  std::vector<double> ptThreshold_, etEcalThreshold_, etHcalThreshold_ ,dRVetoTrk_, dRTrk_, dREcal_ , dRHcal_,  alpha_, beta_;
  std::vector<double> relativeIsolation_;
  std::vector<string> hltPath_;
  int counter;


};

template<typename T>
  double isolation(const T * t, double ptThreshold, double etEcalThreshold, double etHcalThreshold , double dRVetoTrk, double dRTrk, double dREcal , double dRHcal,  double alpha, double beta, bool relativeIsolation) {
   // on 34X:
const pat::IsoDeposit * trkIso = t->isoDeposit(pat::TrackIso);
//  const pat::IsoDeposit * trkIso = t->trackerIsoDeposit();
   // on 34X
const pat::IsoDeposit * ecalIso = t->isoDeposit(pat::EcalIso);
//  const pat::IsoDeposit * ecalIso = t->ecalIsoDeposit();
//    on 34X
const pat::IsoDeposit * hcalIso = t->isoDeposit(pat::HcalIso);
//    const pat::IsoDeposit * hcalIso = t->hcalIsoDeposit();

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
    edm::EDGetTokenT<CandidateView> zToken = consumes<CandidateView>( i->getParameter<InputTag>( "z" ) );
    edm::EDGetTokenT<GenParticleMatch> zGenParticlesMatchToken = consumes<GenParticleMatch>( i->getParameter<InputTag>( "zGenParticlesMatch" ) );
    beamSpotToken_ = consumes<BeamSpot>( i->getParameter<InputTag>( "beamSpot" ) );
    primaryVerticesToken_= consumes<VertexCollection>( i->getParameter<InputTag>( "primaryVertices" ) ) ;
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
    string hltPath = i ->getParameter<std::string >("hltPath");
    zName_.push_back( zName );
    zTokens_.push_back( zToken );
    zGenParticlesMatchTokens_.push_back( zGenParticlesMatchToken );
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
    hltPath_.push_back(hltPath);
    produces<vector<unsigned int> >( alias = zName + "EventNumber" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "RunNumber" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "LumiBlock" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Mass" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "MassSa" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Pt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Eta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Phi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Y" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Pt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Pt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1SaPt" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2SaPt" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau1HLTBit" ).setBranchAlias( alias );
    produces<vector<unsigned int> >( alias = zName + "Dau2HLTBit" ).setBranchAlias( alias );
    produces<vector<int> >( alias = zName + "Dau1Q" ).setBranchAlias( alias );
    produces<vector<int> >( alias = zName + "Dau2Q" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Eta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Eta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1SaEta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2SaEta" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Phi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Phi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1SaPhi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2SaPhi" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1Iso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2Iso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1TrkIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2TrkIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1EcalIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2EcalIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1HcalIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2HcalIso" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1MuEnergyEm" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau1MuEnergyHad" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2MuEnergyEm" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2MuEnergyHad" ).setBranchAlias( alias );

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
    produces<vector<float> >( alias = zName + "Dau1TrkChi2" ).setBranchAlias( alias );
    produces<vector<float> >( alias = zName + "Dau2TrkChi2" ).setBranchAlias( alias );
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
  if (!evt.getByToken(beamSpotToken_, beamSpotHandle)) {
    std::cout << ">>> No beam spot found !!!"<<std::endl;
  }
  Handle<reco::VertexCollection> primaryVertices;  // Collection of primary Vertices
  if (!evt.getByToken(primaryVerticesToken_, primaryVertices)){
    std::cout << ">>> No primary verteces  found !!!"<<std::endl;
  }

  unsigned int size = zTokens_.size();
  for( unsigned int c = 0; c < size; ++ c ) {
    Handle<CandidateView> zColl;
    evt.getByToken( zTokens_[c], zColl );
    bool isMCMatchTrue = false;
    //if (zGenParticlesMatchTokens_[c] != "")  isMCMatchTrue = true;
    Handle<GenParticleMatch> zGenParticlesMatch;
    if (evt.getByToken( zGenParticlesMatchTokens_[c], zGenParticlesMatch )) {
      isMCMatchTrue=true;
    }
    unsigned int zSize = zColl->size();
    auto_ptr<vector<unsigned int> > event( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > run( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > lumi( new vector<unsigned int > );
    auto_ptr<vector<float> > zMass( new vector<float> );
    auto_ptr<vector<float> > zMassSa( new vector<float> );
    auto_ptr<vector<float> > zPt( new vector<float> );
    auto_ptr<vector<float> > zEta( new vector<float> );
    auto_ptr<vector<float> > zPhi( new vector<float> );
    auto_ptr<vector<float> > zY( new vector<float> );
    auto_ptr<vector<float> > zDau1Pt( new vector<float> );
    auto_ptr<vector<float> > zDau2Pt( new vector<float> );
    auto_ptr<vector<float> > zDau1SaPt( new vector<float> );
    auto_ptr<vector<float> > zDau2SaPt( new vector<float> );
    auto_ptr<vector<unsigned int> > zDau1HLTBit( new vector<unsigned int> );
    auto_ptr<vector<unsigned int> > zDau2HLTBit( new vector<unsigned int> );
    auto_ptr<vector<int> > zDau1Q( new vector<int> );
    auto_ptr<vector<int> > zDau2Q( new vector<int> );
    auto_ptr<vector<float> > zDau1Eta( new vector<float> );
    auto_ptr<vector<float> > zDau2Eta( new vector<float> );
    auto_ptr<vector<float> > zDau1SaEta( new vector<float> );
    auto_ptr<vector<float> > zDau2SaEta( new vector<float> );
    auto_ptr<vector<float> > zDau1Phi( new vector<float> );
    auto_ptr<vector<float> > zDau2Phi( new vector<float> );
    auto_ptr<vector<float> > zDau1SaPhi( new vector<float> );
    auto_ptr<vector<float> > zDau2SaPhi( new vector<float> );
    auto_ptr<vector<float> > zDau1Iso( new vector<float> );
    auto_ptr<vector<float> > zDau2Iso( new vector<float> );
    auto_ptr<vector<float> > zDau1TrkIso( new vector<float> );
    auto_ptr<vector<float> > zDau2TrkIso( new vector<float> );
    auto_ptr<vector<float> > zDau1EcalIso( new vector<float> );
    auto_ptr<vector<float> > zDau2EcalIso( new vector<float> );
    auto_ptr<vector<float> > zDau1HcalIso( new vector<float> );
    auto_ptr<vector<float> > zDau2HcalIso( new vector<float> );
    auto_ptr<vector<float> > zDau1MuEnergyEm( new vector<float> );
    auto_ptr<vector<float> > zDau2MuEnergyEm( new vector<float> );
    auto_ptr<vector<float> > zDau1MuEnergyHad( new vector<float> );
    auto_ptr<vector<float> > zDau2MuEnergyHad( new vector<float> );
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
    auto_ptr<vector<float> > zDau1TrkChi2( new vector<float> );
    auto_ptr<vector<float> > zDau2TrkChi2( new vector<float> );
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
      zDau1Q->push_back( dau1->charge() );
      zDau2Q->push_back( dau2->charge() );
      zDau1Eta->push_back( dau1->eta() );
      zDau2Eta->push_back( dau2->eta() );
      zDau1Phi->push_back( dau1->phi() );
      zDau2Phi->push_back( dau2->phi() );
      if(!(dau1->hasMasterClone()&&dau2->hasMasterClone()))
	throw edm::Exception(edm::errors::InvalidReference)
	  << "Candidate daughters have no master clone\n";
      const CandidateBaseRef & mr1 = dau1->masterClone(), & mr2 = dau2->masterClone();

       const Candidate * m1 = &*mr1, * m2 = &*mr2;

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
        // protection for standalone and trackerMuon
      if (mu1->isGlobalMuon() == true){
	zDau1NofHit->push_back(mu1->numberOfValidHits());
	zDau1NofHitTk->push_back(mu1->innerTrack()->numberOfValidHits());
	zDau1NofHitSta->push_back(mu1->outerTrack()->numberOfValidHits());
	zDau1Chi2->push_back(mu1->normChi2());
	TrackRef mu1TrkRef = mu1->innerTrack();
	zDau1TrkChi2->push_back( mu1TrkRef->normalizedChi2());
	zDau1dxyFromBS->push_back(mu1TrkRef->dxy(beamSpotHandle->position()));
	zDau1dzFromBS->push_back(mu1TrkRef->dz(beamSpotHandle->position()));
	zDau1dxyFromPV->push_back(mu1TrkRef->dxy(primaryVertices->begin()->position() ));
	zDau1dzFromPV->push_back(mu1TrkRef->dz(primaryVertices->begin()->position() ));
	zDau1MuEnergyEm->push_back( mu1->calEnergy().em);
	zDau1MuEnergyHad->push_back( mu1->calEnergy().had);

      } else  if (mu1->isStandAloneMuon() == true) {
        // the muon is a standalone
	TrackRef mu1StaRef = mu1->outerTrack();
  	zDau1NofHit->push_back(mu1StaRef->numberOfValidHits());
	zDau1NofHitTk->push_back(0);
	zDau1NofHitSta->push_back(mu1StaRef->numberOfValidHits());
	zDau1Chi2->push_back(mu1StaRef->normalizedChi2());
	zDau1TrkChi2->push_back(0);
	zDau1dxyFromBS->push_back(mu1StaRef->dxy(beamSpotHandle->position()));
	zDau1dzFromBS->push_back(mu1StaRef->dz(beamSpotHandle->position()));
	zDau1dxyFromPV->push_back(mu1StaRef->dxy(primaryVertices->begin()->position() ));
	zDau1dzFromPV->push_back(mu1StaRef->dz(primaryVertices->begin()->position() ));
	zDau1MuEnergyEm->push_back( -1);
	zDau1MuEnergyHad->push_back( -1);
      } else  if (mu1->isTrackerMuon() == true) {
        // the muon is a trackerMuon
	TrackRef mu1TrkRef = mu1->innerTrack();
	zDau1NofHit->push_back(mu1TrkRef->numberOfValidHits());
	zDau1NofHitTk->push_back(mu1TrkRef->numberOfValidHits());
	zDau1NofHitSta->push_back(0);
	zDau1Chi2->push_back(mu1TrkRef->normalizedChi2());
	zDau1TrkChi2->push_back(mu1TrkRef->normalizedChi2());
	zDau1dxyFromBS->push_back(mu1TrkRef->dxy(beamSpotHandle->position()));
	zDau1dzFromBS->push_back(mu1TrkRef->dz(beamSpotHandle->position()));
	zDau1dxyFromPV->push_back(mu1TrkRef->dxy(primaryVertices->begin()->position() ));
	zDau1dzFromPV->push_back(mu1TrkRef->dz(primaryVertices->begin()->position() ));
	zDau1MuEnergyEm->push_back( mu1->calEnergy().em);
	zDau1MuEnergyHad->push_back( mu1->calEnergy().had);
      }
      zDau1NofMuChambers->push_back(mu1->numberOfChambers());
      zDau1NofMuMatches->push_back(mu1->numberOfMatches());

      // would we like to add another variables???
      // HLT trigger  bit
      const pat::TriggerObjectStandAloneCollection mu1HLTMatches =  mu1->triggerObjectMatchesByPath( hltPath_[c] );

      int dimTrig1 = mu1HLTMatches.size();
      if(dimTrig1 !=0 ){
	zDau1HLTBit->push_back(1);
      } else {
	zDau1HLTBit->push_back(0);
      }
      const pat::Muon * mu2 = dynamic_cast<const pat::Muon*>(m2);
      if (mu2!=0 ) {
	if (mu2->isGlobalMuon() == true) {
	  zDau2NofHit->push_back(mu2->numberOfValidHits());
	  zDau2NofHitTk->push_back(mu2->innerTrack()->numberOfValidHits());
	  zDau2NofHitSta->push_back(mu2->outerTrack()->numberOfValidHits());
	  zDau2Chi2->push_back(mu2->normChi2());
	  TrackRef mu2TrkRef = mu2->innerTrack();
	  zDau1TrkChi2->push_back( mu2TrkRef->normalizedChi2());
	  zDau2dxyFromBS->push_back(mu2TrkRef->dxy(beamSpotHandle->position()));
	  zDau2dzFromBS->push_back(mu2TrkRef->dz(beamSpotHandle->position()));
	  zDau2dxyFromPV->push_back(mu2TrkRef->dxy(primaryVertices->begin()->position() ));
	  zDau2dzFromPV->push_back(mu2TrkRef->dz(primaryVertices->begin()->position() ));
	  zDau2MuEnergyEm->push_back( mu2->calEnergy().em);
	  zDau2MuEnergyHad->push_back( mu2->calEnergy().had);
	} else if (mu2->isStandAloneMuon() == true){
	  // its' a standalone
	  zDau2HLTBit->push_back(0);
	  TrackRef mu2StaRef = mu2->outerTrack();
	  zDau2NofHit->push_back(mu2StaRef->numberOfValidHits());
	  zDau2NofHitTk->push_back(0);
	  zDau2NofHitSta->push_back(mu2StaRef->numberOfValidHits());
	  zDau2Chi2->push_back(mu2StaRef->normalizedChi2());
	  zDau2TrkChi2->push_back(0);
	  zDau2dxyFromBS->push_back(mu2StaRef->dxy(beamSpotHandle->position()));
	  zDau2dzFromBS->push_back(mu2StaRef->dz(beamSpotHandle->position()));
	  zDau2dxyFromPV->push_back(mu2StaRef->dxy(primaryVertices->begin()->position() ));
	  zDau2dzFromPV->push_back(mu2StaRef->dz(primaryVertices->begin()->position() ));
	  zDau1MuEnergyEm->push_back( -1);
	  zDau1MuEnergyHad->push_back(  -1);
	} else  if (mu2->isTrackerMuon() == true) {
	  // the muon is a trackerMuon
	  TrackRef mu2TrkRef = mu2->innerTrack();
	  zDau2NofHit->push_back(mu2TrkRef->numberOfValidHits());
	  zDau2NofHitSta->push_back(0);
	  zDau2NofHitTk->push_back(mu2TrkRef->numberOfValidHits());
	  zDau2Chi2->push_back(mu2TrkRef->normalizedChi2());
	  zDau2TrkChi2->push_back(mu2TrkRef->normalizedChi2());
	  zDau2dxyFromBS->push_back(mu2TrkRef->dxy(beamSpotHandle->position()));
	  zDau2dzFromBS->push_back(mu2TrkRef->dz(beamSpotHandle->position()));
	  zDau2dxyFromPV->push_back(mu2TrkRef->dxy(primaryVertices->begin()->position() ));
	  zDau2dzFromPV->push_back(mu2TrkRef->dz(primaryVertices->begin()->position() ));
	  zDau2MuEnergyEm->push_back( mu2->calEnergy().em);
	  zDau2MuEnergyHad->push_back( mu2->calEnergy().had);
	}

	// HLT trigger  bit
	  const pat::TriggerObjectStandAloneCollection mu2HLTMatches = mu2->triggerObjectMatchesByPath( hltPath_[c] );
	  int dimTrig2 = mu2HLTMatches.size();
	  if(dimTrig2 !=0 ){
	    zDau2HLTBit->push_back(1);
	  }
	  else {
	    zDau2HLTBit->push_back(0);
	  }
	  /// only for ZGolden evaluated zMassSa for the mu+sta pdf, see zmumuSaMassHistogram.cc
	  if ( mu1->isGlobalMuon() && mu2->isGlobalMuon() ) {
          TrackRef stAloneTrack1;
	  TrackRef stAloneTrack2;
          Vector momentum;
	  Candidate::PolarLorentzVector p4_1;
	  double mu_mass;
	  stAloneTrack1 = dau1->get<TrackRef,reco::StandAloneMuonTag>();
	  stAloneTrack2 = dau2->get<TrackRef,reco::StandAloneMuonTag>();
	  zDau1SaEta->push_back(stAloneTrack1->eta());
  	  zDau2SaEta->push_back(stAloneTrack2->eta());
	  zDau1SaPhi->push_back(stAloneTrack1->phi());
  	  zDau2SaPhi->push_back(stAloneTrack2->phi());
	  if(counter % 2 == 0) {
	    momentum = stAloneTrack1->momentum();
	    p4_1 = dau2->polarP4();
	    mu_mass = dau1->mass();
	    /// I fill the dau1 with positive and dau2 with negatove values for the pt, in order to flag the muons used for building zMassSa
	    zDau1SaPt->push_back(stAloneTrack1 ->pt());
	    zDau2SaPt->push_back(- stAloneTrack2->pt());
	  }else{
	    momentum = stAloneTrack2->momentum();
	    p4_1= dau1->polarP4();
	    mu_mass = dau2->mass();
	    /// I fill the dau1 with negatove and dau2 with positive values for the pt
	    zDau1SaPt->push_back( - stAloneTrack1->pt());
	    zDau2SaPt->push_back( stAloneTrack2->pt());
	  }

	  Candidate::PolarLorentzVector p4_2(momentum.rho(), momentum.eta(),momentum.phi(), mu_mass);
	  double mass = (p4_1+p4_2).mass();
	  zMassSa->push_back(mass);
	  ++counter;
	  }


	  zDau2NofMuChambers->push_back(mu2->numberOfChambers());
	  zDau2NofMuMatches->push_back(mu2->numberOfMatches());
      } else{
	// for ZMuTk case...
	// it's a track......
      const pat::GenericParticle * trk2 = dynamic_cast<const pat::GenericParticle*>(m2);
      TrackRef mu2TrkRef = trk2->track();
      zDau2NofHit->push_back(mu2TrkRef->numberOfValidHits());
      zDau2NofHitTk->push_back( mu2TrkRef->numberOfValidHits());
      zDau2NofHitSta->push_back( 0);
      zDau2NofMuChambers->push_back(0);
      zDau2NofMuMatches->push_back(0);
      zDau2Chi2->push_back(mu2TrkRef->normalizedChi2());
      zDau2dxyFromBS->push_back(mu2TrkRef->dxy(beamSpotHandle->position()));
      zDau2dzFromBS->push_back(mu2TrkRef->dz(beamSpotHandle->position()));
      zDau2dxyFromPV->push_back(mu2TrkRef->dxy(primaryVertices->begin()->position() ));
      zDau2dzFromPV->push_back(mu2TrkRef->dz(primaryVertices->begin()->position() ));
	zDau1MuEnergyEm->push_back( -1);
	zDau1MuEnergyHad->push_back( -1);
    }
  }
  const string & zName = zName_[c];
  evt.put( event,zName + "EventNumber" );
  evt.put( run, zName + "RunNumber" );
  evt.put( lumi,zName + "LumiBlock" );
  evt.put( zMass, zName +  "Mass" );
  evt.put( zMassSa, zName +  "MassSa" );
  evt.put( zPt, zName + "Pt" );
  evt.put( zEta, zName + "Eta" );
  evt.put( zPhi, zName + "Phi" );
  evt.put( zY, zName + "Y" );
  evt.put( zDau1Pt, zName + "Dau1Pt" );
  evt.put( zDau2Pt, zName + "Dau2Pt" );
  evt.put( zDau1SaPt, zName + "Dau1SaPt" );
  evt.put( zDau2SaPt, zName + "Dau2SaPt" );
  evt.put( zDau1HLTBit, zName + "Dau1HLTBit" );
  evt.put( zDau2HLTBit, zName + "Dau2HLTBit" );
  evt.put( zDau1Q, zName + "Dau1Q" );
  evt.put( zDau2Q, zName + "Dau2Q" );
  evt.put( zDau1Eta, zName + "Dau1Eta" );
  evt.put( zDau2Eta, zName + "Dau2Eta" );
  evt.put( zDau1SaEta, zName + "Dau1SaEta" );
  evt.put( zDau2SaEta, zName + "Dau2SaEta" );
  evt.put( zDau1Phi, zName + "Dau1Phi" );
  evt.put( zDau2Phi, zName + "Dau2Phi" );
  evt.put( zDau1SaPhi, zName + "Dau1SaPhi" );
  evt.put( zDau2SaPhi, zName + "Dau2SaPhi" );
  evt.put( zDau1Iso, zName + "Dau1Iso" );
  evt.put( zDau2Iso, zName + "Dau2Iso" );
  evt.put( zDau1TrkIso, zName + "Dau1TrkIso" );
  evt.put( zDau2TrkIso, zName + "Dau2TrkIso" );
  evt.put( zDau1EcalIso, zName + "Dau1EcalIso" );
  evt.put( zDau2EcalIso, zName + "Dau2EcalIso" );
  evt.put( zDau1HcalIso, zName + "Dau1HcalIso" );
  evt.put( zDau2HcalIso, zName + "Dau2HcalIso" );
  evt.put( zDau1MuEnergyEm, zName + "Dau1MuEnergyEm" );
  evt.put( zDau2MuEnergyEm, zName + "Dau2MuEnergyEm" );
  evt.put( zDau1MuEnergyHad, zName + "Dau1MuEnergyHad" );
  evt.put( zDau2MuEnergyHad, zName + "Dau2MuEnergyHad" );
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
  evt.put( zDau1TrkChi2, zName + "Dau1TrkChi2" );
  evt.put( zDau2TrkChi2, zName + "Dau2TrkChi2" );
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

