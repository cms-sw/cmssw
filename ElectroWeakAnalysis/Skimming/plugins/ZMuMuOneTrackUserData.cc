#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <vector>

using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;
//using namespace pat;

class ZMuMuUserDataOneTrack : public edm::EDProducer {
public:
  ZMuMuUserDataOneTrack( const edm::ParameterSet & );
  typedef math::XYZVector Vector;
private:
  void produce( edm::Event &, const edm::EventSetup & ) override;

  EDGetTokenT<std::vector<reco::CompositeCandidate> > srcToken_;
  EDGetTokenT<BeamSpot> beamSpotToken_;
  EDGetTokenT<VertexCollection> primaryVerticesToken_;
  EDGetTokenT<GenParticleMatch> zGenParticlesMatchToken_;
  double alpha_, beta_;
  string hltPath_;
  int counter;


};



ZMuMuUserDataOneTrack::ZMuMuUserDataOneTrack( const ParameterSet & cfg ):
  srcToken_(consumes<std::vector<reco::CompositeCandidate> > ( cfg.getParameter<InputTag>( "src" ) ) ),
  beamSpotToken_(consumes<BeamSpot> (cfg.getParameter<InputTag>( "beamSpot" ) ) ),
  primaryVerticesToken_(consumes<VertexCollection> (cfg.getParameter<InputTag>( "primaryVertices" ) ) ),
  zGenParticlesMatchToken_(consumes<GenParticleMatch> (cfg.getParameter<InputTag>( "zGenParticlesMatch" ) ) ),
  alpha_(cfg.getParameter<double>("alpha") ),
  beta_(cfg.getParameter<double>("beta") ),
  hltPath_(cfg.getParameter<std::string >("hltPath") ){
  produces<vector<pat::CompositeCandidate> >();
}

void ZMuMuUserDataOneTrack::produce( Event & evt, const EventSetup & ) {
  Handle<std::vector<reco::CompositeCandidate> > dimuons;
  evt.getByToken(srcToken_,dimuons);

  Handle<BeamSpot> beamSpotHandle;
  if (!evt.getByToken(beamSpotToken_, beamSpotHandle)) {
    std::cout << ">>> No beam spot found !!!"<<std::endl;
  }

  Handle<VertexCollection> primaryVertices;  // Collection of primary Vertices
  if (!evt.getByToken(primaryVerticesToken_, primaryVertices)){
    std::cout << ">>> No primary vertices  found !!!"<<std::endl;
  }

  bool isMCMatchTrue=false;

  Handle<GenParticleMatch> zGenParticlesMatch;
  if(evt.getByToken( zGenParticlesMatchToken_, zGenParticlesMatch )){
    isMCMatchTrue=true;
  }

  //cout<<"isMCMatchTrue"<<isMCMatchTrue <<endl;
  auto_ptr<vector<pat::CompositeCandidate> > dimuonColl( new vector<pat::CompositeCandidate> () );


  for (unsigned int i = 0; i< dimuons->size();++i){
    const CompositeCandidate & z = (*dimuons)[i];
    //CandidateBaseRef zRef = dimuons ->refAt(i);
    edm::Ref<std::vector<reco::CompositeCandidate> > zRef(dimuons, i);
    pat::CompositeCandidate dimuon(z);

    float trueMass,truePt,trueEta,truePhi,trueY;
    if (isMCMatchTrue){
      GenParticleRef trueZRef  = (*zGenParticlesMatch)[zRef];
      //CandidateRef trueZRef = trueZIter->val;
      if( trueZRef.isNonnull() ) {
	const Candidate & z = * trueZRef;
	trueMass = z.mass();
	truePt   = z.pt();
	trueEta  = z.eta();
	truePhi  = z.phi();
	trueY    = z.rapidity();
      } else {
	trueMass = -100;
	truePt   = -100;
	trueEta  = -100;
	truePhi  = -100;
	trueY    = -100;
      }

      dimuon.addUserFloat("TrueMass",trueMass);
      dimuon.addUserFloat("TruePt",truePt);
      dimuon.addUserFloat("TrueEta",trueEta);
      dimuon.addUserFloat("TruePhi",truePhi);
      dimuon.addUserFloat("TrueY",trueY);

    }

    dimuonColl->push_back(dimuon);

  }

  evt.put( dimuonColl);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZMuMuUserDataOneTrack );

