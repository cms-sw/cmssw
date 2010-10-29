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

class ZMuMuUserData : public edm::EDProducer {
public:
  ZMuMuUserData( const edm::ParameterSet & );   
  typedef math::XYZVector Vector;
private:
  void produce( edm::Event &, const edm::EventSetup & );
  
  InputTag src_,beamSpot_, primaryVertices_, zGenParticlesMatch_;
  double alpha_, beta_; 
  string hltPath_; 
  int counter;
  

};



ZMuMuUserData::ZMuMuUserData( const ParameterSet & cfg ):
  src_( cfg.getParameter<InputTag>( "src" ) ),
  beamSpot_(cfg.getParameter<InputTag>( "beamSpot" ) ),
  primaryVertices_(cfg.getParameter<InputTag>( "primaryVertices" ) ),
  zGenParticlesMatch_(cfg.getParameter<InputTag>( "zGenParticlesMatch" ) ),
  alpha_(cfg.getParameter<double>("alpha") ),
  beta_(cfg.getParameter<double>("beta") ), 
  hltPath_(cfg.getParameter<std::string >("hltPath") ){
  counter = 0;
  produces<vector<pat::CompositeCandidate> >();
}

void ZMuMuUserData::produce( Event & evt, const EventSetup & ) {
  Handle<std::vector<reco::CompositeCandidate> > dimuons;
  evt.getByLabel(src_,dimuons);

  Handle<BeamSpot> beamSpotHandle;
  evt.getByLabel(beamSpot_, beamSpotHandle);
  
  Handle<VertexCollection> primaryVertices;  // Collection of primary Vertices
  evt.getByLabel(primaryVertices_, primaryVertices);

  
  bool isMCMatchTrue=false;
   
  Handle<GenParticleMatch> zGenParticlesMatch;
  if(evt.getByLabel( zGenParticlesMatch_, zGenParticlesMatch )){
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
    const Candidate * dau1 = z.daughter(0); 
    const Candidate * dau2 = z.daughter(1); 
    const pat::Muon & mu1 = dynamic_cast<const pat::Muon&>(*dau1->masterClone());
    const pat::Muon & mu2 = dynamic_cast<const pat::Muon&>(*dau2->masterClone());
    
    /*cout<<"mu1 is null? "<<mu1.isMuon()<<endl;
    cout<<"mu2 is null? "<<mu2.isMuon()<<endl;
    cout<<"mu1 is global?"<<mu1.isGlobalMuon()<<endl;
    cout<<"mu2 is global?"<<mu2.isGlobalMuon()<<endl;
    */

    if(mu1.isGlobalMuon()==true && mu2.isGlobalMuon()==true){
      ++counter;

      // StandAlone part of the Z daughters
      TrackRef stAloneTrack1 = dau1->get<TrackRef,reco::StandAloneMuonTag>();
      TrackRef stAloneTrack2 = dau2->get<TrackRef,reco::StandAloneMuonTag>();
      // StandAlone kinematics
      Vector momentumSta1 = stAloneTrack1->momentum();
      Vector momentumSta2 = stAloneTrack2->momentum();
      float zDau1SaEta = stAloneTrack1->eta();
      float zDau2SaEta = stAloneTrack2->eta();
      float zDau1SaPhi = stAloneTrack1->phi();
      float zDau2SaPhi = stAloneTrack2->phi();
      float zDau1SaPt,zDau2SaPt;

      Candidate::PolarLorentzVector p4_Sta1(momentumSta1.rho(), momentumSta1.eta(),momentumSta1.phi(), dau1->mass()); 
      Candidate::PolarLorentzVector p4_Sta2(momentumSta2.rho(), momentumSta2.eta(),momentumSta2.phi(), dau2->mass()); 

      float zSaGlbMass = ( p4_Sta1 + (dau2->polarP4()) ).mass() ;
      float zGlbSaMass = ( p4_Sta2 + (dau1->polarP4()) ).mass() ;
      float zSaSaMass = ( p4_Sta1 + p4_Sta2 ).mass() ;
      float zMassSaPdf;

      if(counter % 2 == 0) {
	/// dau2 Pt filled with negative value in order to flag the downgraded muon for building zMassSa
	zMassSaPdf = zSaGlbMass;
	zDau1SaPt = stAloneTrack1->pt();
	zDau2SaPt = - stAloneTrack2->pt();
      }else{
	/// dau1 Pt filled with negative value in order to flag the downgraded muon for building zMassSa
	zMassSaPdf = zGlbSaMass;
	zDau1SaPt = - stAloneTrack1->pt();
	zDau2SaPt =  stAloneTrack2->pt();
      }
      
      dimuon.addUserFloat("SaGlbMass",zSaGlbMass);  	
      dimuon.addUserFloat("GlbSaMass",zGlbSaMass);  	
      dimuon.addUserFloat("SaSaMass",zSaSaMass);  	
      dimuon.addUserFloat("MassSa",zMassSaPdf);  	
      dimuon.addUserFloat("Dau1SaPt",zDau1SaPt);  	
      dimuon.addUserFloat("Dau2SaPt",zDau2SaPt);  	
      dimuon.addUserFloat("Dau1SaPhi",zDau1SaPhi);  	
      dimuon.addUserFloat("Dau2SaPhi",zDau2SaPhi);  	
      dimuon.addUserFloat("Dau1SaEta",zDau1SaEta);  	
      dimuon.addUserFloat("Dau2SaEta",zDau2SaEta);  	
    }

    dimuonColl->push_back(dimuon);
    
  }
  
  evt.put( dimuonColl );
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZMuMuUserData );

