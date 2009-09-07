// -*- C++ -*-
//
// Package:    Castor
// Class:      CastorTowerCandidateProducer
// 

/**\class CastorTowerCandidateProducer CastorTowerCandidateProducer.cc RecoLocalCalo/Castor/src/CastorTowerCandidateProducer.cc

 Description: CastorTowerCandidate Reconstruction Producer. Produce CastorTowerCandidates from CastorTowers.
 Implementation:
*/

//
// Original Author:  Hans Van Haevermaet, Benoit Roland
//         Created:  Wed Jul  9 14:00:40 CEST 2008
// $Id: CastorTowerCandidateProducer.cc,v 1.1 2009/02/27 16:13:18 hvanhaev Exp $
//
//

#define debug 0
#define debugTowervariable 0

// system include 
#include <memory>
#include <vector>
#include <iostream>
#include <TMath.h>
#include <TRandom3.h>

// user include 
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/Point3D.h"


// Castor Object include
#include "DataFormats/CastorReco/interface/CastorCell.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/RecoCandidate/interface/RecoCastorTowerCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"


//
// class declaration
//

class CastorTowerCandidateProducer : public edm::EDProducer {
   public:
      explicit CastorTowerCandidateProducer(const edm::ParameterSet&);
      ~CastorTowerCandidateProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      
      // member data
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;
      typedef ROOT::Math::RhoZPhiPoint CellPoint;
      typedef std::vector<reco::CastorCell> CastorCellCollection;
      typedef std::vector<reco::CastorTower> CastorTowerCollection;
      typedef edm::RefVector<CastorCellCollection>  CastorCellRefVector;
      //typedef edm::OwnVector<Candidate> CandidateCollection;

      edm::InputTag mSource;
      double mEThreshold;
};

//
// constants, enums and typedefs
//

const double MYR2D = 180/M_PI;
const double theta = 3.136113777658;

//
// static data member definitions
//

//
// constructor and destructor
//

CastorTowerCandidateProducer::CastorTowerCandidateProducer(const edm::ParameterSet& iConfig) :
  mSource (iConfig.getParameter<edm::InputTag> ("src")),
  mEThreshold (iConfig.getParameter<double> ("minimumE"))

{
  //register your products
  produces<std::vector<reco::RecoCastorTowerCandidate> >();

  //now do what ever other initialization is needed
}


CastorTowerCandidateProducer::~CastorTowerCandidateProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void CastorTowerCandidateProducer::produce(edm::Event& evt, const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace reco;
  using namespace std;
  using namespace TMath;
  
  // Produce CastorTowerCandidates from CastorTowers
  
  int mVerbose = 0;

  if(debug) cout<<""<<endl;
  if(debug) cout<<"-------------------------------"<<endl;
  if(debug) cout<<"2. entering CastorTowerCandidateProducer"<<endl;
  if(debug) cout<<"-------------------------------"<<endl;
  if(debug) cout<<""<<endl;
  
  Handle<CastorTowerCollection> castorTowers;
  evt.getByLabel( mSource, castorTowers );

  if (castorTowers->size() == 0) cout <<"Warning: You are trying to run the Candidate algorithm with 0 input towers. \n";
  
  auto_ptr<vector<RecoCastorTowerCandidate> > cands( new vector<RecoCastorTowerCandidate> );
  cands->reserve( castorTowers->size() );

  unsigned idx = 0;
  for (; idx < castorTowers->size (); idx++) {
    const CastorTower* cal = &((*castorTowers) [idx]);
    if (mVerbose >= 2) {
      std::cout << "CastorTowerCandidateCreator::produce-> " << idx << " tower eta/phi/E: " << cal->eta() << '/' << cal->phi() << '/' << cal->energy() << " is...";
    }
    if (cal->energy() >= mEThreshold ) {
      //cout << " debug 1" << endl;
      math::PtEtaPhiMLorentzVector p( (cal->energy())*sin(theta), cal->eta(), cal->phi(), 0 );
      //cout << " debug 2" << endl;
      RecoCastorTowerCandidate *c = new RecoCastorTowerCandidate( 0, Candidate::LorentzVector( p ) );
      //cout << " debug 3" << endl;
      c->setCastorTower(CastorTowerRef(castorTowers,idx)); 
      //cout << " debug 4" << endl;
      cands->push_back( *c );
      //cout << " debug 5" << endl;
      if (mVerbose >= 2) std::cout << "accepted " << std::endl;
      //cout << " debug 6" << endl;
    }
    else {
      if (mVerbose >= 2) std::cout << "rejected" << std::endl;
    }
  }
  
  if (mVerbose >= 1) {
    std::cout << "CastorTowerCandidateCreator::produce-> " << cands->size() << " candidates created" << std::endl;
  }
  
  evt.put( cands );

} 


// ------------ method called once each job just before starting event loop  ------------
void CastorTowerCandidateProducer::beginJob(const edm::EventSetup&) {
  if(debug) std::cout<<"Starting CastorTowerCandidateProducer"<<std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void CastorTowerCandidateProducer::endJob() {
  if(debug) std::cout<<"Ending CastorTowerCandidateProducer"<<std::endl;
}


//define this as a plug-in
DEFINE_FWK_MODULE(CastorTowerCandidateProducer);
