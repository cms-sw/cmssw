// 
// Translation of BTag MCJetFlavour tool to identify real flavour of a jet 
// to work with Candidate objects
// Author: Attilio  
// Date: 29.05.2007
//

//=======================================================================

// user include files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
//#include "SimDataFormats/JetMatching/interface/CandJetFlavourMatching.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <Math/VectorUtil.h>
#include <TMath.h>

using namespace std;
using namespace reco;
using namespace edm;
using namespace ROOT::Math::VectorUtil;

class CandJetFlavourIdentifier : public edm::EDProducer {
  public:
    CandJetFlavourIdentifier( const edm::ParameterSet & );
    ~CandJetFlavourIdentifier();

  private:
    virtual void produce(edm::Event&, const edm::EventSetup& );

    int fillAlgoritDefinition( const Candidate& ) const;
    int fillPhysicsDefinition( const Candidate& ) const;

    Handle<CandidateCollection> jets;
    Handle<CandidateCollection> particles;

    edm::InputTag m_jetsSrc;
    double coneSizeToAssociate;
    bool physDefinition;
    bool vetoB, vetoC, vetoL, vetoG;
    bool debug;
};

//=========================================================================

CandJetFlavourIdentifier::CandJetFlavourIdentifier( const edm::ParameterSet& iConfig )
{
    produces<CandMatchMap>();
    m_jetsSrc           = iConfig.getParameter<edm::InputTag>("jets");
    coneSizeToAssociate = iConfig.getParameter<double>("coneSizeToAssociate");
    physDefinition      = iConfig.getParameter<bool>("physicsDefinition");
    debug               = iConfig.getParameter<bool>("debug");
}

//=========================================================================

CandJetFlavourIdentifier::~CandJetFlavourIdentifier() 
{
}

// ------------ method called to produce the data  ------------

void CandJetFlavourIdentifier::produce( Event& iEvent, const EventSetup& iEs ) 
{

  iEvent.getByLabel(m_jetsSrc, jets);
  iEvent.getByLabel ("genParticleCandidates", particles );

  auto_ptr<CandMatchMap> matchJetParton( new CandMatchMap( CandMatchMap::ref_type( CandidateRefProd( jets      ),
                                                                                   CandidateRefProd( particles )
						       		             ) ) );
  for (size_t j = 0; j < jets->size(); j++) {
       const int theMappedPartonAlg = fillAlgoritDefinition( (*jets)[j] );
       const int theMappedPartonPhy = fillPhysicsDefinition( (*jets)[j] );
       if(debug) {
         if(theMappedPartonAlg >= 0) {
           const Candidate & aPartAlg = (*particles)[theMappedPartonAlg] ;
           cout << "the mapped partonAlg=" << theMappedPartonAlg << " pT=" << aPartAlg.et() << " id=" << aPartAlg.pdgId()<< endl;
         }
         if(theMappedPartonPhy >= 0) {
           const Candidate & aPartPhy = (*particles)[theMappedPartonPhy] ;
           cout << "the mapped partonPhy=" << theMappedPartonPhy << " pT=" << aPartPhy.et() << " id=" << aPartPhy.pdgId()<< endl;
         }
       }
       if(physDefinition) {
         if ( theMappedPartonPhy < 0 ) continue;
         matchJetParton->insert( CandidateRef( jets, j   ), CandidateRef( particles, theMappedPartonPhy ) ); 
       } else {
         if ( theMappedPartonAlg < 0 ) continue;
         matchJetParton->insert( CandidateRef( jets, j   ), CandidateRef( particles, theMappedPartonAlg ) );
       }         
  }

  iEvent.put(matchJetParton);

}


//
// Algorithmic Definition: 
// Output: define one associatedParton
// Loop on all particle.
// A particle is a parton if its daughter is a string(code=92) or a cluster(code=93) 
// If (parton is within the cone defined by coneSizeToAssociate) then:
//           if (parton is a b)                                   then associatedParton is the b
//      else if (associatedParton =! b and parton is a c)         then associatedParton is the c
//      else if (associatedParton =! b and associatedParton =! c) then associatedParton is the one with the highest pT
// associatedParton can be -1 --> no partons in the cone
// True Flavour of the jet --> flavour of the associatedParton
//
// ToDo: if more than one b(c) in the cone --> the selected parton is not always the one with highest pT
//
int CandJetFlavourIdentifier::fillAlgoritDefinition( const Candidate& theJet ) const {

  int tempParticle = -1;
  int tempPartonHighestPt = -1;
  float maxPt = 0;
  for( size_t m = 0; m != particles->size(); ++ m ) {
    const Candidate & aParton = (*particles)[ m ];
    if( aParton.numberOfDaughters() > 0  && ( aParton.daughter(0)->pdgId() == 91 || aParton.daughter(0)->pdgId() == 92 ) ) {
      double dist = DeltaR( theJet.p4(), aParton.p4() );
      if( dist <= coneSizeToAssociate ) {
        if( tempParticle == -1 && ( abs( aParton.pdgId() ) == 4 )  ) tempParticle = m;
        if(                         abs( aParton.pdgId() ) == 5    ) tempParticle = m;
        if( aParton.pt() > maxPt ) {
           maxPt = aParton.pt();
           tempPartonHighestPt = m;
        }
      }
    }
  }
  if ( tempParticle == -1 ) tempParticle = tempPartonHighestPt;
  return tempParticle;
}

//
// Physics Definition: 
// A initialParticle is a particle with status=3
// Output: define one associatedInitialParticle
// Loop on all particles
// A particle is a parton if its daughter is a string(code=92) or a cluster(code=93)
// if( only one initialParticle within the cone defined by coneSizeToAssociate) associatedInitialParticle is the initialParticle
// TheBiggerConeSize = 0.7 --> it's hard coded!
// if( a parton not coming from associatedInitialParticle in TheBiggerConeSize is a b or a c) reject the association
//
// associatedInitialParticle can be -1 --> no initialParticle in the cone or rejected association
// True Flavour of the jet --> flavour of the associatedInitialParticle
//
int CandJetFlavourIdentifier::fillPhysicsDefinition( const Candidate& theJet ) const {

  float TheBiggerConeSize = 0.7; // In HepMC it's 0.3 --> it's a mistake: value has to be 0.7
  int tempParticle = -1;
  int nInTheCone = 0;

  vector<const reco::Candidate *> theContaminations;
  theContaminations.clear();

  for( size_t m = 0; m != particles->size(); ++ m ) {
    //skip first 6 particles (2 protons + 4 initial state partons)
    if(m<6) continue;
    const Candidate & aParticle = (*particles)[ m ];
    // skipping all particle but udscbg (is this correct/enough?!?!)
    bool isAParton = false;
    int flavour = abs(aParticle.pdgId());
    if(flavour == 1 || 
       flavour == 2 ||
       flavour == 3 ||
       flavour == 4 ||
       flavour == 5 ||
       flavour == 21 ) isAParton = true;
    if(!isAParton) continue;
    double dist = DeltaR( theJet.p4(), aParticle.p4() );
    if( aParticle.status() == 3 && dist <= coneSizeToAssociate ) {
      //cout << "particle in small cone=" << aParticle.pdgId() << endl;
      tempParticle = m;
      nInTheCone++;
    }
    // Look for heavy partons in TheBiggerConeSize now
    if( aParticle.numberOfDaughters() > 0  && ( aParticle.daughter(0)->pdgId() == 91 || aParticle.daughter(0)->pdgId() == 92 ) ) {
      if( flavour ==  1 ||
          flavour ==  2 ||
          flavour ==  3 ||
          flavour == 21 ) continue;
      if( dist < TheBiggerConeSize ) theContaminations.push_back( &aParticle );
    }
  }

  //cout << "theJet=" << theJet.p4().Et() << " " << theJet.p4().eta() << " " << theJet.p4().phi() << endl;

  if(nInTheCone != 1) return -1; // rejected --> only one initialParton requested
  if(theContaminations.size() == 0 ) return tempParticle; //no contamination
  int initialPartonFlavour = abs( ((*particles)[ tempParticle ]).pdgId() );

  //cout << "nInTheCone=" << nInTheCone << " theContaminations.size=" << theContaminations.size() << " iniFla=" << initialPartonFlavour << endl;

  vector<const Candidate *>::const_iterator itCont = theContaminations.begin();
  for( ; itCont != theContaminations.end(); itCont++ ) {
    int contaminatingFlavour = abs( (*itCont)->pdgId() );
    if( (*itCont)->numberOfMothers()>0 && (*itCont)->mother(0) == &( (*particles)[ tempParticle ]) ) continue; // mother is the initialParton --> OK
    if( initialPartonFlavour == 4 ) {  
      if( contaminatingFlavour == 4 ) continue; // keep association --> the initialParton is a c --> the contaminated parton is a c
      tempParticle = -1; // all the other cases reject!
      return tempParticle;
    }
  } 

  return tempParticle;   
}

//define this as a plug-in
DEFINE_FWK_MODULE(CandJetFlavourIdentifier);

