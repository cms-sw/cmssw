// 
// Translation of BTag MCJetFlavour tool to identify real flavour of a jet 
// work with CaloJet objects
// Store Infos by Values in JetFlavour.h
// Author: Attilio  
// Date: 05.10.2007
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

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "SimDataFormats/JetMatching/interface/MatchedPartons.h"
#include "SimDataFormats/JetMatching/interface/JetMatchedPartons.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/getRef.h" 
//#include "DataFormats/Candidate/interface/Candidate.h"
//#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

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

class JetPartonMatcher : public edm::EDProducer 
{
  public:
    JetPartonMatcher( const edm::ParameterSet & );
    ~JetPartonMatcher();

  private:
    virtual void produce(edm::Event&, const edm::EventSetup& );

    int fillAlgoritDefinition( const Jet& );
    int fillPhysicsDefinition( const Jet& );
    int theHeaviest;
    int theNearest2;
    int theNearest3;
    int theHardest;

    Handle <GenParticleRefVector> particles;

    edm::InputTag m_jetsSrc, m_ParticleSrc;
    double coneSizeToAssociate;
    bool physDefinition;

};

//=========================================================================

JetPartonMatcher::JetPartonMatcher( const edm::ParameterSet& iConfig ) :
  theHeaviest(0),
  theNearest2(0),
  theNearest3(0),
  theHardest(0)
{
    produces<JetMatchedPartonsCollection>();
    m_jetsSrc           = iConfig.getParameter<edm::InputTag>("jets");
    m_ParticleSrc       = iConfig.getParameter<edm::InputTag>("partons");
    coneSizeToAssociate = iConfig.getParameter<double>("coneSizeToAssociate");
}

//=========================================================================

JetPartonMatcher::~JetPartonMatcher() 
{
}

// ------------ method called to produce the data  ------------

void JetPartonMatcher::produce( Event& iEvent, const EventSetup& iEs ) 
{
  edm::Handle <edm::View <reco::Jet> > jets_h;
  iEvent.getByLabel(m_jetsSrc,     jets_h    );
  iEvent.getByLabel(m_ParticleSrc, particles );

  edm::LogVerbatim("JetPartonMatcher") << "=== Partons size:" << particles->size();

  for( size_t m = 0; m != particles->size(); ++ m ) {
    const GenParticle & aParton = *(particles->at(m).get());
    edm::LogVerbatim("JetPartonMatcher") <<  aParton.status() << " " <<
                                             aParton.pdgId()  << " " <<
                                             aParton.pt()     << " " << 
                                             aParton.eta()    << " " <<
                                             aParton.phi()    << endl;
  }


  auto_ptr<reco::JetMatchedPartonsCollection> jetMatchedPartons( new JetMatchedPartonsCollection(reco::JetRefBaseProd(jets_h)));
  
  for (size_t j = 0; j < jets_h->size(); j++) {

    const int theMappedPartonAlg = fillAlgoritDefinition( (*jets_h)[j] );
    const int theMappedPartonPhy = fillPhysicsDefinition( (*jets_h)[j] );

    GenParticleRef pHV;
    GenParticleRef pN2;
    GenParticleRef pN3;
    GenParticleRef pPH;
    GenParticleRef pAL;

    if(theHeaviest>=0)        pHV = particles->at( theHeaviest        );
    if(theNearest2>=0)        pN2 = particles->at( theNearest2        );
    if(theNearest3>=0)        pN3 = particles->at( theNearest3        );
    if(theMappedPartonPhy>=0) pPH = particles->at( theMappedPartonPhy );
    if(theMappedPartonAlg>=0) pAL = particles->at( theMappedPartonAlg );

    (*jetMatchedPartons)[jets_h->refAt(j)]=MatchedPartons(pHV,pN2,pN3,pPH,pAL);
  }
  
  iEvent.put(  jetMatchedPartons );

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
int JetPartonMatcher::fillAlgoritDefinition( const Jet& theJet ) {

  int tempParticle = -1;
  int tempPartonHighestPt = -1;
  int tempNearest = -1;
  float maxPt = 0;
  float minDr = 1000;
  for( size_t m = 0; m != particles->size(); ++ m ) {
    const Candidate & aParton = *(particles->at(m).get());
    if( aParton.numberOfDaughters() > 0  && ( aParton.daughter(0)->pdgId() == 91 || aParton.daughter(0)->pdgId() == 92 ) ) {
      double dist = DeltaR( theJet.p4(), aParton.p4() );
      if( dist <= coneSizeToAssociate ) {
        if( dist < minDr ) {
           minDr = dist;
           tempNearest = m;
        }
        if( tempParticle == -1 && ( abs( aParton.pdgId() ) == 4 )  ) tempParticle = m;
        if(                         abs( aParton.pdgId() ) == 5    ) tempParticle = m;
        if( aParton.pt() > maxPt ) {
           maxPt = aParton.pt();
           tempPartonHighestPt = m;
        }
      }
    }
  }
  theHeaviest = tempParticle;
  theHardest  = tempPartonHighestPt;
  theNearest2 = tempNearest;
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
int JetPartonMatcher::fillPhysicsDefinition( const Jet& theJet ) {

  float TheBiggerConeSize = 0.7; // In HepMC it's 0.3 --> it's a mistake: value has to be 0.7
  int tempParticle = -1;
  int nInTheCone = 0;
  int tempNearest = -1;
  float minDr = 1000;

  vector<const reco::Candidate *> theContaminations;
  theContaminations.clear();

  for( size_t m = 0; m != particles->size(); ++ m ) {
    //skip first 6 particles (2 protons + 4 initial state partons)
    if(m<6) continue;
    const Candidate & aParticle = *(particles->at(m).get());
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
    if( aParticle.status() == 3 && dist < minDr ) {
      minDr = dist;
      tempNearest = m;
    }
    if( aParticle.status() == 3 && dist <= coneSizeToAssociate ) {
      //cout << "particle in small cone=" << aParticle.pdgId() << endl;
      tempParticle = m;
      nInTheCone++;
    }
    // Look for partons in TheBiggerConeSize now
    if( aParticle.numberOfDaughters() > 0  && ( aParticle.daughter(0)->pdgId() == 91 || aParticle.daughter(0)->pdgId() == 92 ) ) {
      if( dist < TheBiggerConeSize ) theContaminations.push_back( &aParticle );
    }
  }

  theNearest3 = tempNearest;

  if(nInTheCone != 1) return -1; // rejected --> only one initialParton requested
  if(theContaminations.size() == 0 ) return tempParticle; //no contamination
  int initialPartonFlavour = abs( (particles->at(tempParticle).get()) ->pdgId() );

  vector<const Candidate *>::const_iterator itCont = theContaminations.begin();
  for( ; itCont != theContaminations.end(); itCont++ ) {
    int contaminatingFlavour = abs( (*itCont)->pdgId() );
    if( (*itCont)->numberOfMothers()>0 && (*itCont)->mother(0) == particles->at(tempParticle).get() ) continue; // mother is the initialParton --> OK
    if( initialPartonFlavour == 4 ) {  
      if( contaminatingFlavour == 4 ) continue; // keep association --> the initialParton is a c --> the contaminated parton is a c
      tempParticle = -1; // all the other cases reject!
      return tempParticle;
    }
  } 

  return tempParticle;   
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetPartonMatcher);

