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

//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

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

class JetFlavourIdentifier : public edm::EDProducer 
{
  public:
    JetFlavourIdentifier( const edm::ParameterSet & );
    ~JetFlavourIdentifier();

  private:
    virtual void produce(edm::Event&, const edm::EventSetup& );

    typedef std::vector<std::pair<reco::CaloJetRef, reco::JetFlavour> > JetTagVector;

    int fillAlgoritDefinition( const Jet& ) const;
    int fillPhysicsDefinition( const Jet& ) const;

    Handle<CaloJetCollection>   jets;
    Handle<CandidateCollection> particles;

    edm::InputTag m_jetsSrc;
    double coneSizeToAssociate;
    bool physDefinition;

    math::XYZTLorentzVector thePartonLorentzVector;
    math::XYZPoint          thePartonVertex;
    int                     thePartonFlavour;        

};

//=========================================================================

JetFlavourIdentifier::JetFlavourIdentifier( const edm::ParameterSet& iConfig )
{
    produces<JetTagVector>();
    m_jetsSrc           = iConfig.getParameter<edm::InputTag>("jets");
    coneSizeToAssociate = iConfig.getParameter<double>("coneSizeToAssociate");
    physDefinition      = iConfig.getParameter<bool>("physicsDefinition");
}

//=========================================================================

JetFlavourIdentifier::~JetFlavourIdentifier() 
{
}

// ------------ method called to produce the data  ------------

void JetFlavourIdentifier::produce( Event& iEvent, const EventSetup& iEs ) 
{

  iEvent.getByLabel(m_jetsSrc, jets);
  iEvent.getByLabel ("genParticleCandidates", particles );

  auto_ptr<JetTagVector> theJetTagVector( new JetTagVector );
  int size = jets->size();
  theJetTagVector->reserve(size);

  for (size_t j = 0; j < jets->size(); j++) {
       const int theMappedPartonAlg = fillAlgoritDefinition( (*jets)[j] );
       const int theMappedPartonPhy = fillPhysicsDefinition( (*jets)[j] );

       if(physDefinition) {
         if ( theMappedPartonPhy < 0 ) continue;
         const Candidate & aPartPhy = (*particles)[theMappedPartonPhy] ;
         thePartonLorentzVector = aPartPhy.p4();         
         thePartonVertex        = aPartPhy.vertex();
         thePartonFlavour       = aPartPhy.pdgId();         
       } else {
         if ( theMappedPartonAlg < 0 ) continue;
         const Candidate & aPartAlg = (*particles)[theMappedPartonAlg] ; 
         thePartonLorentzVector = aPartAlg.p4();
         thePartonVertex        = aPartAlg.vertex();
         thePartonFlavour       = aPartAlg.pdgId();
       }

       auto_ptr<JetFlavour> matchJetParton( new JetFlavour(thePartonLorentzVector, thePartonVertex, thePartonFlavour) );
       theJetTagVector->push_back( make_pair( CaloJetRef(jets,j) , *matchJetParton ) ); 

  }

  iEvent.put( theJetTagVector );

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
int JetFlavourIdentifier::fillAlgoritDefinition( const Jet& theJet ) const {

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
int JetFlavourIdentifier::fillPhysicsDefinition( const Jet& theJet ) const {

  float TheBiggerConeSize = 0.7; // In HepMC it's 0.3 --> it's a mistake: value has to be 0.7
  int tempParticle = -1;
  int nInTheCone = 0;

  vector<const reco::Candidate *> theContaminations;
  theContaminations.clear();

  for( size_t m = 0; m != particles->size(); ++ m ) {
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
    // Look for partons in TheBiggerConeSize now
    if( aParticle.numberOfDaughters() > 0  && ( aParticle.daughter(0)->pdgId() == 91 || aParticle.daughter(0)->pdgId() == 92 ) ) {
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
    }
  } 

  return tempParticle;   
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetFlavourIdentifier);

