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
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "SimDataFormats/JetMatching/interface/MatchedPartons.h"
#include "SimDataFormats/JetMatching/interface/JetMatchedPartons.h"

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

    Handle<JetMatchedPartonsCollection> theTagByRef;
    InputTag sourceByRefer_;
    bool physDefinition;

};

//=========================================================================

JetFlavourIdentifier::JetFlavourIdentifier( const edm::ParameterSet& iConfig )
{
    produces<JetFlavourMatchingCollection>();
    sourceByRefer_ = iConfig.getParameter<InputTag>("srcByReference");
    physDefinition = iConfig.getParameter<bool>("physicsDefinition");
}

//=========================================================================

JetFlavourIdentifier::~JetFlavourIdentifier() 
{
}

// ------------ method called to produce the data  ------------

void JetFlavourIdentifier::produce( Event& iEvent, const EventSetup& iEs ) 
{

  iEvent.getByLabel (sourceByRefer_ , theTagByRef   );  

  JetFlavourMatchingCollection * jfmc;
  if(theTagByRef.product()->size()>0) {
    RefToBase<Jet> jj = theTagByRef->begin()->first;
    jfmc = new JetFlavourMatchingCollection(RefToBaseProd<Jet>(jj));
  } else {
    jfmc = new JetFlavourMatchingCollection();
  }
  auto_ptr<reco::JetFlavourMatchingCollection> jetFlavMatching(jfmc);

  for ( JetMatchedPartonsCollection::const_iterator j  = theTagByRef->begin();
                                                    j != theTagByRef->end();
                                                    j ++ ) {

    const MatchedPartons aMatch = (*j).second;

    math::XYZTLorentzVector thePartonLorentzVector(0,0,0,0);
    math::XYZPoint          thePartonVertex(0,0,0);
    int                     thePartonFlavour = 0;  

    if( physDefinition ) {
      const GenParticleRef aPartPhy = aMatch.physicsDefinitionParton() ;
      if(aPartPhy.isNonnull()) {  
        thePartonLorentzVector = aPartPhy.get()->p4();         
        thePartonVertex        = aPartPhy.get()->vertex();
        thePartonFlavour       = aPartPhy.get()->pdgId(); 
      }
    }
    if( !physDefinition ) {
      const GenParticleRef aPartAlg = aMatch.algoDefinitionParton() ;
      if(aPartAlg.isNonnull()) {
        thePartonLorentzVector = aPartAlg.get()->p4();
        thePartonVertex        = aPartAlg.get()->vertex();
        thePartonFlavour       = aPartAlg.get()->pdgId();
      }
    }

    (*jetFlavMatching)[(*j).first]=JetFlavour(thePartonLorentzVector, thePartonVertex, thePartonFlavour); 

  }

  iEvent.put(  jetFlavMatching );

}

//define this as a plug-in
DEFINE_FWK_MODULE(JetFlavourIdentifier);

