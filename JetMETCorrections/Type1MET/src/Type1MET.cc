// -*- C++ -*-
//
// Package:    Type1MET
// Class:      Type1MET
// 
/**\class Type1MET Type1MET.cc JetMETCorrections/Type1MET/src/Type1MET.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Oct 12 08:23
//         Created:  Wed Oct 12 12:16:04 CDT 2005
// $Id: Type1MET.cc,v 1.3 2006/04/24 07:42:16 cavana Exp $
//
//


// system include files
#include <memory>

#include <string.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Handle.h"

#include "JetMETCorrections/Type1MET/interface/Type1METAlgo.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
//#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

//using namespace std;

namespace cms 
{
  // PRODUCER CLASS DEFINITION -------------------------------------
  class Type1MET : public edm::EDProducer 
  {
  public:
    explicit Type1MET( const edm::ParameterSet& );
    explicit Type1MET();
    virtual ~Type1MET();
    virtual void produce( edm::Event&, const edm::EventSetup& );
  private:
    Type1METAlgo alg_;
    std::string inputLabel;
    std::string inputUncorrCandLabel;
    std::string inputCorrCandLabel;
  };

  // PRODUCER CONSTRUCTORS ------------------------------------------
  Type1MET::Type1MET( const edm::ParameterSet& iConfig ) : alg_() 
  {
    produces<METCollection>();
    inputLabel = iConfig.getParameter<std::string>("inputUncorrMetLabel");
    inputUncorrCandLabel = iConfig.getParameter<std::string>("inputUncorrCandLabel");
    inputCorrCandLabel = iConfig.getParameter<std::string>("inputCorrCandLabel");
  }
  Type1MET::Type1MET() : alg_() {}
  // PRODUCER DESTRUCTORS -------------------------------------------
  Type1MET::~Type1MET() {}

  // PRODUCER METHODS -----------------------------------------------
  void Type1MET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    using namespace edm;
    Handle<CaloMETCollection> pIn;                                //Define Inputs
    iEvent.getByLabel(inputLabel, pIn);                       //Get Inputs
    std::auto_ptr<METCollection> pOut( new METCollection() ); //Create empty output
    
    edm::Handle<CaloJetCollection> inputUncorrJetCollection;
    iEvent.getByLabel( inputUncorrCandLabel, inputUncorrJetCollection );
    vector <const CaloJet*> inputUncorrJet;
    inputUncorrJet.reserve( inputUncorrJetCollection->size() );
    CaloJetCollection::const_iterator input_object = inputUncorrJetCollection->begin();
    for( ; input_object != inputUncorrJetCollection->end(); input_object++ )
      inputUncorrJet.push_back( &*input_object );
    
    edm::Handle<CaloJetCollection> inputCorrJetCollection;
    iEvent.getByLabel( inputCorrCandLabel, inputCorrJetCollection );
    vector <const CaloJet*> inputCorrJet;
    inputCorrJet.reserve( inputCorrJetCollection->size() );
    input_object = inputCorrJetCollection->begin();
    for( ; input_object != inputCorrJetCollection->end(); input_object++ )
      inputCorrJet.push_back( &*input_object );
    
    alg_.run( pIn.product(), inputUncorrJet, inputCorrJet, *pOut );                         //Invoke the algorithm
    iEvent.put( pOut );                                       //Put output into Event
  }

  //define this as a plug-in
  DEFINE_FWK_MODULE(Type1MET)

}//end namespace cms

