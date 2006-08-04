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
// $Id: Type1MET.cc,v 1.4 2006/08/03 22:18:14 cavana Exp $
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
    std::string inputUncorMetLabel;
    std::string inputUncorJetsLabel;
    std::string inputCorJetsLabel;
  };

  // PRODUCER CONSTRUCTORS ------------------------------------------
  Type1MET::Type1MET( const edm::ParameterSet& iConfig ) : alg_() 
  {
    produces<METCollection>();
    inputUncorMetLabel  = iConfig.getParameter<std::string>("inputUncorrMetLabel");
    inputUncorJetsLabel = iConfig.getParameter<std::string>("inputUncorrJetsLabel");
    inputCorJetsLabel   = iConfig.getParameter<std::string>("inputCorrJetsLabel");
  }
  Type1MET::Type1MET() : alg_() {}
  // PRODUCER DESTRUCTORS -------------------------------------------
  Type1MET::~Type1MET() {}

  // PRODUCER METHODS -----------------------------------------------
  void Type1MET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    using namespace edm;
    Handle<CaloMETCollection> inputUncorMet;                     //Define Inputs
    Handle<CaloJetCollection> inputUncorJets;
    Handle<CaloJetCollection> inputCorJets;
    iEvent.getByLabel( inputUncorMetLabel,  inputUncorMet );     //Get Inputs
    iEvent.getByLabel( inputUncorJetsLabel, inputUncorJets );
    iEvent.getByLabel( inputCorJetsLabel,   inputCorJets );
    std::auto_ptr<METCollection> output( new METCollection() );  //Create empty output
    alg_.run( inputUncorMet.product(), inputUncorJets.product(), 
	      inputCorJets.product(), *output );                 //Invoke the algorithm
    iEvent.put( output );                                        //Put output into Event
  }

  DEFINE_FWK_MODULE(Type1MET)  //define this as a plug-in

}//end namespace cms

