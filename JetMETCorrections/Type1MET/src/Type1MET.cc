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
// $Id$
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
#include "DataFormats/METObjects/interface/METCollection.h"

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
  };

  // PRODUCER CONSTRUCTORS ------------------------------------------
  Type1MET::Type1MET( const edm::ParameterSet& iConfig ) : alg_() 
  {
    produces<METCollection>();
    inputLabel = iConfig.getParameter<std::string>("inputLabel");
  }
  Type1MET::Type1MET() : alg_() {}
  // PRODUCER DESTRUCTORS -------------------------------------------
  Type1MET::~Type1MET() {}

  // PRODUCER METHODS -----------------------------------------------
  void Type1MET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    using namespace edm;
    Handle<METCollection> pIn;                                //Define Inputs
    iEvent.getByLabel(inputLabel, pIn);                       //Get Inputs
    std::auto_ptr<METCollection> pOut( new METCollection() ); //Create empty output
    alg_.run( pIn.product(), *pOut );                         //Invoke the algorithm
    iEvent.put( pOut );                                       //Put output into Event
  }

  //define this as a plug-in
  DEFINE_FWK_MODULE(Type1MET)

    }//end namespace cms

