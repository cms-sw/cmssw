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
// $Id: Type1MET.cc,v 1.14 2007/10/09 18:09:44 scurlock Exp $
//
//


// user include files
#include "Type1MET.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"



//using namespace std;

namespace cms 
{
  // PRODUCER CONSTRUCTORS ------------------------------------------
  Type1MET::Type1MET( const edm::ParameterSet& iConfig ) : alg_() 
  {
    metType             = iConfig.getParameter<std::string>("metType");
    
    inputUncorMetLabel  = iConfig.getParameter<std::string>("inputUncorMetLabel");
    inputUncorJetsLabel = iConfig.getParameter<std::string>("inputUncorJetsLabel");
    correctorLabel   = iConfig.getParameter<std::string>("corrector");
    jetPTthreshold      = iConfig.getParameter<double>("jetPTthreshold");
    jetEMfracLimit      = iConfig.getParameter<double>("jetEMfracLimit");
    if( metType == "CaloMET" )
      produces<CaloMETCollection>();
    else
      produces<METCollection>();
  }
  Type1MET::Type1MET() : alg_() {}
  // PRODUCER DESTRUCTORS -------------------------------------------
  Type1MET::~Type1MET() {}

  // PRODUCER METHODS -----------------------------------------------
  void Type1MET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    using namespace edm;
    Handle<CaloJetCollection> inputUncorJets;
    iEvent.getByLabel( inputUncorJetsLabel, inputUncorJets );
    const JetCorrector* corrector = JetCorrector::getJetCorrector (correctorLabel, iSetup);
    if( metType == "CaloMET")
      {
	Handle<CaloMETCollection> inputUncorMet;                     //Define Inputs
	iEvent.getByLabel( inputUncorMetLabel,  inputUncorMet );     //Get Inputs
	std::auto_ptr<CaloMETCollection> output( new CaloMETCollection() );  //Create empty output
	alg_.run( *(inputUncorMet.product()), *corrector, *(inputUncorJets.product()), 
		  jetPTthreshold, jetEMfracLimit, 
		  &*output );                                         //Invoke the algorithm
	iEvent.put( output );                                        //Put output into Event
      }
    else
      {
	Handle<METCollection> inputUncorMet;                     //Define Inputs
	iEvent.getByLabel( inputUncorMetLabel,  inputUncorMet );     //Get Inputs
	std::auto_ptr<METCollection> output( new METCollection() );  //Create empty output
	alg_.run( *(inputUncorMet.product()), *corrector, *(inputUncorJets.product()), 
		  jetPTthreshold, jetEMfracLimit,
		  &*output );                                         //Invoke the algorithm
	iEvent.put( output );                                        //Put output into Event
      }
  }

  //  DEFINE_FWK_MODULE(Type1MET);  //define this as a plug-in

}//end namespace cms

