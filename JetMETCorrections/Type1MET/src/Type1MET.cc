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
// $Id: Type1MET.cc,v 1.26 2011/08/23 19:43:46 lacroix Exp $
//
//


// user include files
#include "JetMETCorrections/Type1MET/interface/Type1MET.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
//#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"


using namespace reco;

namespace cms 
{
  // PRODUCER CONSTRUCTORS ------------------------------------------
  Type1MET::Type1MET( const edm::ParameterSet& iConfig ) : alg_() 
  {
    metType             = iConfig.getParameter<std::string>("metType");
    
    inputUncorMetLabel  = iConfig.getParameter<std::string>("inputUncorMetLabel");
    inputUncorJetsLabel = iConfig.getParameter<std::string>("inputUncorJetsLabel");
    edm::InputTag inputUncorUnlusteredLabelDefault("", "");
    inputUncorUnlusteredLabel = iConfig.getUntrackedParameter<edm::InputTag> ("inputUncorUnlusteredLabel", inputUncorUnlusteredLabelDefault);
    jetPTthreshold      = iConfig.getParameter<double>("jetPTthreshold");
    jetEMfracLimit      = iConfig.getParameter<double>("jetEMfracLimit");
    UscaleA      = iConfig.getParameter<double>("UscaleA");
    UscaleB      = iConfig.getParameter<double>("UscaleB");
    UscaleC      = iConfig.getParameter<double>("UscaleC");
    useTypeII      = iConfig.getParameter<bool>("useTypeII");
    hasMuonsCorr   = iConfig.getParameter<bool>("hasMuonsCorr");
    correctorLabel   = iConfig.getParameter<std::string>("corrector");
    refcorrectorLabel   = iConfig.getParameter<std::string>("refcorrector");
    subtractL1Fast_ = iConfig.getParameter<bool>("subtractL1Fast");
    if( metType == "CaloMET" )
      produces<CaloMETCollection>();
    else
      produces<PFMETCollection>();
  }
  Type1MET::Type1MET() : alg_() {}
  // PRODUCER DESTRUCTORS -------------------------------------------
  Type1MET::~Type1MET() {}

  // PRODUCER METHODS -----------------------------------------------
  void Type1MET::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
  {
    using namespace edm;

    if( metType == "CaloMET")
      {

       Handle<View<reco::Muon> > inputMuons;
       iEvent.getByLabel( "muons", inputMuons );
       // Remove lable from the code
       Handle<ValueMap<reco::MuonMETCorrectionData> > vm_muCorrData_h;
       iEvent.getByLabel( "muonMETValueMapProducer","muCorrData", vm_muCorrData_h);

       Handle<CaloJetCollection> inputUncorJets;
       iEvent.getByLabel( inputUncorJetsLabel, inputUncorJets );
       const JetCorrector* corrector = JetCorrector::getJetCorrector (correctorLabel, iSetup);
       const JetCorrector* corrector2=NULL;
       if (subtractL1Fast_)corrector2 = JetCorrector::getJetCorrector (refcorrectorLabel, iSetup);

       Handle<CaloMETCollection> inputUncorMet;                     //Define Inputs
       iEvent.getByLabel( inputUncorMetLabel,  inputUncorMet );     //Get Inputs
       std::auto_ptr<CaloMETCollection> output( new CaloMETCollection() );  //Create empty output
       alg_.run( *(inputUncorMet.product()), *corrector, *corrector2, *(inputUncorJets.product()), 
		 jetPTthreshold, jetEMfracLimit, UscaleA, UscaleB, UscaleC, useTypeII, hasMuonsCorr,
		 *(inputMuons.product()), *(vm_muCorrData_h.product()),
		 &*output, iEvent, iSetup, subtractL1Fast_ );       //Invoke the algorithm
       iEvent.put( output );                                        //Put output into Event
      }
    else if (metType == "PFMET")
      {
      Handle<PFJetCollection> inputUncorJets;
      iEvent.getByLabel( inputUncorJetsLabel, inputUncorJets );
      const JetCorrector* corrector = JetCorrector::getJetCorrector (correctorLabel, iSetup);
      const JetCorrector* corrector2=NULL;
      if (subtractL1Fast_)corrector2 = JetCorrector::getJetCorrector (refcorrectorLabel, iSetup);

      Handle < PFCandidateCollection > inputUncorUnlustered;
      if (useTypeII) {
	iEvent.getByLabel(inputUncorUnlusteredLabel, inputUncorUnlustered);
	  if (!inputUncorUnlustered.isValid()) {
	   useTypeII = false;
	  }
      }

      Handle<PFMETCollection> inputUncorMet;                     //Define Inputs
      iEvent.getByLabel( inputUncorMetLabel,  inputUncorMet );     //Get Inputs
      std::auto_ptr<PFMETCollection> output( new PFMETCollection() );  //Create empty output
      alg_.run( *(inputUncorMet.product()), *corrector, *corrector2, *(inputUncorJets.product()), *(inputUncorUnlustered.product()),
		jetPTthreshold, jetEMfracLimit, UscaleA, UscaleB, UscaleC, useTypeII,
		&*output, iEvent, iSetup, subtractL1Fast_ );       //Invoke the algorithm
      iEvent.put( output );                                        //Put output into Event
      }
//    else
//      {
//	Handle<pat::JetCollection> inputUncorJets;
//	iEvent.getByLabel( inputUncorJetsLabel, inputUncorJets );
//	const JetCorrector* corrector = JetCorrector::getJetCorrector (correctorLabel, iSetup);
//	const JetCorrector* corrector2=NULL;
//	if (subtractL1Fast_)corrector2 = JetCorrector::getJetCorrector (refcorrectorLabel, iSetup);
//
//	Handle<PFMETCollection> inputUncorMet;                     //Define Inputs
//	iEvent.getByLabel( inputUncorMetLabel,  inputUncorMet );     //Get Inputs
//	std::auto_ptr<PFMETCollection> output( new PFMETCollection() );  //Create empty output
//	alg_.run( *(inputUncorMet.product()), *corrector, *(inputUncorJets.product()), 
//		  jetPTthreshold, jetEMfracLimit, UscaleA, UscaleB, UscaleC, useTypeII,
//		  &*output, iEvent, iSetup );                        //Invoke the algorithm
//	iEvent.put( output );                                        //Put output into Event
//      }
  }

  //  DEFINE_FWK_MODULE(Type1MET);  //define this as a plug-in

}//end namespace cms

