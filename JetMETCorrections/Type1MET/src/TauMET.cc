// -*- C++ -*-
//
// Package:    TauMET
// Class:      TauMET
// 
/**\class TauMET TauMET.cc TauMET.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chi Nhan Nguyen
//         Created:  Mon Oct 22 15:20:51 CDT 2007
// $Id$
//
//


#include "TauMET.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CorrMETData.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

using namespace std;
namespace cms 
{

  TauMET::TauMET(const edm::ParameterSet& iConfig) : _algo()
  {

    _InputPFJetsLabel    = iConfig.getParameter<string>("InputPFJetsLabel");
    _InputCaloJetsLabel    = iConfig.getParameter<string>("InputCaloJetsLabel");
    _correctorLabel      = iConfig.getParameter<string>("correctorLabel");
    _UseCorrectedJets    = iConfig.getParameter<bool>("UseCorrectedJets");
    _JetMatchDeltaR      = iConfig.getParameter<double>("JetMatchDeltaR");

    produces< METCollection   >("DeltaTauMETCorr");

  }
  

  TauMET::~TauMET()
  {
    
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)
    
  }
  
  

  // ------------ method called to produce the data  ------------
  void
  TauMET::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {

    using namespace edm;

    Handle<PFJetCollection> pfjetHandle;
    iEvent.getByLabel(_InputPFJetsLabel, pfjetHandle);
    Handle<CaloJetCollection> calojetHandle;
    iEvent.getByLabel(_InputCaloJetsLabel, calojetHandle);
    const JetCorrector* correctedjets = JetCorrector::getJetCorrector (_correctorLabel, iSetup);

    //Handle<CaloMETCollection> metHandle;
    //iEvent.getByLabel(_InputTyp1MetLabel, metHandle);

    // Handle<JetTagCollection> jetTagHandle;
    // iEvent.getByLabel(_InputJetTagLabel, jetTagHandle);

    // std::cout << "I'm starting" << std::endl;

    std::auto_ptr< METCollection > output( new METCollection );
    _algo.run(iEvent,iSetup,pfjetHandle,calojetHandle,*correctedjets, _UseCorrectedJets, _JetMatchDeltaR,&*output);

    iEvent.put( output, "DeltaTauMETCorr" ); 
  }
  
  // ------------ method called once each job just before starting event loop  ------------
  void 
  TauMET::beginJob(const edm::EventSetup&)
  {
  }
  
  // ------------ method called once each job just after ending the event loop  ------------
  void 
  TauMET::endJob() {
  }

  //DEFINE_FWK_MODULE(TauMET);  //define this as a plug-in
}

  
