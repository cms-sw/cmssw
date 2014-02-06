///
/// \class l1t::L1uGtRecordDump.cc
///
/// Description: Dump/Analyze Input Collections for GT.
///
/// Implementation:
///    Based off of Michael Mulhearn's YellowParamTester
///
/// \author: Brian Winer Ohio State
///


//
//  This simple module simply retreives the YellowParams object from the event
//  setup, and sends its payload as an INFO message, for debugging purposes.
//


#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
//#include "FWCore/ParameterSet/interface/InputTag.h"

// system include files
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/L1uGtRecBlk.h"
#include "DataFormats/L1Trigger/interface/L1uGtAlgBlk.h"
#include "DataFormats/L1Trigger/interface/L1uGtExtBlk.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

using namespace edm;
using namespace std;

namespace l1t {

  // class declaration
  class L1uGtRecordDump : public edm::EDAnalyzer {
  public:
    explicit L1uGtRecordDump(const edm::ParameterSet&);
    virtual ~L1uGtRecordDump(){};
    virtual void analyze(const edm::Event&, const edm::EventSetup&);  
    
    EDGetToken egToken;
    EDGetToken muToken;
    EDGetToken tauToken;
    EDGetToken jetToken;
    EDGetToken etsumToken;
    EDGetToken uGtRecToken; 
    EDGetToken uGtAlgToken;
    EDGetToken uGtExtToken;
  };

  L1uGtRecordDump::L1uGtRecordDump(const edm::ParameterSet& iConfig)
  {
      egToken     = consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<InputTag>("egInputTag"));
      muToken     = consumes<BXVector<l1t::Muon>>(iConfig.getParameter<InputTag>("muInputTag"));
      tauToken    = consumes<BXVector<l1t::Tau>>(iConfig.getParameter<InputTag>("tauInputTag"));
      jetToken    = consumes<BXVector<l1t::Jet>>(iConfig.getParameter<InputTag>("jetInputTag"));
      etsumToken  = consumes<BXVector<l1t::EtSum>>(iConfig.getParameter<InputTag>("etsumInputTag"));
      uGtRecToken = consumes<std::vector<L1uGtRecBlk>>(iConfig.getParameter<InputTag>("uGtRecInputTag"));
      uGtAlgToken = consumes<BXVector<L1uGtAlgBlk>>(iConfig.getParameter<InputTag>("uGtAlgInputTag"));
      uGtExtToken = consumes<BXVector<L1uGtExtBlk>>(iConfig.getParameter<InputTag>("uGtExtInputTag"));
  }
  
  // loop over events
  void L1uGtRecordDump::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup){

    
 //inputs
  Handle<BXVector<l1t::EGamma>> egammas;
  iEvent.getByToken(egToken,egammas);

  Handle<BXVector<l1t::Muon>> muons;
  iEvent.getByToken(muToken,muons);
 
   Handle<BXVector<l1t::Tau>> taus;
   iEvent.getByToken(tauToken,taus);

  Handle<BXVector<l1t::Jet>> jets;
  iEvent.getByToken(jetToken,jets);
 
  Handle<BXVector<l1t::EtSum>> etsums;
  iEvent.getByToken(etsumToken,etsums); 
  
  Handle<std::vector<L1uGtRecBlk>> uGtRec;
  iEvent.getByToken(uGtRecToken,uGtRec);   

  Handle<BXVector<L1uGtAlgBlk>> uGtAlg;
  iEvent.getByToken(uGtAlgToken,uGtAlg);   

  Handle<BXVector<L1uGtExtBlk>> uGtExt;
  iEvent.getByToken(uGtExtToken,uGtExt);   
  
 
    printf("\n -------------------------------------- \n");
    printf(" ***********  New Event  ************** \n");
    printf(" -------------------------------------- \n"); 
 //Loop over BX
    for(int i = egammas->getFirstBX(); i <= egammas->getLastBX(); ++i) {
    
       printf("\n ========== BX %i =============================\n",i);
    
       //Loop over EGamma
       printf(" ------ EGammas --------\n");
       for(std::vector<l1t::EGamma>::const_iterator eg = egammas->begin(i); eg != egammas->end(i); ++eg) {
           printf("   Pt %i Eta %i Phi %i Qual %i  Isol %i\n",eg->hwPt(),eg->hwEta(),eg->hwPhi(),eg->hwQual(),eg->hwIso());
       }    

       //Loop over Muons
       printf("\n ------ Muons --------\n");
       for(std::vector<l1t::Muon>::const_iterator mu = muons->begin(i); mu != muons->end(i); ++mu) {
           printf("   Pt %i Eta %i Phi %i Qual %i  Iso %i \n",mu->hwPt(),mu->hwEta(),mu->hwPhi(),mu->hwQual(),mu->hwIso());
       }

       //Loop over Taus
       printf("\n ------ Taus ----------\n");
       for(std::vector<l1t::Tau>::const_iterator tau = taus->begin(i); tau != taus->end(i); ++tau) {
           printf("   Pt %i Eta %i Phi %i Qual %i  Iso %i \n",tau->hwPt(),tau->hwEta(),tau->hwPhi(),tau->hwQual(),tau->hwIso());
       }        

       //Loop over Jets
       printf("\n ------ Jets ----------\n");
       for(std::vector<l1t::Jet>::const_iterator jet = jets->begin(i); jet != jets->end(i); ++jet) {
          printf("   Pt %i Eta %i Phi %i Qual %i \n",jet->hwPt(),jet->hwEta(),jet->hwPhi(),jet->hwQual());
       }
                  //Dump Content
	printf("\n ------ EtSums ----------\n");	  
       for(std::vector<l1t::EtSum>::const_iterator etsum = etsums->begin(i); etsum != etsums->end(i); ++etsum) {
            printf("   Pt %i Eta %i Phi %i Qual %i \n",etsum->hwPt(),etsum->hwEta(),etsum->hwPhi(),etsum->hwQual());
       }        

  // Dump the output record
 	printf("\n ------ uGtRec ----------\n");	  
       for(std::vector<L1uGtRecBlk>::const_iterator recBlk = uGtRec->begin(); recBlk != uGtRec->end(); ++recBlk) {
           recBlk->print(std::cout);
       }        
        
  // Dump the output record
 	printf("\n ------ uGtAlg ----------\n");	  
       for(std::vector<L1uGtAlgBlk>::const_iterator algBlk = uGtAlg->begin(i); algBlk != uGtAlg->end(i); ++algBlk) {
            algBlk->print(std::cout);
       }        
 
   // Dump the output record
 	printf("\n ------ uGtExt ----------\n");	  
       for(std::vector<L1uGtExtBlk>::const_iterator extBlk = uGtExt->begin(i); extBlk != uGtExt->end(i); ++extBlk) {
            extBlk->print(std::cout);
       }        
       
       

    }
    printf("\n");
  }

}


DEFINE_FWK_MODULE(l1t::L1uGtRecordDump);

