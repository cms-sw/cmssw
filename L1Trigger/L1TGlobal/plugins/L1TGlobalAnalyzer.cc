// -*- C++ -*-
//
// Package:    L1Trigger/L1TGloba
// Class:      L1TGlobalAnalyzer
// 
/**\class L1TGlobalAnalyzer L1TGlobalAnalyzer.cc L1Trigger/L1TGlobal/plugins/L1TGlobalAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  James Brooke
//         Created:  Tue, 11 Mar 2014 14:55:45 GMT
//
// Modifying Author:  Brian Winer
//         Created: Tue, 10 Mar 2015 based off L1TCaloAnalyzer
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "TH1F.h"
#include "TH2F.h"

//
// class declaration
//

namespace l1t {

class L1TGlobalAnalyzer : public edm::EDAnalyzer {
public:
  explicit L1TGlobalAnalyzer(const edm::ParameterSet&);
  ~L1TGlobalAnalyzer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  
  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------
  edm::EDGetToken m_dmxEGToken;
  edm::EDGetToken m_dmxTauToken;
  edm::EDGetToken m_dmxJetToken;
  edm::EDGetToken m_dmxSumToken;
  edm::EDGetToken m_egToken;
  edm::EDGetToken m_tauToken;
  edm::EDGetToken m_jetToken;
  edm::EDGetToken m_sumToken;

  edm::EDGetToken m_gtAlgToken;
  edm::EDGetToken m_emulGtAlgToken;
  edm::EDGetToken m_emulDxAlgToken;


  bool m_doDmxEGs;
  bool m_doDmxTaus;
  bool m_doDmxJets;
  bool m_doDmxSums;
  bool m_doEGs;
  bool m_doTaus;
  bool m_doJets;
  bool m_doSums;

  bool m_doGtAlg;
  bool m_doEmulGtAlg;
  bool m_doEmulDxAlg;
  
  bool doText_;
  bool doHistos_;
  bool doEvtDisp_;

  enum ObjectType{
		  EG=0x1,
		  Tau=0x2,
		  Jet=0x3,
		  Sum=0x4,
		  DmxEG=0x5,
		  DmxTau=0x6,
		  DmxJet=0x7,
		  DmxSum=0x8,
                  GtAlg=0x9,
                  EmulGtAlg=0x10};
  
  std::vector< ObjectType > types_;
  std::vector< std::string > typeStr_;
  
  std::map< ObjectType, TFileDirectory > dirs_;
  std::map< ObjectType, TH1F* > het_;
  std::map< ObjectType, TH1F* > heta_;
  std::map< ObjectType, TH1F* > hphi_;
  std::map< ObjectType, TH1F* > hbx_;
  std::map< ObjectType, TH2F* > hetaphi_;

  TFileDirectory evtDispDir_;
  TFileDirectory algDir_;
  TFileDirectory dmxVGtDir_;
  TH1F* hAlgoBits_;
  TH1F* hEmulGtAlgoBits_;
  TH1F* hEmulDxAlgoBits_;
  TH2F* hAlgoBitsEmulGtVsHw_;
  TH2F* hAlgoBitsEmulDxVsHw_;
  TH2F* hDmxVsGTJetEt_;
  TH2F* hDmxVsGTJetEta_;
  TH2F* hDmxVsGTJetPhi_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
  L1TGlobalAnalyzer::L1TGlobalAnalyzer(const edm::ParameterSet& iConfig) :
    doText_(iConfig.getUntrackedParameter<bool>("doText", true)),
    doHistos_(iConfig.getUntrackedParameter<bool>("doHistos", true))
{
   //now do what ever initialization is needed

  // register what you consume and keep token for later access:
  edm::InputTag nullTag("None");

  edm::InputTag dmxEGTag  = iConfig.getParameter<edm::InputTag>("dmxEGToken");
  m_dmxEGToken          = consumes<l1t::EGammaBxCollection>(dmxEGTag);
  m_doDmxEGs            = !(dmxEGTag==nullTag);

  edm::InputTag dmxTauTag = iConfig.getParameter<edm::InputTag>("dmxTauToken");
  m_dmxTauToken         = consumes<l1t::TauBxCollection>(dmxTauTag);
  m_doDmxTaus           = !(dmxTauTag==nullTag);

  edm::InputTag dmxJetTag = iConfig.getParameter<edm::InputTag>("dmxJetToken");
  m_dmxJetToken         = consumes<l1t::JetBxCollection>(dmxJetTag);
  m_doDmxJets           = !(dmxJetTag==nullTag);  

  edm::InputTag dmxSumTag = iConfig.getParameter<edm::InputTag>("dmxEtSumToken");
  m_dmxSumToken         = consumes<l1t::EtSumBxCollection>(dmxSumTag);
  m_doDmxSums           = !(dmxSumTag==nullTag);

  edm::InputTag egTag  = iConfig.getParameter<edm::InputTag>("egToken");
  m_egToken          = consumes<l1t::EGammaBxCollection>(egTag);
  m_doEGs            = !(egTag==nullTag);

  edm::InputTag tauTag = iConfig.getParameter<edm::InputTag>("tauToken");
  m_tauToken         = consumes<l1t::TauBxCollection>(tauTag);
  m_doTaus           = !(tauTag==nullTag);

  edm::InputTag jetTag = iConfig.getParameter<edm::InputTag>("jetToken");
  m_jetToken         = consumes<l1t::JetBxCollection>(jetTag);
  m_doJets           = !(jetTag==nullTag);  

  edm::InputTag sumTag = iConfig.getParameter<edm::InputTag>("etSumToken");
  m_sumToken         = consumes<l1t::EtSumBxCollection>(sumTag);
  m_doSums           = !(sumTag==nullTag);

  edm::InputTag gtAlgTag = iConfig.getParameter<edm::InputTag>("gtAlgToken");
  m_gtAlgToken         = consumes<GlobalAlgBlkBxCollection>(gtAlgTag);
  m_doGtAlg            = !(gtAlgTag==nullTag); 

  edm::InputTag emulGtAlgTag = iConfig.getParameter<edm::InputTag>("emulGtAlgToken");
  m_emulGtAlgToken         = consumes<GlobalAlgBlkBxCollection>(emulGtAlgTag);
  m_doEmulGtAlg            = !(emulGtAlgTag==nullTag);

  edm::InputTag emulDxAlgTag = iConfig.getParameter<edm::InputTag>("emulDxAlgToken");
  m_emulDxAlgToken         = consumes<GlobalAlgBlkBxCollection>(emulDxAlgTag);
  m_doEmulDxAlg            = !(emulDxAlgTag==nullTag);

  types_.push_back( DmxEG );
  types_.push_back( DmxTau );
  types_.push_back( DmxJet );
  types_.push_back( DmxSum );
  types_.push_back( EG );
  types_.push_back( Tau );
  types_.push_back( Jet );
  types_.push_back( Sum );

  typeStr_.push_back( "dmxeg" );
  typeStr_.push_back( "dmxtau" );
  typeStr_.push_back( "dmxjet" );
  typeStr_.push_back( "dmxsum" );
  typeStr_.push_back( "eg" );
  typeStr_.push_back( "tau" );
  typeStr_.push_back( "jet" );
  typeStr_.push_back( "sum" );

}


L1TGlobalAnalyzer::~L1TGlobalAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
L1TGlobalAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  std::stringstream text;

  TH2F* hEvtEG = new TH2F();
  TH2F* hEvtTau = new TH2F();
  TH2F* hEvtJet = new TH2F();
  TH2F* hEvtDmxEG = new TH2F();
  TH2F* hEvtDmxTau = new TH2F();
  TH2F* hEvtDmxJet = new TH2F();

  if (doEvtDisp_) {
    std::stringstream ss;
    ss << iEvent.run() << "-" << iEvent.id().event();
    TFileDirectory dir = evtDispDir_.mkdir(ss.str());
    hEvtEG = dir.make<TH2F>("EG", "", 83, -41.5, 41.5, 72, .5, 72.5);
    hEvtTau = dir.make<TH2F>("Tau", "", 83, -41.5, 41.5, 72, .5, 72.5);
    hEvtJet = dir.make<TH2F>("Jet", "", 83, -41.5, 41.5, 72, .5, 72.5);
    hEvtDmxEG = dir.make<TH2F>("DmxEG", "", 227, -113.5, 113.5, 144, -0.5, 143.5);
    hEvtDmxTau = dir.make<TH2F>("DmxTau", "", 227, -113.5, 113.5, 144, -0.5, 143.5);
    hEvtDmxJet = dir.make<TH2F>("DmxJet", "", 227, -113.5, 113.5, 144, -0.5, 143.5);
  }

  // get EG
  if (m_doDmxEGs) {
    Handle< BXVector<l1t::EGamma> > dmxegs;
    iEvent.getByToken(m_dmxEGToken,dmxegs);
    
    for ( int ibx=dmxegs->getFirstBX(); ibx<=dmxegs->getLastBX(); ++ibx) {

      for ( auto itr = dmxegs->begin(ibx); itr != dmxegs->end(ibx); ++itr ) {
        hbx_.at(DmxEG)->Fill( ibx );
	het_.at(DmxEG)->Fill( itr->hwPt() );
	heta_.at(DmxEG)->Fill( itr->hwEta() );
	hphi_.at(DmxEG)->Fill( itr->hwPhi() );
        hetaphi_.at(DmxEG)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	text << "Dmx EG : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;      

	if (doEvtDisp_) hEvtDmxEG->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

      }
      
    }

  }

  // get tau
  if (m_doDmxTaus) {
    Handle< BXVector<l1t::Tau> > dmxtaus;
    iEvent.getByToken(m_dmxTauToken,dmxtaus);
    
    for ( int ibx=dmxtaus->getFirstBX(); ibx<=dmxtaus->getLastBX(); ++ibx) {

      for ( auto itr = dmxtaus->begin(ibx); itr != dmxtaus->end(ibx); ++itr ) {
        hbx_.at(DmxTau)->Fill( ibx );
	het_.at(DmxTau)->Fill( itr->hwPt() );
	heta_.at(DmxTau)->Fill( itr->hwEta() );
	hphi_.at(DmxTau)->Fill( itr->hwPhi() );
        hetaphi_.at(DmxTau)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	text << "Dmx Tau : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;      

	if (doEvtDisp_) hEvtDmxTau->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }
      
    }
    
  }

  // get jet
  if (m_doDmxJets) {
    Handle< BXVector<l1t::Jet> > dmxjets;
    iEvent.getByToken(m_dmxJetToken,dmxjets);
    
    for ( int ibx=dmxjets->getFirstBX(); ibx<=dmxjets->getLastBX(); ++ibx) {

      for ( auto itr = dmxjets->begin(ibx); itr != dmxjets->end(ibx); ++itr ) {
        hbx_.at(DmxJet)->Fill( ibx );
	het_.at(DmxJet)->Fill( itr->hwPt() );
	heta_.at(DmxJet)->Fill( itr->hwEta() );
	hphi_.at(DmxJet)->Fill( itr->hwPhi() );
        hetaphi_.at(DmxJet)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	text << "Dmx Jet : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

	if (doEvtDisp_) hEvtDmxJet->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }
      
    }

  }

  // get sums
  if (m_doDmxSums) {
    Handle< BXVector<l1t::EtSum> > dmxsums;
    iEvent.getByToken(m_dmxSumToken,dmxsums);
    
    for ( int ibx=dmxsums->getFirstBX(); ibx<=dmxsums->getLastBX(); ++ibx) {

      for ( auto itr = dmxsums->begin(ibx); itr != dmxsums->end(ibx); ++itr ) {
	hbx_.at(DmxSum)->Fill( ibx );
	het_.at(DmxSum)->Fill( itr->hwPt() );
	heta_.at(DmxSum)->Fill( itr->hwEta() );
	hphi_.at(DmxSum)->Fill( itr->hwPhi() );
        hetaphi_.at(DmxSum)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	text << "Dmx Sum : " << " type=" << itr->getType() << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

      }

    }

  }

  // get EG
  if (m_doEGs) {
    Handle< BXVector<l1t::EGamma> > egs;
    iEvent.getByToken(m_egToken,egs);
    
    for ( int ibx=egs->getFirstBX(); ibx<=egs->getLastBX(); ++ibx) {

      for ( auto itr = egs->begin(ibx); itr != egs->end(ibx); ++itr ) {
        hbx_.at(EG)->Fill( ibx );
	het_.at(EG)->Fill( itr->hwPt() );
	heta_.at(EG)->Fill( itr->hwEta() );
	hphi_.at(EG)->Fill( itr->hwPhi() );
        hetaphi_.at(EG)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	text << "EG : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

	if (doEvtDisp_) hEvtEG->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }
      
    }

  }

  // get tau
  if (m_doTaus) {
    Handle< BXVector<l1t::Tau> > taus;
    iEvent.getByToken(m_tauToken,taus);
    
    for ( int ibx=taus->getFirstBX(); ibx<=taus->getLastBX(); ++ibx) {

      for ( auto itr = taus->begin(ibx); itr != taus->end(ibx); ++itr ) {
        hbx_.at(Tau)->Fill( ibx );
	het_.at(Tau)->Fill( itr->hwPt() );
	heta_.at(Tau)->Fill( itr->hwEta() );
	hphi_.at(Tau)->Fill( itr->hwPhi() );
        hetaphi_.at(Tau)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	text << "Tau : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

	if (doEvtDisp_) hEvtTau->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }
      
    }
    
  }

  // get jet
  if (m_doJets) {
    Handle< BXVector<l1t::Jet> > jets;
    iEvent.getByToken(m_jetToken,jets);
    
    for ( int ibx=jets->getFirstBX(); ibx<=jets->getLastBX(); ++ibx) {

      for ( auto itr = jets->begin(ibx); itr != jets->end(ibx); ++itr ) {
        hbx_.at(Jet)->Fill( ibx );
	het_.at(Jet)->Fill( itr->hwPt() );
	heta_.at(Jet)->Fill( itr->hwEta() );
	hphi_.at(Jet)->Fill( itr->hwPhi() );
        hetaphi_.at(Jet)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );

	text << "Jet : " << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;

	if (doEvtDisp_) hEvtJet->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
      }
      
    }

  }

  // get sums
  if (m_doSums) {
    Handle< BXVector<l1t::EtSum> > sums;
    iEvent.getByToken(m_sumToken,sums);
    
    for ( int ibx=sums->getFirstBX(); ibx<=sums->getLastBX(); ++ibx) {

      for ( auto itr = sums->begin(ibx); itr != sums->end(ibx); ++itr ) {
	hbx_.at(Sum)->Fill( ibx );
	het_.at(Sum)->Fill( itr->hwPt() );
	heta_.at(Sum)->Fill( itr->hwEta() );
	hphi_.at(Sum)->Fill( itr->hwPhi() );
        hetaphi_.at(Sum)->Fill( itr->hwEta(), itr->hwPhi(), itr->hwPt() );
	text << "Sum : " << " type=" << itr->getType() << " BX=" << ibx << " ipt=" << itr->hwPt() << " ieta=" << itr->hwEta() << " iphi=" << itr->hwPhi() << std::endl;
      }

    }

  }
  
  //Algorithm Bits
  if (m_doGtAlg) {
    Handle< BXVector<GlobalAlgBlk> > algs;
    iEvent.getByToken(m_gtAlgToken,algs);
    
    for ( int ibx=algs->getFirstBX(); ibx<=algs->getLastBX(); ++ibx) {

      for ( auto itr = algs->begin(ibx); itr != algs->end(ibx); ++itr ) {
           
	for(int algBit=0; algBit<128; algBit++) {  //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
	  if(itr->getAlgoDecisionFinal(algBit)) hAlgoBits_->Fill(algBit);
        }
      }

    }

  }

  //Algorithm Bits (Emulation seeded by GT input)
  if (m_doEmulGtAlg) {
    Handle< BXVector<GlobalAlgBlk> > emulGtAlgs;
    iEvent.getByToken(m_emulGtAlgToken,emulGtAlgs);
    
    for ( int ibx=emulGtAlgs->getFirstBX(); ibx<=emulGtAlgs->getLastBX(); ++ibx) {

      for ( auto itr = emulGtAlgs->begin(ibx); itr != emulGtAlgs->end(ibx); ++itr ) {
           
	for(int algBit=0; algBit<128; algBit++) { //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
	  if(itr->getAlgoDecisionFinal(algBit)) hEmulGtAlgoBits_->Fill(algBit);
        }
      }

    }

  }

  //Algorithm Bits (Emulation seeded by Demux Output)
  if (m_doEmulDxAlg) {
    Handle< BXVector<GlobalAlgBlk> > emulDxAlgs;
    iEvent.getByToken(m_emulDxAlgToken,emulDxAlgs);
    
    for ( int ibx=emulDxAlgs->getFirstBX(); ibx<=emulDxAlgs->getLastBX(); ++ibx) {

      for ( auto itr = emulDxAlgs->begin(ibx); itr != emulDxAlgs->end(ibx); ++itr ) {
           
	for(int algBit=0; algBit<128; algBit++) { //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
	  if(itr->getAlgoDecisionFinal(algBit)) hEmulDxAlgoBits_->Fill(algBit);
        }
      }

    }

  }



  
  // Jets (Dmx vs GT)
  if (m_doJets & m_doDmxJets) {
    Handle< BXVector<l1t::Jet> > jets;
    iEvent.getByToken(m_jetToken,jets);

    Handle< BXVector<l1t::Jet> > dmxjets;
    iEvent.getByToken(m_dmxJetToken,dmxjets);
    
    for ( int ibx=jets->getFirstBX(); ibx<=jets->getLastBX(); ++ibx) {

      // Cycle through all GT jets
      for (unsigned int igtJ=0; igtJ<jets->size(ibx); igtJ++) {
 
	double gtJetEt = jets->at(ibx,igtJ).hwPt();
        double dmxJetEt = 0.0;
        if(dmxjets->size(ibx)>igtJ) dmxJetEt = dmxjets->at(ibx,igtJ).hwPt();
        hDmxVsGTJetEt_->Fill(gtJetEt,dmxJetEt);        

	double gtJetEta = jets->at(ibx,igtJ).hwEta();
        double dmxJetEta = 0.0;
        if(dmxjets->size(ibx)>igtJ) dmxJetEta = dmxjets->at(ibx,igtJ).hwEta();
        hDmxVsGTJetEta_->Fill(gtJetEta,dmxJetEta);        

	double gtJetPhi = jets->at(ibx,igtJ).hwPhi();
        double dmxJetPhi = 0.0;
        if(dmxjets->size(ibx)>igtJ) dmxJetPhi = dmxjets->at(ibx,igtJ).hwPhi();
        hDmxVsGTJetPhi_->Fill(gtJetPhi,dmxJetPhi);        



      }
      // if there are extra jets in the dmx record them
      for (unsigned int idmJ=jets->size(ibx); idmJ<dmxjets->size(ibx); idmJ++) {

        
	double gtJetEt = 0.0; //no GT jet exists
        double dmxJetEt = dmxjets->at(ibx,idmJ).hwPt();
        hDmxVsGTJetEt_->Fill(gtJetEt,dmxJetEt);        

	double gtJetEta = 0.0;
        double dmxJetEta = dmxjets->at(ibx,idmJ).hwEta();
        hDmxVsGTJetEta_->Fill(gtJetEta,dmxJetEta);        

	double gtJetPhi = 0.0;
        double dmxJetPhi  = dmxjets->at(ibx,idmJ).hwPhi();
        hDmxVsGTJetPhi_->Fill(gtJetPhi,dmxJetPhi);        
         
      }      
       
    }

  }
  
  
  //Algorithm Bits (Emulation vs HW)
  if (m_doGtAlg && m_doEmulGtAlg) {
    Handle< BXVector<GlobalAlgBlk> > hwalgs;
    iEvent.getByToken(m_gtAlgToken,hwalgs);

    Handle< BXVector<GlobalAlgBlk> > emulAlgs;
    iEvent.getByToken(m_emulGtAlgToken,emulAlgs);

       
    //for ( int ibx=hwalgs->getFirstBX(); ibx<=hwalgs->getLastBX(); ++ibx) {
    int ibx=0;

        auto itr = hwalgs->begin(ibx);
        auto itr_emul = emulAlgs->begin(ibx);        
   
	for(int algBit=0; algBit<128; algBit++) { //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
          bool hw = itr->getAlgoDecisionFinal(algBit);
          bool em = itr_emul->getAlgoDecisionFinal(algBit);
	  if(hw & em) {
	    hAlgoBitsEmulGtVsHw_->Fill(algBit,algBit);
	  }else if(hw & !em) {
            hAlgoBitsEmulGtVsHw_->Fill(algBit,-1.0);
	  }else if(!hw & em) {
            hAlgoBitsEmulGtVsHw_->Fill(-1.0,algBit);
	  }
        }
      
  // }

  }

  //Algorithm Bits (Emulation vs HW)
  if (m_doGtAlg && m_doEmulDxAlg) {
    Handle< BXVector<GlobalAlgBlk> > hwalgs;
    iEvent.getByToken(m_gtAlgToken,hwalgs);

    Handle< BXVector<GlobalAlgBlk> > emulAlgs;
    iEvent.getByToken(m_emulDxAlgToken,emulAlgs);

       
    //for ( int ibx=hwalgs->getFirstBX(); ibx<=hwalgs->getLastBX(); ++ibx) {
    int ibx=0;

        auto itr = hwalgs->begin(ibx);
        auto itr_emul = emulAlgs->begin(ibx);        
   
	for(int algBit=0; algBit<128; algBit++) { //Fix Me: Should access size of algo vector...need method in GlobalAlgBlk class
          bool hw = itr->getAlgoDecisionFinal(algBit);
          bool em = itr_emul->getAlgoDecisionFinal(algBit);
	  if(hw & em) {
	    hAlgoBitsEmulDxVsHw_->Fill(algBit,algBit);
	  }else if(hw & !em) {
            hAlgoBitsEmulDxVsHw_->Fill(algBit,-1.0);
	  }else if(!hw & em) {
            hAlgoBitsEmulDxVsHw_->Fill(-1.0,algBit);
	  }
        }
      
  // }

  }  

  


  if (doText_) edm::LogInfo("L1TGlobalEvents") << text.str();

}



// ------------ method called once each job just before starting event loop  ------------
void 
L1TGlobalAnalyzer::beginJob()
{

  edm::Service<TFileService> fs;

  auto itr = types_.cbegin();
  auto str = typeStr_.cbegin();

  for (; itr!=types_.end(); ++itr, ++str ) {
    
    double etmax=99.5;
    if (*itr==Jet || *itr==DmxJet || *itr==Sum || *itr==DmxSum) etmax=249.5;

    dirs_.insert( std::pair< ObjectType, TFileDirectory >(*itr, fs->mkdir(*str) ) );
    
    het_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("et", "", 100, -0.5, etmax) ));

    hbx_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("bx", "", 11, -5.5, 5.5) ));


    heta_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("eta", "", 227, -113.5, 113.5) ));
    hphi_.insert( std::pair< ObjectType, TH1F* >(*itr, dirs_.at(*itr).make<TH1F>("phi", "", 144, -0.5, 143.5) ));
    hetaphi_.insert( std::pair< ObjectType, TH2F* >(*itr, dirs_.at(*itr).make<TH2F>("etaphi", "", 227, -113.5, 113.5, 144, -0.5, 143.5) ));

  }

  algDir_ = fs->mkdir("Algorithms");
  hAlgoBits_ = algDir_.make<TH1F>("hAlgoBits","Algorithm Bits",100, -0.5,99.5);
  hEmulGtAlgoBits_ = algDir_.make<TH1F>("hEmulGtAlgoBits","GT Emulated Algorithm Bits",100, -0.5,99.5);
  hAlgoBitsEmulGtVsHw_ = algDir_.make<TH2F>("hAlgoBitsEmulGtVsHw","Algorithm Bits (GT) Emulation vs Hardware",101, -1.5,99.5,101,-1.5,99.5);
  hEmulDxAlgoBits_ = algDir_.make<TH1F>("hEmulDxAlgoBits","Dx Emulated Algorithm Bits",100, -0.5,99.5);
  hAlgoBitsEmulDxVsHw_ = algDir_.make<TH2F>("hAlgoBitsEmulDxVsHw","Algorithm Bits (Dx) Emulation vs Hardware",101, -1.5,99.5,101,-1.5,99.5);
 
  dmxVGtDir_ = fs->mkdir("DmxVsGT");
  hDmxVsGTJetEt_  = dmxVGtDir_.make<TH2F>("hDmxVsGTJetEt","Dmx Jet Et versus GT Jet Et",200,-0.5,199.5,200,-0.5,199.5);
  hDmxVsGTJetEta_ = dmxVGtDir_.make<TH2F>("hDmxVsGTJetEta","Dmx Jet Eta versus GT Jet Eta",227,-113.5,113.5,227,-113.5,113.5);
  hDmxVsGTJetPhi_ = dmxVGtDir_.make<TH2F>("hDmxVsGTJetPhi","Dmx Jet Phi versus GT Jet Phi",144,-0.5,143.5,144,-0.5,143.5);

  if (doEvtDisp_) {
    evtDispDir_ = fs->mkdir("Events");
  }

}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TGlobalAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
L1TGlobalAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
L1TGlobalAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
L1TGlobalAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
L1TGlobalAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TGlobalAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

}

using namespace l1t;

//define this as a plug-in
DEFINE_FWK_MODULE(L1TGlobalAnalyzer);
