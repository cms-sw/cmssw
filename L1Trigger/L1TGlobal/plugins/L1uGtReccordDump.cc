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
#include <fstream>
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
    
    
    void dumpTestVectors(int bx, std::ofstream& myCout,
                         Handle<BXVector<l1t::Muon>> muons,
			 Handle<BXVector<l1t::EGamma>> egammas,
			 Handle<BXVector<l1t::Tau>> taus,
			 Handle<BXVector<l1t::Jet>> jets,
			 Handle<BXVector<l1t::EtSum>> etsums,
			 Handle<BXVector<L1uGtAlgBlk>> uGtAlg,
			 Handle<BXVector<L1uGtExtBlk>> uGtExt );
				          
    cms_uint64_t formatMuon(std::vector<l1t::Muon>::const_iterator mu);
    unsigned int formatEG(std::vector<l1t::EGamma>::const_iterator eg);
    unsigned int formatTau(std::vector<l1t::Tau>::const_iterator tau);
    unsigned int formatJet(std::vector<l1t::Jet>::const_iterator jet);
    unsigned int formatMissET(std::vector<l1t::EtSum>::const_iterator etSum);
    unsigned int formatTotalET(std::vector<l1t::EtSum>::const_iterator etSum);
    
    unsigned int m_absBx;
    
    std::ofstream m_testVectorFile;
    
    bool m_dumpTestVectors;
    bool m_dumpGTRecord;
    int m_minBx;
    int m_maxBx;
    int m_minBxVectors;
    int m_maxBxVectors;

    
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

      m_minBx           = iConfig.getParameter<int>("minBx");
      m_maxBx           = iConfig.getParameter<int>("maxBx");     
      m_dumpGTRecord    = iConfig.getParameter<bool>("dumpGTRecord");

      m_minBxVectors    = iConfig.getParameter<int>("minBxVec");
      m_maxBxVectors    = iConfig.getParameter<int>("maxBxVec"); 
      m_dumpTestVectors = iConfig.getParameter<bool>("dumpVectors");        
      std::string fileName = iConfig.getParameter<std::string>("tvFileName");
      if(m_dumpTestVectors) m_testVectorFile.open(fileName.c_str());
      
      
      m_absBx = 0;
      
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
  

  if(m_dumpGTRecord) {
   
       cout << " -----------------------------------------------------  " << endl;
       cout << " *********** Run " << iEvent.id().run()  <<" Event " << iEvent.id().event()  << " **************  " << endl;
       cout << " ----------------------------------------------------- " << endl; 

      // Dump the output record	  
       for(std::vector<L1uGtRecBlk>::const_iterator recBlk = uGtRec->begin(); recBlk != uGtRec->end(); ++recBlk) {
           recBlk->print(std::cout);
       }    

    //Loop over BX
       for(int i =m_minBx; i <= m_maxBx; ++i) {

	  cout << " ========== BX = " << std::dec << i << " =============================" << endl;

	  //Loop over EGamma
	  cout << " ------ EGammas -------- " << endl;
	  if(i>=egammas->getFirstBX() && i<=egammas->getLastBX()) {
	      for(std::vector<l1t::EGamma>::const_iterator eg = egammas->begin(i); eg != egammas->end(i); ++eg) {
	          cout << "   Pt "  << std::dec << std::setw(3) << eg->hwPt()  << " (0x" << std::hex << std::setw(3) << std::setfill('0') << eg->hwPt()  << ")";
		  cout << "   Eta " << std::dec << std::setw(3) << eg->hwEta() << " (0x" << std::hex << std::setw(2) << std::setfill('0') << eg->hwEta() << ")";
		  cout << "   Phi " << std::dec << std::setw(3) << eg->hwPhi() << " (0x" << std::hex << std::setw(2) << std::setfill('0') << eg->hwPhi() << ")";
		  cout << "   Iso " << std::dec << std::setw(1) << eg->hwIso() ;
		  cout << "   Qual "<< std::dec << std::setw(1) << eg->hwQual() ;
		  cout << endl;
	      } 
	  } else {
	      cout << "No EG stored for this bx " << i << endl;
	  }  

	  //Loop over Muons
	  cout << " ------ Muons --------" << endl;
	  if(i>=muons->getFirstBX() && i<=muons->getLastBX()) {
	      for(std::vector<l1t::Muon>::const_iterator mu = muons->begin(i); mu != muons->end(i); ++mu) {
	          cout << "   Pt "  << std::dec << std::setw(3) << mu->hwPt()  << " (0x" << std::hex << std::setw(3) << std::setfill('0') << mu->hwPt()  << ")";
		  cout << "   Eta " << std::dec << std::setw(3) << mu->hwEta() << " (0x" << std::hex << std::setw(3) << std::setfill('0') << mu->hwEta() << ")";
		  cout << "   Phi " << std::dec << std::setw(3) << mu->hwPhi() << " (0x" << std::hex << std::setw(3) << std::setfill('0') << mu->hwPhi() << ")";
		  cout << "   Iso " << std::dec << std::setw(1) << mu->hwIso() ;
		  cout << "   Qual "<< std::dec << std::setw(1) << mu->hwQual() ;
		  cout << endl;
	      }
	  }else {
	      cout << "No Muons stored for this bx " << i << endl;
	  }

	  //Loop over Taus
	  cout << " ------ Taus ----------" << endl;
	  if(i>=taus->getFirstBX() && i<=taus->getLastBX()) {
	      for(std::vector<l1t::Tau>::const_iterator tau = taus->begin(i); tau != taus->end(i); ++tau) {
	          cout << "   Pt "  << std::dec << std::setw(3) << tau->hwPt()  << " (0x" << std::hex << std::setw(3) << std::setfill('0') << tau->hwPt()  << ")";
		  cout << "   Eta " << std::dec << std::setw(3) << tau->hwEta() << " (0x" << std::hex << std::setw(2) << std::setfill('0') << tau->hwEta() << ")";
		  cout << "   Phi " << std::dec << std::setw(3) << tau->hwPhi() << " (0x" << std::hex << std::setw(2) << std::setfill('0') << tau->hwPhi() << ")";
		  cout << "   Iso " << std::dec << std::setw(1) << tau->hwIso() ;
		  cout << "   Qual "<< std::dec << std::setw(1) << tau->hwQual() ;
		  cout << endl;
	      } 
	  } else {
	     cout << "No Taus stored for this bx " << i << endl;
	  }       

	  //Loop over Jets
	  cout << " ------ Jets ----------" << endl;
	  if(i>=jets->getFirstBX() && i<=jets->getLastBX()) {
	     for(std::vector<l1t::Jet>::const_iterator jet = jets->begin(i); jet != jets->end(i); ++jet) {
	          cout << "   Pt "  << std::dec << std::setw(3) << jet->hwPt()  << " (0x" << std::hex << std::setw(3) << std::setfill('0') << jet->hwPt()  << ")";
		  cout << "   Eta " << std::dec << std::setw(3) << jet->hwEta() << " (0x" << std::hex << std::setw(2) << std::setfill('0') << jet->hwEta() << ")";
		  cout << "   Phi " << std::dec << std::setw(3) << jet->hwPhi() << " (0x" << std::hex << std::setw(2) << std::setfill('0') << jet->hwPhi() << ")";
		  cout << "   Qual "<< std::dec << std::setw(1) << jet->hwQual() ;
		  cout << endl;
	     }
	  } else {
	      cout << "No Jets stored for this bx " << i << endl;
	  }
                     //Dump Content
	  cout << " ------ EtSums ----------" << endl;
	  if(i>=etsums->getFirstBX() && i<=etsums->getLastBX()) {	  
	     for(std::vector<l1t::EtSum>::const_iterator etsum = etsums->begin(i); etsum != etsums->end(i); ++etsum) {
	          switch ( etsum->getType() ) {
		     case l1t::EtSum::EtSumType::kMissingEt:
		       cout << " ETM: ";
		       break; 
		     case l1t::EtSum::EtSumType::kMissingHt:
		       cout << " HTM: ";
		       break; 		     
		     case l1t::EtSum::EtSumType::kTotalEt:
		       cout << " ETT: ";
		       break; 		     
		     case l1t::EtSum::EtSumType::kTotalHt:
		       cout << " HTT: ";
		       break; 		     
		  }
	          cout << " Et "  << std::dec << std::setw(3) << etsum->hwPt()  << " (0x" << std::hex << std::setw(3) << std::setfill('0') << etsum->hwPt()  << ")";
		  if(etsum->getType() == l1t::EtSum::EtSumType::kMissingEt || etsum->getType() == l1t::EtSum::EtSumType::kMissingHt)
		     cout << " Phi " << std::dec << std::setw(3) << etsum->hwPhi() << " (0x" << std::hex << std::setw(2) << std::setfill('0') << etsum->hwPhi() << ")";
		  cout << endl;
	     } 
	  } else {
	      cout << "No EtSums stored for this bx " << i << endl;
	  }           

     // Dump the output record
 	  cout << " ------ uGtAlg ----------" << endl;
	  if(i>=uGtAlg->getFirstBX() && i<=uGtAlg->getLastBX()) {	  
	     for(std::vector<L1uGtAlgBlk>::const_iterator algBlk = uGtAlg->begin(i); algBlk != uGtAlg->end(i); ++algBlk) {
        	  algBlk->print(std::cout);
	     } 
	  } else {
	      cout << "No Alg Decisions stored for this bx " << i << endl;
	  }       

      // Dump the output record
 	  cout << " ------ uGtExt ----------" << endl;
	  if(i>=uGtExt->getFirstBX() && i<=uGtExt->getLastBX()) { 	  
	     for(std::vector<L1uGtExtBlk>::const_iterator extBlk = uGtExt->begin(i); extBlk != uGtExt->end(i); ++extBlk) {
        	  extBlk->print(std::cout);
	     } 
	  } else {
	      cout << "No Ext Conditions stored for this bx " << i << endl;
	  }       



       }
       cout << std::dec <<endl;
    } //if dumpGtRecord
    
    // Dump Test Vectors for this bx       
    if(m_dumpTestVectors) {
       for(int i=m_minBxVectors; i<=m_maxBxVectors; i++) {
         if(  (i>=egammas->getFirstBX() && i<=egammas->getLastBX())&&
	      (i>=muons->getFirstBX()   && i<=muons->getLastBX())  &&
	      (i>=taus->getFirstBX()    && i<=taus->getLastBX())   &&
	      (i>=jets->getFirstBX()    && i<=jets->getLastBX())   &&
	      (i>=etsums->getFirstBX()  && i<=etsums->getLastBX()) &&
	      (i>=uGtAlg->getFirstBX()  && i<=uGtAlg->getLastBX()) &&
	      (i>=uGtAlg->getFirstBX()  && i<=uGtAlg->getLastBX()) ) {	    
                  dumpTestVectors(i, m_testVectorFile, muons, egammas, taus, jets, etsums, uGtAlg, uGtExt);
	 } else {
	      edm::LogWarning("L1uGtRecordDump") << "WARNING: Not enough information to dump test vectors for this bx=" << i << endl;
	 } 	  
       }
    }   
    
    
    
  }

void L1uGtRecordDump::dumpTestVectors(int bx, std::ofstream& myOutFile, 
                                      Handle<BXVector<l1t::Muon>> muons,
				      Handle<BXVector<l1t::EGamma>> egammas,
				      Handle<BXVector<l1t::Tau>> taus,
				      Handle<BXVector<l1t::Jet>> jets,
				      Handle<BXVector<l1t::EtSum>> etsums,
				      Handle<BXVector<L1uGtAlgBlk>> uGtAlg,
				      Handle<BXVector<L1uGtExtBlk>> uGtExt
				      ) {


   const int empty = 0;
      
// Dump Bx (4 digits)
   myOutFile << std::hex << std::setw(4) << std::setfill('0') << m_absBx;

// Dump 8 Muons (16 digits + space)
   for(std::vector<l1t::Muon>::const_iterator mu = muons->begin(bx); mu != muons->end(bx); ++mu) {
      cms_uint64_t packedWd = formatMuon(mu);
      myOutFile << " " << std::hex << std::setw(16) << std::setfill('0') << packedWd;
   }   
   for(int i=muons->size(bx); i<8; i++) {
      myOutFile << " " << std::hex << std::setw(16) << std::setfill('0') << empty;
   }

// Dump 12 EG (8 digits + space)
   for(std::vector<l1t::EGamma>::const_iterator eg = egammas->begin(bx); eg != egammas->end(bx); ++eg) {
      unsigned int packedWd = formatEG(eg);
      myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << packedWd;
   }    
   for(int i=egammas->size(bx); i<12; i++) {
      myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << empty;
   }


// Dump 8 tau (8 digits + space)
   for(std::vector<l1t::Tau>::const_iterator tau = taus->begin(bx); tau != taus->end(bx); ++tau) {
      unsigned int packedWd = formatTau(tau);
      myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << packedWd;
   }    
   for(int i=taus->size(bx); i<8; i++) {
      myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << empty;
   }
   
// Dump 12 Jets (8 digits + space)
   for(std::vector<l1t::Jet>::const_iterator jet = jets->begin(bx); jet != jets->end(bx); ++jet) {
      unsigned int packedWd = formatJet(jet);
      myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << packedWd;
   }    
   for(int i=jets->size(bx); i<12; i++) {
      myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << empty;
   }

// Dump Et Sums (Order ETT, HT, ETM, HTM) (Each 8 digits + space)
   unsigned int ETTpackWd = 0;
   unsigned int HTTpackWd = 0;
   unsigned int ETMpackWd = 0;
   unsigned int HTMpackWd = 0;
   for(std::vector<l1t::EtSum>::const_iterator etsum = etsums->begin(bx); etsum != etsums->end(bx); ++etsum) {

      switch ( etsum->getType() ) {
	 case l1t::EtSum::EtSumType::kMissingEt:
	   ETMpackWd = formatMissET(etsum);
	   break; 
	 case l1t::EtSum::EtSumType::kMissingHt:
	   HTMpackWd = formatMissET(etsum);
	   break; 		     
	 case l1t::EtSum::EtSumType::kTotalEt:
	   ETTpackWd = formatTotalET(etsum);
	   break; 		     
	 case l1t::EtSum::EtSumType::kTotalHt:
	   HTTpackWd = formatTotalET(etsum);
	   break; 		     
      } //end switch statement
   } //end loop over etsums
   // Fill in the words in appropriate order
   myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << ETTpackWd;
   myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << HTTpackWd;
   myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << ETMpackWd;
   myOutFile << " " << std::hex << std::setw(8) << std::setfill('0') << HTMpackWd;
   
// External Condition (64 digits + space)
    int digit = 0;
    myOutFile << " ";
    for(std::vector<L1uGtExtBlk>::const_iterator extBlk = uGtExt->begin(bx); extBlk != uGtExt->end(bx); ++extBlk) {
        for(int i=255; i>-1; i--) {
          if(extBlk->getExternalDecision(i)) digit |= (1 << (i%4));
             if((i%4) == 0){
                  myOutFile << std::hex << std::setw(1) << digit;
	          digit = 0; 
             }  
         } //end loop over external bits
    }
   
// Algorithm Dump (128 digits + space)
    digit = 0;
    myOutFile << " ";
    for(std::vector<L1uGtAlgBlk>::const_iterator algBlk = uGtAlg->begin(bx); algBlk != uGtAlg->end(bx); ++algBlk) {
        for(int i=511; i>-1; i--) {
          if(algBlk->getAlgoDecisionFinal(i)) digit |= (1 << (i%4));
             if((i%4) == 0){
                  myOutFile << std::hex << std::setw(1) << digit;
	          digit = 0; 
             }  
         } //end loop over algorithm bits       
    
// Final OR (1 digit + space) 
         unsigned int finalOr = (algBlk->getFinalOR() & 0xf);    
         myOutFile << " " << std::hex << std::setw(1) << std::setfill('0') << finalOr;
    }
   
    myOutFile << endl;
    
    m_absBx++; 
    
}

cms_uint64_t L1uGtRecordDump::formatMuon(std::vector<l1t::Muon>::const_iterator mu){

  cms_uint64_t packedVal = 0;

// Order specified in Table 1 of CNS IN-2013/005 ("Input Specification to the Global Trigger Upgrade")
  packedVal |= ((mu->hwPhi()              & 0x3ff) <<0);
  packedVal |= ((mu->hwEta()              & 0x1ff) <<10);
  packedVal |= ((mu->hwPt()               & 0x1ff) <<19);
  packedVal |= ((mu->hwChargeValid()      & 0x1)   <<28);
  packedVal |= ((mu->hwCharge()           & 0x1)   <<29);
  packedVal |= ((mu->hwQual()             & 0xf)   <<30);
  packedVal |= ((cms_uint64_t)(mu->hwIso()& 0x3)   <<34);  
  
  return packedVal;
}

unsigned int L1uGtRecordDump::formatEG(std::vector<l1t::EGamma>::const_iterator eg){

  unsigned int packedVal = 0;

// Order specified in Table 1 of CNS IN-2013/005 ("Input Specification to the Global Trigger Upgrade")  
  packedVal |= ((eg->hwPhi()   & 0xff)   <<0);
  packedVal |= ((eg->hwEta()   & 0xff)   <<8);
  packedVal |= ((eg->hwPt()    & 0x1ff)  <<16);
  packedVal |= ((eg->hwIso()   & 0x1)    <<25);
  packedVal |= ((eg->hwQual()  & 0x1)    <<26);
  
  return packedVal;
}

unsigned int L1uGtRecordDump::formatTau(std::vector<l1t::Tau>::const_iterator tau){

  unsigned int packedVal = 0;
  
// Order specified in Table 1 of CNS IN-2013/005 ("Input Specification to the Global Trigger Upgrade")  
  packedVal |= ((tau->hwPhi()   & 0xff)   <<0);
  packedVal |= ((tau->hwEta()   & 0xff)   <<8);
  packedVal |= ((tau->hwPt()    & 0x1ff)  <<16);
  packedVal |= ((tau->hwIso()   & 0x1)    <<25);
  packedVal |= ((tau->hwQual()  & 0x1)    <<26);  
  
  return packedVal;
}

unsigned int L1uGtRecordDump::formatJet(std::vector<l1t::Jet>::const_iterator jet){

  unsigned int packedVal = 0;

// Order specified in Table 1 of CNS IN-2013/005 ("Input Specification to the Global Trigger Upgrade")  
  packedVal |= ((jet->hwPhi()    & 0xff)    <<0);
  packedVal |= ((jet->hwEta()    & 0xff)    <<8);
  packedVal |= ((jet->hwPt()     & 0x7ff)   <<16);
  packedVal |= ((jet->hwQual()   & 0x1)     <<27);
    
  return packedVal;
}

unsigned int L1uGtRecordDump::formatMissET(std::vector<l1t::EtSum>::const_iterator etSum){

  unsigned int packedVal = 0;

// Order specified in Table 1 of CNS IN-2013/005 ("Input Specification to the Global Trigger Upgrade")
  packedVal |= ((etSum->hwPhi()    & 0xff)    <<0);
  packedVal |= ((etSum->hwPt()     & 0xfff)   <<8); 
  
  return packedVal;
}

unsigned int L1uGtRecordDump::formatTotalET(std::vector<l1t::EtSum>::const_iterator etSum){

  unsigned int packedVal = 0;

// Order specified in Table 1 of CNS IN-2013/005 ("Input Specification to the Global Trigger Upgrade")
  packedVal |= ((etSum->hwPt()     & 0xfff)   <<0); 
  
  return packedVal;
}


}


DEFINE_FWK_MODULE(l1t::L1uGtRecordDump);

