//-------------------------------------------------
//
//   Class: L1PromptAnalysis
//
//
//   \class L1PromptAnalysis
/**
 *   Description:  This code is designed for l1 prompt analysis
//                 starting point is a GMTTreeMaker By Ivan Mikulec now
//                 extended from Lorenzo Agostino. 
*/
//
//
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerOffline/L1Analyzer/interface/L1PromptAnalysis.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"


#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"


using namespace std;

//----------------
// Constructors --
//----------------
L1PromptAnalysis::L1PromptAnalysis(const edm::ParameterSet& ps) : m_file(0), m_tree(0) {

  verbose_ = ps.getUntrackedParameter < bool > ("verbose", false);

//gt, gmt  
  m_GMTInputTag = ps.getParameter<edm::InputTag>("GMTInputTag");
  m_GTEvmInputTag = ps.getParameter<edm::InputTag>("GTEvmInputTag");
  m_GTInputTag = ps.getParameter<edm::InputTag>("GTInputTag");
  m_GeneratorInputTag = ps.getParameter<edm::InputTag>("GeneratorInputTag");
  m_SimulationInputTag = ps.getParameter<edm::InputTag>("SimulationInputTag");
  m_PhysVal = ps.getParameter<bool>("PhysVal");
  m_outfilename = ps.getUntrackedParameter<string>("OutputFile","L1PromptAnalysis.root");
//gct
  gctCenJetsSource_ = ps.getParameter<edm::InputTag>("gctCentralJetsSource");
  gctForJetsSource_ = ps.getParameter<edm::InputTag>("gctForwardJetsSource");
  gctTauJetsSource_ = ps.getParameter<edm::InputTag>("gctTauJetsSource");
  gctEnergySumsSource_ = ps.getParameter<edm::InputTag>("gctEnergySumsSource");
  gctIsoEmSource_ = ps.getParameter<edm::InputTag>("gctIsoEmSource");
  gctNonIsoEmSource_ = ps.getParameter<edm::InputTag>("gctNonIsoEmSource");
//rct
  rctSource_= ps.getParameter< edm::InputTag >("rctSource");
//dt  
  dttfSource_ =  ps.getParameter< edm::InputTag >("dttfSource") ;

}

//--------------
// Destructor --
//--------------
L1PromptAnalysis::~L1PromptAnalysis() {}

void L1PromptAnalysis::beginJob(const edm::EventSetup& es) {
  m_file = TFile::Open(m_outfilename.c_str(),"RECREATE");
  m_tree = new TTree("h1","GMT Tree");
  book();
}

void L1PromptAnalysis::endJob() {
  m_file->Write();
  m_file->Close();
}

//--------------
// Operations --
//--------------

void L1PromptAnalysis::analyze(const edm::Event& e, const edm::EventSetup& es) {

  //
  // GENERAL block
  //
  runn = e.id().run();
  eventn = e.id().event();
  timest = e.time().value();
  bx = e.bunchCrossing();   //overwritten by EVM info until fixed by fw
  lumi = e.luminosityBlock();
  orbitn = e.orbitNumber();   //overwritten by EVM info until fixed by fw

//  edm::LogVerbatim("GMTDump") << "run: " << runn << ", event: " << eventn << endl;

  // generetor block
  HepMC::GenEvent const* genevent = NULL;

  if(m_GeneratorInputTag.label() != "none") {
    edm::Handle<edm::HepMCProduct> vtxSmeared_handle;
    e.getByLabel(m_GeneratorInputTag,vtxSmeared_handle);
    genevent = vtxSmeared_handle.product()->GetEvent();

    weight = 1.;
    if(genevent->weights().size() > 0) weight = genevent->weights()[0];
    pthat = genevent->event_scale();
  }

  if(m_SimulationInputTag.label() != "none") {
    edm::Handle<edm::SimVertexContainer> simvertices_handle;
    e.getByLabel(m_SimulationInputTag,simvertices_handle);
    edm::SimVertexContainer const* simvertices = simvertices_handle.product();

    edm::Handle<edm::SimTrackContainer> simtracks_handle;
    e.getByLabel(m_SimulationInputTag,simtracks_handle);
    edm::SimTrackContainer const* simtracks = simtracks_handle.product();

    edm::SimTrackContainer::const_iterator isimtr;
    int igen = 0;
    for(isimtr=simtracks->begin(); isimtr!=simtracks->end(); isimtr++) {
      if(abs((*isimtr).type())!=13 || igen>=MAXGEN) continue;
      pxgen[igen]=(*isimtr).momentum().px();
      pygen[igen]=(*isimtr).momentum().py();
      pzgen[igen]=(*isimtr).momentum().pz();
      ptgen[igen]=(*isimtr).momentum().pt();
      etagen[igen]=(*isimtr).momentum().eta();
      phigen[igen]=(*isimtr).momentum().phi()>0 ? (*isimtr).momentum().phi() : (*isimtr).momentum().phi()+2*3.14159265359;
      chagen[igen]=(*isimtr).type()>0 ? -1 : 1 ;
      vxgen[igen]=(*simvertices)[(*isimtr).vertIndex()].position().x();
      vygen[igen]=(*simvertices)[(*isimtr).vertIndex()].position().y();
      vzgen[igen]=(*simvertices)[(*isimtr).vertIndex()].position().z();
      pargen[igen]=-1;
      //      if(genevent && (*isimtr).genpartIndex()!=-1 && genevent->particle((*isimtr).genpartIndex())->listParents().size()>0) {
      //        pargen[igen] = genevent->particle((*isimtr).genpartIndex())->listParents()[0]->pdg_id();
      //      }
      
      igen++;
    }
    ngen=igen;  
  } 

  // Get GMTReadoutCollection

  if(m_GMTInputTag.label() != "none") {
    edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle; 
    e.getByLabel(m_GMTInputTag,gmtrc_handle);
    L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
    
    int idt = 0;
    int icsc = 0;
    int irpcb = 0;
    int irpcf = 0;
    int igmt = 0;
    vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
    vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;
    for(igmtrr=gmt_records.begin(); igmtrr!=gmt_records.end(); igmtrr++) {
  
      vector<L1MuRegionalCand>::const_iterator iter1;
      vector<L1MuRegionalCand> rmc;;
  
      if(igmtrr->getBxInEvent()==0) {
        bxgmt = igmtrr->getBxNr();
      }
  
      //
      // DTBX Trigger
      //
  
      int iidt = 0;
      rmc = igmtrr->getDTBXCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
        if ( idt < MAXDTBX && !(*iter1).empty() ) {
      	  bxd[idt]=(*iter1).bx();
      	  if(m_PhysVal) {
            etad[idt]=float((*iter1).etaValue());
            phid[idt]=float((*iter1).phiValue());
            ptd[idt]=float((*iter1).ptValue());
      	  } else {
            etad[idt]=float((*iter1).eta_packed());
            phid[idt]=float((*iter1).phi_packed());
            ptd[idt]=float((*iter1).pt_packed());
      	  }
      	  chad[idt]=(*iter1).chargeValue(); if(!(*iter1).chargeValid()) chad[idt]=0;
      	  etafined[idt]=0; // etafined[idt]=(*iter1).fineEtaBit();
      	  quald[idt]=(*iter1).quality();
          dwd[idt]=(*iter1).getDataWord();
          chd[idt]=iidt;
            
      	  idt++;
        }
        iidt++;
      }
  
      //
      // CSC Trigger
      //
  
      rmc = igmtrr->getCSCCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
        if ( icsc < MAXCSC && !(*iter1).empty() ) {
          bxc[icsc]=(*iter1).bx();
          if(m_PhysVal) {            
            etac[icsc]=(*iter1).etaValue();
            phic[icsc]=(*iter1).phiValue();
            ptc[icsc]=(*iter1).ptValue();
          } else {
            etac[icsc]=(*iter1).eta_packed();
            phic[icsc]=(*iter1).phi_packed();
            ptc[icsc]=(*iter1).pt_packed();
          }
          chac[icsc]=(*iter1).chargeValue(); if(!(*iter1).chargeValid()) chac[icsc]=0;
          qualc[icsc]=(*iter1).quality();
          dwc[icsc]=(*iter1).getDataWord();
 
          icsc++;
        }
      }
  
      //
      // RPCb Trigger
      //
      rmc = igmtrr->getBrlRPCCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
        if ( irpcb < MAXRPC && !(*iter1).empty() ) {
          bxrb[irpcb]=(*iter1).bx();
          if(m_PhysVal) {            
            etarb[irpcb]=(*iter1).etaValue();
            phirb[irpcb]=(*iter1).phiValue();
            ptrb[irpcb]=(*iter1).ptValue();
          } else {
            etarb[irpcb]=(*iter1).eta_packed();
            phirb[irpcb]=(*iter1).phi_packed();
            ptrb[irpcb]=(*iter1).pt_packed();
          }
          charb[irpcb]=(*iter1).chargeValue(); if(!(*iter1).chargeValid()) charb[irpcb]=0;
          qualrb[irpcb]=(*iter1).quality();
          dwrb[irpcb]=(*iter1).getDataWord();
 
          irpcb++;
        }
      }
 
      //
      // RPCf Trigger
      //
      rmc = igmtrr->getFwdRPCCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
        if ( irpcf < MAXRPC && !(*iter1).empty() ) {
          bxrf[irpcf]=(*iter1).bx();
          if(m_PhysVal) {
            etarf[irpcf]=(*iter1).etaValue();
            phirf[irpcf]=(*iter1).phiValue();
            ptrf[irpcf]=(*iter1).ptValue();
          } else {
            etarf[irpcf]=(*iter1).eta_packed();
            phirf[irpcf]=(*iter1).phi_packed();
            ptrf[irpcf]=(*iter1).pt_packed();
          }
          charf[irpcf]=(*iter1).chargeValue(); if(!(*iter1).chargeValid()) charf[irpcf]=0;
          qualrf[irpcf]=(*iter1).quality();
          dwrf[irpcf]=(*iter1).getDataWord();
 
          irpcf++;
        }
      }
  
      //
      // GMT Trigger
      //
  
      vector<L1MuGMTExtendedCand>::const_iterator gmt_iter;
      vector<L1MuGMTExtendedCand> exc = igmtrr->getGMTCands();
      for(gmt_iter=exc.begin(); gmt_iter!=exc.end(); gmt_iter++) {
        if ( igmt < MAXGMT && !(*gmt_iter).empty() ) {
          bxg[igmt]=(*gmt_iter).bx();
          if(m_PhysVal) {
            etag[igmt]=(*gmt_iter).etaValue();
            phig[igmt]=(*gmt_iter).phiValue(); 
            ptg[igmt]=(*gmt_iter).ptValue();
          } else {
            etag[igmt]=(*gmt_iter).etaIndex();
            phig[igmt]=(*gmt_iter).phiIndex(); 
            ptg[igmt]=(*gmt_iter).ptIndex();
          }
          chag[igmt]=(*gmt_iter).charge(); if(!(*gmt_iter).charge_valid()) chag[igmt]=0;
          qualg[igmt]=(*gmt_iter).quality();
          detg[igmt]=(*gmt_iter).detector();
          rankg[igmt]=(*gmt_iter).rank();
          isolg[igmt]=(*gmt_iter).isol();
          mipg[igmt]=(*gmt_iter).mip();
          dwg[igmt]=(*gmt_iter).getDataWord();
                
          idxRPCb[igmt]=-1;
          idxRPCf[igmt]=-1;
          idxDTBX[igmt]=-1;
          idxCSC[igmt]=-1;
  
          if ( (*gmt_iter).isMatchedCand() || (*gmt_iter).isRPC() ) {
            if((*gmt_iter).isFwd()) {
              idxRPCf[igmt] = (*gmt_iter).getRPCIndex();
            } else {
              idxRPCb[igmt] = (*gmt_iter).getRPCIndex();
            }
          }
  
          if ( (*gmt_iter).isMatchedCand() || ( !(*gmt_iter).isRPC() ) ) {
            if ( (*gmt_iter).isFwd() )  {
              idxCSC[igmt] = (*gmt_iter).getDTCSCIndex();
            } else {
              idxDTBX[igmt] = (*gmt_iter).getDTCSCIndex();
            }
          }
          igmt++;
        }
      }
    }
    ndt = idt;
    ncsc = icsc;
    nrpcb = irpcb;
    nrpcf = irpcf;
    ngmt = igmt;
  }

  //////////////////////////////////////////////////////////////////////
  if (m_GTEvmInputTag.label() != "none") {
    edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtevmrr_handle;
    e.getByLabel(m_GTEvmInputTag.label(), gtevmrr_handle);
    L1GlobalTriggerEvmReadoutRecord const* gtevmrr = gtevmrr_handle.product();

    L1TcsWord tcsw = gtevmrr->tcsWord();

    bx = tcsw.bxNr();
//    lumi = tcsw.luminositySegmentNr();
//    runn = tcsw.partRunNr();
//    eventn = tcsw.partTrigNr();
    orbitn = tcsw.orbitNr();
  }


   //////////////////////////////////////////////////////////////////////
  if (m_GTInputTag.label() != "none") {
    edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
    e.getByLabel(m_GTInputTag.label(), gtrr_handle);
    L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();

    int iel = 0;
    int ijet = 0;
    for (int ibx=-1; ibx<=1; ibx++) {
      const L1GtPsbWord psb = gtrr->gtPsbWord(0xbb0d, ibx);
      vector<int> psbel;
      psbel.push_back(psb.aData(4));
      psbel.push_back(psb.aData(5));
      psbel.push_back(psb.bData(4));
      psbel.push_back(psb.bData(5));
      std::vector<int>::const_iterator ipsbel;
      for(ipsbel=psbel.begin(); ipsbel!=psbel.end(); ipsbel++) {
        float rank = (*ipsbel)&0x3f;
        if(rank>0) {
          bxel[iel] = ibx;
          rankel[iel] = rank;
          phiel[iel] = ((*ipsbel)>>10)&0x1f;
          etael[iel] = (((*ipsbel)>>6)&7) * ( ((*ipsbel>>9)&1) ? -1 : 1 );
          iel++;
        }
      }
      vector<int> psbjet;
      psbjet.push_back(psb.aData(2));
      psbjet.push_back(psb.aData(3));
      psbjet.push_back(psb.bData(2));
      psbjet.push_back(psb.bData(3));
      std::vector<int>::const_iterator ipsbjet;
      for(ipsbjet=psbjet.begin(); ipsbjet!=psbjet.end(); ipsbjet++) {
        float rank = (*ipsbjet)&0x3f;
        if(rank>0) {
          bxjet[ijet] = ibx;
          rankjet[ijet] = rank;
          phijet[ijet] = ((*ipsbjet)>>10)&0x1f;
          etajet[ijet] = (((*ipsbjet)>>6)&7) * ( ((*ipsbjet>>9)&1) ? -1 : 1 );
          ijet++;
        }
      }
    }
    nele = iel;
    njet = ijet;


    L1GtFdlWord fdlWord = gtrr->gtFdlWord();
    
    /// get Global Trigger algo and technical triger bit statistics
    for(int iebx=0; iebx<=2; iebx++) {
      DecisionWord gtDecisionWord = gtrr->decisionWord(iebx-1);

      int dbitNumber = 0;
      gttw1[iebx] = 0;
      gttw2[iebx] = 0;
      gttt[iebx] = 0;
      DecisionWord::const_iterator GTdbitItr;
      for(GTdbitItr = gtDecisionWord.begin(); GTdbitItr != gtDecisionWord.end(); GTdbitItr++) {
        if (*GTdbitItr) {
          if(dbitNumber<64) { gttw1[iebx] |= (1LL<<dbitNumber); }
          else { gttw2[iebx] |= (1LL<<(dbitNumber-64)); }
        }
        dbitNumber++; 
      }

      dbitNumber = 0;
      TechnicalTriggerWord gtTTWord = gtrr->technicalTriggerWord(iebx-1);
      TechnicalTriggerWord::const_iterator GTtbitItr;
      for(GTtbitItr = gtTTWord.begin(); GTtbitItr != gtTTWord.end(); GTtbitItr++) {
        if (*GTtbitItr) {
          gttt[iebx] |= (1LL<<dbitNumber);
        }
        dbitNumber++;
      }
    }
  }
  
//////////////////////////////////////////////// GCT /////////////////////////////////////////////////

  bool doJet = true;
  bool doEm = true;
  bool doHFminbias = true;
  bool doES = true;
    gctIsoEmSize=-999;
    gctNonIsoEmSize=-999;
    gctCJetSize=-999;
    gctFJetSize=-999;
    gctTJetSize=-999;
    gctEtMiss=-999;
    gctEtMissPhi=-999.;
    gctEtHad=-999.;
    gctEtTot=-999.;
    gctHFRingEtSumSize=-999;
    gctHFBitCountsSize=-999;
    for(int ii=0;ii<4;ii++ ){
    gctIsoEmEta[ii]=-999.;
    gctIsoEmPhi[ii]=-999.;
    gctIsoEmRnk[ii]=-999.;
    gctNonIsoEmEta[ii]=-999.;
    gctNonIsoEmPhi[ii]=-999.;
    gctNonIsoEmRnk[ii]=-999.;
    gctCJetEta[ii]=-999.;
    gctCJetPhi[ii]=-999.;
    gctCJetRnk[ii]=-999.;
    gctFJetEta[ii]=-999.;
    gctFJetPhi[ii]=-999.;
    gctFJetRnk[ii]=-999.;
    gctTJetEta[ii]=-999.;
    gctTJetPhi[ii]=-999.;
    gctTJetRnk[ii]=-999.;
    gctHFRingEtSumEta[ii]=-999.;
    gctHFBitCountsEta[ii]=-999.;
    }
  
  edm::Handle < L1GctEmCandCollection > l1IsoEm;
  e.getByLabel(gctIsoEmSource_, l1IsoEm);
  if (!l1IsoEm.isValid()) {
    edm::LogWarning("DataNotFound") << " Could not find l1IsoEm "
      " elements, label was " << gctIsoEmSource_ ;
    doEm = false;
  }

  edm::Handle < L1GctEmCandCollection > l1NonIsoEm;
  e.getByLabel(gctNonIsoEmSource_, l1NonIsoEm);
  if (!l1NonIsoEm.isValid()) {
    edm::LogWarning("DataNotFound") << " Could not find l1NonIsoEm "
      " elements, label was " << gctNonIsoEmSource_ ;
    doEm = false;
  }

  edm::Handle < L1GctJetCandCollection > l1CenJets;
  e.getByLabel(gctCenJetsSource_, l1CenJets);
  if (!l1CenJets.isValid())  {
    edm::LogWarning("DataNotFound") << " Could not find l1CenJets"
      ", label was " << gctCenJetsSource_ ;
    doJet = false;
  }

  edm::Handle < L1GctJetCandCollection > l1ForJets;
  e.getByLabel(gctForJetsSource_, l1ForJets);
  if (!l1ForJets.isValid())  {
    edm::LogWarning("DataNotFound") << " Could not find l1ForJets"
      ", label was " << gctForJetsSource_ ;
    doJet = false;
  }

  edm::Handle < L1GctJetCandCollection > l1TauJets;
  e.getByLabel(gctTauJetsSource_, l1TauJets);
  if (!l1TauJets.isValid())  {
    edm::LogWarning("DataNotFound") << " Could not find l1TauJets"
      ", label was " << gctTauJetsSource_ ;
    doJet = false;
  }

  edm::Handle < L1GctHFRingEtSumsCollection > l1HFSums; 
  e.getByLabel(gctEnergySumsSource_, l1HFSums);
  if (!l1HFSums.isValid())  {
    edm::LogWarning("DataNotFound") << " Could not find l1HFSums"
      ", label was " << gctEnergySumsSource_ ;
    doHFminbias = false;
  }

  edm::Handle < L1GctHFBitCountsCollection > l1HFCounts;
  e.getByLabel(gctEnergySumsSource_, l1HFCounts);  
  if (!l1HFCounts.isValid())  {
    edm::LogWarning("DataNotFound") << " Could not find l1HFCounts"
      ", label was " << gctEnergySumsSource_ ;
    doHFminbias = false;
  }   


  edm::Handle < L1GctEtMissCollection >  l1EtMiss;
  e.getByLabel(gctEnergySumsSource_, l1EtMiss);
  if (!l1EtMiss.isValid())  {
    edm::LogWarning("DataNotFound") << " Could not find l1EtMiss"
      ", label was " << gctEnergySumsSource_ ;
    doES = false;
  }

  edm::Handle < L1GctEtHadCollection >   l1EtHad;
  e.getByLabel(gctEnergySumsSource_, l1EtHad);
  if (!l1EtHad.isValid())  {
    edm::LogWarning("DataNotFound") << " Could not find l1EtHad"
      ", label was " << gctEnergySumsSource_ ;
    doES = false;
  }

  edm::Handle < L1GctEtTotalCollection > l1EtTotal;
  e.getByLabel(gctEnergySumsSource_, l1EtTotal);
  if (!l1EtTotal.isValid())  {
    edm::LogWarning("DataNotFound") << " Could not find l1EtTotal"
      ", label was " << gctEnergySumsSource_ ;
    doES = false;
  }

  if ( doJet ) {
    // Central jets
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1PromptAnalysis: number of central jets = " 
		<< l1CenJets->size() << std::endl;
    }
    gctCJetSize= l1CenJets->size();//1
    int icj=0;
    for (L1GctJetCandCollection::const_iterator cj = l1CenJets->begin();
	 cj != l1CenJets->end(); cj++) {
      gctCJetEta[icj]=cj->regionId().ieta();//2
      gctCJetPhi[icj]=cj->regionId().iphi();//3
      gctCJetRnk[icj]=cj->rank();//4
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1PromptAnalysis: Central jet " 
		  << cj->regionId().iphi() << ", " << cj->regionId().ieta()
		  << ", " << cj->rank() << std::endl;
      }
      icj++;
    }

    // Forward jets
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1PromptAnalysis: number of forward jets = " 
		<< l1ForJets->size() << std::endl;
    }
    gctFJetSize= l1ForJets->size();//5
    int ifj=0;
    for (L1GctJetCandCollection::const_iterator fj = l1ForJets->begin();
	 fj != l1ForJets->end(); fj++) {
      gctFJetEta[ifj]=fj->regionId().ieta();//6
      gctFJetPhi[ifj]=fj->regionId().iphi();//7
      gctFJetRnk[ifj]=fj->rank();//8
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1PromptAnalysis: Forward jet " 
		  << fj->regionId().iphi() << ", " << fj->regionId().ieta()
		  << ", " << fj->rank() << std::endl;
      }
      ifj++;
    }

    // Tau jets
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1PromptAnalysis: number of tau jets = " 
		<< l1TauJets->size() << std::endl;
    }
    gctFJetSize= l1TauJets->size();//9
    int itj=0;
    for (L1GctJetCandCollection::const_iterator tj = l1TauJets->begin();
	 tj != l1TauJets->end(); tj++) {
      //if ( tj->rank() == 0 ) continue;
      gctTJetEta[itj]=tj->regionId().ieta();//10
      gctTJetPhi[itj]=tj->regionId().iphi();//11
      gctTJetRnk[itj]=tj->rank();//12
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1PromptAnalysis: Tau jet " 
			       << tj->regionId().iphi() << ", " << tj->regionId().ieta()
			       << ", " << tj->rank() << std::endl;
      }
      itj++;
    }
        
  }

  if (doES) {
    // Energy sums
    if ( l1EtMiss->size() ) {
      gctEtMiss= l1EtMiss->at(0).et();//
      gctEtMissPhi= l1EtMiss->at(0).phi();//
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1PromptAnalysis: Et Miss " 
			       << l1EtMiss->size() << ", " << l1EtMiss->at(0).et()
			       << ", " << l1EtMiss->at(0).phi() << std::endl;
      }
    }
    // these don't have phi values
    if ( l1EtHad->size() ) {
      gctEtHad= l1EtHad->at(0).et();//
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1PromptAnalysis: Et Had " 
			       << l1EtHad->size() << ", " << l1EtHad->at(0).et() << std::endl;
      }
    }
    if ( l1EtTotal->size() ) {
      gctEtTot=l1EtTotal->at(0).et();//
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1PromptAnalysis: Et Total " 
			       << l1EtTotal->size() << ", " << l1EtTotal->at(0).et() << std::endl;
      }
    }
  }

  if (doHFminbias) {

    //Fill HF Ring Histograms
    gctHFRingEtSumSize=l1HFSums->size();
    int ies=0;
    for (L1GctHFRingEtSumsCollection::const_iterator hfs=l1HFSums->begin(); hfs!=l1HFSums->end(); hfs++){ 
       gctHFRingEtSumEta[ies]= hfs->etSum(ies);
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1PromptAnalysis: HF Sums " 
			       << l1HFSums->size() << ", " << hfs->etSum(ies) << std::endl;
      }
      ies++;
    }
    
    int ibc=0;
    gctHFBitCountsSize=l1HFCounts->size();
    for (L1GctHFBitCountsCollection::const_iterator hfc=l1HFCounts->begin(); hfc!=l1HFCounts->end(); hfc++){ 
      gctHFBitCountsEta[ibc]=hfc->bitCount(ibc);
      if ( verbose_ ) {
	edm::LogInfo("L1Prompt") << "L1PromptAnalysis: HF Counts " 
			       << l1HFCounts->size() << ", " << hfc->bitCount(ibc) << std::endl;
      }
      ibc++;
    }


  }


  if ( doEm ) {

    // Isolated EM
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1TGCT: number of iso em cands: " 
		<< l1IsoEm->size() << std::endl;
    }
    int iie=0;
    gctIsoEmSize = l1IsoEm->size();
    for (L1GctEmCandCollection::const_iterator ie=l1IsoEm->begin(); ie!=l1IsoEm->end(); ie++) {
      //if ( ie->rank() == 0 ) continue;
      gctIsoEmEta[iie] = ie->regionId().ieta();
      gctIsoEmPhi[iie] = ie->regionId().iphi();
      gctIsoEmRnk[iie] = ie->rank();
     iie++;
    } 

    // Non-isolated EM
    if ( verbose_ ) {
      edm::LogInfo("L1Prompt") << "L1TGCT: number of non-iso em cands: " 
		<< l1NonIsoEm->size() << std::endl;
    }
    gctNonIsoEmSize = l1NonIsoEm->size();
    int ine=0;
    for (L1GctEmCandCollection::const_iterator ne=l1NonIsoEm->begin(); ne!=l1NonIsoEm->end(); ne++) {
      gctNonIsoEmEta[ine] = ne->regionId().ieta();
      gctNonIsoEmPhi[ine] = ne->regionId().iphi();
      gctNonIsoEmRnk[ine] = ne->rank();
      ine++;  
    } 

   }
   
///////////////////////RCT///////////////////////////
  bool doEmRCT = true; 
  bool doHdRCT = true;
    rctRegSize=-999;
    rctEmSize=-999;
    for(int ii=0;ii<MAXRCTREG;ii++){
    rctRegEta[ii]=-999.;
    rctRegPhi[ii]=-999.;
    rctRegRnk[ii]=-999.;
    rctRegVeto[ii]=-999;
    rctRegBx[ii]=-999;
    rctRegOverFlow[ii]=-999;
    rctRegMip[ii]=-999;
    rctRegFGrain[ii]=-999;
    rctIsIsoEm[ii]=-999;
    rctEmEta[ii]=-999.;
    rctEmPhi[ii]=-999.;
    rctEmRnk[ii]=-999.;
    rctEmBx[ii]=-999;
    }

  edm::Handle < L1CaloEmCollection > em;
  e.getByLabel(rctSource_,em);
  
  if (!em.isValid()) {
    edm::LogInfo("L1Prompt") << "can't find L1CaloEmCollection with label "
			       << rctSource_.label() ;
    doEmRCT = false;
  }
  
  
  edm::Handle < L1CaloRegionCollection > rgn;
  e.getByLabel(rctSource_,rgn);
  if (!rgn.isValid()) {
    edm::LogInfo("L1Prompt") << "can't find L1CaloRegionCollection with label "
			       << rctSource_.label() ;
    doHdRCT = false;
  }
  
  if ( doHdRCT ) {
    // Regions
    int irg=0;
    rctRegSize=rgn->size();
    for (L1CaloRegionCollection::const_iterator ireg = rgn->begin();
	 ireg != rgn->end(); ireg++) {

      rctRegEta[irg]=ireg->rctEta();
      rctRegPhi[irg]=ireg->rctPhi();
      rctRegRnk[irg]=ireg->et();
      rctRegVeto[irg]=ireg->tauVeto();
      rctRegBx[irg]=ireg->bx();
      rctRegOverFlow[irg]=ireg->overFlow();
      rctRegMip[irg]=ireg->mip();
      rctRegFGrain[irg]=ireg->fineGrain();
     irg++;
     }

  }

  if ( doEmRCT ) {
  // Isolated and non-isolated EM
  rctEmSize = em->size();
  int iem=0;
  for (L1CaloEmCollection::const_iterator emit = em->begin(); emit != em->end(); emit++) {
      rctIsIsoEm[iem]= emit->isolated();
      rctEmEta[iem]=emit->regionId().ieta();
      rctEmPhi[iem]=emit->regionId().iphi();
      rctEmRnk[iem]=emit->rank();
      rctEmBx[iem]=emit->bx();
      iem++;
  }
  }

///////////////////////DTTF///////////////////////////

  bool doDTPH = true; 
  bool doDTTH = true; 
  bool doDTTR = true; 
   dttf_phSize = 0;
   dttf_thSize = 0;
   dttf_trSize = 0;

  
  edm::Handle<L1MuDTChambPhContainer > myL1MuDTChambPhContainer;  
  e.getByLabel(dttfSource_,myL1MuDTChambPhContainer);
  
  if (!myL1MuDTChambPhContainer.isValid()) {
    edm::LogInfo("L1Prompt") << "can't find L1MuDTChambPhContainer with label "
			     << dttfSource_.label() ;
    doDTPH=false;
  }
  
  if ( doDTPH ) {
  L1MuDTChambPhContainer::Phi_Container *myPhContainer =  
    myL1MuDTChambPhContainer->getContainer();

   
  for( int ii=0; ii<MAXDTPH;  ii++) 
    {           		
      dttf_phBx[ii] = -999;
      dttf_phWh[ii] = -999;
      dttf_phSe[ii] = -999;
      dttf_phSt[ii] = -999;
      dttf_phAng[ii] = -999.;
      dttf_phBandAng[ii] = -999.;
      dttf_phCode[ii] = -999;
      dttf_phX[ii] = -999.;
      dttf_phY[ii] = -999.;
    }


  dttf_phSize = myPhContainer->size();
   int iphtr=0;
   for( L1MuDTChambPhContainer::Phi_Container::const_iterator 
	 DTPhDigiItr =  myPhContainer->begin() ;
       DTPhDigiItr != myPhContainer->end() ;
       ++DTPhDigiItr ) 
    {        
      if(iphtr>MAXDTPH-1) continue;
      dttf_phBx[iphtr] = DTPhDigiItr->bxNum() - DTPhDigiItr->Ts2Tag()+1;
      dttf_phWh[iphtr] = DTPhDigiItr->whNum();
      dttf_phSe[iphtr] = DTPhDigiItr->scNum();
      dttf_phSt[iphtr] = DTPhDigiItr->stNum();
      dttf_phAng[iphtr] = DTPhDigiItr->phi();
      dttf_phBandAng[iphtr] = DTPhDigiItr->phiB();
      dttf_phCode[iphtr] = DTPhDigiItr->code();
      dttf_phX[iphtr] = DTPhDigiItr->scNum();
      dttf_phY[iphtr] = DTPhDigiItr->stNum()+4*(DTPhDigiItr->whNum()+2);
      
      iphtr++;
    }
    }


//  const L1MuDTChambPhDigi* bestPhQualMap[5][12][4];
//  memset(bestPhQualMap,0,240*sizeof(L1MuDTChambPhDigi*));
   
////

  edm::Handle<L1MuDTChambThContainer > myL1MuDTChambThContainer;  
  e.getByLabel(dttfSource_,myL1MuDTChambThContainer);
  
  if (!myL1MuDTChambThContainer.isValid()) {
    edm::LogInfo("L1Prompt") << "can't find L1MuDTChambThContainer with label "
			     << dttfSource_.label() ;
    edm::LogInfo("L1Prompt") << "if this fails try to add DATA to the process name." ;

    doDTTH =false;
  }

  if ( doDTTH ) {
  L1MuDTChambThContainer::The_Container* myThContainer =  
    myL1MuDTChambThContainer->getContainer();


  for( int ii=0; ii<MAXDTTH;  ii++) 
    {           		
      dttf_thBx[ii] = -999;
      dttf_thWh[ii] =  -999;
      dttf_thSe[ii] =  -999;
      dttf_thSt[ii] =  -999;
      dttf_thX[ii] = -999.;
      dttf_thY[ii] =  -999.;
      for (int j = 0; j < 7; j++)
	{
         dttf_thTheta[ii][j] =  -999.;
         dttf_thCode[ii][j] =  -999;
	}      
      
    }


//  int bestThQualMap[5][12][3];
//  memset(bestThQualMap,0,180*sizeof(int));

   int ithtr=0;
   dttf_thSize = myThContainer->size();

   for( L1MuDTChambThContainer::The_Container::const_iterator 
	 DTThDigiItr =  myThContainer->begin() ;
       DTThDigiItr != myThContainer->end() ;
       ++DTThDigiItr ) 
     {  
     
      if(ithtr>MAXDTTH-1) continue;
      dttf_thBx[ithtr] = DTThDigiItr->bxNum() + 1;
      dttf_thWh[ithtr] = DTThDigiItr->whNum();
      dttf_thSe[ithtr] = DTThDigiItr->scNum();
      dttf_thSt[ithtr] = DTThDigiItr->stNum();
      dttf_thX[ithtr] = DTThDigiItr->stNum()+4*(DTThDigiItr->whNum()+2);
//	  int xpos = iwh*4+ist+1; ????
      dttf_thY[ithtr] = DTThDigiItr->scNum();
      for (int j = 0; j < 7; j++)
	{
         dttf_thTheta[ithtr][j] = DTThDigiItr->position(j);
         dttf_thCode[ithtr][j] = DTThDigiItr->code(j);
	}
      ithtr++;
     
    }
    }

//

  edm::Handle<L1MuDTTrackContainer > myL1MuDTTrackContainer;

  std::string trstring;
  trstring = dttfSource_.label()+":"+"DATA"+":"+dttfSource_.process();
  edm::InputTag trInputTag(trstring);
  e.getByLabel(trInputTag,myL1MuDTTrackContainer);
  
  if (!myL1MuDTTrackContainer.isValid()) {
    edm::LogInfo("L1Prompt") << "can't find L1MuDTTrackContainer with label "
                               << dttfSource_.label() ;
    doDTTR=false;
  }

  if ( doDTTR ) {
  L1MuDTTrackContainer::TrackContainer *tr =  myL1MuDTTrackContainer->getContainer();

  

  for( int ii=0; ii<MAXDTTR;  ii++) 
    {           		
	
	dttf_trBx[ii] =-999;
	dttf_trTag[ii] =-999; 
	dttf_trQual[ii] =-999;
	dttf_trPtPck[ii] =-999;
	dttf_trPtVal[ii] =-999.;
	dttf_trPhiPck[ii] =-999;
	dttf_trPhiVal[ii] =-999.;
	dttf_trPhiGlob[ii] =-999;
	dttf_trChPck[ii] =-999;
	dttf_trWh[ii] =-999;
	dttf_trSc[ii] =-999;

  }

  int idttr=0; 
  dttf_trSize = tr->size();
  for ( L1MuDTTrackContainer::TrackContainer::const_iterator i 
	  = tr->begin(); i != tr->end(); ++i ) {
        if(idttr>MAXDTTR-1) continue;	
	dttf_trBx[idttr] = i->bx()+1;  
	dttf_trTag[idttr] = i->TrkTag();  
	dttf_trQual[idttr] = i->quality_packed(); 
	dttf_trPtPck[idttr] = i->pt_packed();
	dttf_trPtVal[idttr] = i->ptValue();
	dttf_trPhiPck[idttr] = i->phi_packed(); 
	dttf_trPhiVal[idttr] = i->phiValue();
        int phi_local = i->phi_packed();//range: 0 < phi_local < 31 
        if(phi_local > 15) phi_local -= 32; //range: -16 < phi_local < 15
        int phi_global = phi_local + 12*i->scNum(); //range: -16 < phi_global < 147
        if(phi_global < 0) phi_global = 144; //range: 0 < phi_global < 147
        if(phi_global > 143) phi_global -= 144; //range: 0 < phi_global < 143
	dttf_trPhiGlob[idttr] = phi_global;
	dttf_trChPck[idttr] = i->charge_packed(); 
	dttf_trWh[idttr] = i->whNum();
	dttf_trSc[idttr] = i->scNum();
        idttr++;	  
  }
  
  }




  m_tree->Fill();

}

//--------------
// Operations --
//--------------
void L1PromptAnalysis::book() {

  // GENERAL block branches
  m_tree->Branch("Run",&runn,"Run/I");
  m_tree->Branch("Event",&eventn,"Event/I");
  m_tree->Branch("Lumi",&lumi,"Lumi/I");
  m_tree->Branch("Bx",&bx,"Bx/I");
  m_tree->Branch("Orbit",&orbitn,"Orbit/l");
  m_tree->Branch("Time",&timest,"Time/l");
  
  // Generator info
  if(m_GeneratorInputTag.label() != "none") {
    m_tree->Branch("Weight",&weight,"Weight/F");  
    m_tree->Branch("Pthat",&pthat,"Pthat/F");
  }

  // GEANT block branches
  if(m_SimulationInputTag.label() != "none") {
    m_tree->Branch("Ngen",&ngen,"Ngen/I");
    m_tree->Branch("Pxgen",pxgen,"Pxgen[Ngen]/F");
    m_tree->Branch("Pygen",pygen,"Pygen[Ngen]/F");
    m_tree->Branch("Pzgen",pzgen,"Pzgen[Ngen]/F");
    m_tree->Branch("Ptgen",ptgen,"Ptgen[Ngen]/F");
    m_tree->Branch("Etagen",etagen,"Etagen[Ngen]/F");
    m_tree->Branch("Phigen",phigen,"Phigen[Ngen]/F");
    m_tree->Branch("Chagen",chagen,"Chagen[Ngen]/I");
    m_tree->Branch("Vxgen",vxgen,"Vxgen[Ngen]/F");
    m_tree->Branch("Vygen",vygen,"Vygen[Ngen]/F");
    m_tree->Branch("Vzgen",vzgen,"Vzgen[Ngen]/F");
    m_tree->Branch("Pargen",pargen,"Pargen[Ngen]/I");
  }
  
  // GMT data 
  if(m_GMTInputTag.label() != "none") {
    m_tree->Branch("gmtEvBx",&bxgmt,"gmtEvBx/I");

    // DTBX Trigger block branches
    m_tree->Branch("gmtNdt",&ndt,"gmtNdt/I");
    m_tree->Branch("gmtBxdt",bxd,"gmtBxdt[gmtNdt]/I");
    m_tree->Branch("gmtPtdt",ptd,"gmtPtdt[gmtNdt]/F");
    m_tree->Branch("gmtChdt",chad,"gmtChdt[gmtNdt]/I");
    m_tree->Branch("gmtEtadt",etad,"gmtEtadt[gmtNdt]/F");
    m_tree->Branch("gmtFineEtadt",etafined,"gmtFineEtadt[gmtNdt]/I");
    m_tree->Branch("gmtPhidt",phid,"gmtPhidt[gmtNdt]/F");
    m_tree->Branch("gmtQualdt",quald,"gmtQualdt[gmtNdt]/I");
    m_tree->Branch("gmtDwdt",dwd,"gmtDwdt[gmtNdt]/I");
//    m_tree->Branch("gmtChdt",chd,"gmtChdt[gmtNdt]/I");  
  
    // CSC Trigger block branches
    m_tree->Branch("gmtNcsc",&ncsc,"gmtNcsc/I");
    m_tree->Branch("gmtBxcsc",bxc,"gmtBxcsc[gmtNcsc]/I");
    m_tree->Branch("gmtPtcsc",ptc,"gmtPtcsc[gmtNcsc]/F");
    m_tree->Branch("gmtChcsc",chac,"gmtChcsc[gmtNcsc]/I");
    m_tree->Branch("gmtEtacsc",etac,"gmtEtacsc[gmtNcsc]/F");
    m_tree->Branch("gmtPhicsc",phic,"gmtPhicsc[gmtNcsc]/F");
    m_tree->Branch("gmtQualcsc",qualc,"gmtQualcsc[gmtNcsc]/I");
    m_tree->Branch("gmtDwcsc",dwc,"gmtDwcsc[gmtNcsc]/I");
   
    // RPC barrel Trigger branches
    m_tree->Branch("gmtNrpcb",&nrpcb,"gmtNrpcb/I");
    m_tree->Branch("gmtBxrpcb",bxrb,"gmtBxrpcb[gmtNrpcb]/I");
    m_tree->Branch("gmtPtrpcb",ptrb,"gmtPtrpcb[gmtNrpcb]/F");
    m_tree->Branch("gmtCharpcb",charb,"gmtCharpcb[gmtNrpcb]/I");
    m_tree->Branch("gmtEtarpcb",etarb,"gmtEtarpcb[gmtNrpcb]/F");
    m_tree->Branch("gmtPhirpcb",phirb,"gmtPhirpcb[gmtNrpcb]/F");
    m_tree->Branch("gmtQualrpcb",qualrb,"gmtQualrpcb[gmtNrpcb]/I");
    m_tree->Branch("gmtDwrpcb",dwrb,"gmtDwrpcb[gmtNrpcb]/I");
  
    // RPC forward Trigger branches
    m_tree->Branch("gmtNrpcf",&nrpcf,"gmtNrpcf/I");
    m_tree->Branch("gmtBxrpcf",bxrf,"gmtBxrpcf[gmtNrpcf]/I");
    m_tree->Branch("gmtPtrpcf",ptrf,"gmtPtrpcf[gmtNrpcf]/F");
    m_tree->Branch("gmtCharpcf",charf,"gmtCharpcf[gmtNrpcf]/I");
    m_tree->Branch("gmtEtarpcf",etarf,"gmtEtarpcf[gmtNrpcf]/F");
    m_tree->Branch("gmtPhirpcf",phirf,"gmtPhirpcf[gmtNrpcf]/F");
    m_tree->Branch("gmtQualrpcf",qualrf,"gmtQualrpcf[gmtNrpcf]/I");
    m_tree->Branch("gmtDwrpcf",dwrf,"gmtDwrpcf[gmtNrpcf]/I");
  
    // Global Muon trigger branches
    m_tree->Branch("gmtN",&ngmt,"gmtN/I");
    m_tree->Branch("gmtCandBx",bxg,"gmtBx[gmtN]/I");
    m_tree->Branch("gmtPt",ptg,"gmtPt[gmtN]/F");
    m_tree->Branch("gmtCha",chag,"gmtCha[gmtN]/I");
    m_tree->Branch("gmtEta",etag,"gmtEta[gmtN]/F");
    m_tree->Branch("gmtPhi",phig,"gmtPhi[gmtN]/F");
    m_tree->Branch("gmtQual",qualg,"gmtQual[gmtN]/I");
    m_tree->Branch("gmtDet",detg,"gmtDet[gmtN]/I");
    m_tree->Branch("gmtRank",rankg,"gmtRank[gmtN]/I");
    m_tree->Branch("gmtIsol",isolg,"gmtIsol[gmtN]/I");
    m_tree->Branch("gmtMip",mipg,"gmtMip[gmtN]/I");
    m_tree->Branch("gmtDw",dwg,"gmtDw[gmtN]/I");
    m_tree->Branch("gmtIdxRPCb",idxRPCb,"gmtIdxRPCb[gmtN]/I");
    m_tree->Branch("gmtIdxRPCf",idxRPCf,"gmtIdxRPCf[gmtN]/I");
    m_tree->Branch("gmtIdxDTBX",idxDTBX,"gmtIdxDTBX[gmtN]/I");
    m_tree->Branch("gmtIdxCSC",idxCSC,"gmtIdxCSC[gmtN]/I");
  }
    
  if(m_GTInputTag.label() != "none") {
    // PSB block branches
    m_tree->Branch("gttw1",gttw1,"gttw1[3]/l");
    m_tree->Branch("gttw2",gttw2,"gttw2[3]/l");
    m_tree->Branch("gttt",gttt,"gttt[3]/l");
    
    m_tree->Branch("gtNele",&nele,"gtNele/I");
    m_tree->Branch("gtBxel",bxel,"gtBxel[gtNele]/I");
    m_tree->Branch("gtRankel",rankel,"gtRankel[gtNele]/F");
    m_tree->Branch("gtPhiel",phiel,"gtPhiel[gtNele]/F");
    m_tree->Branch("gtEtael",etael,"gtEtael[gtNele]/F");
    
    m_tree->Branch("gtNjet",&njet,"gtNjet/I");
    m_tree->Branch("gtBxjet",bxjet,"gtBxjet[gtNjet]/I");
    m_tree->Branch("gtRankjet",rankjet,"gtRankjet[gtNjet]/F");
    m_tree->Branch("gtPhijet",phijet,"gtPhijet[gtNjet]/F");
    m_tree->Branch("gtEtajet",etajet,"gtEtajet[gtNjet]/F");
 }
  
  if(gctIsoEmSource_.label() != "none") {
    
    m_tree->Branch("gctIsoEmSize",&gctIsoEmSize,"gctIsoEmSize/I");
    m_tree->Branch("gctIsoEmEta",gctIsoEmEta,"gctIsoEmEta[4]/F");
    m_tree->Branch("gctIsoEmPhi",gctIsoEmPhi,"gctIsoEmPhi[4]/F");
    m_tree->Branch("gctIsoEmRnk",gctIsoEmRnk,"gctIsoEmRnk[4]/F");
    
 }
  if(gctNonIsoEmSource_.label() != "none") {
    m_tree->Branch("gctNonIsoEmSize",&gctNonIsoEmSize,"gctNonIsoEmSize/I");
    m_tree->Branch("gctNonIsoEmEta",gctNonIsoEmEta,"gctNonIsoEmEta[4]/F");
    m_tree->Branch("gctNonIsoEmPhi",gctNonIsoEmPhi,"gctNonIsoEmPhi[4]/F");
    m_tree->Branch("gctNonIsoEmRnk",gctNonIsoEmRnk,"gctNonIsoEmRnk[4]/F");
 }
  if(gctCenJetsSource_.label() != "none") {
    m_tree->Branch("gctCJetSize",&gctCJetSize,"gctCJetSize/I");
    m_tree->Branch("gctCJetEta",gctCJetEta,"gctCJetEta[4]/F");
    m_tree->Branch("gctCJetPhi",gctCJetPhi,"gctCJetPhi[4]/F");
    m_tree->Branch("gctCJetRnk",gctCJetRnk,"gctCJetRnk[4]/F");
 }
  if(gctForJetsSource_.label() != "none") {
    m_tree->Branch("gctFJetSize",&gctFJetSize,"gctFJetSize/I");
    m_tree->Branch("gctFJetEta",gctFJetEta,"gctFJetEta[4]/F");
    m_tree->Branch("gctFJetPhi",gctFJetPhi,"gctFJetPhi[4]/F");
    m_tree->Branch("gctFJetRnk",gctFJetRnk,"gctFJetRnk[4]/F");
 }
  if(gctTauJetsSource_.label() != "none") {
    m_tree->Branch("gctTJetSize",&gctTJetSize,"gctTJetSize/I");
    m_tree->Branch("gctTJetEta",gctTJetEta,"gctTJetEta[4]/F");
    m_tree->Branch("gctTJetPhi",gctTJetPhi,"gctTJetPhi[4]/F");
    m_tree->Branch("gctTJetRnk",gctTJetRnk,"gctTJetRnk[4]/F");
 }
  if(gctEnergySumsSource_.label() != "none") {
    m_tree->Branch("gctEtMiss",&gctEtMiss,"gctEtMiss/F");
    m_tree->Branch("gctEtMissPhi",&gctEtMissPhi,"gctEtMissPhi/F");
    m_tree->Branch("gctEtHad",&gctEtHad,"gctEtHad/F");
    
    m_tree->Branch("gctEtTot",&gctEtTot,"gctEtTot/F");
    m_tree->Branch("gctHFRingEtSumSize",&gctHFRingEtSumSize,"gctHFRingEtSumSize/I");
    m_tree->Branch("gctHFRingEtSumEta",gctHFRingEtSumEta,"gctHFRingEtSumEta[4]/F");
    m_tree->Branch("gctHFBitCountsSize",&gctHFBitCountsSize,"gctHFBitCountsSize/I");
    m_tree->Branch("gctHFBitCountsEta",gctHFBitCountsEta,"gctHFBitCountsEta[4]/F");
 }
 
 
  if(rctSource_.label() != "none") {
    m_tree->Branch("rctRegSize",&rctRegSize,"rctRegSize/I");
    m_tree->Branch("rctRegEta",rctRegEta,"rctRegEta[rctRegSize]/F");
    m_tree->Branch("rctRegPhi",rctRegPhi,"rctRegPhi[rctRegSize]/F");
    m_tree->Branch("rctRegRnk",rctRegRnk,"rctRegRnk[rctRegSize]/F");
    m_tree->Branch("rctRegVeto",rctRegVeto,"rctRegVeto[rctRegSize]/I");
    m_tree->Branch("rctRegBx",rctRegBx,"rctRegBx[rctRegSize]/I");
    m_tree->Branch("rctRegOverFlow",rctRegOverFlow,"rctRegOverFlow[rctRegSize]/I");
    m_tree->Branch("rctRegMip",rctRegMip,"rctRegMip[rctRegSize]/I");
    m_tree->Branch("rctRegFGrain",rctRegFGrain,"rctRegFGrain[rctRegSize]/I");
    m_tree->Branch("rctEmSize",&rctEmSize,"rctEmSize/I");
    m_tree->Branch("rctIsIsoEm",&rctIsIsoEm,"rctIsIsoEm[rctEmSize]/I");
    m_tree->Branch("rctEmEta",rctEmEta,"rctEmEta[rctEmSize]/F");
    m_tree->Branch("rctEmPhi",rctEmPhi,"rctEmPhi[rctEmSize]/F");
    m_tree->Branch("rctEmRnk",rctEmRnk,"rctEmRnk[rctEmSize]/F");
    m_tree->Branch("rctEmBx",rctEmBx,"rctEmBx[rctEmSize]/I");
 }

  if(dttfSource_.label() != "none") {
//dtph
    m_tree->Branch("dttf_phSize",&dttf_phSize,"dttf_phSize/I");
    m_tree->Branch("dttf_phBx",dttf_phBx,"dttf_phBx[dttf_phSize]/I");
    m_tree->Branch("dttf_phWh",dttf_phWh,"dttf_phWh[dttf_phSize]/I");
    m_tree->Branch("dttf_phSe",dttf_phSe,"dttf_phSe[dttf_phSize]/I");
    m_tree->Branch("dttf_phSt",dttf_phSt,"dttf_phSt[dttf_phSize]/I");
    m_tree->Branch("dttf_phAng",dttf_phAng,"dttf_phAng[dttf_phSize]/F");
    m_tree->Branch("dttf_phBandAng",dttf_phBandAng,"dttf_phBandAng[dttf_phSize]/F");
    m_tree->Branch("dttf_phCode",dttf_phCode,"dttf_phCode[dttf_phSize]/I");
    m_tree->Branch("dttf_phX",dttf_phX,"dttf_phX[dttf_phSize]/F");
    m_tree->Branch("dttf_phY",dttf_phY,"dttf_phY[dttf_phSize]/F");
//dtth
    m_tree->Branch("dttf_thSize",&dttf_thSize,"dttf_thSize/I");
    m_tree->Branch("dttf_thBx",dttf_thBx,"dttf_thBx[dttf_thSize]/I");
    m_tree->Branch("dttf_thWh",dttf_thWh,"dttf_thWh[dttf_thSize]/I");
    m_tree->Branch("dttf_thSe",dttf_thSe,"dttf_thSe[dttf_thSize]/I");
    m_tree->Branch("dttf_thSt",dttf_thSt,"dttf_thSt[dttf_thSize]/I");
    m_tree->Branch("dttf_thX",dttf_thX,"dttf_thX[dttf_thSize]/F");
    m_tree->Branch("dttf_thY",dttf_thY,"dttf_thY[dttf_thSize]/F");
    m_tree->Branch("dttf_thTheta",dttf_thTheta,"dttf_thTheta[dttf_thSize][7]/F");
    m_tree->Branch("dttf_thCode",dttf_thCode,"dttf_thCode[dttf_thSize][7]/I");
//dttr
    m_tree->Branch("dttf_trSize",&dttf_trSize,"dttf_trSize/I");
    m_tree->Branch("dttf_trBx"  , dttf_trBx,  "dttf_trBx[dttf_trSize]/I");
    m_tree->Branch("dttf_trQual", dttf_trQual,"dttf_trQual[dttf_trSize]/I");
    m_tree->Branch("dttf_trTag" , dttf_trTag, "dttf_trTag[dttf_trSize]/I");
    m_tree->Branch("dttf_trPtPck",dttf_trPtPck,"dttf_trPtPck[dttf_trSize]/I");
    m_tree->Branch("dttf_trPtVal",dttf_trPtVal,"dttf_trPtVal[dttf_trSize]/F");
    m_tree->Branch("dttf_trPhiPck",dttf_trPhiPck,"dttf_trPhiPck[dttf_trSize]/I");
    m_tree->Branch("dttf_trPhiVal",dttf_trPhiVal,"dttf_trPhiVal[dttf_trSize]/F");
    m_tree->Branch("dttf_trPhiGlob",dttf_trPhiGlob,"dttf_trPhiGlob[dttf_trSize]/I");
    m_tree->Branch("dttf_trChPck",dttf_trChPck,"dttf_trChPck[dttf_trSize]/I");
    m_tree->Branch("dttf_trWh",dttf_trWh,"dttf_trWh[dttf_trSize]/I");
    m_tree->Branch("dttf_trSc",dttf_trSc,"dttf_trSc[dttf_trSize]/I");
 
  }
 
}



