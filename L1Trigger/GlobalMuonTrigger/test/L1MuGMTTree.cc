//-------------------------------------------------
//
//   Class: L1MuGMTTree
//
//   Description:   Build GMT tree
//                  
//                
//   $Date: 2009/12/18 20:44:58 $
//   $Revision: 1.17 $
//
//   I. Mikulec            HEPHY Vienna
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/GlobalMuonTrigger/test/L1MuGMTTree.h"

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
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"


#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"


using namespace std;

//----------------
// Constructors --
//----------------
L1MuGMTTree::L1MuGMTTree(const edm::ParameterSet& ps) : m_file(0), m_tree(0) {
  
  m_GMTInputTag = ps.getParameter<edm::InputTag>("GMTInputTag");
  m_GTEvmInputTag = ps.getParameter<edm::InputTag>("GTEvmInputTag");
  m_GTInputTag = ps.getParameter<edm::InputTag>("GTInputTag");
  m_GeneratorInputTag = ps.getParameter<edm::InputTag>("GeneratorInputTag");
  m_SimulationInputTag = ps.getParameter<edm::InputTag>("SimulationInputTag");
  
  m_PhysVal = ps.getParameter<bool>("PhysVal");
  
  m_outfilename = ps.getUntrackedParameter<string>("OutputFile","L1MuGMTTree.root");
}

//--------------
// Destructor --
//--------------
L1MuGMTTree::~L1MuGMTTree() {}

void L1MuGMTTree::beginJob() {
  m_file = TFile::Open(m_outfilename.c_str(),"RECREATE");
  m_tree = new TTree("h1","GMT Tree");
  book();
}

void L1MuGMTTree::endJob() {
  m_file->Write();
  m_file->Close();
}

//--------------
// Operations --
//--------------

void L1MuGMTTree::analyze(const edm::Event& e, const edm::EventSetup& es) {

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
  
  m_tree->Fill();

}

//--------------
// Operations --
//--------------
void L1MuGMTTree::book() {

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
    m_tree->Branch("Bxgmt",&bxgmt,"Bxgmt/I");

    // DTBX Trigger block branches
    m_tree->Branch("Ndt",&ndt,"Ndt/I");
    m_tree->Branch("Bxd",bxd,"Bxd[Ndt]/I");
    m_tree->Branch("Ptd",ptd,"Ptd[Ndt]/F");
    m_tree->Branch("Chad",chad,"Chad[Ndt]/I");
    m_tree->Branch("Etad",etad,"Etad[Ndt]/F");
    m_tree->Branch("Etafined",etafined,"Etafined[Ndt]/I");
    m_tree->Branch("Phid",phid,"Phid[Ndt]/F");
    m_tree->Branch("Quald",quald,"Quald[Ndt]/I");
    m_tree->Branch("Dwd",dwd,"Dwd[Ndt]/I");
    m_tree->Branch("Chd",chd,"Chd[Ndt]/I");  
  
    // CSC Trigger block branches
    m_tree->Branch("Ncsc",&ncsc,"Ncsc/I");
    m_tree->Branch("Bxc",bxc,"Bxc[Ncsc]/I");
    m_tree->Branch("Ptc",ptc,"Ptc[Ncsc]/F");
    m_tree->Branch("Chac",chac,"Chac[Ncsc]/I");
    m_tree->Branch("Etac",etac,"Etac[Ncsc]/F");
    m_tree->Branch("Phic",phic,"Phic[Ncsc]/F");
    m_tree->Branch("Qualc",qualc,"Qualc[Ncsc]/I");
    m_tree->Branch("Dwc",dwc,"Dwc[Ncsc]/I");
   
    // RPC barrel Trigger branches
    m_tree->Branch("Nrpcb",&nrpcb,"Nrpcb/I");
    m_tree->Branch("Bxrb",bxrb,"Bxrb[Nrpcb]/I");
    m_tree->Branch("Ptrb",ptrb,"Ptrb[Nrpcb]/F");
    m_tree->Branch("Charb",charb,"Charb[Nrpcb]/I");
    m_tree->Branch("Etarb",etarb,"Etarb[Nrpcb]/F");
    m_tree->Branch("Phirb",phirb,"Phirb[Nrpcb]/F");
    m_tree->Branch("Qualrb",qualrb,"Qualrb[Nrpcb]/I");
    m_tree->Branch("Dwrb",dwrb,"Dwrb[Nrpcb]/I");
  
    // RPC forward Trigger branches
    m_tree->Branch("Nrpcf",&nrpcf,"Nrpcf/I");
    m_tree->Branch("Bxrf",bxrf,"Bxrf[Nrpcf]/I");
    m_tree->Branch("Ptrf",ptrf,"Ptrf[Nrpcf]/F");
    m_tree->Branch("Charf",charf,"Charf[Nrpcf]/I");
    m_tree->Branch("Etarf",etarf,"Etarf[Nrpcf]/F");
    m_tree->Branch("Phirf",phirf,"Phirf[Nrpcf]/F");
    m_tree->Branch("Qualrf",qualrf,"Qualrf[Nrpcf]/I");
    m_tree->Branch("Dwrf",dwrf,"Dwrf[Nrpcf]/I");
  
    // Global Muon trigger branches
    m_tree->Branch("Ngmt",&ngmt,"Ngmt/I");
    m_tree->Branch("Bxg",bxg,"Bxg[Ngmt]/I");
    m_tree->Branch("Ptg",ptg,"Ptg[Ngmt]/F");
    m_tree->Branch("Chag",chag,"Chag[Ngmt]/I");
    m_tree->Branch("Etag",etag,"Etag[Ngmt]/F");
    m_tree->Branch("Phig",phig,"Phig[Ngmt]/F");
    m_tree->Branch("Qualg",qualg,"Qualg[Ngmt]/I");
    m_tree->Branch("Detg",detg,"Detg[Ngmt]/I");
    m_tree->Branch("Rankg",rankg,"Rankg[Ngmt]/I");
    m_tree->Branch("Isolg",isolg,"Isolg[Ngmt]/I");
    m_tree->Branch("Mipg",mipg,"Mipg[Ngmt]/I");
    m_tree->Branch("Dwg",dwg,"Dwg[Ngmt]/I");
    m_tree->Branch("IdxRPCb",idxRPCb,"IdxRPCb[Ngmt]/I");
    m_tree->Branch("IdxRPCf",idxRPCf,"IdxRPCf[Ngmt]/I");
    m_tree->Branch("IdxDTBX",idxDTBX,"IdxDTBX[Ngmt]/I");
    m_tree->Branch("IdxCSC",idxCSC,"IdxCSC[Ngmt]/I");
  }
    
  if(m_GTInputTag.label() != "none") {
    // PSB block branches
    m_tree->Branch("Gttw1",gttw1,"Gttw1[3]/l");
    m_tree->Branch("Gttw2",gttw2,"Gttw2[3]/l");
    m_tree->Branch("Gttt",gttt,"Gttt[3]/l");
    
    m_tree->Branch("Nele",&nele,"Nele/I");
    m_tree->Branch("Bxel",bxel,"Bxel[Nele]/I");
    m_tree->Branch("Rankel",rankel,"Rankel[Nele]/F");
    m_tree->Branch("Phiel",phiel,"Phiel[Nele]/F");
    m_tree->Branch("Etael",etael,"Etael[Nele]/F");
    
    m_tree->Branch("Njet",&njet,"Njet/I");
    m_tree->Branch("Bxjet",bxjet,"Bxjet[Njet]/I");
    m_tree->Branch("Rankjet",rankjet,"Rankjet[Njet]/F");
    m_tree->Branch("Phijet",phijet,"Phijet[Njet]/F");
    m_tree->Branch("Etajet",etajet,"Etajet[Njet]/F");
 }
  
}



//define this as a plug-in
DEFINE_FWK_MODULE(L1MuGMTTree);
