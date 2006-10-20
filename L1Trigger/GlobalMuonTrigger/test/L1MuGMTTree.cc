//-------------------------------------------------
//
//   Class: L1MuGMTTree
//
//   Description:   Build GMT tree
//                  
//                
//   $Date: 2006/08/25 16:51:53 $
//   $Revision: 1.2 $
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
//----------------
// Constructors --
//----------------
L1MuGMTTree::L1MuGMTTree(const edm::ParameterSet& ps) : m_ps(ps), m_file(0), m_tree(0) {}

//--------------
// Destructor --
//--------------
L1MuGMTTree::~L1MuGMTTree() {}

void L1MuGMTTree::beginJob(const edm::EventSetup& es) {
  string output = m_ps.getUntrackedParameter<string>("OutputFile","L1MuGMTTree.root");
  m_file = new TFile(output.c_str(),"RECREATE");
  m_tree = new TTree("h1","GMT Tree");
  m_inputTag = m_ps.getUntrackedParameter<edm::InputTag>("GMTInputTag", edm::InputTag("gmt"));
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
  weight = 1.;

  edm::LogVerbatim("GMTDump") << "run: " << runn << ", event: " << eventn << endl;

  // generetor block

  try {
    edm::Handle<edm::SimVertexContainer> simvertices_handle;
    e.getByLabel("g4SimHits",simvertices_handle);
    edm::SimVertexContainer const* simvertices = simvertices_handle.product();

    edm::Handle<edm::SimTrackContainer> simtracks_handle;
    e.getByLabel("g4SimHits",simtracks_handle);
    edm::SimTrackContainer const* simtracks = simtracks_handle.product();

    edm::SimTrackContainer::const_iterator isimtr;
    int igen = 0;
    for(isimtr=simtracks->begin(); isimtr!=simtracks->end(); isimtr++) {
      if(abs((*isimtr).type())!=13 || igen>=MAXGEN) continue;
      pxgen[igen]=(*isimtr).momentum().px();
      pygen[igen]=(*isimtr).momentum().py();
      pzgen[igen]=(*isimtr).momentum().pz();
      ptgen[igen]=(*isimtr).momentum().perp();
      etagen[igen]=(*isimtr).momentum().eta();
      phigen[igen]=(*isimtr).momentum().phi()>0 ? (*isimtr).momentum().phi() : (*isimtr).momentum().phi()+2*3.14159265359;
      chagen[igen]=(*isimtr).type()>0 ? 1 : -1 ;
      vxgen[igen]=(*simvertices)[(*isimtr).vertIndex()].position().x();
      vygen[igen]=(*simvertices)[(*isimtr).vertIndex()].position().y();
      vzgen[igen]=(*simvertices)[(*isimtr).vertIndex()].position().z();
      igen++;
    }
    ngen=igen;  
  }
  catch(...) {
    edm::LogWarning("BlockMissing") << "Simulated vertex/track block missing" << endl;
    ngen=0;
  }


  // Get GMTReadoutCollection

  edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle; 
  e.getByLabel(m_inputTag.label(),gmtrc_handle);
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

    //
    // DTBX Trigger
    //

    rmc = igmtrr->getDTBXCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( idt < MAXDTBX ) {
	bxd[idt]=(*iter1).bx();
	ptd[idt]=(*iter1).ptValue();
	chad[idt]=(*iter1).chargeValue();
	etad[idt]=(*iter1).etaValue();
	etafined[idt]=0; // etafined[idt]=(*iter1).fineEtaBit();
	phid[idt]=(*iter1).phiValue();
	quald[idt]=(*iter1).quality();
	tclassd[idt]=0; // tclassd[idt]=(*iter1).tc();
	ntsd[idt]=0; // ntsd[idt]=(*iter1).numberOfTSphi();
      
	idt++;
      }
    }

    //
    // CSC Trigger
    //

    rmc = igmtrr->getCSCCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( icsc < MAXCSC ) {
	bxc[icsc]=(*iter1).bx();
	ptc[icsc]=(*iter1).ptValue();
	chac[icsc]=(*iter1).chargeValue();
	etac[icsc]=(*iter1).etaValue();
	phic[icsc]=(*iter1).phiValue();
	qualc[icsc]=(*iter1).quality();
      
        ntsc[icsc]= 0; //(*iter2).trackStubList().size();
        rankc[icsc]=0; //(*iter2).trackId().rank();

	icsc++;
      }
    }

    //
    // RPCb Trigger
    //
    rmc = igmtrr->getBrlRPCCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( irpcb < MAXRPC ) {
	bxrb[irpcb]=(*iter1).bx();
	ptrb[irpcb]=(*iter1).ptValue();
	charb[irpcb]=(*iter1).chargeValue();
	etarb[irpcb]=(*iter1).etaValue();
	phirb[irpcb]=(*iter1).phiValue();
	qualrb[irpcb]=(*iter1).quality();

	irpcb++;
      }
    }

    //
    // RPCf Trigger
    //
    rmc = igmtrr->getFwdRPCCands();
    for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
      if ( irpcf < MAXRPC ) {
	bxrf[irpcf]=(*iter1).bx();
	ptrf[irpcf]=(*iter1).ptValue();
	charf[irpcf]=(*iter1).chargeValue();
	etarf[irpcf]=(*iter1).etaValue();
	phirf[irpcf]=(*iter1).phiValue();
	qualrf[irpcf]=(*iter1).quality();

	irpcf++;
      }
    }

    //
    // GMT Trigger
    //

    vector<L1MuGMTExtendedCand>::const_iterator gmt_iter;
    vector<L1MuGMTExtendedCand> exc = igmtrr->getGMTCands();
    for(gmt_iter=exc.begin(); gmt_iter!=exc.end(); gmt_iter++) {
      if ( igmt < MAXGMT ) {
	bxg[igmt]=(*gmt_iter).bx();
	ptg[igmt]=(*gmt_iter).ptValue();
	chag[igmt]=(*gmt_iter).charge();
	etag[igmt]=(*gmt_iter).etaValue();
	phig[igmt]=(*gmt_iter).phiValue(); 
	qualg[igmt]=(*gmt_iter).quality();
	detg[igmt]=(*gmt_iter).detector();
	rankg[igmt]=(*gmt_iter).rank();
	isolg[igmt]=(*gmt_iter).isol();
	mipg[igmt]=(*gmt_iter).mip();
	int data = (*gmt_iter).getDataWord();
	datawordg[igmt]=data;
      
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
	  if ( (*gmt_iter).isFwd() ) 
	    idxCSC[igmt] = (*gmt_iter).getDTCSCIndex();
	  else
	    idxDTBX[igmt] = (*gmt_iter).getDTCSCIndex();	  
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

  m_tree->Fill();

  if ( L1MuGMTConfig::Debug(1) ) {

    // Generator print
  
    edm::LogVerbatim("GMTDump") << "Number of muons generated: " << ngen << endl;
    edm::LogVerbatim("GMTDump") << "Generated muons:" << endl;
    for(int igen=0; igen<ngen; igen++) {
      edm::LogVerbatim("GMTDump") << setiosflags(ios::showpoint | ios::fixed)
				  << setw(2) << igen+1 << " : "
				  << "pt = " << setw(5) << setprecision(1) << ptgen[igen] << " GeV  "
				  << "charge = " << setw(2) << chagen[igen] << " "
				  << "eta = " << setw(6) << setprecision(3) << etagen[igen] << "  "
				  << "phi = " << setw(5) << setprecision(3) << phigen[igen] << " rad  "
				  << "vx = " << setw(5) << setprecision(3) << vxgen[igen] << " cm "
				  << "vy = " << setw(5) << setprecision(3) << vygen[igen] << " cm "
				  << "vz = " << setw(5) << setprecision(3) << vzgen[igen] << " cm "
				  << endl;
    }

    //
    // DT Trigger print
    //
    edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 DTBX TRIGGER: "
				<< ndt << endl;
    edm::LogVerbatim("GMTDump") << "L1 DT TRIGGER muons: " << endl;
    for(idt=0; idt<ndt; idt++) {
      edm::LogVerbatim("GMTDump") << setiosflags(ios::showpoint | ios::fixed)
				  << setw(2) << idt+1 << " : "
				  << "pt = " << setw(5) << setprecision(1) << ptd[idt] << " GeV  "
				  << "charge = " << setw(2) << chad[idt] << " "
				  << "eta = " << setw(6) << setprecision(3) << etad[idt] << "  "
				  << "phi = " << setw(5) << setprecision(3) << phid[idt] << " rad  "
				  << "quality = " << setw(1) << quald[idt] << "  "
				  << "bx = " << setw(2) << bxd[idt] << endl;
    }

    //
    // CSC Trigger print
    //
    edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 CSC  TRIGGER: "
				<< ncsc << endl;
    edm::LogVerbatim("GMTDump") << "L1 CSC TRIGGER muons: " << endl;
    for(icsc=0; icsc<ncsc; icsc++) {
      edm::LogVerbatim("GMTDump") << setiosflags(ios::showpoint | ios::fixed)
				  << setw(2) << icsc+1 << " : "
				  << "pt = " << setw(5) << setprecision(1) << ptc[icsc] << " GeV  "
				  << "charge = " << setw(2) << chac[icsc] << " "
				  << "eta = " << setw(6) << setprecision(3) << etac[icsc] << "  "
				  << "phi = " << setw(5) << setprecision(3) << phic[icsc] << " rad  "
				  << "quality = " << setw(1) << qualc[icsc] << "  "
				  << "bx = " << setw(2) << bxc[icsc] << endl;
    }

    //
    // RPCb Trigger print
    //
    edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 RPCb  TRIGGER: "
				<< nrpcb << endl;
    edm::LogVerbatim("GMTDump") << "L1 RPCb TRIGGER muons: " << endl;
    for(irpcb=0; irpcb<nrpcb; irpcb++) {
      edm::LogVerbatim("GMTDump") << setiosflags(ios::showpoint | ios::fixed)
				  << setw(2) << irpcb+1 << " : "
				  << "pt = " << setw(5) << setprecision(1) << ptrb[irpcb] << " GeV  "
				  << "charge = " << setw(2) << charb[irpcb] << " "
				  << "eta = " << setw(6) << setprecision(3) << etarb[irpcb] << "  "
				  << "phi = " << setw(5) << setprecision(3) << phirb[irpcb] << " rad  "
				  << "quality = " << setw(1) << qualrb[irpcb] << "  "
				  << "bx = " << setw(2) << bxrb[irpcb] << endl;
    }

    //
    // Rpcf Trigger print
    //
    edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 RPCf  TRIGGER: "
				<< nrpcf << endl;
    edm::LogVerbatim("GMTDump") << "L1 RPCf TRIGGER muons: " << endl;
    for(irpcf=0; irpcf<nrpcf; irpcf++) {
      edm::LogVerbatim("GMTDump") << setiosflags(ios::showpoint | ios::fixed)
				  << setw(2) << irpcf+1 << " : "
				  << "pt = " << setw(5) << setprecision(1) << ptrf[irpcf] << " GeV  "
				  << "charge = " << setw(2) << charf[irpcf] << " "
				  << "eta = " << setw(6) << setprecision(3) << etarf[irpcf] << "  "
				  << "phi = " << setw(5) << setprecision(3) << phirf[irpcf] << " rad  "
				  << "quality = " << setw(1) << qualrf[irpcf] << "  "
				  << "bx = " << setw(2) << bxrf[irpcf] << endl;
    }

    //
    // GMT Trigger print
    //
    edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 Global Muon TRIGGER: "
				<< ngmt << endl;
    edm::LogVerbatim("GMTDump") << "L1 GMT muons: " << endl;
    for(igmt=0; igmt<ngmt; igmt++) {
      if(igmt==4) {edm::LogVerbatim("GMTDump") << "Additional muon candidates" << endl;}
      edm::LogVerbatim("GMTDump") << setiosflags(ios::showpoint | ios::fixed)
				  << setw(2) << igmt+1 << " : "
				  << "pt = " << setw(5) << setprecision(1) << ptg[igmt] << " GeV  "
				  << "charge = " << setw(2) << chag[igmt] << " "
				  << "eta = " << setw(6) << setprecision(3) << etag[igmt]<< "  " 
				  << "phi = " << setw(5) << setprecision(3) << phig[igmt] << " rad  "
				  << "quality = " << setw(1) << qualg[igmt] << "  "
				  << "rank = " << setw(3) << rankg[igmt] << "  "
				  << "bx = " << setw(2) << bxg[igmt] << "  "
				  << "detectors = " << setw(2) << idxDTBX[igmt] << idxRPCb[igmt] 
				  << idxCSC[igmt] << idxRPCf[igmt] << endl;
    }
  }

}

//--------------
// Operations --
//--------------
void L1MuGMTTree::book() {

  // GENERAL block branches
  m_tree->Branch("Run",&runn,"Run/I");
  m_tree->Branch("Event",&eventn,"Event/I");
  m_tree->Branch("Weight",&weight,"Weight/F");  

  // GEANT block branches
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
  
  // DTBX Trigger block branches
  m_tree->Branch("Ndt",&ndt,"Ndt/I");
  m_tree->Branch("Bxd",bxd,"Bxd[Ndt]/I");
  m_tree->Branch("Ptd",ptd,"Ptd[Ndt]/F");
  m_tree->Branch("Chad",chad,"Chad[Ndt]/I");
  m_tree->Branch("Etad",etad,"Etad[Ndt]/F");
  m_tree->Branch("Etafined",etafined,"Etafined[Ndt]/I");
  m_tree->Branch("Phid",phid,"Phid[Ndt]/F");
  m_tree->Branch("Quald",quald,"Quald[Ndt]/I");
  m_tree->Branch("TClassd",tclassd,"TClassd[Ndt]/I");
  m_tree->Branch("Ntsd",ntsd,"Ntsd[Ndt]/I");  

  // CSC Trigger block branches
  m_tree->Branch("Ncsc",&ncsc,"Ncsc/I");
  m_tree->Branch("Bxc",bxc,"Bxc[Ncsc]/I");
  m_tree->Branch("Ptc",ptc,"Ptc[Ncsc]/F");
  m_tree->Branch("Chac",chac,"Chac[Ncsc]/I");
  m_tree->Branch("Etac",etac,"Etac[Ncsc]/F");
  m_tree->Branch("Phic",phic,"Phic[Ncsc]/F");
  m_tree->Branch("Qualc",qualc,"Qualc[Ncsc]/I");
  m_tree->Branch("Ntsc",ntsc,"Ntsc[Ncsc]/I");
  m_tree->Branch("Rankc",rankc,"Rankc[Ncsc]/I");

  // RPC barrel Trigger branches
  m_tree->Branch("Nrpcb",&nrpcb,"Nrpcb/I");
  m_tree->Branch("Ptrb",ptrb,"Ptrb[Nrpcb]/F");
  m_tree->Branch("Charb",charb,"Charb[Nrpcb]/I");
  m_tree->Branch("Etarb",etarb,"Etarb[Nrpcb]/F");
  m_tree->Branch("Phirb",phirb,"Phirb[Nrpcb]/F");
  m_tree->Branch("Qualrb",qualrb,"Qualrb[Nrpcb]/I");

  // RPC forward Trigger branches
  m_tree->Branch("Nrpcf",&nrpcf,"Nrpcf/I");
  m_tree->Branch("Ptrf",ptrf,"Ptrf[Nrpcf]/F");
  m_tree->Branch("Charf",charf,"Charf[Nrpcf]/I");
  m_tree->Branch("Etarf",etarf,"Etarf[Nrpcf]/F");
  m_tree->Branch("Phirf",phirf,"Phirf[Nrpcf]/F");
  m_tree->Branch("Qualrf",qualrf,"Qualrf[Nrpcf]/I");

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
  m_tree->Branch("DataWordg",datawordg,"DataWordg[Ngmt]/I");
  m_tree->Branch("IdxRPCb",idxRPCb,"IdxRPCb[Ngmt]/I");
  m_tree->Branch("IdxRPCf",idxRPCf,"IdxRPCf[Ngmt]/I");
  m_tree->Branch("IdxDTBX",idxDTBX,"IdxDTBX[Ngmt]/I");
  m_tree->Branch("IdxCSC",idxCSC,"IdxCSC[Ngmt]/I");


}


//define this as a plug-in
DEFINE_FWK_MODULE(L1MuGMTTree)
