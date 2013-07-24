//-------------------------------------------------
//
//   Class: L1MuGMTDump
//
//   Description:   Dump GMT readout
//                  
//                
//   $Date: 2010/10/31 17:40:04 $
//   $Revision: 1.12 $
//
//   I. Mikulec            HEPHY Vienna
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/GlobalMuonTrigger/test/L1MuGMTDump.h"

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

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

using namespace std;

//----------------
// Constructors --
//----------------
L1MuGMTDump::L1MuGMTDump(const edm::ParameterSet& ps) {
  m_inputTag = ps.getUntrackedParameter<edm::InputTag>("GMTInputTag", edm::InputTag("gmt"));
}

//--------------
// Destructor --
//--------------
void L1MuGMTDump::endJob() {

}

//--------------
// Operations --
//--------------

void L1MuGMTDump::analyze(const edm::Event& e, const edm::EventSetup& es) {

  //
  // GENERAL block
  //
  runn = e.id().run();
  eventn = e.id().event();

  edm::LogVerbatim("GMTDump") << "run: " << runn << ", event: " << eventn << endl;

  // generetor block

  edm::Handle<edm::SimVertexContainer> simvertices_handle;
  e.getByLabel("g4SimHits",simvertices_handle);
  if (simvertices_handle.isValid()) {
    edm::SimVertexContainer const* simvertices = simvertices_handle.product();

    edm::Handle<edm::SimTrackContainer> simtracks_handle;
    e.getByLabel("g4SimHits",simtracks_handle);
    if (simtracks_handle.isValid()) { 
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
        igen++;
      }
      ngen=igen;
    } else {
      edm::LogWarning("BlockMissing") << "Simulated track block missing" << endl;
      ngen=0;
    }
  } else {
    edm::LogWarning("BlockMissing") << "Simulated vertex block missing" << endl;
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
      if ( idt < MAXDTBX && !(*iter1).empty() ) {
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
      if ( icsc < MAXCSC && !(*iter1).empty() ) {
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
      if ( irpcb < MAXRPC && !(*iter1).empty() ) {
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
      if ( irpcf < MAXRPC && !(*iter1).empty() ) {
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
      if ( igmt < MAXGMT && !(*gmt_iter).empty() ) {
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

  // Header
  edm::LogVerbatim("GMTDump") << "************** GMTDump from " << m_inputTag.label() << ": *************************";

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

  edm::LogVerbatim("GMTDump") << "**************************************************************";
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1MuGMTDump);
