//-------------------------------------------------
//
//   Class: L1MuGMTDump
//
//   Description:   Dump GMT readout
//                  
//                
//   $Date: 2006/08/17 16:08:16 $
//   $Revision: 1.2 $
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
//----------------
// Constructors --
//----------------
L1MuGMTDump::L1MuGMTDump(const edm::ParameterSet& ps) {

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

  //  const int MAXGEN  = 10;
  const int MAXRPC  = 20;
  const int MAXDTBX = 20;
  const int MAXCSC  = 20;    
  const int MAXGMT  = 20;
      
  //
  // GENERAL block
  //
  runn = e.id().run();
  eventn = e.id().event();

  edm::LogVerbatim("GMTDump") << "run: " << runn << ", event: " << eventn << endl;


  
  // Get GMTReadoutCollection

  edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle; 
  e.getByType(gmtrc_handle);
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


  //
  // DT Trigger print
  //
  ndt = idt;
  edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 DTBX TRIGGER : "
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
  ncsc = icsc;
  edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 CSC  TRIGGER : "
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
  nrpcb = irpcb;
  edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 RPCb  TRIGGER : "
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
  nrpcf = irpcf;
  edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 RPCf  TRIGGER : "
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
  ngmt = igmt;
  edm::LogVerbatim("GMTDump") << "Number of muons found by the L1 Global Muon TRIGGER : "
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

//define this as a plug-in
DEFINE_FWK_MODULE(L1MuGMTDump)
