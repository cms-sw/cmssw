
//-------------------------------------------------
//
//   \class L1MuGMTDump
/**
 *   Description:  Dump GMT readout
*/
//                
//   $Date$
//   $Revision$
//
//   I. Mikulec            HEPHY Vienna
//
//--------------------------------------------------
#ifndef L1MU_GMT_DUMP_H
#define L1MU_GMT_DUMP_H

//---------------
// C++ Headers --
//---------------

#include <memory>

//----------------------
// Base Class Headers --
//----------------------
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"


//              ---------------------
//              -- Class Interface --
//              ---------------------
 

class L1MuGMTDump : public edm::EDAnalyzer {

  public:

    // constructor
    explicit L1MuGMTDump(const edm::ParameterSet&);

    // fill tree
    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    virtual void endJob();

  public:

    //GENERAL block
    int             runn;
    int             eventn;
    float           weight;
 
    //GEANT block
    int             ngen;
    float           pxgen[10];
    float           pygen[10];
    float           pzgen[10];
    float           ptgen[10];
    float           etagen[10];
    float           phigen[10];
    int             chagen[10];
    float           vxgen[10];
    float           vygen[10];
    float           vzgen[10];
    
    //DTBX Trigger block
    int             ndt;
    int             bxd[20];
    float           ptd[20];
    int             chad[20];
    float           etad[20];
    int             etafined[20];
    float           phid[20];
    int             quald[20];
    int             tclassd[20];
    int             ntsd[20];

    //CSC Trigger block
    int             ncsc;
    int             bxc[20];
    float           ptc[20];
    int             chac[20];
    float           etac[20];
    float           phic[20];
    int             qualc[20];
    int             ntsc[20];
    int             rankc[20];

    //RPCb Trigger
    int             nrpcb ;
    int             bxrb[20];
    float           ptrb[20];
    int             charb[20];
    float           etarb[20];
    float           phirb[20];
    int             qualrb[20];

    //RPCf Trigger
    int             nrpcf ;
    int             bxrf[20];
    float           ptrf[20];
    int             charf[20];
    float           etarf[20];
    float           phirf[20];
    int             qualrf[20];

    //Global Muon Trigger
    int             ngmt;
    int             bxg[20];
    float           ptg[20];
    int             chag[20];
    float           etag[20];
    float           phig[20];
    int             qualg[20];
    int             detg[20];
    int             rankg[20];
    int             isolg[20];
    int             mipg[20];
    int             datawordg[20];
    int             idxRPCb[20];
    int             idxRPCf[20];
    int             idxDTBX[20];
    int             idxCSC[20];


 
  private:
 
};


#endif
