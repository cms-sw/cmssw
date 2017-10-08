//-------------------------------------------------
//
//   Class: L1TwinMuxAlgortithm
//
//   L1TwinMuxAlgortithm
//
//
//   Author :
//   G. Flouris               U Ioannina    Feb. 2015
//   modifications: G Karathanasis
//--------------------------------------------------

#include <iostream>
#include <iomanip>
#include <iterator>

#include "L1Trigger/L1TTwinMux/interface/L1TwinMuxAlgortithm.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/L1TTwinMux/interface/AlignTrackSegments.h"
#include "L1Trigger/L1TTwinMux/interface/RPCtoDTTranslator.h"
#include "L1Trigger/L1TTwinMux/interface/DTRPCBxCorrection.h"
#include "L1Trigger/L1TTwinMux/interface/DTLowQMatching.h"
#include "L1Trigger/L1TTwinMux/interface/RPCHitCleaner.h"
#include "L1Trigger/L1TTwinMux/interface/IOPrinter.h"
#include  "L1Trigger/L1TTwinMux/interface/L1MuTMChambPhContainer.h"

using namespace std;


void L1TwinMuxAlgortithm::run(
                                                            edm::Handle<L1MuDTChambPhContainer> inphiDigis,
                                                            edm::Handle<L1MuDTChambThContainer> thetaDigis,
                                                            edm::Handle<RPCDigiCollection> rpcDigis,
                                                            const edm::EventSetup& c) {


  ///ES Parameters
  const L1TwinMuxParamsRcd& tmParamsRcd = c.get<L1TwinMuxParamsRcd>();
  tmParamsRcd.get(tmParamsHandle);
  const L1TwinMuxParams& tmParams = *tmParamsHandle.product();
  bool onlyRPC = tmParams.get_UseOnlyRPC();
  bool onlyDT = tmParams.get_UseOnlyDT();
  bool useLowQDT = tmParams.get_UseLowQDT();
  bool correctBX = tmParams.get_CorrectDTBxwRPC();
  bool verbose = tmParams.get_Verbose();



  ///Align track segments that are coming in bx-1.
  AlignTrackSegments *alignedDTs = new AlignTrackSegments(*inphiDigis);
  alignedDTs->run(c);
  L1MuDTChambPhContainer phiDigis = alignedDTs->getDTContainer();
  //if only DTs are required without bx correction
  //return the aligned track segments
  if(onlyDT && !correctBX && !useLowQDT) {
    m_tm_phi_output = phiDigis;
    return;
  }
  ///Clean RPC hits
  RPCHitCleaner *rpcHitCl = new RPCHitCleaner(*rpcDigis);
  rpcHitCl->run(c);
  RPCDigiCollection rpcDigisCleaned = rpcHitCl->getRPCCollection();

  ///Translate RPC digis to DT primitives.
  RPCtoDTTranslator *dt_from_rpc = new RPCtoDTTranslator(rpcDigisCleaned);
  dt_from_rpc->run(c);
  L1MuDTChambPhContainer rpcPhiDigis = dt_from_rpc->getDTContainer();            //Primitves used for RPC->DT (only station 1 and 2)
  L1MuDTChambPhContainer rpcHitsPhiDigis = dt_from_rpc->getDTRPCHitsContainer(); //Primitves used for bx correction

  ///Match low q DT primitives with RPC hits in dphiWindow
  DTLowQMatching *dtlowq = new DTLowQMatching(&phiDigis, rpcHitsPhiDigis);
  dtlowq->run(c);

  if(onlyDT && !correctBX && useLowQDT) {
    m_tm_phi_output = phiDigis;
    if(verbose){
       IOPrinter ioPrinter;
       cout<<"======DT========"<<endl;
       ioPrinter.run(inphiDigis, m_tm_phi_output, rpcDigis, c);
       cout<<"======RPC========"<<endl;
       ioPrinter.run(&rpcHitsPhiDigis, m_tm_phi_output, &rpcDigisCleaned, c);
       cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
    }

    return;
  }


  ///Correct(in bx) DT primitives by comparing them to RPC.
  DTRPCBxCorrection *rpc_dt_bx = new DTRPCBxCorrection(phiDigis,rpcHitsPhiDigis);
  rpc_dt_bx->run(c);

  L1MuDTChambPhContainer phiDigiscp = rpc_dt_bx->getDTContainer();

  ///Add RPC primitives in case that there are no DT primitives.
  std::vector<L1MuDTChambPhDigi> l1ttma_out;
  L1MuDTChambPhDigi const* dtts1=0;
  L1MuDTChambPhDigi const* dtts2=0;

  L1MuDTChambPhDigi const* rpcts1=0;

  int bx=0, wheel=0, sector=0, station=1;

  for(bx=-3; bx<=3; bx++){
    for (wheel=-2;wheel<=2; wheel++ ){
      for (sector=0;sector<12; sector++ ){
        for (station=1; station<=4; station++){

          dtts1=0; dtts2=0; rpcts1=0;

          dtts1 = phiDigiscp.chPhiSegm1(wheel,station,sector,bx);
          dtts2 = phiDigiscp.chPhiSegm2(wheel,station,sector,bx);
          rpcts1 = rpcPhiDigis.chPhiSegm1(wheel,station,sector,bx);

            if(!onlyRPC) {
              if(!dtts1 && !dtts2 && !rpcts1 ) continue;
              if(dtts1 && dtts1->code()!=7) {
                l1ttma_out.push_back(*dtts1);
              }
              if(dtts2 && dtts2->code()!=7) {
                l1ttma_out.push_back(*dtts2);
              }
              if(!onlyDT){
                if(!dtts1 && !dtts2 && rpcts1 && station<=2 ) {
                  L1MuDTChambPhDigi dt_rpc( rpcts1->bxNum() , rpcts1->whNum(), rpcts1->scNum(), rpcts1->stNum(),rpcts1->phi(), rpcts1->phiB(), rpcts1->code(), rpcts1->Ts2Tag(), rpcts1->BxCnt(),2);
                  l1ttma_out.push_back(dt_rpc);
                }
              }
            }

            if(onlyRPC){
              if( rpcts1 && station<=2 ) {
              L1MuDTChambPhDigi dt_rpc( rpcts1->bxNum() , rpcts1->whNum(), rpcts1->scNum(), rpcts1->stNum(),rpcts1->phi(), rpcts1->phiB(), rpcts1->code(), rpcts1->Ts2Tag(), rpcts1->BxCnt(),2);
              l1ttma_out.push_back(dt_rpc);

              }
            }

        }
      }
    }
  }

m_tm_phi_output.setContainer(l1ttma_out);

  if(verbose){
     IOPrinter ioPrinter;
     cout<<"======DT========"<<endl;
     ioPrinter.run(inphiDigis, m_tm_phi_output, rpcDigis, c);
     cout<<"======RPC========"<<endl;
     ioPrinter.run(&rpcHitsPhiDigis, m_tm_phi_output, &rpcDigisCleaned, c);
     cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
  }

}
