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

#include "L1Trigger/L1TTwinMux/interface/L1TTwinMuxAlgorithm.h"
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


void L1TTwinMuxAlgorithm::run(edm::Handle<L1MuDTChambPhContainer> inphiDigis,
edm::Handle<L1MuDTChambThContainer> thetaDigis, edm::Handle<RPCDigiCollection> rpcDigis,const edm::EventSetup& c) {


  ///ES Parameters
  const L1TTwinMuxParamsRcd& tmParamsRcd = c.get<L1TTwinMuxParamsRcd>();
  tmParamsRcd.get(tmParamsHandle);
  const L1TTwinMuxParams& tmParams = *tmParamsHandle.product();
  bool onlyRPC = tmParams.get_UseOnlyRPC();
  bool onlyDT = tmParams.get_UseOnlyDT();
  bool useLowQDT = tmParams.get_UseLowQDT();
  bool correctBX = tmParams.get_CorrectDTBxwRPC();
  bool verbose = tmParams.get_Verbose();



  ///Align track segments that are coming in bx-1.
  AlignTrackSegments alignedDTs{*inphiDigis};
  alignedDTs.run(c);
  L1MuDTChambPhContainer const& phiDigis = alignedDTs.getDTContainer();
  //if only DTs are required without bx correction
  //return the aligned track segments
  if(onlyDT && !correctBX && !useLowQDT) {
    m_tm_phi_output = phiDigis;
    return;
  }
  ///Clean RPC hits
  RPCHitCleaner rpcHitCl{*rpcDigis};
  rpcHitCl.run(c);
  RPCDigiCollection const& rpcDigisCleaned = rpcHitCl.getRPCCollection();

  ///Translate RPC digis to DT primitives.
  RPCtoDTTranslator dt_from_rpc{rpcDigisCleaned};
  dt_from_rpc.run(c);
  L1MuDTChambPhContainer const& rpcPhiDigis = dt_from_rpc.getDTContainer();            //Primitves used for RPC->DT (only station 1 and 2)
  L1MuDTChambPhContainer const& rpcHitsPhiDigis = dt_from_rpc.getDTRPCHitsContainer(); //Primitves used for bx correction

  ///Match low q DT primitives with RPC hits in dphiWindow
  DTLowQMatching dtlowq{&phiDigis, rpcHitsPhiDigis};
  dtlowq.run(c);

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
//  DTRPCBxCorrection *rpc_dt_bx = new DTRPCBxCorrection(phiDigis,rpcHitsPhiDigis);
  DTRPCBxCorrection rpc_dt_bx{phiDigis,rpcHitsPhiDigis};
  rpc_dt_bx.run(c);

  L1MuDTChambPhContainer const& phiDigiscp = rpc_dt_bx.getDTContainer();

  ///Add RPC primitives in case that there are no DT primitives.
  std::vector<L1MuDTChambPhDigi> l1ttma_out;

  int bx=0, wheel=0, sector=0, station=1;

  for(bx=-3; bx<=3; bx++){
    for (wheel=-2;wheel<=2; wheel++ ){
      for (sector=0;sector<12; sector++ ){
        for (station=1; station<=4; station++){

           L1MuDTChambPhDigi const* dtts1 = phiDigiscp.chPhiSegm1(wheel,station,sector,bx);
           L1MuDTChambPhDigi const* dtts2 = phiDigiscp.chPhiSegm2(wheel,station,sector,bx);
           L1MuDTChambPhDigi const* rpcts1 = rpcPhiDigis.chPhiSegm1(wheel,station,sector,bx);

           if(!onlyRPC) {
              if(!dtts1 && !dtts2 && !rpcts1) continue;
              if(dtts1 && dtts1->code()!=7) {
                l1ttma_out.push_back(*dtts1);
              }
              if(dtts2 && dtts2->code()!=7) {
                l1ttma_out.push_back(*dtts2);
              }
              if(!onlyDT){
                if(!dtts1 && !dtts2 && rpcts1 && station<=2 ) {
                  l1ttma_out.emplace_back(rpcts1->bxNum() , rpcts1->whNum(), rpcts1->scNum(), rpcts1->stNum(),rpcts1->phi(), rpcts1->phiB(), rpcts1->code(), rpcts1->Ts2Tag(), rpcts1->BxCnt(),2);
                }
              }
            }

            else if(onlyRPC){
              if( rpcts1 && station<=2 ) {
                l1ttma_out.emplace_back(rpcts1->bxNum() , rpcts1->whNum(), rpcts1->scNum(), rpcts1->stNum(),rpcts1->phi(), rpcts1->phiB(), rpcts1->code(), rpcts1->Ts2Tag(), rpcts1->BxCnt(),2);
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
