//-------------------------------------------------
//
//   Class: DTLowQMatching
//
//   DTLowQMatching
//
//
//   Author :
//   G. Flouris               U Ioannina    Nov. 2016
//   modifications: GKarathanasis  UAthens
//--------------------------------------------------
#include <iostream>
#include <iomanip>
#include <iterator>

#include "L1Trigger/L1TTwinMux/interface/DTLowQMatching.h"
#include "L1Trigger/L1TTwinMux/interface/DTRPCBxCorrection.h"
#include "L1Trigger/L1TTwinMux/interface/L1MuTMChambPhContainer.h"

using namespace std;

DTLowQMatching::DTLowQMatching(L1MuDTChambPhContainer const* inphiDTDigis, L1MuDTChambPhContainer const& inphiRPCDigis): m_phiDTDigis(inphiDTDigis),m_phiRPCDigis(inphiRPCDigis) {
// m_phiDTDigis=inphiDTDigis;
// m_phiRPCDigis=inphiRPCDigis;
};


void DTLowQMatching::run( const edm::EventSetup& c) {

  const L1TTwinMuxParamsRcd& tmParamsRcd = c.get<L1TTwinMuxParamsRcd>();
  tmParamsRcd.get(tmParamsHandle);
  const L1TTwinMuxParams& tmParams = *tmParamsHandle.product();

  m_DphiWindow   = tmParams.get_DphiWindowBxShift();

  Matching(0);
  Matching(1);
}

void DTLowQMatching::Matching(int track_seg){

  L1MuDTChambPhDigi * dtts=nullptr;
  L1MuDTChambPhDigi * rpcts1=nullptr;
  L1MuTMChambPhContainer  m_phiRPCDigis_tm;
  const std::vector<L1MuDTChambPhDigi> *phiChambVector;
  phiChambVector=m_phiRPCDigis.getContainer(); 
  m_phiRPCDigis_tm.setContainer(*phiChambVector );
  
  L1MuTMChambPhContainer m_phiDTDigis_tm;
  const std::vector<L1MuDTChambPhDigi> *phiChambVectorDT;
  phiChambVectorDT=m_phiDTDigis->getContainer();
  m_phiDTDigis_tm.setContainer(*phiChambVectorDT );

  int bx=0, wheel=0, sector=0, station=1;
   //cout<<"LowQ Matching  "<<track_seg<<endl;
  for (wheel=-2;wheel<=2; wheel++ ){
   for (sector=0;sector<12; sector++ ){
    for (station=1; station<=4; station++){
     for(bx=-3; bx<=3; bx++){
         int matched = 0;
         for(int rpcbx=bx-1; rpcbx<=bx+1; rpcbx++){
            dtts=nullptr; rpcts1=nullptr; 
            dtts = m_phiDTDigis_tm.chPhiSegm(wheel,station,sector,bx ,track_seg);
            if(!dtts || dtts->code()>=2) continue;
            int nhits = 0;    
            nhits = DTRPCBxCorrection::nRPCHits(m_phiRPCDigis, rpcbx, wheel, sector, station);
            for(int hit=0; hit<nhits; hit++){
              rpcts1 = m_phiRPCDigis_tm.chPhiSegm(wheel, station, sector, rpcbx,hit);
              //If DT primitives with q<2 match with rpc hits do nothing else
              //'remove' the primitive by setting is quality to 7
              if(rpcts1 && DTRPCBxCorrection::deltaPhi(dtts->phi(),rpcts1->phi()) < m_DphiWindow) {
               matched++;
               continue;}
          }
        }//end of rpc bx
     if(matched == 0 && dtts && dtts->code()<2) {//dtts->setCode(7); 
       // int bx=dtts->bxNum(); cout<<bx<<endl;
       L1MuDTChambPhDigi dtts2(dtts->bxNum(),dtts->whNum(),dtts->scNum(),dtts->stNum(),dtts->phi(),dtts->phiB(),7,dtts->Ts2Tag(),dtts->BxCnt(),dtts->RpcBit());
       *dtts=dtts2;      
       } 
   }//end of dt bx
   }//end of station
   }//end of sc
  }//end of wheel
}
