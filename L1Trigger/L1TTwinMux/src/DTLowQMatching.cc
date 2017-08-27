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

DTLowQMatching::DTLowQMatching(L1MuDTChambPhContainer* inphiDTDigis, L1MuDTChambPhContainer inphiRPCDigis)  {
m_phiDTDigis=inphiDTDigis;
 m_phiRPCDigis=inphiRPCDigis;
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

  //L1MuDTChambPhDigi * dtts=0;
  //L1MuDTChambPhDigi * rpcts1=0;
  auto dtts=std::make_shared<L1MuDTChambPhDigi>(0,0,0,0,0,0,7,0,0,0);
  auto rpcts1=std::make_shared<L1MuDTChambPhDigi>(0,0,0,0,0,0,7,0,0,0);
  std::auto_ptr<L1MuTMChambPhContainer> m_phiRPCDigis_tm(new L1MuTMChambPhContainer);
  const std::vector<L1MuDTChambPhDigi> *phiChambVector;
  phiChambVector=m_phiRPCDigis.getContainer(); 
  m_phiRPCDigis_tm->setContainer(*phiChambVector );
  std::auto_ptr<L1MuTMChambPhContainer> m_phiDTDigis_tm(new L1MuTMChambPhContainer);
  const std::vector<L1MuDTChambPhDigi> *phiChambVectorDT;
  phiChambVectorDT=m_phiDTDigis->getContainer();
  m_phiDTDigis_tm->setContainer(*phiChambVectorDT );

  int bx=0, wheel=0, sector=0, station=1;
   //cout<<"LowQ Matching  "<<track_seg<<endl;
  for (wheel=-2;wheel<=2; wheel++ ){
   for (sector=0;sector<12; sector++ ){
    for (station=1; station<=4; station++){
     for(bx=-3; bx<=3; bx++){
         vector<int> delta_m, delta_p, delta_0;
         int matched = 0;
         for(int rpcbx=bx-1; rpcbx<=bx+1; rpcbx++){
           // dtts=0; rpcts1=0; 
           // dtts = m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx ,track_seg);
            dtts.reset(new L1MuDTChambPhDigi(0,0,0,0,0,0,7,0,0,0));
            rpcts1.reset(new L1MuDTChambPhDigi(0,0,0,0,0,0,7,0,0,0));
            if (m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx ,track_seg))
              dtts.reset(new L1MuDTChambPhDigi(m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->bxNum(),m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->whNum(),m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->scNum(),m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->stNum(),m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->phi(),m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->phiB(),m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->code(),m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->Ts2Tag(),m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->BxCnt(),m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg)->RpcBit()));
            if(!dtts || dtts->code()>=2) continue;
            int nhits = 0;    
            nhits = DTRPCBxCorrection::noRPCHits(m_phiRPCDigis, rpcbx, wheel, sector, station);
            for(int hit=0; hit<nhits; hit++){
              //rpcts1 = m_phiRPCDigis_tm->chPhiSegm(wheel, station, sector, rpcbx,hit);
              if(m_phiRPCDigis_tm->chPhiSegm(wheel, station, sector, rpcbx,hit))
                 rpcts1.reset(new L1MuDTChambPhDigi(m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->bxNum(),m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->whNum(),m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->scNum(),m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->stNum(),m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->phi(),m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->phiB(),m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->code(),m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->Ts2Tag(),m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->BxCnt(),m_phiRPCDigis_tm->chPhiSegm(wheel,station,sector,rpcbx,hit)->RpcBit()));
              //If DT primitives with q<2 match with rpc hits do nothing else
              //'remove' the primitive by setting is quality to 7
              if(rpcts1 &&rpcts1->code()!=7 && DTRPCBxCorrection::deltaPhi(dtts->phi(),rpcts1->phi()) < m_DphiWindow) {
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
