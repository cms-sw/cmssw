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
  const std::vector<L1MuDTChambPhDigi> * vInCon=inphiDTDigis->getContainer();
  m_phiDTDigis->setContainer( * vInCon);
 const std::vector<L1MuDTChambPhDigi> * vInCon2=inphiRPCDigis.getContainer();
 m_phiRPCDigis.setContainer(*vInCon2);

 m_phiRPCDigis2=inphiRPCDigis;
};


void DTLowQMatching::run( const edm::EventSetup& c) {

  const L1TwinMuxParamsRcd& tmParamsRcd = c.get<L1TwinMuxParamsRcd>();
  tmParamsRcd.get(tmParamsHandle);
  const L1TwinMuxParams& tmParams = *tmParamsHandle.product();

	m_DphiWindow   = tmParams.get_DphiWindowBxShift();

  Matching(0);
  Matching(1);
}

void DTLowQMatching::Matching(int track_seg){

  L1MuDTChambPhDigi * dtts=0;
  L1MuDTChambPhDigi * rpcts1=0;
  


  int bx=0, wheel=0, sector=0, station=1;
   //cout<<"LowQ Matching  "<<track_seg<<endl;
	for (wheel=-2;wheel<=2; wheel++ ){
    for (sector=0;sector<12; sector++ ){
     for (station=1; station<=4; station++){

        for(bx=-3; bx<=3; bx++){
        vector<int> delta_m, delta_p, delta_0;
				  int matched = 0;
          for(int rpcbx=bx-1; rpcbx<=bx+1; rpcbx++){
            dtts=0; rpcts1=0; 
            dtts = m_phiDTDigis->chPhiSegm(wheel,station,sector,bx ,track_seg);
            if(!dtts || dtts->code()>=2) continue;
						
            int nhits = 0;
	    
            nhits = DTRPCBxCorrection::noRPCHits(m_phiRPCDigis, rpcbx, wheel, sector, station);

            for(int hit=0; hit<nhits; hit++){
            	rpcts1 = m_phiRPCDigis.chPhiSegm(wheel, station, sector, rpcbx,hit);
							///If DT primitives with q<2 match with rpc hits do nothing else
							///'remove' the primitive by setting is quality to 7
            if(rpcts1 && DTRPCBxCorrection::deltaPhi(dtts->phi(),rpcts1->phi()) < m_DphiWindow) {
								matched++;
								continue;
							}
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
