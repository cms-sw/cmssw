//-------------------------------------------------
//
//   Class: IOPrinter
//
//   IOPrinter
//
//
//   Author :
//   G. Flouris               U Ioannina    Feb. 2015
//   modifications:  G Karathanasis    U Athens
//--------------------------------------------------
#include "L1Trigger/L1TTwinMux/interface/IOPrinter.h"
#include "L1Trigger/L1TTwinMux/interface/L1MuTMChambPhContainer.h"

using namespace std;

void IOPrinter::run(edm::Handle<L1MuDTChambPhContainer> inphiDigis, const L1MuDTChambPhContainer & outphiDigis,edm::Handle<RPCDigiCollection> rpcDigis, const edm::EventSetup& c) {

    cout<<"======================================================"<<endl;
    int bx=0, wheel=0, sector=0, station=1;

  ///Align track segments that are coming in bx-1.
  cout<<"DT Inputs/RPCDT Inputs"<<endl;
  cout<<"bx\twheel\tsector\tstation\tphi\tphib\tcode\tts2tag\tbxcnt\trpcbit"<<endl;

  for(bx=-2; bx<=2; bx++){
    for (wheel=-2;wheel<=2; wheel++ ){
      for (sector=0;sector<12; sector++ ){
        for (station=1; station<=4; station++){

          auto dtts1 = inphiDigis->chPhiSegm1(wheel,station,sector,bx);
          auto dtts2 = inphiDigis->chPhiSegm2(wheel,station,sector,bx - 1 );
          if(dtts1 && dtts1->code()!=7) {
            L1MuDTChambPhDigi dt_ts1 = *dtts1;
            cout<<dtts1->bxNum()<<"\t"<<dtts1->whNum()<<"\t"<<dtts1->scNum()<<"\t"<<dtts1->stNum()<<"\t"<<dtts1->phi()<<"\t"<<dtts1->phiB()<<"\t"<<dtts1->code()<<"\t"<<dtts1->Ts2Tag()<<"\t"<<dtts1->BxCnt()<<"\t0"<<endl;
          }
          if(dtts2 && dtts2->code()!=7) {
            L1MuDTChambPhDigi dt_ts2 = *dtts2;
            cout<<dtts2->bxNum()<<"\t"<<dtts2->whNum()<<"\t"<<dtts2->scNum()<<"\t"<<dtts2->stNum()<<"\t"<<dtts2->phi()<<"\t"<<dtts2->phiB()<<"\t"<<dtts2->code()<<"\t"<<dtts2->Ts2Tag()<<"\t"<<dtts2->BxCnt()<<"\t0"<<endl;
          }
        }
      }
    }
  }


  cout<<endl;

  cout<<"RPC Inputs"<<endl;
  cout<<"bx\tring\tsector\tstation\troll\tlayer\tstrip\tphi\tlocalX"<<endl;
  //cout<<"RPCHitCleaner"<<endl;
  for( auto chamber = rpcDigis->begin(); chamber != rpcDigis->end(); ++chamber ) {
     RPCDetId detid = (*chamber).first;
     for( auto digi = (*chamber).second.first ; digi != (*chamber).second.second; ++digi ) {
        RPCDigi digi_out(digi->strip(), digi->bx());
       //if(digi->bx()!=0) continue;
        int phi = RPCtoDTTranslator::radialAngle(detid, c, digi->strip()) << 2 ;
        float localx = RPCtoDTTranslator::localX(detid, c, digi->strip());
        cout<<digi->bx()<<"\t"<<detid.ring()<<"\t"<<detid.sector()-1<<"\t"<<detid.station()<<"\t"<<detid.roll()<<"\t"<<detid.layer()<<"\t"<<digi->strip()<<"\t"<<phi<<"\t"<<localx<<endl;
       }///for digicout
    }///for chamber
  cout<<endl;



  cout<<"TwinMux Output"<<endl;
  cout<<"bx\twheel\tsector\tstation\tphi\tphib\tcode\tts2tag\tbxcnt\trpcbit"<<endl;


  for(bx=-2; bx<=2; bx++){
    for (wheel=-2;wheel<=2; wheel++ ){
      for (sector=0;sector<12; sector++ ){
        for (station=1; station<=4; station++){

          auto dtts1 = outphiDigis.chPhiSegm1(wheel,station,sector,bx);
          auto dtts2 = outphiDigis.chPhiSegm2(wheel,station,sector,bx - 1 );
          if(dtts1&& dtts1->code()!=7) {
            L1MuDTChambPhDigi dt_ts1 = *dtts1;
            cout<<dtts1->bxNum()<<"\t"<<dtts1->whNum()<<"\t"<<dtts1->scNum()<<"\t"<<dtts1->stNum()<<"\t"<<dtts1->phi()<<"\t"<<dtts1->phiB()<<"\t"<<dtts1->code()<<"\t"<<dtts1->Ts2Tag()<<"\t"<<dtts1->BxCnt()<<"\t"<<dtts1->RpcBit()<<endl;
          }
          if(dtts2&& dtts2->code()!=7) {
            L1MuDTChambPhDigi dt_ts2 = *dtts2;
            cout<<dtts2->bxNum()<<"\t"<<dtts2->whNum()<<"\t"<<dtts2->scNum()<<"\t"<<dtts2->stNum()<<"\t"<<dtts2->phi()<<"\t"<<dtts2->phiB()<<"\t"<<dtts2->code()<<"\t"<<dtts2->Ts2Tag()<<"\t"<<dtts2->BxCnt()<<"\t"<<dtts2->RpcBit()<<endl;
          }
        }
      }
    }
  }


cout<<"======================================================"<<endl;
}


void IOPrinter::run(L1MuDTChambPhContainer const* inphiDigis,const L1MuDTChambPhContainer & outphiDigis,RPCDigiCollection const* rpcDigis,const edm::EventSetup& c) {

  cout<<"======================================================"<<endl;
  int bx=0, wheel=0, sector=0, station=1;
  L1MuTMChambPhContainer  inphiDigis_tm;
  const std::vector<L1MuDTChambPhDigi> * vInCon=inphiDigis->getContainer();
  inphiDigis_tm.setContainer(*vInCon);


  ///Align track segments that are coming in bx-1.
  cout<<"RPC->DT Inputs"<<endl;
  cout<<"bx\twheel\tsector\tstation\tphi\tphib\tcode\tts2tag\tbxcnt\trpcbit"<<endl;

  for(bx=-2; bx<=2; bx++){
    for (wheel=-2;wheel<=2; wheel++ ){
      for (sector=0;sector<12; sector++ ){
        for (station=1; station<=4; station++){
          int nhits = DTRPCBxCorrection::nRPCHits(*inphiDigis, bx, wheel, sector, station);
          for(int hit=0; hit<nhits; hit++){
            auto dtts1 = inphiDigis_tm.chPhiSegm(wheel,station,sector,bx,hit);
            if(dtts1) {
              cout<<dtts1->bxNum()<<"\t"<<dtts1->whNum()<<"\t"<<dtts1->scNum()<<"\t"<<dtts1->stNum()<<"\t"<<dtts1->phi()<<"\t"<<dtts1->phiB()<<"\t"<<dtts1->code()<<"\t"<<dtts1->Ts2Tag()<<"\t"<<dtts1->BxCnt()<<"\t0"<<endl;
          }
        }
        }
      }
    }
  }


  cout<<endl;

  cout<<"RPC Inputs"<<endl;
  cout<<"bx\tring\tsector\tstation\troll\tlayer\tstrip\tphi\tlocalX"<<endl;
  //cout<<"RPCHitCleaner"<<endl;
  for( auto chamber = rpcDigis->begin(); chamber != rpcDigis->end(); ++chamber ) {
   RPCDetId detid = (*chamber).first;
   for( auto digi = (*chamber).second.first ; digi != (*chamber).second.second; ++digi ) {
       RPCDigi digi_out(digi->strip(), digi->bx());
       //if(digi->bx()!=0) continue;
       int phi = RPCtoDTTranslator::radialAngle(detid, c, digi->strip()) <<2 ;
       float localx = RPCtoDTTranslator::localX(detid, c, digi->strip());
       cout<<digi->bx()<<"\t"<<detid.ring()<<"\t"<<detid.sector()-1<<"\t"<<detid.station()<<"\t"<<detid.roll()<<"\t"<<detid.layer()<<"\t"<<digi->strip()<<"\t"<<phi<<"\t"<<localx<<endl;
        }///for digicout
    }///for chamber
  cout<<endl;



  cout<<"TwinMux Output"<<endl;
  cout<<"bx\twheel\tsector\tstation\tphi\tphib\tcode\tts2tag\tbxcnt\trpcbit"<<endl;


  for(bx=-2; bx<=2; bx++){
    for (wheel=-2;wheel<=2; wheel++ ){
      for (sector=0;sector<12; sector++ ){
        for (station=1; station<=4; station++){

          auto dtts1 = outphiDigis.chPhiSegm1(wheel,station,sector,bx);
          auto dtts2 = outphiDigis.chPhiSegm2(wheel,station,sector,bx - 1 );
          if(dtts1&& dtts1->code()!=7) {
            L1MuDTChambPhDigi dt_ts1 = *dtts1;
            cout<<dtts1->bxNum()<<"\t"<<dtts1->whNum()<<"\t"<<dtts1->scNum()<<"\t"<<dtts1->stNum()<<"\t"<<dtts1->phi()<<"\t"<<dtts1->phiB()<<"\t"<<dtts1->code()<<"\t"<<dtts1->Ts2Tag()<<"\t"<<dtts1->BxCnt()<<"\t"<<dtts1->RpcBit()<<endl;
          }
          if(dtts2&& dtts2->code()!=7) {
            L1MuDTChambPhDigi dt_ts2 = *dtts2;
            cout<<dtts2->bxNum()<<"\t"<<dtts2->whNum()<<"\t"<<dtts2->scNum()<<"\t"<<dtts2->stNum()<<"\t"<<dtts2->phi()<<"\t"<<dtts2->phiB()<<"\t"<<dtts2->code()<<"\t"<<dtts2->Ts2Tag()<<"\t"<<dtts2->BxCnt()<<"\t"<<dtts2->RpcBit()<<endl;
          }
        }
      }
    }
  }


cout<<"======================================================"<<endl;
}
