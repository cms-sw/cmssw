//-------------------------------------------------
//
//   Class: DTRPCBxCorrection
//
//   DTRPCBxCorrection
//
//
//   Author :
//   G. Flouris               U Ioannina    Mar. 2015
//modifications:  G Karathanasis   U Athens
//--------------------------------------------------
#include <iostream>
#include <iomanip>
#include <iterator>

#include "L1Trigger/L1TTwinMux/interface/DTRPCBxCorrection.h"
#include "L1Trigger/L1TTwinMux/interface/L1MuTMChambPhContainer.h"

using namespace std;

DTRPCBxCorrection::DTRPCBxCorrection(L1MuDTChambPhContainer inphiDTDigis, L1MuDTChambPhContainer inphiRPCDigis) :m_phiDTDigis(inphiDTDigis),m_phiRPCDigis(inphiRPCDigis) {
//  m_phiDTDigis=inphiDTDigis;
//  m_phiRPCDigis=inphiRPCDigis;
};


void DTRPCBxCorrection::run( const edm::EventSetup& c) {

  const L1TTwinMuxParamsRcd& tmParamsRcd = c.get<L1TTwinMuxParamsRcd>();
  tmParamsRcd.get(tmParamsHandle);
  const L1TTwinMuxParams& tmParams = *tmParamsHandle.product();

  m_QualityLimit = tmParams.get_USERPCBXFORDTBELOWQUALITY();
  m_DphiWindow   = tmParams.get_DphiWindowBxShift();

  BxCorrection(0);
  BxCorrection(1);

  m_dt_tsshifted.setContainer(m_l1ttma_out);
}

void DTRPCBxCorrection::BxCorrection(int track_seg){

  L1MuDTChambPhDigi * dtts=0;
  L1MuDTChambPhDigi * dttsnew=0;
  L1MuDTChambPhDigi * rpcts1=0;
  
  auto m_phiDTDigis_tm=std::make_shared<L1MuTMChambPhContainer>();
  //std::shared_ptr<L1MuTMChambPhContainer> m_phiDTDigis_tm (new L1MuTMChambPhContainer);
  const std::vector<L1MuDTChambPhDigi> *phiChambVectorDT;
  phiChambVectorDT= m_phiDTDigis.getContainer();
  m_phiDTDigis_tm->setContainer(*phiChambVectorDT);
  auto m_phiRPCDigis_tm=std::make_shared<L1MuTMChambPhContainer>();
  //std::shared_ptr<L1MuTMChambPhContainer> m_phiRPCDigis_tm (new L1MuTMChambPhContainer);
  const std::vector<L1MuDTChambPhDigi> *phiChambVectorRPC;
  phiChambVectorRPC= m_phiRPCDigis.getContainer();
  m_phiRPCDigis_tm->setContainer(*phiChambVectorRPC);

  int bx=0, wheel=0, sector=0, station=1;
  int ibx_dtm = 0, fbx_dtm = 0;
  int ibx_dtp = 0, fbx_dtp = 0;

  for (wheel=-2;wheel<=2; wheel++ ){
   for (sector=0;sector<12; sector++ ){
     for (station=1; station<=4; station++){
       bool shifted[7] = {false, false, false, false,false, false, false};
       bool dups[7] = {false, false, false, false,false, false, false};
       bool secondTs[7] = {false, false, false, false,false, false, false};
       L1MuTMChambPhContainer shiftedPhiDTDigis;
       L1MuDTChambPhDigi *dtts_sh2nd = 0;
       for(bx=3; bx>=-3; bx--){
         vector<int> delta_m, delta_p, delta_0;
         for(int rpcbx=bx-1; rpcbx<=bx+1; rpcbx++){
           dtts=0; rpcts1=0; dttsnew = 0;
           dtts = m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx ,track_seg);

           if(!dtts ) continue;
           int nhits = noRPCHits(*m_phiRPCDigis_tm, rpcbx, wheel, sector, station);
           for(int hit=0; hit<nhits; hit++){
             rpcts1 = m_phiRPCDigis_tm->chPhiSegm(wheel, station, sector, rpcbx,hit);
             //Store in vectors the dphi of matched dt/rpc
             if(rpcts1 && dtts && dtts->code()<m_QualityLimit && deltaPhi(dtts->phi(),rpcts1->phi()) < m_DphiWindow){
               if((dtts->bxNum()-rpcbx)==-1  ) {
                  delta_m.push_back( deltaPhi(dtts->phi(),rpcts1->phi()) );
                  ibx_dtm = dtts->bxNum();
                  fbx_dtm = rpcbx;
                  }
               if((dtts->bxNum()-rpcbx)==0 )  {
                  delta_0.push_back( deltaPhi(dtts->phi(),rpcts1->phi()) );
                  }
               if((dtts->bxNum()-rpcbx)==1 )  {
                  delta_p.push_back( deltaPhi(dtts->phi(),rpcts1->phi()) );
                  ibx_dtp = dtts->bxNum(); 
                  fbx_dtp = rpcbx;
                  }
              }//end if dtts and quality
            }
        }//end of rpc bx

    ///Concatanate all vectors in one
    vector<int> delta = concat_delta(delta_0, delta_p, delta_m);
    ///Shift primitives if vector>0
    if(delta.size() != 0){
       L1MuDTChambPhDigi *dtts_sh = 0;
       std::vector<L1MuDTChambPhDigi> l1ttma_outsh;
       //Find the pair the min dphi(rpc,dt)
       unsigned int min_index = std::distance(delta.begin(), std::min_element(delta.begin(), delta.end())) + 0;
       int init_bx = 0, final_bx = 0;

       if ( ((delta_0.size() <= min_index) && ( min_index < (delta_0.size() + delta_p.size()) ) && delta_p.size()!=0 ) ) {
           init_bx = ibx_dtp;
           final_bx = fbx_dtp;
           }
       else if ( (delta_0.size() + delta_p.size()) <= min_index  && delta_m.size()!=0  ) {
           init_bx = ibx_dtm;
           final_bx = fbx_dtm;
          }
       else continue;
        //Primitve to be shifted in place of dttsnew
       dtts = m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,init_bx,track_seg);
       dttsnew = m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,final_bx,track_seg);
       bool shift_1 = false;
        ///dtts exists and qual lt m_QualityLimit and (there is no primitve in the new bx or it was shifted or it is going to be removed as duplicate)
      // if(dtts && dtts->code()<m_QualityLimit && (!dttsnew  || shifted[final_bx+3] || dups[final_bx+3])) {
        if(dtts && dtts->code()<m_QualityLimit  && (!dttsnew  || shifted[final_bx+3] || dups[final_bx+3])) {
          dtts_sh = new L1MuDTChambPhDigi( final_bx , dtts->whNum(), dtts->scNum(), dtts->stNum(),dtts->phi(), dtts->phiB(), dtts->code(), dtts->Ts2Tag(), dtts->BxCnt(),1);
          l1ttma_outsh.push_back(*dtts_sh);
          shifted[init_bx+3] = true;
          shift_1 = true;}
       if(dtts && dtts->code()<m_QualityLimit && dttsnew) dups[init_bx+3] = true;

        //dtts exists and qual lt m_QualityLimit and dttsnew exists and the previous (shift_1) prim was not shifted and there is empty space in second TS
       if(dtts && dtts->code()<m_QualityLimit && dttsnew && !shift_1 && !m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,final_bx,flipBit(track_seg)) ) {
        ///XXX: Source of discrepancies
        ///in order to send as second TS the two prims must come from different halves of the station
       ///this information does not exist in data
       ///'simulate' this information by requiring different sign in phis and dphi>100
           if(sign(dtts->phi())!=sign(dttsnew->phi())&& deltaPhi(-(dtts->phi()),dttsnew->phi())>100) {
              dtts_sh2nd  = new L1MuDTChambPhDigi( final_bx , dtts->whNum(), dtts->scNum(), dtts->stNum(),dtts->phi(), dtts->phiB(), dtts->code(), flipBit(track_seg), dtts->BxCnt(),1);
              secondTs[final_bx+3] = true;
              dups[init_bx+3] = false;
              shifted[init_bx+3] = true;
              }
          }
        shiftedPhiDTDigis.setContainer(l1ttma_outsh);
   }
 }//end of bx

       for(int bx=-3; bx<=3; bx++){
         L1MuDTChambPhDigi * dtts=0;
         if(secondTs[bx+3] ) 
           if(dtts_sh2nd) {m_l1ttma_out.push_back(*dtts_sh2nd); }
         dtts = shiftedPhiDTDigis.chPhiSegm(wheel,station,sector,bx,track_seg);
         if(dtts){m_l1ttma_out.push_back(*dtts); 
                  continue;}
         if(dups[bx+3]) continue;
         ///if there is no shift then put the original primitive
         dtts = m_phiDTDigis_tm->chPhiSegm(wheel,station,sector,bx,track_seg);
         if(!shifted[bx+3] && dtts) {
            m_l1ttma_out.push_back(*dtts);
         }
        }

   }//end of station
   }//end of sc
  }//end of wheel
}

int DTRPCBxCorrection::deltaPhi(int dt_phi, int rpc2dt_phi ){
  int delta_phi = abs( dt_phi - rpc2dt_phi );
  return delta_phi;
}

std::vector<int> DTRPCBxCorrection::concat_delta(vector<int> delta_0, vector<int> delta_p, vector<int> delta_m){
    vector<int> delta;
    delta.insert(delta.end(), delta_0.begin(), delta_0.end());
    delta.insert(delta.end(), delta_p.begin(), delta_p.end());
    delta.insert(delta.end(), delta_m.begin(), delta_m.end());
    return delta;
}

int DTRPCBxCorrection::sign(float inv){
  if(inv<0) return -1;
  if(inv>0) return 1;
  return 0;
}

int DTRPCBxCorrection::noRPCHits(L1MuTMChambPhContainer inCon, int bx, int wh, int sec, int st){
  int size = 0;
  const std::vector<L1MuDTChambPhDigi>* vInCon = inCon.getContainer();
  for ( auto &i: *vInCon){
    if  (bx == i.bxNum() && i.code() != 7 && i.whNum()==wh && i.scNum()==sec && i.stNum()==st) size++;
  }

  return size;
}

int DTRPCBxCorrection::noRPCHits(L1MuDTChambPhContainer inCon, int bx, int wh, int sec, int st){
  int size = 0;
  const std::vector<L1MuDTChambPhDigi>* vInCon = inCon.getContainer();
  for ( auto &i:* vInCon){
    if  (bx == i.bxNum() && i.code() != 7 && i.whNum()==wh && i.scNum()==sec && i.stNum()==st) size++;
  }

  return size;
}


int DTRPCBxCorrection::flipBit(int inv){
   return (inv^1);
}
