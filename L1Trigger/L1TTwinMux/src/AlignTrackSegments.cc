//-------------------------------------------------
//
//   Class: AlignTrackSegments
//
//   AlignTrackSegments
//
//
//   Author :
//   G. Flouris               U Ioannina    Mar. 2015
//modifications:              GKarathanasis U Athens
//--------------------------------------------------
#include <iostream>
#include <iomanip>
#include <iterator>
#include <memory>
#include "L1Trigger/L1TTwinMux/interface/AlignTrackSegments.h"

using namespace std;

AlignTrackSegments::AlignTrackSegments(L1MuDTChambPhContainer inm_phiDigis) :m_phiDigis(inm_phiDigis)  {
//   m_phiDigis = inm_phiDigis;
};


void AlignTrackSegments::run( const edm::EventSetup& c) {



  std::vector<L1MuDTChambPhDigi> l1ttma_out;

 L1MuDTChambPhDigi const* ts1=0;
 L1MuDTChambPhDigi const* ts2=0;
//  std::auto_ptr<const L1MuDTChambPhDigi> ts1(new L1MuDTChambPhDigi);
 // std::unique_ptr<L1MuDTChambPhDigi> ts2(new L1MuDTChambPhDigi);

  int bx=0, wheel=0, sector=0, station=1;


  for(bx=-3; bx<=3; bx++){
   for (wheel=-3;wheel<=3; wheel++ ){
    for (sector=0;sector<12; sector++ ){
     for (station=1; station<=4; station++){
        ts1=0; ts2=0;
        ///chPhiSegm1 reads the first ts from bx
        ts1=m_phiDigis.chPhiSegm1(wheel,station,sector,bx);
        ///chPhiSegm2 reads the second ts from bx-1
        ts2 = m_phiDigis.chPhiSegm2(wheel,station,sector,bx);
        ///Code = 7 NULL primitive
        if(ts1 && ts1->code()!=7) {
          l1ttma_out.emplace_back(bx , ts1->whNum(), ts1->scNum(), ts1->stNum(),ts1->phi(), ts1->phiB(), ts1->code(), ts1->Ts2Tag(), ts1->BxCnt(),0);

        }
        if((!ts1|| ts1->code()==7 ) && ts2 && ts2->code()!=7) {
           l1ttma_out.emplace_back(ts2->bxNum() , ts2->whNum(), ts2->scNum(), ts2->stNum(),ts2->phi(), ts2->phiB(), ts2->code(), ts2->Ts2Tag(), ts2->BxCnt(),0);
        }
        ///if the second ts (bx-1) co-exist with ts1 shift it to the correct bx
        if(ts1 && ts1->code()!=7 && ts2 && ts2->code()!=7) {
         l1ttma_out.emplace_back(ts2->bxNum() - ts2->Ts2Tag() , ts2->whNum(), ts2->scNum(), ts2->stNum(),ts2->phi(), ts2->phiB(), ts2->code(), ts2->Ts2Tag(), ts2->BxCnt(),0);
         }



     }}}}

  m_dt_tsshifted.setContainer(l1ttma_out);
}

