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

  for(int bx=-3; bx<=3; bx++){
   for (int wheel=-3;wheel<=3; wheel++ ){
    for (int sector=0;sector<12; sector++ ){
     for (int station=1; station<=4; station++){
        ///chPhiSegm1 reads the first ts from bx
        L1MuDTChambPhDigi const* ts1 = m_phiDigis.chPhiSegm1(wheel,station,sector,bx);
        ///chPhiSegm2 reads the second ts from bx-1
        L1MuDTChambPhDigi const* ts2 = m_phiDigis.chPhiSegm2(wheel,station,sector,bx);
        ///Code = 7 NULL primitive
        if(ts1!=nullptr && ts1->code()!=7) {
          l1ttma_out.emplace_back(bx , ts1->whNum(), ts1->scNum(), ts1->stNum(),ts1->phi(), ts1->phiB(), ts1->code(), ts1->Ts2Tag(), ts1->BxCnt(),0);

        }
        if((ts1==nullptr || ts1->code()==7 ) && ts2 && ts2->code()!=7) {
           l1ttma_out.emplace_back(ts2->bxNum() , ts2->whNum(), ts2->scNum(), ts2->stNum(),ts2->phi(), ts2->phiB(), ts2->code(), ts2->Ts2Tag(), ts2->BxCnt(),0);
        }
        ///if the second ts (bx-1) co-exist with ts1 shift it to the correct bx
        if(ts1!=nullptr && ts1->code()!=7 && ts2 && ts2->code()!=7) {
         l1ttma_out.emplace_back(ts2->bxNum() - ts2->Ts2Tag() , ts2->whNum(), ts2->scNum(), ts2->stNum(),ts2->phi(), ts2->phiB(), ts2->code(), ts2->Ts2Tag(), ts2->BxCnt(),0);
         }



     }}}}

  m_dt_tsshifted.setContainer(l1ttma_out);
}

