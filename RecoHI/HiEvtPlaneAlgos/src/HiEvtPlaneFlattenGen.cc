// -*- C++ -*-
//
// Package:    HiEvtPlaneFlattenGen
// Class:      HiEvtPlaneFlattenGen
// 
/**\class HiEvtPlaneFlatten HiEvtPlaneFlatten.cc HiEvtPlaneFlatten/HiEvtPlaneFlatten/src/HiEvtPlaneFlatten.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Stephen Sanders
//         Created:  Mon Jun  7 14:40:12 EDT 2010
// $Id: HiEvtPlaneFlattenGen.cc,v 1.8 2011/09/15 16:43:56 ssanders Exp $
//
//

// system include files

#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneFlattenGen.h"
using namespace std;
using std::rand;
using std::cout;
using std::endl;


//
// constructors and destructor
//
HiEvtPlaneFlattenGen::HiEvtPlaneFlattenGen()
{
  hcentbins = 1;
  hOrder = 20;
  vorder = 2;    //sets order of event plane
}


HiEvtPlaneFlattenGen::~HiEvtPlaneFlattenGen()
{

}

void HiEvtPlaneFlattenGen::Init(int order, int ncentbins, const double * wcent,  string tag, int vord) {
  hOrder = order;  //order of flattening
  vorder = vord;   //1(v1), 2(v2), 3(v3), 4(v4)	
  hcentbins = ncentbins;
  if(hcentbins<=0) hcentbins = 1;
  for(int i = 0; i<= ncentbins; i++) {
    hwcent[i]=wcent[i];
  }
  hcent = new TH1D("hcent","hcent",ncentbins,hwcent);
  hvtx = new TH1D("hvtx","hvtx",nvtxbins,vtxbins);
  hbins = hcentbins*nvtxbins*hOrder;
  if(hbins>MAXCUT) {
    cout<<"Too many cuts for flattening calculation.  RESET to deaults"<<endl;
    hcentbins = 1;
    hOrder = 21;
  }
  for(int i = 0; i<hbins; i++) {
    flatX[i]=0;
    flatY[i]=0;
    flatCnt[i]=0;
  }

}

int HiEvtPlaneFlattenGen::GetCutIndx(double cent, double vtx, int iord){
  int cut;
  int icent = hcent->GetXaxis()->FindBin(cent) - 1;
  if(icent < 0 || icent > hcentbins) return -1;
  int ivtx = hvtx->GetXaxis()->FindBin(vtx) - 1;
  if(ivtx < 0 || ivtx > nvtxbins) return -1;
  cut = hOrder*nvtxbins*icent + hOrder*ivtx + iord;
  if(cut<0 || cut>hbins) return -1;
  return cut;
}

void HiEvtPlaneFlattenGen::Fill(double psi, double vtx, double cent) {
  if(fabs(psi)>4 ) return;
  for(int k = 0; k<hOrder; k++) {
    double fsin = sin(vorder*(k+1)*psi);
    double fcos = cos(vorder*(k+1)*psi);
    int indx = GetCutIndx(cent,vtx,k);
    if(indx>=0) {
      flatX[indx]+=fcos;
      flatY[indx]+=fsin;
      ++flatCnt[indx];
    }
  }
}

double HiEvtPlaneFlattenGen::GetFlatPsi(double psi, double vtx, double cent) {
  double correction = 0;
  for(int k = 0; k<hOrder; k++) {
    int indx = GetCutIndx(cent,vtx,k);
    correction+=(2./(double)((k+1)*vorder))*(flatXDB[indx]*sin(vorder*(k+1)*psi)-flatYDB[indx]*cos(vorder*(k+1)*psi));
  }
  psi+=correction;
  psi=bounds(psi);
  psi=bounds2(psi);
  return psi;
}
//
// member functions
//

// ------------ method called to produce the data  ------------
