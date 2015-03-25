#ifndef __HiEvtPlaneFlatten__
#define __HiEvtPlaneFlatten__
// -*- C++ -*-
//
// Package:    HiEvtPlaneFlatten
// Class:      HiEvtPlaneFlatten
// 

//
//
// Original Author:  Stephen Sanders
//         Created:  Mon Jun  7 14:40:12 EDT 2010
// $Id: HiEvtPlaneFlatten.h,v 1.4 2011/11/06 23:17:27 ssanders Exp $
//
//

// system include files
#include <memory>
#include <iostream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

#include "TMath.h"
#include <vector>

#define MAXCUT 10000
#define MAXCUTOFF 1000

//
// class declaration
//

class HiEvtPlaneFlatten {
public:

  explicit HiEvtPlaneFlatten()
  {
    pi = TMath::Pi();
    hbins = 1;
    hOrder = 9;
    vorder = 2;    //sets default order of event plane
    minvtx = -25;
    delvtx = 5;
    nvtxbins = 10;
  }


  void Init(int order, int nbins,   std::string tag, int vord)
  {
    hOrder = order;  //order of flattening
    vorder = vord;   //1(v1), 2(v2), 3(v3), 4(v4)	
    caloCentRefMinBin_ = -1;
    caloCentRefMaxBin_ = -1;
    hbins = nbins*nvtxbins*hOrder;
    obins = nbins*nvtxbins;
    if(hbins>MAXCUT) {
      hbins = 1;
      hOrder = 9;
    }
    for(int i = 0; i<hbins; i++) {
      flatX[i]=0;
      flatY[i]=0;
      flatXDB[i]=0;
      flatYDB[i]=0;
      flatCnt[i]=0;
    } 
    for(int i = 0; i<obins; i++) {
      xoff[i]=0;
      yoff[i]=0;
      xoffDB[i]=0;
      yoffDB[i]=0;
      xyoffcnt[i]=0;
      xyoffmult[i]=0;
      pt[i]=0;
      pt2[i]=0;
      ptDB[i]=0;
      pt2DB[i]=0;
      ptcnt[i]=0;
    }
  }

  int GetCutIndx(int centbin, double vtx, int iord)
  {
    int cut;
    if(centbin < 0 ) return -1;
    //int ietbin = hfetbins*log10( 9.*(et/scale)+1.);
    //if(ietbin>hfetbins) ietbin=hfetbins-1;
    int ibin = centbin;
    int ivtx = (vtx-minvtx)/delvtx;
    if(vtx < minvtx || ivtx >= nvtxbins) return -1;
    cut = hOrder*nvtxbins*ibin + hOrder*ivtx + iord;
    if(cut<0 || cut>=hbins) return -1;
    return cut;
  }

  int GetOffsetIndx(int centbin, double vtx)
  {
    int cut;
    if(centbin < 0 ) return -1;
    //int ietbin = hfetbins*log10( 9.*(et/scale)+1.);
    //if(ietbin>hfetbins) ietbin=hfetbins-1;
    int ibin = centbin;
    int ivtx = (vtx-minvtx)/delvtx;
    if(ivtx < 0 || ivtx > nvtxbins) return -1;
    cut = nvtxbins*ibin + ivtx ;
    if(cut<0 || cut>hbins) return -1;
    return cut;
  }
  
  void Fill(double psi, double vtx, int centbin)
  {
    if(fabs(psi)>4 ) return;
    for(int k = 0; k<hOrder; k++) {
      double fsin = sin(vorder*(k+1)*psi);
      double fcos = cos(vorder*(k+1)*psi);
      int indx = GetCutIndx(centbin,vtx,k);
      if(indx>=0) {
	flatX[indx]+=fcos;
	flatY[indx]+=fsin;
	++flatCnt[indx];
      }
    }
  }
  void FillOffset(double s, double c, uint m, double vtx, int centbin)
  {
    int indx = GetOffsetIndx(centbin,vtx);
    if(indx>=0) {
      xoff[indx]+=c;
      yoff[indx]+=s;
      xyoffmult[indx]+=m;
      ++xyoffcnt[indx];
    }
  }
  void FillPt(double ptval, double vtx, int centbin)
  {
  
    int indx = GetOffsetIndx(centbin,vtx);
    if(indx>=0) {
      pt[indx]+=ptval;
      pt2[indx]+=ptval*ptval;
      ++ptcnt[indx];
    }
  }

  void SetCaloCentRefBins(const int caloCentRefMinBin, const int caloCentRefMaxBin) {
    caloCentRefMinBin_ = caloCentRefMinBin;
    caloCentRefMaxBin_ = caloCentRefMaxBin;
    caloCentRefVal_ = 1.;
  }

  double GetEtScale(double vtx, int centbin) {
    if(caloCentRefMinBin_<0) return 1.;
    int indx = GetOffsetIndx(centbin,vtx);
    int refmin = GetOffsetIndx(caloCentRefMinBin_,vtx);
    int refmax = GetOffsetIndx(caloCentRefMaxBin_,vtx);
    caloCentRefVal_ = 0;
    for(int i = refmin; i<=refmax; i++) {
      caloCentRefVal_+=GetPtDB(i);
    }    
    caloCentRefVal_/=refmax-refmin+1.;
    if(caloCentRefVal_==0 || GetPtDB(indx)==0) return 1.;
    return caloCentRefVal_/GetPtDB(indx);
   }

  double GetW(double pt, double vtx, int centbin)
  {
  
    int indx = GetOffsetIndx(centbin,vtx);
    if(indx>=0) {
      double scale = GetEtScale(vtx,centbin);
      double ptval = GetPtDB(indx)*scale;
      double pt2val = GetPt2DB(indx)*pow(scale,2);
      if(ptval>0) return pt*scale-pt2val/ptval;
    }
    return 0.;
  }

  double GetFlatPsi(double psi, double vtx, int centbin)
  {
    double correction = 0;
    for(int k = 0; k<hOrder; k++) {
      int indx = GetCutIndx(centbin,vtx,k);
      if(indx>=0) correction+=(2./(double)((k+1)*vorder))*(flatXDB[indx]*sin(vorder*(k+1)*psi)-flatYDB[indx]*cos(vorder*(k+1)*psi));
    }
    psi+=correction;
    psi=bounds(psi);
    psi=bounds2(psi);
    return psi;
  }
  
  double GetOffsetPsi(double s, double c, double w, uint m,  double vtx, int centbin)
  {
    int indx = GetOffsetIndx(centbin,vtx);
    double snew = s-yoffDB[indx];
    double cnew = c-xoffDB[indx];
    double psi = atan2(snew,cnew)/vorder;
    if((fabs(snew)<1e-4) && (fabs(cnew)<1e-4)) psi = 0.;
    psi=bounds(psi);
    psi=bounds2(psi);
    soff_ = snew;
    coff_ = cnew;
    w_ = w;
    mult_ = m;

    return psi;
  }
  
  ~HiEvtPlaneFlatten(){}
  int GetHBins(){return hbins;}
  int GetOBins(){return obins;}
  int GetNvtx(){return nvtxbins;}
  double GetVtxMin(){return minvtx;}
  double GetVtxMax(){return minvtx+nvtxbins*delvtx;}
  int GetNcent(){return hbins;}

  double GetX(int bin){return flatX[bin];}
  double GetY(int bin){return flatY[bin];}
  double GetXoff(int bin){return xoff[bin];}
  double GetYoff(int bin){return yoff[bin];}
  double GetXoffDB(int bin){return xoffDB[bin];}
  double GetYoffDB(int bin){return yoffDB[bin];}
  double GetXYoffcnt(int bin){return xyoffcnt[bin];}
  double GetXYoffmult(int bin){return xyoffmult[bin];}
  double GetPt(int bin){return pt[bin];}
  double GetPt2(int bin){return pt2[bin];}
  double GetPtDB(int bin){return ptDB[bin];}
  double GetPt2DB(int bin){return pt2DB[bin];}
  double GetPtcnt(int bin){return ptcnt[bin];}
  double GetXDB(int bin) {return flatXDB[bin];}
  double GetYDB(int bin) {return flatYDB[bin];}


  double GetCnt(int bin) {return flatCnt[bin];}
  void SetXDB(int indx, double val) {flatXDB[indx]=val;}
  void SetYDB(int indx, double val) {flatYDB[indx]=val;}
  void SetXoffDB(int indx, double val) {xoffDB[indx]=val;}
  void SetYoffDB(int indx, double val) {yoffDB[indx]=val;}
  void SetPtDB(int indx, double val) {ptDB[indx]=val;}
  void SetPt2DB(int indx, double val) {pt2DB[indx]=val;}
  double sumSin() const { return soff_; }
  double sumCos() const { return coff_; }
  double sumw()  const { return w_; }
  uint mult()  const {return mult_;}
  double      qx()      const { return (w_>0)? coff_/w_:0.;};
  double      qy()      const { return (w_>0)? soff_/w_:0.;};
  double      q()      const { return ((pow(qx(),2)+pow(qy(),2))>0)? sqrt(pow(qx(),2)+pow(qy(),2)): 0.;};
  Double_t bounds(Double_t ang) {
    if(ang<-pi) ang+=2.*pi;
    if(ang>pi)  ang-=2.*pi;
    return ang;
  }
  Double_t bounds2(Double_t ang) {
    double range = pi/(double) vorder;
    while(ang<-range) { ang+=2*range; }
    while(ang>range)  {ang-=2*range; }
    return ang;
  }
  void SetCentRes1(int bin, double res, double err){ if(bin<100 && bin>=0) {centRes1[bin]=res; centResErr1[bin]=err;}}
  void SetCentRes2(int bin, double res, double err){ if(bin<50 && bin>=0) {centRes2[bin]=res; centResErr2[bin]=err;}}
  void SetCentRes5(int bin, double res, double err){ if(bin<20 && bin>=0) {centRes5[bin]=res; centResErr5[bin]=err;}}
  void SetCentRes10(int bin, double res, double err){ if(bin<10 && bin>=0) {centRes10[bin]=res; centResErr10[bin]=err;}}
  void SetCentRes20(int bin, double res, double err){ if(bin<5 && bin>=0) {centRes20[bin]=res; centResErr20[bin]=err;}}
  void SetCentRes25(int bin, double res, double err){ if(bin<4 && bin>=0) {centRes25[bin]=res; centResErr25[bin]=err;}}
  void SetCentRes30(int bin, double res, double err){ if(bin<3 && bin>=0) {centRes30[bin]=res; centResErr30[bin]=err;}}
  void SetCentRes40(int bin, double res, double err){ if(bin<2 && bin>=0) {centRes40[bin]=res; centResErr40[bin]=err;}}

  double GetCentRes1(int bin){ if(bin<100 && bin>=0) {return centRes1[bin];} else {return 0.;}}
  double GetCentRes2(int bin){ if(bin<50 && bin>=0)  {return centRes2[bin];} else {return 0.;}}
  double GetCentRes5(int bin){ if(bin<20 && bin>=0)  {return centRes5[bin];} else {return 0.;}}
  double GetCentRes10(int bin){ if(bin<10 && bin>=0) {return centRes10[bin];} else {return 0.;}}
  double GetCentRes20(int bin){ if(bin<5 && bin>=0)  {return centRes20[bin];} else {return 0.;}}
  double GetCentRes25(int bin){ if(bin<4 && bin>=0)  {return centRes25[bin];} else {return 0.;}}
  double GetCentRes30(int bin){ if(bin<3 && bin>=0)  {return centRes30[bin];} else {return 0.;}}
  double GetCentRes40(int bin){ if(bin<2 && bin>=0)  {return centRes40[bin];} else {return 0.;}}

  double GetCentResErr1(int bin){ if(bin<100 && bin>=0) {return centResErr1[bin];} else {return 0.;}}
  double GetCentResErr2(int bin){ if(bin<50 && bin>=0)  {return centResErr2[bin];} else {return 0.;}}
  double GetCentResErr5(int bin){ if(bin<20 && bin>=0)  {return centResErr5[bin];} else {return 0.;}}
  double GetCentResErr10(int bin){ if(bin<10 && bin>=0) {return centResErr10[bin];} else {return 0.;}}
  double GetCentResErr20(int bin){ if(bin<5 && bin>=0)  {return centResErr20[bin];} else {return 0.;}}
  double GetCentResErr25(int bin){ if(bin<4 && bin>=0)  {return centResErr25[bin];} else {return 0.;}}
  double GetCentResErr30(int bin){ if(bin<3 && bin>=0)  {return centResErr30[bin];} else {return 0.;}}
  double GetCentResErr40(int bin){ if(bin<2 && bin>=0)  {return centResErr40[bin];} else {return 0.;}}

private:
  double flatX[MAXCUT];
  double flatY[MAXCUT];
  double flatXDB[MAXCUT];
  double flatYDB[MAXCUT];
  double flatCnt[MAXCUT];



  double xoff[MAXCUTOFF];
  double yoff[MAXCUTOFF];
  double xoffDB[MAXCUTOFF];
  double yoffDB[MAXCUTOFF];
  double xyoffcnt[MAXCUTOFF];
  uint xyoffmult[MAXCUTOFF]; 

  double pt[MAXCUTOFF];
  double pt2[MAXCUTOFF];
  double ptDB[MAXCUTOFF];
  double pt2DB[MAXCUTOFF];
  double ptcnt[MAXCUTOFF];

  double centRes1[100];
  double centResErr1[100];

  double centRes2[50];
  double centResErr2[50];

  double centRes5[20];
  double centResErr5[20];

  double centRes10[10];
  double centResErr10[10];

  double centRes20[5];
  double centResErr20[5];

  double centRes25[4];
  double centResErr25[4];

  double centRes30[3];
  double centResErr30[3];

  double centRes40[2];
  double centResErr40[2];


  int hOrder;    //flattening order
  double scale;
  int hbins; //number of bins needed for flattening
  int obins; //number of (x,y) offset bins
  int vorder; //order of flattened event plane
  int caloCentRefMinBin_; //min ref centrality bin for calo weight scale
  int caloCentRefMaxBin_; //max ref centrality bin for calo weight scale
  double caloCentRefVal_; //reference <pt> or <et>
  double pi;

  int nvtxbins;
  double minvtx;
  double delvtx;
  double soff_ ;
  double coff_ ;
  double w_ ;
  uint mult_ ;

};



#endif
