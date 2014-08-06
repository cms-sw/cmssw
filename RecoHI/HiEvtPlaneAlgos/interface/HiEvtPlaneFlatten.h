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
// $Id: HiEvtPlaneFlatten.h,v 1.5 2012/02/15 10:33:36 eulisse Exp $
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

#define MAXCUT 5000

//
// class declaration
//

class HiEvtPlaneFlatten {
public:

  explicit HiEvtPlaneFlatten()
  {
    pi = TMath::Pi();
    hcentbins = 1;
    hOrder = 20;
    vorder = 2;    //sets order of event plane
    minvtx = -25;
    delvtx = 5;
    nvtxbins = 10;
  }
  void Init(int order, int ncentbins,const int centbinCompression, std::string tag, int vord)
  {
    hOrder = order;  //order of flattening
    vorder = vord;   //1(v1), 2(v2), 3(v3), 4(v4)	
    hcentbins = ncentbins;
    centbinComp = centbinCompression;
    if(hcentbins<=0) hcentbins = 1;
    hbins = hcentbins*nvtxbins*hOrder;
    if(hbins>MAXCUT) {
      std::cout<<"Too many cuts for flattening calculation.  RESET to deaults"<<std::endl;
      hcentbins = 1;
      hOrder = 21;
    }
    for(int i = 0; i<hbins; i++) {
      flatX[i]=0;
      flatY[i]=0;
      flatCnt[i]=0;
    } 
  }

  int GetCutIndx(int centbin, double vtx, int iord)
  {
    int cut;
    int icent = centbin/centbinComp;
    if(icent < 0 || icent > hcentbins) return -1;
    int ivtx = (vtx-minvtx)/delvtx;
    if(ivtx < 0 || ivtx > nvtxbins) return -1;
    cut = hOrder*nvtxbins*icent + hOrder*ivtx + iord;
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

  double GetFlatPsi(double psi, double vtx, double cent)
  {
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
  
  ~HiEvtPlaneFlatten(){}
  int GetHBins(){return hbins;}
  double GetX(int bin){return flatX[bin];}
  double GetY(int bin){return flatY[bin];}
  double GetCnt(int bin) {return flatCnt[bin];}
  void SetXDB(int indx, double val) {flatXDB[indx]=val;}
  void SetYDB(int indx, double val) {flatYDB[indx]=val;}
  Double_t bounds(Double_t ang) {
    if(ang<-pi) ang+=2.*pi;
    if(ang>pi)  ang-=2.*pi;
    return ang;
  }
  Double_t bounds2(Double_t ang) {
    double range = pi/(double) vorder;
    if(ang<-range) ang+=2*range;
    if(ang>range)  ang-=2*range;
    return ang;
  }
private:
  double flatX[MAXCUT];
  double flatY[MAXCUT];
  double flatXDB[MAXCUT];
  double flatYDB[MAXCUT];
  double flatCnt[MAXCUT];
  int hOrder;    //flattening order
  int hcentbins; //# of centrality bins
  int centbinComp;
  int hbins;
  int vorder; //order of flattened event plane
  double pi;

  int nvtxbins;
  double minvtx;
  double delvtx;

};



#endif
