///////////////////////////////////////////////////////////////////////////////
// File: TrackProducerFP420.cc
// Date: 12.2006
// Description: TrackProducerFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "RecoRomanPot/RecoFP420/interface/TrackProducerFP420.h"
#include <stdio.h>
#include <gsl/gsl_fit.h>
#include<vector>
#include <iostream>
using namespace std;

//#define debugmaxampl
//#define debugvar1
//#define debugvar2
//#define debugvar22
//#define debugsophisticated
//#define debugsophisticated
//#define debug3d
//#define debug3d30

TrackProducerFP420::TrackProducerFP420(int asn0, int apn0, int arn0, int axytype, double az420, double azD2, double azD3, double apitchX, double apitchY, double apitchXW, double apitchYW, double aZGapLDet, double aZSiStep, double aZSiPlane, double aZSiDet, double azBlade, double agapBlade, bool aUseHalfPitchShiftInX, bool aUseHalfPitchShiftInY, bool aUseHalfPitchShiftInXW, bool aUseHalfPitchShiftInYW, double adXX, double adYY, float achiCutX, float achiCutY, double azinibeg, int verbosity, double aXsensorSize, double aYsensorSize) {
  //
  // Everything that depend on the det
  //
  verbos=verbosity;
  sn0 = asn0;
  pn0 = apn0;
  rn0 = arn0;
  xytype = axytype;
  z420= az420;
  zD2 = azD2;
  zD3 = azD3;
  //zUnit= azUnit;
  pitchX = apitchX;
  pitchY = apitchY;
  pitchXW = apitchXW;
  pitchYW = apitchYW;
  ZGapLDet = aZGapLDet;
  ZSiStep = aZSiStep; 
  ZSiPlane = aZSiPlane;
  ZSiDet = aZSiDet;
  zBlade = azBlade;
  gapBlade = agapBlade;

  UseHalfPitchShiftInX = aUseHalfPitchShiftInX;
  UseHalfPitchShiftInY = aUseHalfPitchShiftInY;
  UseHalfPitchShiftInXW = aUseHalfPitchShiftInXW;
  UseHalfPitchShiftInYW = aUseHalfPitchShiftInYW;
  dXX = adXX;
  dYY = adYY;
  chiCutX = achiCutX;
  chiCutY = achiCutY;
  zinibeg = azinibeg;
  XsensorSize = aXsensorSize;
  YsensorSize = aYsensorSize;

  if (verbos > 0) {
    std::cout << "TrackProducerFP420: call constructor" << std::endl;
    std::cout << " sn0= " << sn0 << " pn0= " << pn0 << " rn0= " << rn0 << " xytype= " << xytype << std::endl;
    std::cout << " zD2= " << zD2 << " zD3= " << zD3 << " zinibeg= " << zinibeg << std::endl;
    //std::cout << " zUnit= " << zUnit << std::endl;
    std::cout << " pitchX= " << pitchX << " pitchY= " << pitchY << std::endl;
    std::cout << " ZGapLDet= " << ZGapLDet << std::endl;
    std::cout << " ZSiStep= " << ZSiStep << " ZSiPlane= " << ZSiPlane << std::endl;
    std::cout << " ZSiDet= " <<ZSiDet  << std::endl;
    std::cout << " UseHalfPitchShiftInX= " << UseHalfPitchShiftInX << " UseHalfPitchShiftInY= " << UseHalfPitchShiftInY << std::endl;
    std::cout << "TrackProducerFP420:----------------------" << std::endl;
    std::cout << " dXX= " << dXX << " dYY= " << dYY << std::endl;
    std::cout << " chiCutX= " << chiCutX << " chiCutY= " << chiCutY << std::endl;
  }

  theFP420NumberingScheme = new FP420NumberingScheme();



}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<TrackFP420> TrackProducerFP420::trackFinderSophisticated(edm::Handle<ClusterCollectionFP420> input, int det){
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<TrackFP420> rhits;
  int restracks = 10;// max # tracks
  rhits.reserve(restracks); 
  rhits.clear();
  double Ax[10]; double Bx[10]; double Cx[10]; int Mx[10];
  double Ay[10]; double By[10]; double Cy[10]; int My[10];
  double AxW[10]; double BxW[10]; double CxW[10]; int MxW[10];
  double AyW[10]; double ByW[10]; double CyW[10]; int MyW[10];
  if (verbos > 0) {
    std::cout << "===============================================================================" << std::endl; 
    std::cout << "=================================================================" << std::endl; 
    std::cout << "==========================================================" << std::endl; 
    std::cout << "=                                                 =" << std::endl; 
    std::cout << "TrackProducerFP420: Start trackFinderSophisticated " << std::endl; 
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// xytype is the sensor grid arrangment
  if( xytype < 1 || xytype > 2 ){
    std::cout << "TrackProducerFP420:ERROR in trackFinderSophisticated: check xytype = " << xytype << std::endl; 
    return rhits;
  }
// sn0= 3 - 2St configuration, sn0= 4 - 3St configuration 
//  if( sn0 < 3 || sn0 > 4 ){
  if( sn0 != 3 ){
    std::cout << "TrackProducerFP420:ERROR in trackFinderSophisticated: check sn0 (configuration) = " << sn0 << std::endl; 
    return rhits;
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int zbeg = 1, zmax=3;// means layer 1 and 2 in superlayer, i.e. for loop: 1,2
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .
  int reshits1 = 12;// is max # cl in sensor of copyinlayer=1
  int reshits2 = 24;// (reshits2-reshits1) is max # cl in sensors of copyinlayers= 2 or 3
  //  int resplanes = 20;
  int nX[20], nY[20];// resplanes =20 NUMBER OF PLANES; nX, nY - # cl for every X and Y plane
  int uX[20], uY[20];// resplanes =20 NUMBER OF PLANES; nX, nY - current # cl used for every X and Y plane
  double zX[24][20], xX[24][20], wX[24][20];
  double zY[24][20], yY[24][20], wY[24][20];
  double             yXW[24][20], wXW[24][20];
  double             xYW[24][20], wYW[24][20];
  bool qX[24][20], qY[24][20];
  //   .
  int txf = 0; int txs1 = 0; int txss = 0;
  int tyf = 0; int tys1 = 0; int tyss = 0;
  //   .
  double pitch=0.;
  double pitchW=0.;
  if(xytype==1){
    pitch=pitchY;
    pitchW=pitchYW;
  }
  else if(xytype==2){
    pitch=pitchX;
    pitchW=pitchXW;
  }


     //current change of geometry:
    float Xshift = pitch/2.;
    float Yshift = pitchW/2.;
    
    //
    int nmetcurx=0;
    int nmetcury=0;
    unsigned int ii0 = 999999;
    int allplacesforsensors=7;
    for (int sector=1; sector < sn0; sector++) {
      for (int zmodule=1; zmodule<pn0; zmodule++) {
	for (int zsideinorder=1; zsideinorder<allplacesforsensors; zsideinorder++) {
	  int zside = theFP420NumberingScheme->FP420NumberingScheme::realzside(rn0, zsideinorder);// 1, 3, 5, 2, 4, 6
	  if (verbos  == -49) {
	    std::cout << "TrackProducerFP420:  sector= " << sector << " zmodule= " << zmodule << " zsideinorder= " << zsideinorder << " zside= " << zside << " det= " << det << std::endl; 
	  }
	  if(zside != 0) {
	    int justlayer = theFP420NumberingScheme->FP420NumberingScheme::unpackLayerIndex(rn0, zside);// 1, 2
	    if(justlayer<1||justlayer>2) {
	      std::cout << "TrackProducerFP420:WRONG  justlayer= " << justlayer << std::endl; 
	    }
	    int copyinlayer = theFP420NumberingScheme->FP420NumberingScheme::unpackCopyIndex(rn0, zside);// 1, 2, 3
	    if(copyinlayer<1||copyinlayer>3) {
	      std::cout << "TrackProducerFP420:WRONG  copyinlayer= " << copyinlayer << std::endl; 
	    }
	    int orientation = theFP420NumberingScheme->FP420NumberingScheme::unpackOrientation(rn0, zside);// Front: = 1; Back: = 2
	    if(orientation<1||orientation>2) {
	      std::cout << "TrackProducerFP420:WRONG  orientation= " << orientation << std::endl; 
	    }
	    // ii is a continues numbering of planes(!)  over two arm FP420 set up
	    //                                                                    and  ...[ii] massives have prepared in use of ii
	    int detfixed=1;// use this treatment for each set up arm, hence no sense to repete the same for +FP420 and -FP420;
	    int nlayers=3;// 2=3-1
	    unsigned int ii = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(nlayers,pn0,sn0,detfixed,justlayer,sector,zmodule)-1;// substruct 1 from 1(+1), 2(+2), 3(+3),4(+4),5...,6...,7...,8...,9...,10... (1st Station)              ,11...,12...,13,...20... (2nd Station)
	    // ii = 0-19   --> 20 items
	    ///////// unsigned int ii=theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0,pn0,sn0,detfixed,justlayer,sector,zmodule)-1;// OLD

	    if (verbos == -49) {
	      std::cout << "TrackProducerFP420:  justlayer= " << justlayer << " copyinlayer= " << copyinlayer << " ii= " << ii << std::endl; 
	    }
	    
	    double zdiststat = 0.;
	    if(sn0<4) {
	      if(sector==2) zdiststat = zD3;
	    }
	    else {
	      if(sector==2) zdiststat = zD2;
	      if(sector==3) zdiststat = zD3;
	    }
	    double kplane = -(pn0-1)/2 - 0.5  +  (zmodule-1); //-3.5 +0...5 = -3.5,-2.5,-1.5,+2.5,+1.5
	    double zcurrent = zinibeg + z420 + (ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
	    //double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + (sector-1)*zUnit;  
	    
	    if(justlayer==1){
	     if(orientation==1) zcurrent += (ZGapLDet+ZSiDet/2);
	     if(orientation==2) zcurrent += zBlade-(ZGapLDet+ZSiDet/2);
	    }
	    if(justlayer==2){
	     if(orientation==1) zcurrent += (ZGapLDet+ZSiDet/2)+zBlade+gapBlade;
	     if(orientation==2) zcurrent += 2*zBlade+gapBlade-(ZGapLDet+ZSiDet/2);
	    }
	    //   .
	    //
	    if(det == 2) zcurrent = -zcurrent;
	    //
	    //
	    //   .
	    // local - global systems with possible shift of every second plate:
	    
	    // for xytype=1
	    float dYYcur = dYY;// XSiDet/2.
	    float dYYWcur = dXX;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 4.7
	    // for xytype=2
	    float dXXcur = dXX;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 4.7
	    float dXXWcur = dYY;// XSiDet/2.
	    //   .
	    if(justlayer==2) {
	      // X-type: x-coord
	      if (UseHalfPitchShiftInX == true){
		dXXcur += Xshift;
	      }
	      // X-type: y-coord
	      if (UseHalfPitchShiftInXW == true){
		dXXWcur -= Yshift;
	      }
	    }
	    //
	    double XXXDelta = 0.0;
	    if(copyinlayer==2) { XXXDelta = XsensorSize;}
	    if(copyinlayer==3) { XXXDelta = 2.*XsensorSize;}
	    double YYYDelta = 0.0;
	    if(copyinlayer==2) { YYYDelta = XsensorSize;}
	    if(copyinlayer==3) { YYYDelta = 2.*XsensorSize;}
	    //   .
	    //   GET CLUSTER collection  !!!!
	    //   .
	    unsigned int iu=theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(rn0,pn0,sn0,det,zside,sector,zmodule);
	    if (verbos > 0 ) {
	      std::cout << "TrackProducerFP420: check        iu = " << iu << std::endl; 
	      std::cout << "TrackProducerFP420:  sector= " << sector << " zmodule= " << zmodule << " zside= " << zside << " det= " << det << " rn0= " << rn0 << " pn0= " << pn0 << " sn0= " << sn0 << " copyinlayer= " << copyinlayer << std::endl; 
	    }
	    //============================================================================================================ put into currentclust
	    std::vector<ClusterFP420> currentclust;
	    currentclust.clear();
	    ClusterCollectionFP420::Range outputRange;
	    outputRange = input->get(iu);
	    // fill output in currentclust vector (for may be sorting? or other checks)
	    ClusterCollectionFP420::ContainerIterator sort_begin = outputRange.first;
	    ClusterCollectionFP420::ContainerIterator sort_end = outputRange.second;
	    for ( ;sort_begin != sort_end; ++sort_begin ) {
	      //  std::sort(currentclust.begin(),currentclust.end());
	      currentclust.push_back(*sort_begin);
	    } // for
	    
	    if (verbos > 0 ) {
	      std::cout << "TrackProducerFP420: currentclust.size = " << currentclust.size() << std::endl; 
	    }
	    //============================================================================================================
	    
	    std::vector<ClusterFP420>::const_iterator simHitIter = currentclust.begin();
	    std::vector<ClusterFP420>::const_iterator simHitIterEnd = currentclust.end();
	    
	    if(xytype ==1){
	      if(ii != ii0) {
		ii0=ii;
		nY[ii] = 0;// # cl in every Y plane (max is reshits)
		uY[ii] = 0;// current used # cl in every X plane 
		nmetcury=0;
	      }
	    }
	    else if(xytype ==2){
	      if(ii != ii0) {
		ii0=ii;
		nX[ii] = 0;// # cl in every X plane (max is reshits)
		uX[ii] = 0;// current used # cl in every X plane 
		nmetcurx=0;
	      }
	    }
	    // loop in #clusters of current sensor
	    for (;simHitIter != simHitIterEnd; ++simHitIter) {
	      const ClusterFP420 icluster = *simHitIter;
	      
	      // fill vectors for track reconstruction
	      
	      //disentangle complicated pattern recognition of hits?
	      // Y:
	      if(xytype ==1){
		nY[ii]++;		
		if(copyinlayer==1 && nY[ii]>reshits1){
		  nY[ii]=reshits1;
		  std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << " zcurrent=" << zY[nY[ii]-1][ii] << " copyinlayer= "  << copyinlayer << " ii= "  << ii << std::endl;
		}
		if(copyinlayer !=1 && nY[ii]>reshits2){
		  nY[ii]=reshits2;
		  std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << " zcurrent=" << zY[nY[ii]-1][ii] << " copyinlayer= "  << copyinlayer << " ii= "  << ii << std::endl;
		}
		zY[nY[ii]-1][ii] = zcurrent;
		yY[nY[ii]-1][ii] = icluster.barycenter()*pitch+0.5*pitch+YYYDelta;
		xYW[nY[ii]-1][ii] = icluster.barycenterW()*pitchW+0.5*pitchW;
		// go to global system:
		yY[nY[ii]-1][ii] = yY[nY[ii]-1][ii] - dYYcur; 
		wY[nY[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
		wY[nY[ii]-1][ii] *= wY[nY[ii]-1][ii];//reciprocal of the variance for each datapoint in y
		if(det == 2) {
		  xYW[nY[ii]-1][ii] =(xYW[nY[ii]-1][ii]+dYYWcur); 
		}
		else {
		  xYW[nY[ii]-1][ii] =-(xYW[nY[ii]-1][ii]+dYYWcur); 
		}
		wYW[nY[ii]-1][ii] = 1./(icluster.barycerrorW()*pitchW);//reciprocal of the variance for each datapoint in y
		wYW[nY[ii]-1][ii] *= wYW[nY[ii]-1][ii];//reciprocal of the variance for each datapoint in y
		qY[nY[ii]-1][ii] = true;
		if(copyinlayer==1 && nY[ii]==reshits1) break;
		if(copyinlayer !=1 && nY[ii]==reshits2) break;
	      }
	      // X:
	      else if(xytype ==2){
		nX[ii]++;	
		if (verbos == -49) {
		  std::cout << "TrackproducerFP420: nX[ii]= " << nX[ii] << " Ncl= " << currentclust.size() << " copyinlayer= "  << copyinlayer << " ii= " << ii << " zcurrent = " << zcurrent << " xX= " << icluster.barycenter()*pitch+0.5*pitch+XXXDelta << " yXW= " << icluster.barycenterW()*pitchW+0.5*pitchW << " det= " << det << " cl.size= " << icluster.amplitudes().size() << " cl.ampl[0]= " << icluster.amplitudes()[0] << std::endl;
		}
		if(copyinlayer==1 && nX[ii]>reshits1){
		  std::cout << "WARNING-ERROR:TrackproducerFP420: nX[ii]= " << nX[ii] <<" bigger reservated number of hits" << " currentclust.size()= " << currentclust.size() << " copyinlayer= "  << copyinlayer << " ii= " << ii << std::endl;
		  nX[ii]=reshits1;
		}
		if(copyinlayer !=1 && nX[ii]>reshits2){
		  std::cout << "WARNING-ERROR:TrackproducerFP420: nX[ii]= " << nX[ii] <<" bigger reservated number of hits" << " currentclust.size()= " << currentclust.size() << " copyinlayer= "  << copyinlayer << " ii= " << ii << std::endl;
		  nX[ii]=reshits2;
		}
		zX[nX[ii]-1][ii] = zcurrent;
		xX[nX[ii]-1][ii] = icluster.barycenter()*pitch+0.5*pitch+XXXDelta;
		yXW[nX[ii]-1][ii] = icluster.barycenterW()*pitchW+0.5*pitchW;
		// go to global system:
		xX[nX[ii]-1][ii] =-(xX[nX[ii]-1][ii]+dXXcur); 
		wX[nX[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
		wX[nX[ii]-1][ii] *= wX[nX[ii]-1][ii];//reciprocal of the variance for each datapoint in y
		if(det == 2) {
		  yXW[nX[ii]-1][ii] = -(yXW[nX[ii]-1][ii] - dXXWcur); 
		}
		else {
		  yXW[nX[ii]-1][ii] = yXW[nX[ii]-1][ii] - dXXWcur; 
		}
		wXW[nX[ii]-1][ii] = 1./(icluster.barycerrorW()*pitchW);//reciprocal of the variance for each datapoint in y
		wXW[nX[ii]-1][ii] *= wXW[nX[ii]-1][ii];//reciprocal of the variance for each datapoint in y
		qX[nX[ii]-1][ii] = true;
		if (verbos == -29) {
		  std::cout << "trackFinderSophisticated: nX[ii]= " << nX[ii]<< " ii = " << ii << " zcurrent = " << zcurrent << " yXW[nX[ii]-1][ii] = " << yXW[nX[ii]-1][ii] << " xX[nX[ii]-1][ii] = " << xX[nX[ii]-1][ii] << std::endl;
		  std::cout << "  XXXDelta= " << XXXDelta << "  dXXcur= " << dXXcur << "  -dXXWcur= " << -dXXWcur << std::endl;
		  std::cout << "  icluster.barycerrorW()*pitchW= " << icluster.barycerrorW()*pitchW << "  wXW[nX[ii]-1][ii]= " <<wXW[nX[ii]-1][ii]  << std::endl;
		  std::cout << " -icluster.barycenterW()*pitchW+0.5*pitchW = " << icluster.barycenterW()*pitchW+0.5*pitchW << std::endl;
		  std::cout << "============================================================" << std::endl;
		}
		if (verbos  > 0) {
		  std::cout << "trackFinderSophisticated: nX[ii]= " << nX[ii]<< " ii = " << ii << " zcurrent = " << zcurrent << " xX[nX[ii]-1][ii] = " << xX[nX[ii]-1][ii] << std::endl;
		  std::cout << " wX[nX[ii]-1][ii] = " << wX[nX[ii]-1][ii] << " wXW[nX[ii]-1][ii] = " << wXW[nX[ii]-1][ii] << std::endl;
		  std::cout << " -icluster.barycenter()*pitch-0.5*pitch = " << -icluster.barycenter()*pitch-0.5*pitch << " -dXXcur = " << -dXXcur << " -XXXDelta = " << -XXXDelta << std::endl;
		  std::cout << "============================================================" << std::endl;
		}

		if(copyinlayer==1 && nX[ii]==reshits1) break;
		if(copyinlayer !=1 && nX[ii]==reshits2) break;
	      }// if(xytype
	      
	    } // for loop in #clusters (can be breaked)
	    
	    // Y:
	    if(xytype ==1){
	      if(nY[ii] > nmetcury) {  /* # Y-planes w/ clusters */
		nmetcury=nY[ii];
		++tyf; if(sector==1) ++tys1; if(sector==(sn0-1)) ++tyss;
	      }	  
	    }
	    // X:
	    else if(xytype ==2){
	      if(nX[ii] > nmetcurx) {  /* # X-planes w/ clusters */
		nmetcurx=nX[ii];
		++txf; if(sector==1) ++txs1; if(sector==(sn0-1)) ++txss;
	      }	  
	    }
	    //================================== end of for loops in continuius number iu:
	  }//if(zside!=0
	}   // for zsideinorder
      }   // for zmodule
    }   // for sector
    if (verbos > 0) {
      std::cout << "trackFinderSophisticated: tyf= " << tyf<< " tys1 = " << tys1 << " tyss = " << tyss << std::endl;
      std::cout << "trackFinderSophisticated: txf= " << txf<< " txs1 = " << txs1 << " txss = " << txss << std::endl;
      std::cout << "============================================================" << std::endl;
    }
    
    //===========================================================================================================================
    //===========================================================================================================================
    //===========================================================================================================================
    //======================    start road finder   =============================================================================
    //===========================================================================================================================

  //  int nitMax=5;// max # iterations to find track
  int nitMax=10;// max # iterations to find track using go over of different XZ and YZ fits to find the good chi2X and chi2Y simultaneously(!!!) 

  // criteria for track selection: 
  // track is selected if for 1st station #cl >=pys1Cut
  //  int  pys1Cut = 5, pyssCut = 5, pyallCut=12;
  //  int  pys1Cut = 1, pyssCut = 1, pyallCut= 3;

//  int  pys1Cut = 3, pyssCut = 3, pyallCut= 6; // before geom. update
//  int  pys1Cut = 2, pyssCut = 2, pyallCut= 4; // bad for 5 layers per station
  int  pys1Cut = 3, pyssCut = 1, pyallCut= 5;
  
  //  double yyyvtx = 0.0, xxxvtx = -15;  //mm
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // for configuration: 3St, 1m for 1-2 St:
  // double sigman=0.1, ssigma = 1.0, sigmam=0.15;/* ssigma is foreseen to match 1st point of 2nd Station*/
  //
  // for equidistant 3 Stations:
  //
  // for tests:
  //  double sigman=118., ssigma = 299., sigmam=118.;
  // RMS1=0.013, RMS2 = 1.0, RMS3 = 0.018 see plots d1XCL, d2XCL, d3XCL
  //
  //  double sigman=0.05, ssigma = 2.5, sigmam=0.06;
  //  double sigman=0.18, ssigma = 1.8, sigmam=0.18;
  //  double sigman=0.18, ssigma = 2.9, sigmam=0.18;
  //
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // for 3 Stations:
  // LAST:
  double sigman=0.18, ssigma = 2.5, sigmam=0.18;
  if( sn0 < 4 ){
    // for 2 Stations:
    // sigman=0.24, ssigma = 4.2, sigmam=0.33;
    //  sigman=0.18, ssigma = 3.9, sigmam=0.18;
    // sigman=0.18, ssigma = 3.6, sigmam=0.18;
    //

    //

    //  sigman=0.18, ssigma = 3.3, sigmam=0.18;// before geometry update for 4 sensors per superlayer
    //    sigman=0.30, ssigma = 7.1, sigmam=0.40;// for update
     sigman=0.30, ssigma = 8.0, sigmam=1.0;// for matching update to find point nearby to fit track in 1st plane of 2nd Station 
    //
  }
  if (verbos > 0) {
    std::cout << "trackFinderSophisticated: ssigma= " << ssigma << std::endl;
  }
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  /* ssigma = 3. * 8000.*(0.025+0.009)/sqrt(pn0-1)/100. = 2.9 mm(!!!)----
     ssigma is reduced by factor k_reduced = (sn0-1)-sector+1 = sn0-sector
     # Stations  currentStation
     2Stations:     sector=2,         sn0=3 , sn0-sector = 1 --> k_reduced = 1
     3Stations:     sector=2,         sn0=4 , sn0-sector = 2 --> k_reduced = 2
     3Stations:     sector=3,         sn0=4 , sn0-sector = 1 --> k_reduced = 1
  */
  int numberXtracks=0, numberYtracks=0, totpl = 2*(pn0-1)*(sn0-1); double sigma;

  for (int xytypecurrent=xytype; xytypecurrent<xytype+1; ++xytypecurrent) {
    if (verbos > 0) {
      std::cout << "trackFinderSophisticated: xytypecurrent= " << xytypecurrent << std::endl;
    }
    
    //
    //
    double tg0 = 0.;
    int qAcl[20], qAii[20], fip=0, niteration = 0;
    int ry = 0, rys1 = 0, ryss = 0;
    int tas1=tys1, tass=tyss, taf=tyf;
    bool SelectTracks = true;
    //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .

  double yA[24][20], zA[24][20], wA[24][20]; int nA[20], uA[20]; bool qA[24][20];
    //
    // Y:
  //======================    start road finder  for xytypecurrent = 1      ===========================================================
    if(xytypecurrent ==1){
  //===========================================================================================================================
      numberYtracks=0;  
      tg0= 3*1./(800.+20.); // for Y: 1cm/...   *3 - 3sigma range
      tas1=tys1;
      tass=tyss;
      taf=tyf;
      for (int ii=0; ii < totpl; ++ii) {
	if (verbos > 0) {
	  std::cout << "trackFinderSophisticated: ii= " << ii << " nY[ii]= " << nY[ii] << std::endl;
	  std::cout << "trackFinderSophisticated: ii= " << ii << " uY[ii]= " << uY[ii] << std::endl;
	}
	nA[ii] = nY[ii];
	uA[ii] = uY[ii];
	for (int cl=0; cl<nA[ii]; ++cl) {
	  if (verbos > 0) {
	    std::cout << " cl= " << cl << " yY[cl][ii]= " << yY[cl][ii] << std::endl;
	    std::cout << " zY[cl][ii]= " << zY[cl][ii] << " wY[cl][ii]= " << wY[cl][ii] << " qY[cl][ii]= " << qY[cl][ii] << std::endl;
	  }
	  yA[cl][ii] = yY[cl][ii];
	  zA[cl][ii] = zY[cl][ii];
	  wA[cl][ii] = wY[cl][ii];
	  qA[cl][ii] = qY[cl][ii];
	}
      }
  //===========================================================================================================================
    }// if xytypecurrent ==1
    // X:
  //======================    start road finder  for superlayer = 2      ===========================================================
    else if(xytypecurrent ==2){
  //===========================================================================================================================
      numberXtracks=0;  
      tg0= 3*2./(800.+20.); // for X: 2cm/...   *3 - 3sigma range
      tas1=txs1;
      tass=txss;
      taf=txf;
      for (int ii=0; ii < totpl; ++ii) {
	if (verbos > 0) {
	  std::cout << "trackFinderSophisticated: ii= " << ii << " nX[ii]= " << nX[ii] << std::endl;
	  std::cout << "trackFinderSophisticated: ii= " << ii << " uX[ii]= " << uX[ii] << std::endl;
	}
	nA[ii] = nX[ii];
	uA[ii] = uX[ii];
	for (int cl=0; cl<nA[ii]; ++cl) {
	  if (verbos == -29) {
	    std::cout << " cl= " << cl << " xX[cl][ii]= " << xX[cl][ii] << std::endl;
	    std::cout << " zX[cl][ii]= " << zX[cl][ii] << " wX[cl][ii]= " << wX[cl][ii] << " qX[cl][ii]= " << qX[cl][ii] << std::endl;
	  }
	  yA[cl][ii] = xX[cl][ii];
	  zA[cl][ii] = zX[cl][ii];
	  wA[cl][ii] = wX[cl][ii];
	  qA[cl][ii] = qX[cl][ii];
	}
      }
  //===========================================================================================================================
    }// if xytypecurrent ==xytype


    
  //======================    start road finder        ====================================================
    if (verbos > 0) {
      std::cout << "                  start road finder                        " << std::endl;
    }
    do {
      double fyY[20], fzY[20], fwY[20];
      double fyYW[20],         fwYW[20];
      int py = 0, pys1 = 0, pyss = 0;
      bool NewStation = false, py1first = false;
      for (int sector=1; sector < sn0; ++sector) {
	double tav=0., t1=0., t2=0., t=0., sm;
	int stattimes=0;
	if( sector != 1 ) {
	  NewStation = true;  
	}
	for (int zmodule=1; zmodule<pn0; ++zmodule) {
	  for (int justlayer=zbeg; justlayer<zmax; justlayer++) {
	    // iu is a continues numbering of planes(!) 
	    int detfixed=1;// use this treatment for each set up arm, hence no sense to do it differently for +FP420 and -FP420;
	    int nlayers=3;// 2=3-1
	    unsigned int ii = theFP420NumberingScheme->FP420NumberingScheme::packMYIndex(nlayers,pn0,sn0,detfixed,justlayer,sector,zmodule)-1;// substruct 1 from 1(+1), 2(+2), 3(+3),4(+4),5...,6...,7...,8...,9...,10... (1st Station)              ,11...,12...,13,...20... (2nd Station)
	    // ii = 0-19   --> 20 items
	    
	    if(nA[ii]!=0  && uA[ii]!= nA[ii]) { 
	      
	      ++py; if(sector==1) ++pys1; if(sector==(sn0-1)) ++pyss;
	      if(py==2 && sector==1) { 
		// find closest cluster in X                   .
		double dymin=9999999., df2; int cl2=-1;
		for (int cl=0; cl<nA[ii]; ++cl) {
		  if(qA[cl][ii]){
		    df2 = std::abs(fyY[fip]-yA[cl][ii]);
		    if(df2 < dymin) {
		      dymin = df2;
		      cl2=cl;
		    }//if(df2		
		  }//if(qA		
		}//for(cl
		// END of finding of closest cluster in X                   .
		if(cl2!=-1){
		  t=(yA[cl2][ii]-fyY[fip])/(zA[cl2][ii]-fzY[fip]);
		  t1 = t*wA[cl2][ii];
		  t2 = wA[cl2][ii];
		  if (verbos > 0) {
		    std::cout << " t= " << t << " tg0= " << tg0 << std::endl;
		  }
		  if(std::abs(t)<tg0) { 
		    qA[cl2][ii] = false;//point is taken, mark it for not using again
		    fyY[py-1]=yA[cl2][ii];
		    fzY[py-1]=zA[cl2][ii];
		    fwY[py-1]=wA[cl2][ii];
		    qAcl[py-1] = cl2;
		    qAii[py-1] = ii;
		    ++uA[ii];
		    if (verbos > 0) {
		      std::cout << " point is taken, mark it for not using again uA[ii]= " << uA[ii] << std::endl;
		    }
		    if(uA[ii]==nA[ii]){/* no points anymore for this plane */
		      ++ry; if(sector==1) ++rys1; if(sector==(sn0-1)) ++ryss;
		    }//if(uA
		  }//if abs
		  else{
		    py--; if(sector==1) pys1--;  if(sector==(sn0-1)) pyss--;
		    t1 -= t*wA[cl2][ii]; t2 -= wA[cl2][ii];
		  }//if(abs
		}//if(cl2!=-1
		else{
		  py--; if(sector==1) pys1--;  if(sector==(sn0-1)) pyss--;
		}//if(cl2!=-1
	      }//if(py==2
	      else {
		//                                                                                                                 .
		bool clLoopTrue = true;
		int clcurr=-1;
		for (int clind=0; clind<nA[ii]; ++clind) {
		  if(clLoopTrue) {
		    int cl=clind;
		    if(qA[cl][ii]){
		      clcurr = cl;
		      if(py<3 ){
			if(py==1) { 
			  py1first = true;
			  fip=py-1;
			  qA[cl][ii] = false;//point is taken, mark it for not using again
			  fyY[py-1]=yA[cl][ii];
			  fzY[py-1]=zA[cl][ii];
			  fwY[py-1]=wA[cl][ii];
			  qAcl[py-1] = cl;
			  qAii[py-1] = ii;
			  ++uA[ii];
			  if (verbos > 0) std::cout << " point is taken, mark it uA[ii]= " << uA[ii] << std::endl;
			}//if py=1
			if(uA[ii]==nA[ii]){/* no points anymore for this plane */
			  ++ry; if(sector==1) ++rys1; if(sector==(sn0-1)) ++ryss;
			}//if(uA
		      }//py<3
		      else {
			if(NewStation){
			  if( sn0 < 4 ) {
			    // stattimes=0 case (point of 1st plane to be matched in new Station)
			    sigma = ssigma;
			  }
			  else {
			    sigma = ssigma/(sn0-1-sector);
			  }
			  //sigma = ssigma/(sn0-sector);
			  //if(stattimes==1 || sector==3 ) sigma = msigma * sqrt(1./wA[cl][ii]);
			  if(stattimes==1 || sector==3 ) sigma = sigmam; // (1st $3rd Stations for 3St. configur. ), 1st only for 2St. conf.
			  //	if(stattimes==1 || sector==(sn0-1) ) sigma = sigmam;
			  
			  double cov00, cov01, cov11, c0Y, c1Y, chisqY;
			  gsl_fit_wlinear (fzY, 1, fwY, 1, fyY, 1, py-1, 
					   &c0Y, &c1Y, &cov00, &cov01, &cov11, 
					   &chisqY);
			  
			  // find closest cluster in X                   .
			  int cl2match=-1;
			  double dymin=9999999., df2; 
			  for (int clmatch=0; clmatch<nA[ii]; ++clmatch) {
			    if(qA[clmatch][ii]){
			      double smmatch = c0Y+ c1Y*zA[clmatch][ii];
			      df2 = std::abs(smmatch-yA[clmatch][ii]);
			      if(df2 < dymin) {
				dymin = df2;
				cl2match=clmatch;
			      }//if(df2		
			    }//if(qA		
			  }//for(clmatch
			  
			  if(cl2match != -1) {
			    cl=cl2match;
			    clLoopTrue = false; // just not continue the clinid loop
			  }
			  
			  sm = c0Y+ c1Y*zA[cl][ii];
			  
			  if (verbos > 0) {
			    std::cout << " sector= " << sector << " sn0= " << sn0 << " sigma= " << sigma << std::endl;
			    std::cout << " stattimes= " << stattimes << " ssigma= " << ssigma << " sigmam= " << sigmam << std::endl;
			    std::cout << " sm= " << sm << " c0Y= " << c0Y << " c1Y= " << c1Y << " chisqY= " << chisqY << std::endl;
			    std::cout << " zA[cl][ii]= " << zA[cl][ii] << " ii= " << ii << " cl= " << cl << std::endl;
			    for (int ct=0; ct<py-1; ++ct) {
			      std::cout << " py-1= " << py-1 << " fzY[ct]= " << fzY[ct] << std::endl;
			      std::cout << " fyY[ct]= " << fyY[ct] << " fwY[ct]= " << fwY[ct] << std::endl;
			    }
			  }
			  
			}//NewStation 1
			else{
			  t=(yA[cl][ii]-fyY[fip])/(zA[cl][ii]-fzY[fip]);
			  t1 += t*wA[cl][ii];
			  t2 += wA[cl][ii];
			  tav=t1/t2;
			  sm = fyY[fip]+tav*(zA[cl][ii]-fzY[fip]);
			  //sigma = nsigma * sqrt(1./wA[cl][ii]);
			  sigma = sigman;
			}
			
			double diffpo = yA[cl][ii]-sm;
			if (verbos > 0) {
			  std::cout << " diffpo= " << diffpo << " yA[cl][ii]= " << yA[cl][ii] << " sm= " << sm << " sigma= " << sigma << std::endl;
			}
			
			if(std::abs(diffpo) < sigma ) {
			  if(NewStation){
			    ++stattimes;
			    if(stattimes==1) {
			      fip=py-1;
			      t1 = 0; t2 = 0;
			    }
			    else if(stattimes==2){
			      NewStation = false; 
			      t=(yA[cl][ii]-fyY[fip])/(zA[cl][ii]-fzY[fip]);
			      //t1 += t*wA[cl][ii];
			      //t2 += wA[cl][ii];
			      t1 = t*wA[cl][ii];
			      t2 = wA[cl][ii];
			    }//if(stattime
			  }//if(NewStation 2
			  fyY[py-1]=yA[cl][ii];
			  fzY[py-1]=zA[cl][ii];
			  fwY[py-1]=wA[cl][ii];
			  qA[cl][ii] = false;//point is taken, mark it for not using again
			  qAcl[py-1] = cl;
			  qAii[py-1] = ii;
			  ++uA[ii];
			  if (verbos > 0) {
			    std::cout << " 3333 point is taken, mark it uA[ii]= " << uA[ii] << std::endl;
			  }
			  if(uA[ii]==nA[ii]){/* no points anymore for this plane */
			    ++ry; if(sector==1) ++rys1; if(sector==(sn0-1)) ++ryss;
			  }//if(cl==
			  //  break; // to go on next plane
			}//if abs
			else{
			  t1 -= t*wA[cl][ii]; t2 -= wA[cl][ii];
			}//if abs
		      }// if py<3 and else py>3
		      
		      if(!qA[cl][ii]) break;// go on next plane if point is found among clusters of current plane;
		    }// if qA
		  } // if clLoopTrue
		}// for cl     --  can be break and return to "for zmodule"
		//                                                                                                                 .
		if( (py!=1 && clcurr != -1 && qA[clcurr][ii]) || (py==1 && !py1first)) { 
		  // if point is not found - continue natural loop, but reduce py 
		  py--; if(sector==1) pys1--;  if(sector==(sn0-1)) pyss--;
		}//if(py!=1
	      }//if(py==2 else 
	    }//if(nA !=0	   : inside  this if( -  ask  ++py
	  }// for justlayer
	}// for zmodule
      }// for sector
      //============
      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      
      if (verbos > 0) {
	std::cout << "END: pys1= " << pys1 << " pyss = " << pyss << " py = " << py << std::endl;
      }
      // apply criteria for track selection: 
      // do not take track if 
      if( pys1 < pys1Cut || pyss < pyssCut || py < pyallCut ){
	//	if( pys1<3 || pyss<2 || py<4 ){
      }
      // do fit:
      else{
	////////////////////////////    main fit for Narrow pixels !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	double cov00, cov01, cov11;
	double c0Y, c1Y, chisqY;
	gsl_fit_wlinear (fzY, 1, fwY, 1, fyY, 1, py, 
			 &c0Y, &c1Y, &cov00, &cov01, &cov11, 
			 &chisqY);
	////////////////////////////
	// collect cases where the nearby points with the same coordinate exists
//	int pyrepete=py+2;
//	if(py < 11 && chisqY/(py-2) < 0.5) {
//	  double fyYold=999999.;
//	  for (int ipy=0; ipy<py; ++ipy) {
//	    if( fyY[ipy]!=fyYold) --pyrepete;
//	    fyYold=fyY[ipy];
//	  }
//	}
	////////////////////////////
	float chindfx;
	if(py>2) {
	  chindfx = chisqY/(py-2);
	}
	else{
	  //	  chindfy = chisqY;
	  chindfx = 9999;
	}//py
	if (verbos  > 0) {
	  //	  std::cout << " Do FIT XZ: chindfx= " << chindfx << " chisqY= " << chisqY << " py= " << py << " pyrepete= " << pyrepete << std::endl;
	  std::cout << " Do FIT XZ: chindfx= " << chindfx << " chisqY= " << chisqY << " py= " << py << std::endl;
	}
	
	////////////////////////////    second order fit for Wide pixels
	if (verbos > 0) {
	  std::cout << " preparation for second order fit for Wide pixels= " << std::endl;
	}
	for (int ipy=0; ipy<py; ++ipy) {
	  if(xytypecurrent ==1){
	    fyYW[ipy]=xYW[qAcl[ipy]][qAii[ipy]];
	    fwYW[ipy]=wYW[qAcl[ipy]][qAii[ipy]];
	    if (verbos > 0) {
	      std::cout << " ipy= " << ipy << std::endl;
	      std::cout << " qAcl[ipy]= " << qAcl[ipy] << " qAii[ipy]= " << qAii[ipy] << std::endl;
	      std::cout << " fyYW[ipy]= " << fyYW[ipy] << " fwYW[ipy]= " << fwYW[ipy] << std::endl;
	    }
	  }
	  else if(xytypecurrent ==2){
	    fyYW[ipy]=yXW[qAcl[ipy]][qAii[ipy]];
	    fwYW[ipy]=wXW[qAcl[ipy]][qAii[ipy]];
	    if (verbos ==-29) {
	      std::cout << " ipy= " << ipy << std::endl;
	      std::cout << " qAcl[ipy]= " << qAcl[ipy] << " qAii[ipy]= " << qAii[ipy] << std::endl;
	      std::cout << " fyYW[ipy]= " << fyYW[ipy] << " fwYW[ipy]= " << fwYW[ipy] << std::endl;
	    }
	  }
	}// for



	if (verbos > 0) {
	  std::cout << " start second order fit for Wide pixels= " << std::endl;
	}
	double wov00, wov01, wov11;
	double w0Y, w1Y, whisqY;
	gsl_fit_wlinear (fzY, 1, fwYW, 1, fyYW, 1, py, 
			 &w0Y, &w1Y, &wov00, &wov01, &wov11, 
			 &whisqY);
	////////////////////////////////////////////////////////////////////////////////////



	float chindfy;
	if(py>2) {
	  chindfy = whisqY/(py-2);
	}
	else{
	  //	  chindfy = chisqY;
	  chindfy = 9999;
	}//py
	
	if (verbos > 0) {
	  std::cout << " chindfy= " << chindfy << " chiCutY= " << chiCutY << std::endl;
	}


	if(xytypecurrent ==1){
	  if(chindfx < chiCutX && chindfy < chiCutY) {
	    ++numberYtracks;
	    Ay[numberYtracks-1] = c0Y; 
	    By[numberYtracks-1] = c1Y; 
	    Cy[numberYtracks-1] = chisqY; 
	    //  My[numberYtracks-1] = py-pyrepete;
	    My[numberYtracks-1] = py;
	    AyW[numberYtracks-1] = w0Y; 
	    ByW[numberYtracks-1] = w1Y; 
	    CyW[numberYtracks-1] = whisqY; 
	    MyW[numberYtracks-1] = py;
	    if (verbos > 0) {
	      if(py>20) {
		std::cout << " niteration = " << niteration << std::endl;
		std::cout << " chindfy= " << chindfy << " py= " << py << std::endl;
		std::cout << " c0Y= " << c0Y << " c1Y= " << c1Y << std::endl;
		std::cout << " pys1= " << pys1 << " pyss = " << pyss << std::endl;
	      }
	    }
	  }//chindfy
	}
	else if(xytypecurrent ==2){
	  if(chindfx < chiCutX && chindfy < chiCutY) {
	    ++numberXtracks;
	    Ax[numberXtracks-1] = c0Y; 
	    Bx[numberXtracks-1] = c1Y; 
	    Cx[numberXtracks-1] = chisqY; 
	    //   Mx[numberXtracks-1] = py-pyrepete;
	    Mx[numberXtracks-1] = py;
	    AxW[numberXtracks-1] = w0Y; 
	    BxW[numberXtracks-1] = w1Y; 
	    CxW[numberXtracks-1] = whisqY; 
	    MxW[numberXtracks-1] = py;
	    if (verbos > 0) {
	      std::cout << " niteration = " << niteration << std::endl;
	      std::cout << " chindfx= " << chindfy << " px= " << py << std::endl;
	      std::cout << " c0X= " << c0Y << " c1X= " << c1Y << std::endl;
	      std::cout << " pxs1= " << pys1 << " pxss = " << pyss << std::endl;
	    }
	  }//chindfy
	}
	
	
      }//  if else
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // do not select tracks anymore if
      if (verbos  > 0) {
	std::cout << "Current iteration, niteration >= " << niteration << std::endl;
	std::cout << " numberYtracks= " << numberYtracks << std::endl;
	std::cout << " numberXtracks= " << numberXtracks << std::endl;
	std::cout << " pys1= " << pys1 << " pyss = " << pyss << " py = " << py << std::endl;
	std::cout << " tas1= " << tas1 << " tass = " << tass << " taf = " << taf << std::endl;
	std::cout << " rys1= " << rys1 << " ryss = " << ryss << " ry = " << ry << std::endl;
	std::cout << " tas1-rys1= " << tas1-rys1 << " tass-ryss = " << tass-ryss << " taf-ry = " << taf-ry << std::endl;
	std::cout << "---------------------------------------------------------- " << std::endl;
      }
      // let's decide: do we continue track finder procedure
      if( tas1-rys1<pys1Cut || tass-ryss<pyssCut || taf-ry<pyallCut  ){
	SelectTracks = false;
      }
      else{
	++niteration;
      }
      
    } while(SelectTracks && niteration < nitMax );      
    //======================    finish do loop finder for  xytypecurrent     ====================================================
    
    //============
    
    //===========================================================================================================================
    
    //===========================================================================================================================
  }// for xytypecurrent 
  //===========================================================================================================================
  
  if (verbos > 0) {
    std::cout << " numberXtracks= " << numberXtracks << " numberYtracks= " << numberYtracks << std::endl;
  }
  //===========================================================================================================================
  //===========================================================================================================================
  //===========================================================================================================================

  // case X and Y plane types are available
  if(xytype>2) {
  //===========================================================================================================================
  // match selected X and Y tracks to each other: tgphi=By/Bx->phi=artg(By/Bx); tgtheta=Bx/cosphi=By/sinphi->  ================
  //                min of |Bx/cosphi-By/sinphi|                                                               ================

  //  
    if (verbos > 0) {
      std::cout << " numberXtracks= " << numberXtracks << " numberYtracks= " << numberYtracks << std::endl;
    }
      if(numberXtracks>0) {
	int newxnum[10], newynum[10];// max # tracks = restracks = 10
	int nmathed=0;
	do {
	  double dthmin= 999999.; 
	  int trminx=-1, trminy=-1;
	  for (int trx=0; trx<numberXtracks; ++trx) {
	    if (verbos > 0) {
	      std::cout << "----------- trx= " << trx << " nmathed= " << nmathed << std::endl;
	    }
	    for (int tr=0; tr<numberYtracks; ++tr) {
	      if (verbos > 0) {
		std::cout << "--- tr= " << tr << " nmathed= " << nmathed << std::endl;
	      }
	      bool YesY=false;
	      for (int nmx=0; nmx<nmathed; ++nmx) {
		if(trx==newxnum[nmx]) YesY=true;
		if(YesY) break;
		for (int nm=0; nm<nmathed; ++nm) {
		  if(tr==newynum[nm]) YesY=true;
		  if(YesY) break;
		}
	      }
	      if(!YesY) {
		//--------------------------------------------------------------------	----	----	----	----	----	----
		//double yyyyyy = 999999.;
		//if(Bx[trx] != 0.) yyyyyy = Ay[tr]-(Ax[trx]-xxxvtx)*By[tr]/Bx[trx];
		//double xxxxxx = 999999.;
		//if(By[tr] != 0.) xxxxxx = Ax[trx]-(Ay[tr]-yyyvtx)*Bx[trx]/By[tr];
		//double  dthdif= std::abs(yyyyyy-yyyvtx) + std::abs(xxxxxx-xxxvtx);
		
		double  dthdif= std::abs(AxW[trx]-Ay[tr]) + std::abs(BxW[trx]-By[tr]);
		
		if (verbos > 0) {
		  //  std::cout << " yyyyyy= " << yyyyyy << " xxxxxx= " << xxxxxx << " dthdif= " << dthdif << std::endl;
		  std::cout << " abs(AxW[trx]-Ay[tr]) = " << std::abs(AxW[trx]-Ay[tr]) << " abs(BxW[trx]-By[tr])= " << std::abs(BxW[trx]-By[tr]) << " dthdif= " << dthdif << std::endl;
		}
		//--------------------------------------------------------------------	    ----	----	----	----	----	----
		if( dthdif < dthmin ) {
		  dthmin = dthdif;
		  trminx = trx;
		  trminy = tr;
		}//if  dthdif
		//--------------------------------------------------------------------	
	      }//if !YesY
	    }//for y
	  }// for x
	  ++nmathed;
	  if(trminx != -1) {
	    newxnum[nmathed-1] = trminx;
	  }
	  else{
	    newxnum[nmathed-1] = nmathed-1;
	  }
	  if (verbos > 0) {
	    std::cout << " trminx= " << trminx << std::endl;
	  }
	  if(nmathed>numberYtracks){
	    newynum[nmathed-1] = -1;
	    if (verbos > 0) {
	      std::cout << "!!!  nmathed= " << nmathed << " > numberYtracks= " << numberYtracks << std::endl;
	    }
	  }
	  else {
	    if (verbos > 0) {
	      std::cout << " trminy= " << trminy << std::endl;
	    }
	    newynum[nmathed-1] = trminy;
	  }    
	} while(nmathed<numberXtracks && nmathed < restracks);      
	
	//
//===========================================================================================================================
//
    for (int tr=0; tr<nmathed; ++tr) {
      int tx=newxnum[tr];
      int ty=newynum[tr];
      if(ty==-1){
	ty=tx;
	Ay[ty]=999.;
	By[ty]=999.;
	Cy[ty]=999.;
	My[ty]=-1;
      }//if ty
      // test:
      //  tx=tr;
      //ty=tr;
      if (verbos > 0) {
	if(Mx[tx]>20) {
	  std::cout << " for track tr= " << tr << " tx= " << tx << " ty= " << ty << std::endl;
	  std::cout << " Ax= " << Ax[tx]   << " Ay= " << Ay[ty]   << std::endl;
	  std::cout << " Bx= " << Bx[tx]   << " By= " << By[ty]   << std::endl;
	  std::cout << " Cx= " << Cx[tx]   << " Cy= " << Cy[ty]   << std::endl;
	  std::cout << " Mx= " << Mx[tx]   << " My= " << My[ty]   << std::endl;
	  std::cout << " AxW= " << AxW[tx]   << " AyW= " << AyW[ty]   << std::endl;
	  std::cout << " BxW= " << BxW[tx]   << " ByW= " << ByW[ty]   << std::endl;
	  std::cout << " CxW= " << CxW[tx]   << " CyW= " << CyW[ty]   << std::endl;
	  std::cout << " MxW= " << MxW[tx]   << " MyW= " << MyW[ty]   << std::endl;
	}
      }
      //   rhits.push_back( TrackFP420(c0X,c1X,chisqX,nhitplanesY,c0Y,c1Y,chisqY,nhitplanesY) );
      rhits.push_back( TrackFP420(Ax[tx],Bx[tx],Cx[tx],Mx[tx],Ay[ty],By[ty],Cy[ty],My[ty]) );
    }//for tr
    //============================================================================================================
      }//in  numberXtracks >0
      //============
      
  }
  // case Y plane types are available only
  else if(xytype==1) {
    for (int ty=0; ty<numberYtracks; ++ty) {
      if (verbos > 0) {
	std::cout << " for track ty= " << ty << std::endl;
	std::cout << " Ay= " << Ay[ty]   << std::endl;
	std::cout << " By= " << By[ty]   << std::endl;
	std::cout << " Cy= " << Cy[ty]   << std::endl;
	std::cout << " My= " << My[ty]   << std::endl;
	std::cout << " AyW= " << AyW[ty]   << std::endl;
	std::cout << " ByW= " << ByW[ty]   << std::endl;
	std::cout << " CyW= " << CyW[ty]   << std::endl;
	std::cout << " MyW= " << MyW[ty]   << std::endl;
      }
      rhits.push_back( TrackFP420(AyW[ty],ByW[ty],CyW[ty],MyW[ty],Ay[ty],By[ty],Cy[ty],My[ty]) );
    }//for ty
    //============
  }
  // case X plane types are available only
  else if(xytype==2) {
    for (int tx=0; tx<numberXtracks; ++tx) {
      if (verbos > 0) {
	std::cout << " for track tx= " << tx << std::endl;
	std::cout << " Ax= " << Ax[tx]   << std::endl;
	std::cout << " Bx= " << Bx[tx]   << std::endl;
	std::cout << " Cx= " << Cx[tx]   << std::endl;
	std::cout << " Mx= " << Mx[tx]   << std::endl;
	std::cout << " AxW= " << AxW[tx]   << std::endl;
	std::cout << " BxW= " << BxW[tx]   << std::endl;
	std::cout << " CxW= " << CxW[tx]   << std::endl;
	std::cout << " MxW= " << MxW[tx]   << std::endl;
      }
      rhits.push_back( TrackFP420(Ax[tx],Bx[tx],Cx[tx],Mx[tx],AxW[tx],BxW[tx],CxW[tx],MxW[tx]) );
    }//for tx
    //============
  }//xytype




///////////////////////////////////////



  return rhits;
  //============
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


