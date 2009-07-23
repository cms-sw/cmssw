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

TrackProducerFP420::TrackProducerFP420(int asn0, int apn0, int azn0, double az420, double azD2, double azD3, double apitchX, double apitchY, double apitchXW, double apitchYW, double aZGapLDet, double aZSiStep, double aZSiPlane, double aZSiDetL, double aZSiDetR, bool aUseHalfPitchShiftInX, bool aUseHalfPitchShiftInY, bool aUseHalfPitchShiftInXW, bool aUseHalfPitchShiftInYW, double adXX, double adYY, float achiCutX, float achiCutY, double azinibeg) {
  //
  // Everything that depend on the det
  //
  sn0 = asn0;
  pn0 = apn0;
  zn0 = azn0;
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
  ZSiDetL = aZSiDetL;
  ZSiDetR = aZSiDetR;
  UseHalfPitchShiftInX = aUseHalfPitchShiftInX;
  UseHalfPitchShiftInY = aUseHalfPitchShiftInY;
  UseHalfPitchShiftInXW = aUseHalfPitchShiftInXW;
  UseHalfPitchShiftInYW = aUseHalfPitchShiftInYW;
  dXX = adXX;
  dYY = adYY;
  chiCutX = achiCutX;
  chiCutY = achiCutY;
  zinibeg = azinibeg;

#ifdef debugmaxampl
  std::cout << "TrackProducerFP420: call constructor" << std::endl;
  std::cout << " sn0= " << sn0 << " pn0= " << pn0 << " zn0= " << zn0 << std::endl;
  std::cout << " zD2= " << zD2 << " zD3= " << zD3 << " zinibeg= " << zinibeg << std::endl;
  //std::cout << " zUnit= " << zUnit << std::endl;
  std::cout << " pitchX= " << pitchX << " pitchY= " << pitchY << std::endl;
  std::cout << " ZGapLDet= " << ZGapLDet << std::endl;
  std::cout << " ZSiStep= " << ZSiStep << " ZSiPlane= " << ZSiPlane << std::endl;
  std::cout << " ZSiDetL= " <<ZSiDetL  << " ZSiDetR= " << ZSiDetR << std::endl;
  std::cout << " UseHalfPitchShiftInX= " << UseHalfPitchShiftInX << " UseHalfPitchShiftInY= " << UseHalfPitchShiftInY << std::endl;
  std::cout << "TrackProducerFP420:----------------------" << std::endl;
  std::cout << " dXX= " << dXX << " dYY= " << dYY << std::endl;
  std::cout << " chiCutX= " << chiCutX << " chiCutY= " << chiCutY << std::endl;
#endif
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
#ifdef debugsophisticated
  std::cout << "TrackProducerFP420: Start trackFinderSophisticated " << std::endl; 
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// zn0 is the same as xytype
  if( zn0 < 1 || zn0 > 4 ){
    std::cout << "TrackProducerFP420:ERROR in trackFinderSophisticated: check zn0 (xytype) = " << zn0 << std::endl; 
    return rhits;
  }
// sn0= 3 - 2St configuration, sn0= 4 - 3St configuration 
  if( sn0 < 3 || zn0 > 4 ){
    std::cout << "TrackProducerFP420:ERROR in trackFinderSophisticated: check sn0 (configuration) = " << sn0 << std::endl; 
    return rhits;
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int zbeg = 1, zmax=3;// XY
  //  if( zn0==1){
  //              zmax=2; // Y
  //  }
  //  else if( zn0==2){
  //    zbeg = 2, zmax=3; // X
  // }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .
  int reshits = 15;// max # cl for every X and Y plane
  //  int resplanes = 30;
  int nX[30], nY[30];// resplanes =30 NUMBER OF PLANES; nX, nY - # cl for every X and Y plane
  int uX[30], uY[30];// resplanes =30 NUMBER OF PLANES; nX, nY - current # cl used for every X and Y plane
  double zX[15][30], xX[15][30], wX[15][30];
  double zY[15][30], yY[15][30], wY[15][30];
  double             yXW[15][30], wXW[15][30];
  double             xYW[15][30], wYW[15][30];
  bool qX[15][30], qY[15][30];
  //   .
  int txf = 0; int txs1 = 0; int txss = 0;
  int tyf = 0; int tys1 = 0; int tyss = 0;
  //   .
  double pitch=0.;
  double pitchW=0.;
  if(zn0==1){
    pitch=pitchY;
    pitchW=pitchYW;
  }
  else if(zn0==2){
    pitch=pitchX;
    pitchW=pitchXW;
  }


     //current change of geometry:
    float Xshift = pitch/2.;
    float Yshift = pitchW/2.;
    
    //
    
    for (int sector=1; sector < sn0; sector++) {
      for (int zmodule=1; zmodule<pn0; zmodule++) {
	for (int zside=zbeg; zside<zmax; zside++) {
	  
	  // index iu is a continues numbering of 3D detector of FP420 (detector ID)
	  int sScale = 2*(pn0-1), dScale = 2*(pn0-1)*(sn0-1);
	  int zScale=2;  unsigned int iu = dScale*(det - 1)+sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	  //	unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) + 1;

	  //	  unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) ; // 0-19   --> 20 items
	  unsigned int ii = iu-1-dScale*(det - 1);// 0-29   --> 30 items
	  
	  double kplane = -(pn0-1)/2 - 0.5  +  (zmodule-1); 
	  
	  
	  double zdiststat = 0.;
	  if(sn0<4) {
	    if(sector==2) zdiststat = zD3;
	  }
	  else {
	    if(sector==2) zdiststat = zD2;
	    if(sector==3) zdiststat = zD3;
	  }
	  double zcurrent = zinibeg + z420 + (ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
	  //double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + (sector-1)*zUnit;  
	  
	  if(zside==1){
	    zcurrent += (ZGapLDet+ZSiDetL/2);
	  }
	  if(zside==2){
	    zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
	  }
	  //   .
	  //
	  if(det == 2) zcurrent = -zcurrent;
	  //
	  //
	  //   .
	  // local - global systems with possible shift of every second plate:
	  
	  // for zn0=1
	  float dYYcur = dYY;// XSiDet/2.
	  float dYYWcur = dXX;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7
	  // for zn0=2
	  float dXXcur = dXX;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7
	  float dXXWcur = dYY;// XSiDet/2.
	  //   .
	  if(zside==2) {
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
	  
	  //   .
	  //   GET CLUSTER collection  !!!!
	  //   .
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
	  
#ifdef debugsophisticated
	  std::cout << "TrackProducerFP420: currentclust.size = " << currentclust.size() << std::endl; 
#endif
	  //============================================================================================================
	  
	  vector<ClusterFP420>::const_iterator simHitIter = currentclust.begin();
	  vector<ClusterFP420>::const_iterator simHitIterEnd = currentclust.end();
	  
	  if(zn0 ==1){
	    nY[ii] = 0;// # cl in every Y plane (max is reshits)
	    uY[ii] = 0;// current used # cl in every X plane 
	  }
	  else if(zn0 ==2){
	    nX[ii] = 0;// # cl in every X plane (max is reshits)
	    uX[ii] = 0;// current used # cl in every X plane 
	  }
	  // loop in #clusters
	  for (;simHitIter != simHitIterEnd; ++simHitIter) {
	    const ClusterFP420 icluster = *simHitIter;
	    
	    // fill vectors for track reconstruction
	    
	    
	    //disentangle complicated pattern recognition of hits?
	    // Y:
	    if(zn0 ==1){
	      nY[ii]++;		
	      if(nY[ii]>reshits){
		nY[ii]=reshits;
		std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << " zcurrent=" << zY[nY[ii]-1][ii] << " ii= "  << ii << std::endl;
	      }
	      zY[nY[ii]-1][ii] = zcurrent;
	      yY[nY[ii]-1][ii] = icluster.barycenter()*pitch;
	      xYW[nY[ii]-1][ii] = icluster.barycenterW()*pitchW;
	      // go to global system:
	      yY[nY[ii]-1][ii] = yY[nY[ii]-1][ii] - dYYcur; 
	      wY[nY[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
	      wY[nY[ii]-1][ii] *= wY[nY[ii]-1][ii];//reciprocal of the variance for each datapoint in y
	      xYW[nY[ii]-1][ii] =-(xYW[nY[ii]-1][ii]+dYYWcur); 
	      wYW[nY[ii]-1][ii] = 1./(icluster.barycerrorW()*pitchW);//reciprocal of the variance for each datapoint in y
	      wYW[nY[ii]-1][ii] *= wYW[nY[ii]-1][ii];//reciprocal of the variance for each datapoint in y
	      qY[nY[ii]-1][ii] = true;
	      if(nY[ii]==reshits) break;
	    }
	    // X:
	    else if(zn0 ==2){
	      nX[ii]++;	
	      if(nX[ii]>reshits){
		std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << std::endl;
		nX[ii]=reshits;
	      }
	      zX[nX[ii]-1][ii] = zcurrent;
	      xX[nX[ii]-1][ii] = icluster.barycenter()*pitch;
	      yXW[nX[ii]-1][ii] = icluster.barycenterW()*pitchW;
	      // go to global system:
	      xX[nX[ii]-1][ii] =-(xX[nX[ii]-1][ii]+dXXcur); 
	      wX[nX[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
	      wX[nX[ii]-1][ii] *= wX[nX[ii]-1][ii];//reciprocal of the variance for each datapoint in y
	      yXW[nX[ii]-1][ii] = yXW[nX[ii]-1][ii] - dXXWcur; 
	      wXW[nX[ii]-1][ii] = 1./(icluster.barycerrorW()*pitchW);//reciprocal of the variance for each datapoint in y
	      wXW[nX[ii]-1][ii] *= wXW[nX[ii]-1][ii];//reciprocal of the variance for each datapoint in y
	      qX[nX[ii]-1][ii] = true;
#ifdef debugsophisticated
	      std::cout << "trackFinderSophisticated: nX[ii]= " << nX[ii]<< " ii = " << ii << " zcurrent = " << zcurrent << " xX[nX[ii]-1][ii] = " << xX[nX[ii]-1][ii] << std::endl;
	      std::cout << " wX[nX[ii]-1][ii] = " << wX[nX[ii]-1][ii] << " wXW[nX[ii]-1][ii] = " << wXW[nX[ii]-1][ii] << std::endl;
	      std::cout << " -icluster.barycenter()*pitch = " << -icluster.barycenter()*pitch << " -dXXcur = " << -dXXcur << std::endl;
	      std::cout << "============================================================" << std::endl;
#endif
	      if(nX[ii]==reshits) break;
	    }
	    
	  } // for loop in #clusters (can be breaked)
	  
	  // Y:
	  if(zn0 ==1){
	    if(nY[ii] != 0) {  /* # Y-planes w/ clusters */
	      ++tyf; if(sector==1) ++tys1; if(sector==(sn0-1)) ++tyss;
	    }	  
	  }
	  // X:
	  else if(zn0 ==2){
	    if(nX[ii] != 0) {  /* # X-planes w/ clusters */
	      ++txf; if(sector==1) ++txs1; if(sector==(sn0-1)) ++txss;
	    }	  
	  }
	  //================================== end of for loops in continuius number iu:
	}   // for zside
      }   // for zmodule
    }   // for sector
#ifdef debugsophisticated
    std::cout << "trackFinderSophisticated: tyf= " << tyf<< " tys1 = " << tys1 << " tyss = " << tyss << std::endl;
    std::cout << "trackFinderSophisticated: txf= " << txf<< " txs1 = " << txs1 << " txss = " << txss << std::endl;
    std::cout << "============================================================" << std::endl;
#endif
    
    //===========================================================================================================================
    //===========================================================================================================================
    //===========================================================================================================================
    //======================    start road finder   =============================================================================
    //===========================================================================================================================

  //  int nitMax=5;// max # iterations to find track
  int nitMax=5;// max # iterations to find track

  // criteria for track selection: 
  // track is selected if for 1st station #cl >=pys1Cut
  //  int  pys1Cut = 5, pyssCut = 5, pyallCut=12;
  //  int  pys1Cut = 1, pyssCut = 1, pyallCut= 3;
  int  pys1Cut = 3, pyssCut = 3, pyallCut= 6;
  
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
    sigman=0.18, ssigma = 3.3, sigmam=0.18;
  }
#ifdef debugsophisticated
  std::cout << "trackFinderSophisticated: ssigma= " << ssigma << std::endl;
#endif

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  /* ssigma = 3. * 8000.*(0.025+0.009)/sqrt(pn0-1)/100. = 2.9 mm(!!!)----
     ssigma is reduced by factor k_reduced = (sn0-1)-sector+1 = sn0-sector
     # Stations  currentStation
     2Stations:     sector=2,         sn0=3 , sn0-sector = 1 --> k_reduced = 1
     3Stations:     sector=2,         sn0=4 , sn0-sector = 2 --> k_reduced = 2
     3Stations:     sector=3,         sn0=4 , sn0-sector = 1 --> k_reduced = 1
  */
  int numberXtracks=0, numberYtracks=0, totpl = 2*(pn0-1)*(sn0-1); double sigma;

  //  for (int zside=zbeg; zside<zmax; ++zside) {
  for (int zsidezn0=zn0; zsidezn0<zn0+1; ++zsidezn0) {
#ifdef debugsophisticated
  std::cout << "trackFinderSophisticated: zsidezn0= " << zsidezn0 << std::endl;
#endif

    //
    //
    double tg0 = 0.;
    int qAcl[30], qAii[30], fip=0, niteration = 0;
    int ry = 0, rys1 = 0, ryss = 0;
    int tas1=tys1, tass=tyss, taf=tyf;
    bool SelectTracks = true;
    //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .

  double yA[15][30], zA[15][30], wA[15][30]; int nA[30], uA[30]; bool qA[15][30];
    //
    // Y:
  //======================    start road finder  for zsidezn0 = 1      ===========================================================
    if(zsidezn0 ==1){
  //===========================================================================================================================
      numberYtracks=0;  
      tg0= 3*1./(800.+20.); // for Y: 1cm/...   *3 - 3sigma range
      tas1=tys1;
      tass=tyss;
      taf=tyf;
      for (int ii=0; ii < totpl; ++ii) {
#ifdef debugsophisticated
  std::cout << "trackFinderSophisticated: ii= " << ii << " nY[ii]= " << nY[ii] << std::endl;
  std::cout << "trackFinderSophisticated: ii= " << ii << " uY[ii]= " << uY[ii] << std::endl;
#endif
	nA[ii] = nY[ii];
	uA[ii] = uY[ii];
	for (int cl=0; cl<nA[ii]; ++cl) {
#ifdef debugsophisticated
  std::cout << " cl= " << cl << " yY[cl][ii]= " << yY[cl][ii] << std::endl;
  std::cout << " zY[cl][ii]= " << zY[cl][ii] << " wY[cl][ii]= " << wY[cl][ii] << " qY[cl][ii]= " << qY[cl][ii] << std::endl;
#endif
	  yA[cl][ii] = yY[cl][ii];
	  zA[cl][ii] = zY[cl][ii];
	  wA[cl][ii] = wY[cl][ii];
	  qA[cl][ii] = qY[cl][ii];
	}
      }
  //===========================================================================================================================
    }// if zsidezn0 ==1
    // X:
  //======================    start road finder  for zside = 2      ===========================================================
    else if(zsidezn0 ==2){
  //===========================================================================================================================
      numberXtracks=0;  
      tg0= 3*2./(800.+20.); // for X: 2cm/...   *3 - 3sigma range
      tas1=txs1;
      tass=txss;
      taf=txf;
      for (int ii=0; ii < totpl; ++ii) {
#ifdef debugsophisticated
  std::cout << "trackFinderSophisticated: ii= " << ii << " nX[ii]= " << nX[ii] << std::endl;
  std::cout << "trackFinderSophisticated: ii= " << ii << " uX[ii]= " << uX[ii] << std::endl;
#endif
	nA[ii] = nX[ii];
	uA[ii] = uX[ii];
	for (int cl=0; cl<nA[ii]; ++cl) {
#ifdef debugsophisticated
  std::cout << " cl= " << cl << " xX[cl][ii]= " << xX[cl][ii] << std::endl;
  std::cout << " zX[cl][ii]= " << zX[cl][ii] << " wX[cl][ii]= " << wX[cl][ii] << " qX[cl][ii]= " << qX[cl][ii] << std::endl;
#endif
	  yA[cl][ii] = xX[cl][ii];
	  zA[cl][ii] = zX[cl][ii];
	  wA[cl][ii] = wX[cl][ii];
	  qA[cl][ii] = qX[cl][ii];
	}
      }
  //===========================================================================================================================
    }// if zsidezn0 ==zn0


    
  //======================    start road finder        ====================================================
    do {
      double fyY[30], fzY[30], fwY[30];
      double fyYW[30],         fwYW[30];
      int py = 0, pys1 = 0, pyss = 0;
      bool NewStation = false, py1first = false;
      for (int sector=1; sector < sn0; ++sector) {
	double tav=0., t1=0., t2=0., t=0., sm;
	int stattimes=0;
	if( sector != 1 ) {
	  NewStation = true;  
	}
	for (int zmodule=1; zmodule<pn0; ++zmodule) {
	  for (int zside=zbeg; zside<zmax; zside++) {
	    
	    // index iu is a continues numbering of 3D detector of FP420 (detector ID)
	    int sScale = 2*(pn0-1);
	    int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	    unsigned int ii = iu-1;// 0-29   --> 30 items
	    
	    
	    if(nA[ii]!=0  && uA[ii]!= nA[ii]) { 
	      
	      ++py; if(sector==1) ++pys1; if(sector==(sn0-1)) ++pyss;
	      if(py==2 && sector==1) { 
		double dymin=9999999., df2; int cl2=-1;
		for (int cl=0; cl<nA[ii]; ++cl) {
		  if(qA[cl][ii]){
		    df2 = abs(fyY[fip]-yA[cl][ii]);
		    if(df2 < dymin) {
		      dymin = df2;
		      cl2=cl;
		    }//if(df2		
		  }//if(qA		
		}//for(cl
		if(cl2!=-1){
		  t=(yA[cl2][ii]-fyY[fip])/(zA[cl2][ii]-fzY[fip]);
		  t1 = t*wA[cl2][ii];
		  t2 = wA[cl2][ii];
#ifdef debugsophisticated
		  std::cout << " t= " << t << " tg0= " << tg0 << std::endl;
#endif
		  if(abs(t)<tg0) { 
		    qA[cl2][ii] = false;//point is taken, mark it for not using again
		    fyY[py-1]=yA[cl2][ii];
		    fzY[py-1]=zA[cl2][ii];
		    fwY[py-1]=wA[cl2][ii];
		    qAcl[py-1] = cl2;
		    qAii[py-1] = ii;
		    ++uA[ii];
#ifdef debugsophisticated
		    std::cout << " point is taken, mark it for not using again uA[ii]= " << uA[ii] << std::endl;
#endif
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
		int clcurr=-1;
		for (int cl=0; cl<nA[ii]; ++cl) {
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
#ifdef debugsophisticated
			std::cout << " point is taken, mark it uA[ii]= " << uA[ii] << std::endl;
#endif
		      }//if py=1
		      if(uA[ii]==nA[ii]){/* no points anymore for this plane */
			++ry; if(sector==1) ++rys1; if(sector==(sn0-1)) ++ryss;
		      }//if(uA
		    }//py<3
		    else {
		      if(NewStation){
			if( sn0 < 4 ) {
			  sigma = ssigma;
			}
			else {
			  sigma = ssigma/(sn0-1-sector);
			}
			//	std::cout << " sector= " << sector << " sn0= " << sn0 << " sigma= " << sigma << std::endl;
			//	std::cout << " stattimes= " << stattimes << " ssigma= " << ssigma << " sigmam= " << sigmam << std::endl;

			//sigma = ssigma/(sn0-sector);
			//if(stattimes==1 || sector==3 ) sigma = msigma * sqrt(1./wA[cl][ii]);

			if(stattimes==1 || sector==3 ) sigma = sigmam; // (1st $3rd Stations for 3St. configur. ), 1st only for 2St. conf.
			//	if(stattimes==1 || sector==(sn0-1) ) sigma = sigmam;

			double cov00, cov01, cov11, c0Y, c1Y, chisqY;
			gsl_fit_wlinear (fzY, 1, fwY, 1, fyY, 1, py-1, 
					 &c0Y, &c1Y, &cov00, &cov01, &cov11, 
					 &chisqY);
			sm = c0Y+ c1Y*zA[cl][ii];
			
#ifdef debugsophisticated
			  std::cout << " sector= " << sector << " sn0= " << sn0 << " sigma= " << sigma << std::endl;
			  std::cout << " stattimes= " << stattimes << " ssigma= " << ssigma << " sigmam= " << sigmam << std::endl;
			  std::cout << " sm= " << sm << " c0Y= " << c0Y << " c1Y= " << c1Y << " chisqY= " << chisqY << std::endl;
			  std::cout << " zA[cl][ii]= " << zA[cl][ii] << " ii= " << ii << " cl= " << cl << std::endl;
			for (int ct=0; ct<py-1; ++ct) {
			  std::cout << " py-1= " << py-1 << " fzY[ct]= " << fzY[ct] << std::endl;
			  std::cout << " fyY[ct]= " << fyY[ct] << " fwY[ct]= " << fwY[ct] << std::endl;
			}
#endif
			
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
#ifdef debugsophisticated
			  std::cout << " diffpo= " << diffpo << " yA[cl][ii]= " << yA[cl][ii] << " sm= " << sm << " sigma= " << sigma << std::endl;
#endif
		      
		      if(abs(diffpo) < sigma ) {
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
#ifdef debugsophisticated
			std::cout << " 3333 point is taken, mark it uA[ii]= " << uA[ii] << std::endl;
#endif
			if(uA[ii]==nA[ii]){/* no points anymore for this plane */
			  ++ry; if(sector==1) ++rys1; if(sector==(sn0-1)) ++ryss;
			}//if(cl==
			//  break; // to go on neyt plane
		      }//if abs
		      else{
			t1 -= t*wA[cl][ii]; t2 -= wA[cl][ii];
		      }//if abs
		    }// if py<3 and else py>3
		    
		    if(!qA[cl][ii]) break;// go on neyt plane if point is found among clusters of current plane;
		  }// if qA
		}// for cl     --  can be break and return to "for zmodule"
		if( (py!=1 && clcurr != -1 && qA[clcurr][ii]) || (py==1 && !py1first)) { 
		  // if point is not found - continue natural loop, but reduce py 
		  py--; if(sector==1) pys1--;  if(sector==(sn0-1)) pyss--;
		}//if(py!=1
	      }//if(py==2 else 
	    }//if(nA !=0	   : inside  this if( -  ask  ++py
	  }// for zside
	}// for zmodule
      }// for sector
      //============
      
      
#ifdef debugsophisticated
      std::cout << "END: pys1= " << pys1 << " pyss = " << pyss << " py = " << py << std::endl;
#endif
      // apply criteria for track selection: 
      // do not take track if 
      if( pys1 < pys1Cut || pyss < pyssCut || py < pyallCut ){
	//	if( pys1<3 || pyss<2 || py<4 ){
      }
      // do fit:
      else{
	////////////////////////////    main fit for Narrow pixels
	double cov00, cov01, cov11;
	double c0Y, c1Y, chisqY;
	gsl_fit_wlinear (fzY, 1, fwY, 1, fyY, 1, py, 
			 &c0Y, &c1Y, &cov00, &cov01, &cov11, 
			 &chisqY);
	  ////////////////////////////
#ifdef debugsophisticated
	float chindfx;
	if(py>2) {
	  chindfx = chisqY/(py-2);
	}
	else{
	  //	  chindfy = chisqY;
	  chindfx = 9999;
	}//py
	std::cout << " Do FIT XZ: chindfx= " << chindfx << std::endl;
#endif

	////////////////////////////    second order fit for Wide pixels
#ifdef debugsophisticated
	std::cout << " preparation for second order fit for Wide pixels= " << std::endl;
#endif
	for (int ipy=0; ipy<py; ++ipy) {
	  if(zsidezn0 ==1){
	    fyYW[ipy]=xYW[qAcl[ipy]][qAii[ipy]];
	    fwYW[ipy]=wYW[qAcl[ipy]][qAii[ipy]];
#ifdef debugsophisticated
	std::cout << " ipy= " << ipy << std::endl;
	std::cout << " qAcl[ipy]= " << qAcl[ipy] << " qAii[ipy]= " << qAii[ipy] << std::endl;
	std::cout << " fyYW[ipy]= " << fyYW[ipy] << " fwYW[ipy]= " << fwYW[ipy] << std::endl;
#endif
	  }
	  else if(zsidezn0 ==2){
	    fyYW[ipy]=yXW[qAcl[ipy]][qAii[ipy]];
	    fwYW[ipy]=wXW[qAcl[ipy]][qAii[ipy]];
#ifdef debugsophisticated
	std::cout << " ipy= " << ipy << std::endl;
	std::cout << " qAcl[ipy]= " << qAcl[ipy] << " qAii[ipy]= " << qAii[ipy] << std::endl;
	std::cout << " fyYW[ipy]= " << fyYW[ipy] << " fwYW[ipy]= " << fwYW[ipy] << std::endl;
#endif
	  }
	}
#ifdef debugsophisticated
	std::cout << " start second order fit for Wide pixels= " << std::endl;
#endif
	double wov00, wov01, wov11;
	double w0Y, w1Y, whisqY;
	gsl_fit_wlinear (fzY, 1, fwYW, 1, fyYW, 1, py, 
			 &w0Y, &w1Y, &wov00, &wov01, &wov11, 
			 &whisqY);
	  ////////////////////////////
	float chindfy;
	if(py>2) {
	  chindfy = chisqY/(py-2);
	}
	else{
	  //	  chindfy = chisqY;
	  chindfy = 9999;
	}//py
	
#ifdef debugsophisticated
	std::cout << " chindfy= " << chindfy << " chiCutY= " << chiCutY << std::endl;
#endif
	if(zsidezn0 ==1){
	  if(chindfy < chiCutX ) {
	    ++numberYtracks;
	    Ay[numberYtracks-1] = c0Y; 
	    By[numberYtracks-1] = c1Y; 
	    Cy[numberYtracks-1] = chisqY; 
	    My[numberYtracks-1] = py;
	    AyW[numberYtracks-1] = w0Y; 
	    ByW[numberYtracks-1] = w1Y; 
	    CyW[numberYtracks-1] = whisqY; 
	    MyW[numberYtracks-1] = py;
#ifdef debugsophisticated
	    if(py>30) {
	      std::cout << " niteration = " << niteration << std::endl;
	      std::cout << " chindfy= " << chindfy << " py= " << py << std::endl;
	      std::cout << " c0Y= " << c0Y << " c1Y= " << c1Y << std::endl;
	      std::cout << " pys1= " << pys1 << " pyss = " << pyss << std::endl;
	    }
#endif
	  }//chindfy
	}
	else if(zsidezn0 ==2){
	  if(chindfy < chiCutY ) {
	    ++numberXtracks;
	    Ax[numberXtracks-1] = c0Y; 
	    Bx[numberXtracks-1] = c1Y; 
	    Cx[numberXtracks-1] = chisqY; 
	    Mx[numberXtracks-1] = py;
	    AxW[numberXtracks-1] = w0Y; 
	    BxW[numberXtracks-1] = w1Y; 
	    CxW[numberXtracks-1] = whisqY; 
	    MxW[numberXtracks-1] = py;
#ifdef debugsophisticated
	      std::cout << " niteration = " << niteration << std::endl;
	      std::cout << " chindfx= " << chindfy << " px= " << py << std::endl;
	      std::cout << " c0X= " << c0Y << " c1X= " << c1Y << std::endl;
	      std::cout << " pxs1= " << pys1 << " pxss = " << pyss << std::endl;
#endif
	  }//chindfy
	}
	
	
      }//  if else
	
      // do not select tracks anymore if
#ifdef debugsophisticated
      std::cout << " numberYtracks= " << numberYtracks << std::endl;
      std::cout << " numberXtracks= " << numberXtracks << std::endl;
      std::cout << " pys1= " << pys1 << " pyss = " << pyss << " py = " << py << std::endl;
      std::cout << " tas1= " << tas1 << " tass = " << tass << " taf = " << taf << std::endl;
      std::cout << " rys1= " << rys1 << " ryss = " << ryss << " ry = " << ry << std::endl;
      std::cout << " tas1-rys1= " << tas1-rys1 << " tass-ryss = " << tass-ryss << " taf-ry = " << taf-ry << std::endl;
      std::cout << "---------------------------------------------------------- " << std::endl;
#endif
      // let's decide: do we continue track finder procedure
      if( tas1-rys1<pys1Cut || tass-ryss<pyssCut || taf-ry<pyallCut  ){
	SelectTracks = false;
      }
      else{
	++niteration;
#ifdef debugsophisticated
	if(niteration > nitMax-1){
	  std::cout << "Neyt iteration, niteration >= " << niteration << std::endl;
	}
#endif
      }
      
    } while(SelectTracks && niteration < nitMax );      
  //======================    finish do loop finder for  zsidezn0     ====================================================
    
    //============
    
    //===========================================================================================================================
    
    //===========================================================================================================================
  }// for zsidezn0 
  //===========================================================================================================================
  
#ifdef debugsophisticated
  std::cout << " numberXtracks= " << numberXtracks << " numberYtracks= " << numberYtracks << std::endl;
#endif
  //===========================================================================================================================
  //===========================================================================================================================
  //===========================================================================================================================

  // case X and Y plane types are available
  if(zn0>2) {
  //===========================================================================================================================
  // match selected X and Y tracks to each other: tgphi=By/Bx->phi=artg(By/Bx); tgtheta=Bx/cosphi=By/sinphi->  ================
  //                min of |Bx/cosphi-By/sinphi|                                                               ================

  //  
#ifdef debugsophisticated
      std::cout << " numberXtracks= " << numberXtracks << " numberYtracks= " << numberYtracks << std::endl;
#endif
      if(numberXtracks>0) {
	int newxnum[10], newynum[10];// max # tracks = restracks = 10
	int nmathed=0;
	do {
	  double dthmin= 999999.; 
	  int trminx=-1, trminy=-1;
	  for (int trx=0; trx<numberXtracks; ++trx) {
#ifdef debugsophisticated
	    std::cout << "----------- trx= " << trx << " nmathed= " << nmathed << std::endl;
#endif
	    for (int tr=0; tr<numberYtracks; ++tr) {
#ifdef debugsophisticated
	      std::cout << "--- tr= " << tr << " nmathed= " << nmathed << std::endl;
#endif
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
       //double  dthdif= abs(yyyyyy-yyyvtx) + abs(xxxxxx-xxxvtx);

       double  dthdif= abs(AxW[trx]-Ay[tr]) + abs(BxW[trx]-By[tr]);

#ifdef debugsophisticated
       //  std::cout << " yyyyyy= " << yyyyyy << " xxxxxx= " << xxxxxx << " dthdif= " << dthdif << std::endl;
  std::cout << " abs(AxW[trx]-Ay[tr]) = " << abs(AxW[trx]-Ay[tr]) << " abs(BxW[trx]-By[tr])= " << abs(BxW[trx]-By[tr]) << " dthdif= " << dthdif << std::endl;
#endif
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
#ifdef debugsophisticated
	  std::cout << " trminx= " << trminx << std::endl;
#endif
	  if(nmathed>numberYtracks){
	    newynum[nmathed-1] = -1;
#ifdef debugsophisticated
	  std::cout << "!!!  nmathed= " << nmathed << " > numberYtracks= " << numberYtracks << std::endl;
#endif
	  }
	  else {
#ifdef debugsophisticated
	    std::cout << " trminy= " << trminy << std::endl;
#endif
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
#ifdef debugsophisticated
	    if(Mx[tx]>30) {
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
#endif
      //   rhits.push_back( TrackFP420(c0X,c1X,chisqX,nhitplanesY,c0Y,c1Y,chisqY,nhitplanesY) );
      rhits.push_back( TrackFP420(Ax[tx],Bx[tx],Cx[tx],Mx[tx],Ay[ty],By[ty],Cy[ty],My[ty]) );
    }//for tr
    //============================================================================================================
  }//in  numberXtracks >0
  //============

  }
  // case Y plane types are available only
  else if(zn0==1) {
    for (int ty=0; ty<numberYtracks; ++ty) {
#ifdef debugsophisticated
      std::cout << " for track ty= " << ty << std::endl;
      std::cout << " Ay= " << Ay[ty]   << std::endl;
      std::cout << " By= " << By[ty]   << std::endl;
      std::cout << " Cy= " << Cy[ty]   << std::endl;
      std::cout << " My= " << My[ty]   << std::endl;
      std::cout << " AyW= " << AyW[ty]   << std::endl;
      std::cout << " ByW= " << ByW[ty]   << std::endl;
      std::cout << " CyW= " << CyW[ty]   << std::endl;
      std::cout << " MyW= " << MyW[ty]   << std::endl;
#endif
      rhits.push_back( TrackFP420(AyW[ty],ByW[ty],CyW[ty],MyW[ty],Ay[ty],By[ty],Cy[ty],My[ty]) );
    }//for ty
    //============
  }
  // case X plane types are available only
  else if(zn0==2) {
    for (int tx=0; tx<numberXtracks; ++tx) {
#ifdef debugsophisticated
      std::cout << " for track tx= " << tx << std::endl;
      std::cout << " Ax= " << Ax[tx]   << std::endl;
      std::cout << " Bx= " << Bx[tx]   << std::endl;
      std::cout << " Cx= " << Cx[tx]   << std::endl;
      std::cout << " Mx= " << Mx[tx]   << std::endl;
      std::cout << " AxW= " << AxW[tx]   << std::endl;
      std::cout << " BxW= " << BxW[tx]   << std::endl;
      std::cout << " CxW= " << CxW[tx]   << std::endl;
      std::cout << " MxW= " << MxW[tx]   << std::endl;
#endif
      rhits.push_back( TrackFP420(Ax[tx],Bx[tx],Cx[tx],Mx[tx],AxW[tx],BxW[tx],CxW[tx],MxW[tx]) );
    }//for tx
    //============
  }//zn0




///////////////////////////////////////



  return rhits;
  //============
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


