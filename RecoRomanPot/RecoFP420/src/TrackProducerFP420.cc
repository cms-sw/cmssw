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



//////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<TrackFP420> TrackProducerFP420::trackFinderMaxAmplitude(const ClusterCollectionFP420 input){
  
  std::vector<TrackFP420> rhits;
  rhits.reserve(50); 
  rhits.clear();
  //   .
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if(zn0 !=3){
    std::cout << "TrackProducerFP420:ERROR in trackFinderMaxAmplitude: zn0 must =3 only, but = " << zn0 << std::endl; 
    return rhits;
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  int zbeg = 1, zmax=3;// XY
  if( zn0==1){
              zmax=2; // Y
  }
  else if( zn0==2){
    zbeg = 2, zmax=3; // X
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .
	double zX[40];
	double xX[40];
	double wX[40];
	double zY[40];
	double yY[40];
	double wY[40];
  //   .
	int  nhitplanesX = 0;
	int  nhitplanesY = 0;
  //   .
  for (int sector=1; sector < sn0; sector++) {
    for (int zmodule=1; zmodule<pn0; zmodule++) {
      for (int zside=zbeg; zside<zmax; zside++) {
	//	int det= 1;
	//	int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);

	// index is a continues numbering of 3D detector of FP420
	int sScale = 2*(pn0-1);
	int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	// int zScale=10;	unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;
	  //   .
	  //   .
	  //   .

	//double kplane = -(pn0-2)/2+(zmodule-1); 
	////	double zcurrent = zinibeg -ZSiPlane/2  + kplane*ZSiStep + (sector-1)*zUnit;  
	//double zcurrent = zinibeg -ZSiPlane -(ZSiStep-ZSiPlane)/2 + kplane*ZSiStep + (sector-1)*zUnit;  

    double kplane = -(pn0-1)/2+(zmodule-1); 

    double zdiststat = 0.;
    if(sector==2) zdiststat = zD2;
    if(sector==3) zdiststat = zD3;
    double zcurrent = zinibeg + z420 + (ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
    //double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + (sector-1)*zUnit;  


	double pitch=0;

	if(zside==1){
	  pitch=pitchY;
	  zcurrent += (ZGapLDet+ZSiDetL/2);
	}
	if(zside==2){
	  pitch=pitchX;
	  //	  zcurrent += (ZGapLDet+ZSiDetL+ZBoundDet+ZSiDetR/2);
	  zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
	}
	  //   .
	  //   GET CLUSTER collection  !!!!
	  //   .
//============================================================================================================ put into currentclust
  std::vector<ClusterFP420> currentclust;
	currentclust.clear();
	ClusterCollectionFP420::Range outputRange;
	outputRange = input.get(iu);
  // fill output in currentclust vector (for may be sorting? or other checks)
  ClusterCollectionFP420::ContainerIterator sort_begin = outputRange.first;
  ClusterCollectionFP420::ContainerIterator sort_end = outputRange.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
  //  std::sort(currentclust.begin(),currentclust.end());
    currentclust.push_back(*sort_begin);
  } // for

#ifdef debugmaxampl
  std::cout << "TrackProducerFP420: currentclust.size = " << currentclust.size() << std::endl; 
#endif
//============================================================================================================

  vector<ClusterFP420>::const_iterator simHitIter = currentclust.begin();
  vector<ClusterFP420>::const_iterator simHitIterEnd = currentclust.end();
  
  int icl = 0;
  float clampmax = 0.;
  ClusterFP420 iclustermax;
  // loop in #clusters
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    const ClusterFP420 icluster = *simHitIter;
    icl++;
    
    float ampmax=0.;
    // loop in strips for each cluster:
    for(unsigned int i = 0; i < icluster.amplitudes().size(); i++ ) {
      if(icluster.amplitudes()[i] > ampmax) ampmax = icluster.amplitudes()[i];      
    }   // for loop in strips for each cluster:
    
    // find( and take info) cluster with max amplitude inside its width  (only one!)
    if(ampmax>clampmax) {
      clampmax = ampmax;
      iclustermax = *simHitIter;
    }
  } // for loop in #clusters

//============================================================================================================
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: clampmax= " << clampmax << std::endl; 
#endif
  if(clampmax != 0){
    // fill vectors for track reconstruction
    
    // local - global systems with possible shift of every second plate:
    //    float dYY = 5.;// XSiDet/2.
    //  float dXX = 12.7+0.05;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7+0.05
    float dYYcur = dYY;
    float dXXcur = dXX;


    if ( zside ==1 && UseHalfPitchShiftInY== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii != 0)  dYYcur -= pitch/2.;
    }
    if (zside ==2 && UseHalfPitchShiftInX== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii == 0)  dXXcur += pitch/2.;
    }
    
    
    //disentangle complicated pattern recognition of hits?
    // Y:
    if(zside ==1){
      nhitplanesY++;		
      zY[nhitplanesY-1] = zcurrent;
      yY[nhitplanesY-1] = iclustermax.barycenter()*pitch;
      // go to global system:
      yY[nhitplanesY-1] = yY[nhitplanesY-1] - dYYcur; 
      wY[nhitplanesY-1] = 1./(iclustermax.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wY[nhitplanesY-1] *= wY[nhitplanesY-1];//reciprocal of the variance for each datapoint in y
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: zside = 1 --> points to be fitted: " << std::endl; 
 std::cout << "TrackProducerFP420: nhitplanesY = " << nhitplanesY << "  zcurrent = " << zcurrent << "  yY[nhitplanesY-1] = " << yY[nhitplanesY-1] << "  wY[nhitplanesY-1] = " << wY[nhitplanesY-1] << std::endl;
#endif
    }
    // X:
    else if(zside ==2){
      nhitplanesX++;		
      zX[nhitplanesX-1] = zcurrent;
      xX[nhitplanesX-1] = iclustermax.barycenter()*pitch;
      // go to global system:
      xX[nhitplanesX-1] =-(xX[nhitplanesX-1]+dXXcur); 
      wX[nhitplanesX-1] = 1./(iclustermax.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wX[nhitplanesX-1] *= wX[nhitplanesX-1];//reciprocal of the variance for each datapoint in y
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: zside = 2 --> points to be fitted: " << std::endl; 
 std::cout << "TrackProducerFP420: nhitplanesX = " << nhitplanesX << "  zcurrent = " << zcurrent << "  xX[nhitplanesX-1] = " << xX[nhitplanesX-1] << "  wX[nhitplanesX-1] = " << wX[nhitplanesX-1] << std::endl;
#endif
    }
  }// if(clampmax
  

  //================================== end of for loops in continuius number iu:
      }   // for
    }   // for
  }   // for


  //============================================================================================================Fits:
  double cov00, cov01, cov11;
  //X                                                                                                   X
  double c0X, c1X, chisqX;
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: before X fit: " << std::endl; 
 std::cout << "nhitplanesX = " << nhitplanesX << std::endl;
#endif
  gsl_fit_wlinear (zX, 1, wX, 1, xX, 1, nhitplanesX, 
		   &c0X, &c1X, &cov00, &cov01, &cov11, 
		   &chisqX);
  float chindfx;
  if(nhitplanesX>2) {
    chindfx = chisqX/(nhitplanesX-2);
  }
  else{
    chindfx = chisqX;
  }
  //Y                                                                                                        Y
  double c0Y, c1Y, chisqY;
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: before Y fit: " << std::endl; 
 std::cout << "nhitplanesY = " << nhitplanesY << std::endl;
#endif
  gsl_fit_wlinear (zY, 1, wY, 1, yY, 1, nhitplanesY, 
		   &c0Y, &c1Y, &cov00, &cov01, &cov11, 
		   &chisqY);
  float chindfy;
  if(nhitplanesY>2) {
    chindfy = chisqY/(nhitplanesY-2);
  }
  else{
    chindfy = chisqY;
  }
  //============================================================================================================
#ifdef debugmaxampl
  std::cout << "TrackProducerFP420: after fit: " << std::endl; 
  std::cout << "nhitplanesX = " << nhitplanesX << "  chisqX = " << chisqX << "  chindfx = " << chindfx << std::endl;
  std::cout << "nhitplanesY = " << nhitplanesY << "  chisqY = " << chisqY << "  chindfy = " << chindfy << std::endl;
#endif
  
  //============================================================================================================
  if(chindfx < chiCutX && chindfy < chiCutY ) {
    //  if(chindfx < 3. && chindfy < 3. ) {
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: start  rhits.push_back " << std::endl; 
#endif
    rhits.push_back( TrackFP420(c0X,c1X,chisqX,nhitplanesX,c0Y,c1Y,chisqY,nhitplanesY)    );
  }
  //      rhits.push_back( TrackFP420(ax,bx,chi2x,nclusterx,ay,by,chi2y,nclustery)  );
  
  
  return rhits;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<TrackFP420> TrackProducerFP420::trackFinderMaxAmplitude2(const ClusterCollectionFP420 input){
  
  std::vector<TrackFP420> rhits;
  rhits.reserve(50); 
  rhits.clear();
  //   .

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if( zn0 < 1 || zn0 >3 ){
    std::cout << "TrackProducerFP420:ERROR in trackFinderMaxAmplitude: check zn0  = " << zn0 << std::endl; 
    return rhits;
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  int zbeg = 1, zmax=3;// XY
  if( zn0==1){
              zmax=2; // Y
  }
  else if( zn0==2){
    zbeg = 2, zmax=3; // X
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .
	double zXY[60];
	double xX[60];
	double wX[60];
	double yY[60];
	double wY[60];
  //   .
	int  nhitplanes = 0;
  //   .
  for (int sector=1; sector < sn0; sector++) {
    for (int zmodule=1; zmodule<pn0; zmodule++) {
      for (int zside=zbeg; zside<zmax; zside++) {
	//	int det= 1;
	//	int index = FP420NumberingScheme::packFP420Index(det, zside, sector, zmodule);

	// index is a continues numbering of 3D detector of FP420
	int sScale = 2*(pn0-1);
	int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	// int zScale=10;	unsigned int intindex = sScale*(sector - 1)+zScale*(zside - 1)+zmodule;
	  //   .
	  //   .
	  //   .

	//double kplane = -(pn0-2)/2+(zmodule-1); 
	////	double zcurrent = zinibeg -ZSiPlane/2  + kplane*ZSiStep + (sector-1)*zUnit;  
	//double zcurrent = zinibeg -ZSiPlane -(ZSiStep-ZSiPlane)/2 + kplane*ZSiStep + (sector-1)*zUnit;  

    double kplane = -(pn0-1)/2+(zmodule-1); 

    double zdiststat = 0.;
    if(sector==2) zdiststat = zD2;
    if(sector==3) zdiststat = zD3;
    double zcurrent = zinibeg + z420 + (ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
    //double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + (sector-1)*zUnit;  


	double pitch=0;
	double pitchW=0;

	if(zside==1){
	  pitch=pitchY;
	  pitchW=pitchYW;
	  zcurrent += (ZGapLDet+ZSiDetL/2);
	}
	if(zside==2){
	  pitch=pitchX;
	  pitchW=pitchXW;
	  //	  zcurrent += (ZGapLDet+ZSiDetL+ZBoundDet+ZSiDetR/2);
	  zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
	}
	  //   .
	  //   GET CLUSTER collection  !!!!
	  //   .
//============================================================================================================ put into currentclust
  std::vector<ClusterFP420> currentclust;
	currentclust.clear();
	ClusterCollectionFP420::Range outputRange;
	outputRange = input.get(iu);
  // fill output in currentclust vector (for may be sorting? or other checks)
  ClusterCollectionFP420::ContainerIterator sort_begin = outputRange.first;
  ClusterCollectionFP420::ContainerIterator sort_end = outputRange.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
  //  std::sort(currentclust.begin(),currentclust.end());
    currentclust.push_back(*sort_begin);
  } // for

#ifdef debugmaxampl
  std::cout << "TrackProducerFP420: currentclust.size = " << currentclust.size() << std::endl; 
#endif
//============================================================================================================

  vector<ClusterFP420>::const_iterator simHitIter = currentclust.begin();
  vector<ClusterFP420>::const_iterator simHitIterEnd = currentclust.end();
  
  int icl = 0;
  float clampmax = 0.;
  ClusterFP420 iclustermax;
  // loop in #clusters
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    const ClusterFP420 icluster = *simHitIter;
    icl++;
    
    float ampmax=0.;
    // loop in strips for each cluster:
    for(unsigned int i = 0; i < icluster.amplitudes().size(); i++ ) {
      if(icluster.amplitudes()[i] > ampmax) ampmax = icluster.amplitudes()[i];      
    }   // for loop in strips for each cluster:
    
    // find( and take info) cluster with max amplitude inside its width  (only one!)
    if(ampmax>clampmax) {
      clampmax = ampmax;
      iclustermax = *simHitIter;
    }
  } // for loop in #clusters

//============================================================================================================
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: clampmax= " << clampmax << std::endl; 
#endif
  if(clampmax != 0){
    // fill vectors for track reconstruction
    
    // local - global systems with possible shift of every second plate:
    //    float dYY = 5.;// XSiDet/2.
    //  float dXX = 12.7+0.05;//(BoxYshft+dYGap) + (YSi - YSiDet)/2. = 12.7+0.05
    float dYYcur = dYY;
    float dXXcur = dXX;


    if ( zside ==1 && UseHalfPitchShiftInY== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii != 0)  dYYcur -= pitch/2.;
    }
    if (zside ==2 && UseHalfPitchShiftInX== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii == 0)  dXXcur += pitch/2.;
    }
    
    
      nhitplanes++;		
    //disentangle complicated pattern recognition of hits?
    // Y:
    if(zside ==1){
      zXY[nhitplanes-1] = zcurrent;
      yY[nhitplanes-1] = iclustermax.barycenter()*pitch;
      xX[nhitplanes-1] = iclustermax.barycenterW()*pitchW;
      // go to global system:
      yY[nhitplanes-1] = yY[nhitplanes-1] - dYYcur; 
      wY[nhitplanes-1] = 1./(iclustermax.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wY[nhitplanes-1] *= wY[nhitplanes-1];//reciprocal of the variance for each datapoint in y
      xX[nhitplanes-1] =-(xX[nhitplanes-1]+dXXcur); 
      wX[nhitplanes-1] = 1./(iclustermax.barycerrorW()*pitchW);//reciprocal of the variance for each datapoint in y
      wX[nhitplanes-1] *= wX[nhitplanes-1];//reciprocal of the variance for each datapoint in y
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: zside = 1 --> points to be fitted: " << std::endl; 
 std::cout << "TrackProducerFP420: nhitplanes = " << nhitplanes << "  zcurrent = " << zcurrent << "  yY[nhitplanes-1] = " << yY[nhitplanes-1] << "  wY[nhitplanes-1] = " << wY[nhitplanes-1] << std::endl;
#endif
    }
    // X:
    else if(zside ==2){
      zXY[nhitplanes-1] = zcurrent;
      xX[nhitplanes-1] = iclustermax.barycenter()*pitch;
      yY[nhitplanes-1] = iclustermax.barycenterW()*pitchW;
      // go to global system:
      xX[nhitplanes-1] =-(xX[nhitplanes-1]+dXXcur); 
      wX[nhitplanes-1] = 1./(iclustermax.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wX[nhitplanes-1] *= wX[nhitplanes-1];//reciprocal of the variance for each datapoint in y
      yY[nhitplanes-1] = yY[nhitplanes-1] - dYYcur; 
      wY[nhitplanes-1] = 1./(iclustermax.barycerrorW()*pitchW);//reciprocal of the variance for each datapoint in y
      wY[nhitplanes-1] *= wY[nhitplanes-1];//reciprocal of the variance for each datapoint in y
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: zside = 2 --> points to be fitted: " << std::endl; 
 std::cout << "TrackProducerFP420: nhitplanes = " << nhitplanes << "  zcurrent = " << zcurrent << "  xX[nhitplanes-1] = " << xX[nhitplanes-1] << "  wX[nhitplanes-1] = " << wX[nhitplanes-1] << std::endl;
#endif
    }
  }// if(clampmax
  

  //================================== end of for loops in continuius number iu:
      }   // for
    }   // for
  }   // for


  //============================================================================================================Fits:
  double cov00, cov01, cov11;
  //X                                                                                                   X
  double c0X, c1X, chisqX;
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: before X fit: " << std::endl; 
 std::cout << "nhitplanes = " << nhitplanes << std::endl;
#endif
  gsl_fit_wlinear (zXY, 1, wX, 1, xX, 1, nhitplanes, 
		   &c0X, &c1X, &cov00, &cov01, &cov11, 
		   &chisqX);
  float chindfx;
  if(nhitplanes>2) {
    chindfx = chisqX/(nhitplanes-2);
  }
  else{
    chindfx = chisqX;
  }
  //Y                                                                                                        Y
  double c0Y, c1Y, chisqY;
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: before Y fit: " << std::endl; 
 std::cout << "nhitplanes = " << nhitplanes << std::endl;
#endif
  gsl_fit_wlinear (zXY, 1, wY, 1, yY, 1, nhitplanes, 
		   &c0Y, &c1Y, &cov00, &cov01, &cov11, 
		   &chisqY);
  float chindfy;
  if(nhitplanes>2) {
    chindfy = chisqY/(nhitplanes-2);
  }
  else{
    chindfy = chisqY;
  }
  //============================================================================================================
#ifdef debugmaxampl
  std::cout << "TrackProducerFP420: after fit: " << std::endl; 
  std::cout << "nhitplanes = " << nhitplanes << "  chisqX = " << chisqX << "  chindfx = " << chindfx << std::endl;
  std::cout << "nhitplanes = " << nhitplanes << "  chisqY = " << chisqY << "  chindfy = " << chindfy << std::endl;
#endif
  
  //============================================================================================================
  if(chindfx < chiCutX && chindfy < chiCutY ) {
    //  if(chindfx < 3. && chindfy < 3. ) {
#ifdef debugmaxampl
	  std::cout << "TrackProducerFP420: start  rhits.push_back " << std::endl; 
#endif
    rhits.push_back( TrackFP420(c0X,c1X,chisqX,nhitplanes,c0Y,c1Y,chisqY,nhitplanes)    );
  }
  //      rhits.push_back( TrackFP420(ax,bx,chi2x,nclusterx,ay,by,chi2y,nclustery)  );
  
  
  return rhits;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////















////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<TrackFP420> TrackProducerFP420::trackFinderSophisticated(const ClusterCollectionFP420 input){
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
	  int sScale = 2*(pn0-1);
	  int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	  //	unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) + 1;

	  //	  unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) ; // 0-19   --> 20 items
	  unsigned int ii = iu-1;// 0-29   --> 30 items
	  
	  double kplane = -(pn0-1)/2 - 0.5  +  (zmodule-1); 
	  
	  
	  double zdiststat = 0.;
	  if(sector==2) zdiststat = zD2;
	  if(sector==3) zdiststat = zD3;
	  double zcurrent = zinibeg + z420 + (ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
	  //double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + (sector-1)*zUnit;  
	  
	  if(zside==1){
	    zcurrent += (ZGapLDet+ZSiDetL/2);
	  }
	  if(zside==2){
	    zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
	  }
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
	  outputRange = input.get(iu);
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
  int  pys1Cut = 1, pyssCut = 1, pyallCut= 4;

//  double yyyvtx = 0.0, xxxvtx = -15;  //mm

// for equidistant 3 Stations:
//  double sigman=0.18, ssigma = 2.5, sigmam=0.18;
//  double sigman=0.18, ssigma = 1.8, sigmam=0.18;

//  double sigman=0.18, ssigma = 2.9, sigmam=0.18;
// for tests:
//  double sigman=118., ssigma = 299., sigmam=118.;
// RMS1=0.013, RMS2 = 1.0, RMS3 = 0.018 see plots d1XCL, d2XCL, d3XCL
  double sigman=0.05, ssigma = 2.5, sigmam=0.06;
//  double sigman=0.18, ssigma = 2.5, sigmam=0.18;


  // for configuration: 3St, 1m for 1-2 St:
 // double sigman=0.1, ssigma = 1.0, sigmam=0.15;/* ssigma is foreseen to match 1st point of 2nd Station*/

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
			sigma = ssigma/(sn0-1-sector);
			//sigma = ssigma/(sn0-sector);
			//if(stattimes==1 || sector==3 ) sigma = msigma * sqrt(1./wA[cl][ii]);
			if(stattimes==1 || sector==3 ) sigma = sigmam;
			
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


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<TrackFP420> TrackProducerFP420::trackFinder3D(const ClusterCollectionFP420 input){
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<TrackFP420> rhits;
  int restracks = 10;// max # tracks
  rhits.reserve(restracks); 
  rhits.clear();
  double Ax[10]; double Bx[10]; double Cx[10]; int Mx[10];
  double Ay[10]; double By[10]; double Cy[10]; int My[10];
  double AxW[10]; double BxW[10]; double CxW[10]; int MxW[10];
  double AyW[10]; double ByW[10]; double CyW[10]; int MyW[10];
#ifdef debug3d
  std::cout << "TrackProducerFP420: Start trackFinder3d " << std::endl; 
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int zbeg = 1, zmax=3;// XY
  if( zn0==1){
              zmax=2; // Y
  }
  else if( zn0==2){
    zbeg = 2, zmax=3; // X
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .
  int reshits = 15;// max # cl for every X and Y plane
  //  int resplanes = 30;
  int nX[30], nY[30];// resplanes =30 NUMBER OF PLANES; nX, nY - # cl for every X and Y plane
  int uX[30], uY[30];// resplanes =30 NUMBER OF PLANES; nX, nY - current # cl used for every X and Y plane
  double zX[15][30], xX[15][30], wX[15][30];
  double zY[15][30], yY[15][30], wY[15][30];
  double             xXW[15][30], wXW[15][30];
  double             yYW[15][30], wYW[15][30];
  bool qX[15][30], qY[15][30];
  //   .
  int tx = 0; int txs1 = 0; int txss = 0;
  int ty = 0; int tys1 = 0; int tyss = 0;
  //   .
  for (int zside=zbeg; zside<zmax; zside++) {
    for (int sector=1; sector < sn0; sector++) {
      for (int zmodule=1; zmodule<pn0; zmodule++) {

	// index is a continues numbering of 3D detector of FP420
	int sScale = 2*(pn0-1);
	int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	//	unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) + 1;
	unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) ; // 0-19   --> 20 items

	double kplane = -(pn0-1)/2+(zmodule-1); 


    double zdiststat = 0.;
    if(sector==2) zdiststat = zD2;
    if(sector==3) zdiststat = zD3;
    double zcurrent = zinibeg + z420 + (ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
    //double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + (sector-1)*zUnit;  

	double pitch=0;
	double pitchW=0;
	if(zside==1){
	  pitch=pitchY;
	  pitchW=pitchYW;
	  zcurrent += (ZGapLDet+ZSiDetL/2);
	}
	if(zside==2){
	  pitch=pitchX;
	  pitchW=pitchXW;
	  zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
	}
	  //   .
	  //   GET CLUSTER collection  !!!!
	  //   .
//============================================================================================================ put into currentclust
  std::vector<ClusterFP420> currentclust;
	currentclust.clear();
	ClusterCollectionFP420::Range outputRange;
	outputRange = input.get(iu);
  // fill output in currentclust vector (for may be sorting? or other checks)
  ClusterCollectionFP420::ContainerIterator sort_begin = outputRange.first;
  ClusterCollectionFP420::ContainerIterator sort_end = outputRange.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
  //  std::sort(currentclust.begin(),currentclust.end());
    currentclust.push_back(*sort_begin);
  } // for

#ifdef debug3d
  std::cout << "TrackProducerFP420: currentclust.size = " << currentclust.size() << std::endl; 
#endif
//============================================================================================================

  vector<ClusterFP420>::const_iterator simHitIter = currentclust.begin();
  vector<ClusterFP420>::const_iterator simHitIterEnd = currentclust.end();
  
    if(zside ==1){
      nY[ii] = 0;// # cl in every Y plane (max is reshits)
      uY[ii] = 0;// current used # cl in every X plane 
    }
    else if(zside ==2){
      nX[ii] = 0;// # cl in every X plane (max is reshits)
      uX[ii] = 0;// current used # cl in every X plane 
    }
  // loop in #clusters
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    const ClusterFP420 icluster = *simHitIter;
    
    // fill vectors for track reconstruction
    
    // local - global systems with possible shift of every second plate:
    float dYYcur = dYY;
    float dXXcur = dXX;
    if (UseHalfPitchShiftInY== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii != 0)  dYYcur -= pitch/2.;
    }
    if (UseHalfPitchShiftInX== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii == 0)  dXXcur += pitch/2.;
    }
    
    
    //disentangle complicated pattern recognition of hits?
    // Y:
    if(zside ==1){
      nY[ii]++;		
      if(nY[ii]>reshits){
	nY[ii]=reshits;
	std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << " zcurrent=" << zY[nY[ii]-1][ii] << " ii= "  << ii << std::endl;
      }
      zY[nY[ii]-1][ii] = zcurrent;
      yY[nY[ii]-1][ii] = icluster.barycenter()*pitch;
      xXW[nY[ii]-1][ii] = icluster.barycenterW()*pitchW;
      // go to global system:
      yY[nY[ii]-1][ii] = yY[nY[ii]-1][ii] - dYYcur; 
      wY[nY[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wY[nY[ii]-1][ii] *= wY[nY[ii]-1][ii];//reciprocal of the variance for each datapoint in y
      xXW[nY[ii]-1][ii] =-(xXW[nY[ii]-1][ii]+dXXcur); 
      wXW[nY[ii]-1][ii] = 1./(icluster.barycerrorW()*pitchW);//reciprocal of the variance for each datapoint in y
      wXW[nY[ii]-1][ii] *= wXW[nY[ii]-1][ii];//reciprocal of the variance for each datapoint in y
      qY[nY[ii]-1][ii] = true;
      if(nY[ii]==reshits) break;
    }
    // X:
    else if(zside ==2){
      nX[ii]++;	
      if(nX[ii]>reshits){
	std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << std::endl;
	nX[ii]=reshits;
      }
      zX[nX[ii]-1][ii] = zcurrent;
      xX[nX[ii]-1][ii] = icluster.barycenter()*pitch;
      yYW[nX[ii]-1][ii] = icluster.barycenterW()*pitchW;
      // go to global system:
      xX[nX[ii]-1][ii] =-(xX[nX[ii]-1][ii]+dXXcur); 
      wX[nX[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wX[nX[ii]-1][ii] *= wX[nX[ii]-1][ii];//reciprocal of the variance for each datapoint in y
      yYW[nX[ii]-1][ii] = yYW[nX[ii]-1][ii] - dYYcur; 
      wYW[nX[ii]-1][ii] = 1./(icluster.barycerrorW()*pitchW);//reciprocal of the variance for each datapoint in y
      wYW[nX[ii]-1][ii] *= wYW[nX[ii]-1][ii];//reciprocal of the variance for each datapoint in y
      qX[nX[ii]-1][ii] = true;
      if(nX[ii]==reshits) break;
    }

  } // for loop in #clusters (can be breaked)

    // Y:
    if(zside ==1){
	  if(nY[ii] != 0) {  /* # Y-planes w/ clusters */
	    ++ty; if(sector==1) ++tys1; if(sector==(sn0-1)) ++tyss;
	  }	  
    }
    // X:
    else if(zside ==2){
	  if(nX[ii] != 0) {  /* # X-planes w/ clusters */
	    ++tx; if(sector==1) ++txs1; if(sector==(sn0-1)) ++txss;
	  }	  
    }
  //================================== end of for loops in continuius number iu:
      }   // for zmodule
    }   // for sector
  }   // for zside
#ifdef debug3d
  std::cout << "trackFinder3d: ty= " << ty << " tys1 = " << tys1 << " tyss = " << tyss << std::endl;
  std::cout << "trackFinder3d: tx= " << tx << " txs1 = " << txs1 << " txss = " << txss << std::endl;
  std::cout << "============================================================" << std::endl;
#endif

  //===========================================================================================================================
  //===========================================================================================================================
  //===========================================================================================================================
  //======================    start road finder   =============================================================================
  //===========================================================================================================================

  int nitMax=3;// max # iterations to find track
  // criteria for track selection: 
  int  pys1Cut = 5, pyssCut = 5, pyallCut=12;

//  double yyyvtx = 0.0, xxxvtx = -15;  //mm

// for equidistant 3 Stations:
//  double sigman=0.18, ssigma = 2.5, sigmam=0.18;

  double sigman=0.18, ssigma = 1.8, sigmam=0.18;


  // for configuration: 3St, 1m for 1-2 St:
 // double sigman=0.1, ssigma = 1.0, sigmam=0.15;/* ssigma is foreseen to match 1st point of 2nd Station*/



/* ssigma = 3. * 8000.*(0.025+0.009)/sqrt(pn0-1)/100. = 2.9 mm(!!!)----
   ssigma is reduced by factor k_reduced = (sn0-1)-sector+1 = sn0-sector
    # Stations  currentStation
    2Stations:     sector=2,         sn0=3 , sn0-sector = 1 --> k_reduced = 1
    3Stations:     sector=2,         sn0=4 , sn0-sector = 2 --> k_reduced = 2
    3Stations:     sector=3,         sn0=4 , sn0-sector = 1 --> k_reduced = 1
*/
  int numberXtracks=0, numberYtracks=0, totpl = (pn0-1)*(sn0-1); double sigma;

  for (int zside=zbeg; zside<zmax; ++zside) {
    //
    //
    double tg0 = 0.;
    int qAcl[30], qAii[30], fip=0, niteration = 0;
    int ry = 0, rys1 = 0, ryss = 0;
    bool SelectTracks = true;
    //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .

  double yA[15][30], zA[15][30], wA[15][30]; int nA[30], uA[30]; bool qA[15][30];
    //
    // Y:
  //======================    start road finder  for zside = 1      ===========================================================
    if(zside ==1){
  //===========================================================================================================================
      numberYtracks=0;  
      tg0= 3*1./(800.+20.); // for Y: 1cm/...   *3 - 3sigma range

      for (int ii=0; ii < totpl; ++ii) {
#ifdef debug3d
  std::cout << "trackFinder3d: ii= " << ii << " nY[ii]= " << nY[ii] << std::endl;
  std::cout << "trackFinder3d: ii= " << ii << " uY[ii]= " << uY[ii] << std::endl;
#endif
	nA[ii] = nY[ii];
	uA[ii] = uY[ii];
	for (int cl=0; cl<nA[ii]; ++cl) {
#ifdef debug3d
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
    }// if zside ==1
    // X:
  //======================    start road finder  for zside = 2      ===========================================================
    else if(zside ==2){
  //===========================================================================================================================
      numberXtracks=0;  
      tg0= 3*2./(800.+20.); // for X: 2cm/...   *3 - 3sigma range

      for (int ii=0; ii < totpl; ++ii) {
	nA[ii] = nX[ii];
	uA[ii] = uX[ii];
	for (int cl=0; cl<nA[ii]; ++cl) {
	  yA[cl][ii] = xX[cl][ii];
	  zA[cl][ii] = zX[cl][ii];
	  wA[cl][ii] = wX[cl][ii];
	  qA[cl][ii] = qX[cl][ii];
	}
      }
  //===========================================================================================================================
    }// if zside ==2
    
  //======================    start road finder  for zside = 1 or 2        ====================================================
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
	  
	  int sScale = 2*(pn0-1); 
	  // unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) + 1;
	  unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) ;
	  
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
		if(abs(t)<tg0) { 
		  qA[cl2][ii] = false;//point is taken, mark it for not using again
		  fyY[py-1]=yA[cl2][ii];
		  fzY[py-1]=zA[cl2][ii];
		  fwY[py-1]=wA[cl2][ii];
		  qAcl[py-1] = cl2;
		  qAii[py-1] = ii;
		  ++uA[ii];
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
		    }//if py=1
		    if(uA[ii]==nA[ii]){/* no points anymore for this plane */
		      ++ry; if(sector==1) ++rys1; if(sector==(sn0-1)) ++ryss;
		    }//if(uA
		  }//py<3
		  else {
		    if(NewStation){
		      sigma = ssigma/(sn0-sector);
		      //if(stattimes==1 || sector==3 ) sigma = msigma * sqrt(1./wA[cl][ii]);
		      if(stattimes==1 || sector==3 ) sigma = sigmam;
		      
		      double cov00, cov01, cov11, c0Y, c1Y, chisqY;
		      gsl_fit_wlinear (fzY, 1, fwY, 1, fyY, 1, py-1, 
				       &c0Y, &c1Y, &cov00, &cov01, &cov11, 
				       &chisqY);
		      sm = c0Y+ c1Y*zA[cl][ii];
		      
		      
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
	}// for zmodule
      }// for sector
      //============
      
      
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

	////////////////////////////    second order fit for Wide pixels
#ifdef debug3d
	std::cout << " preparation for second order fit for Wide pixels= " << std::endl;
#endif
	for (int ipy=0; ipy<py; ++ipy) {
	  if(zside ==1){
	    fyYW[ipy]=yYW[qAcl[ipy]][qAii[ipy]];
	    fwYW[ipy]=wYW[qAcl[ipy]][qAii[ipy]];
	  }
	  else if(zside ==2){

	    fyYW[ipy]=xXW[qAcl[ipy]][qAii[ipy]];
	    fwYW[ipy]=wXW[qAcl[ipy]][qAii[ipy]];
#ifdef debug3d
	std::cout << " ipy= " << ipy << std::endl;
	std::cout << " qAcl[ipy]= " << qAcl[ipy] << " qAii[ipy]= " << qAii[ipy] << std::endl;
	std::cout << " fyYW[ipy]= " << fyYW[ipy] << " fwYW[ipy]= " << fwYW[ipy] << std::endl;
#endif
	  }
	}
#ifdef debug3d
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
	
#ifdef debug3d
	std::cout << " chindfy= " << chindfy << " chiCutY= " << chiCutY << std::endl;
#endif
	if(zside ==1){
	  if(chindfy < chiCutY ) {
	    ++numberYtracks;
	    Ay[numberYtracks-1] = c0Y; 
	    By[numberYtracks-1] = c1Y; 
	    Cy[numberYtracks-1] = chisqY; 
	    My[numberYtracks-1] = py;
	    AyW[numberYtracks-1] = w0Y; 
	    ByW[numberYtracks-1] = w1Y; 
	    CyW[numberYtracks-1] = whisqY; 
	    MyW[numberYtracks-1] = py;
#ifdef debug3d30
	    if(py>30) {
	      std::cout << " niteration = " << niteration << std::endl;
	      std::cout << " chindfy= " << chindfy << " py= " << py << std::endl;
	      std::cout << " c0Y= " << c0Y << " c1Y= " << c1Y << std::endl;
	      std::cout << " pys1= " << pys1 << " pyss = " << pyss << std::endl;
	    }
#endif
	  }//chindfy
	}
	else if(zside ==2){
	  if(chindfy < chiCutX ) {
	    ++numberXtracks;
	    Ax[numberXtracks-1] = c0Y; 
	    Bx[numberXtracks-1] = c1Y; 
	    Cx[numberXtracks-1] = chisqY; 
	    Mx[numberXtracks-1] = py;
	    AxW[numberXtracks-1] = w0Y; 
	    BxW[numberXtracks-1] = w1Y; 
	    CxW[numberXtracks-1] = whisqY; 
	    MxW[numberXtracks-1] = py;
#ifdef debug3d30
	      std::cout << " niteration = " << niteration << std::endl;
	      std::cout << " chindfx= " << chindfy << " px= " << py << std::endl;
	      std::cout << " c0X= " << c0Y << " c1X= " << c1Y << std::endl;
	      std::cout << " pxs1= " << pys1 << " pxss = " << pyss << std::endl;
#endif
	  }//chindfy
	}
	
	
      }//  if else
	
      // do not select tracks anymore if
#ifdef debug3d30
      std::cout << " numberYtracks= " << numberYtracks << std::endl;
      std::cout << " numberXtracks= " << numberXtracks << std::endl;
      std::cout << " pys1= " << pys1 << " pyss = " << pyss << " py = " << py << std::endl;
      std::cout << " tys1= " << tys1 << " tyss = " << tyss << " ty = " << ty << std::endl;
      std::cout << " rys1= " << rys1 << " ryss = " << ryss << " ry = " << ry << std::endl;
      std::cout << " tys1-rys1= " << tys1-rys1 << " tyss-ryss = " << tyss-ryss << " ty-ry = " << ty-ry << std::endl;
      std::cout << "---------------------------------------------------------- " << std::endl;
#endif
      // let's decide: do we continue track finder procedure
      if( tys1-rys1<pys1Cut || tyss-ryss<pyssCut || ty-ry<pyallCut  ){
	SelectTracks = false;
      }
      else{
	++niteration;
#ifdef debug3d30
	if(niteration > nitMax-1){
	  std::cout << "Neyt iteration, niteration >= " << niteration << std::endl;
	}
#endif
      }
      
    } while(SelectTracks && niteration < nitMax );      
  //======================    finish do loop finder for  zside =  1, 2      ====================================================
    
    //============
    
    //===========================================================================================================================
    
    //===========================================================================================================================
  }// for zside 1,2
  //===========================================================================================================================
  
#ifdef debug3d30
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
#ifdef debug3d30
      std::cout << " numberXtracks= " << numberXtracks << " numberYtracks= " << numberYtracks << std::endl;
#endif
      if(numberXtracks>0) {
	int newxnum[10], newynum[10];// max # tracks = restracks = 10
	int nmathed=0;
	do {
	  double dthmin= 999999.; 
	  int trminx=-1, trminy=-1;
	  for (int trx=0; trx<numberXtracks; ++trx) {
#ifdef debug3d30
	    std::cout << "----------- trx= " << trx << " nmathed= " << nmathed << std::endl;
#endif
	    for (int tr=0; tr<numberYtracks; ++tr) {
#ifdef debug3d30
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

#ifdef debug3d30
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
#ifdef debug3d30
	  std::cout << " trminx= " << trminx << std::endl;
#endif
	  if(nmathed>numberYtracks){
	    newynum[nmathed-1] = -1;
#ifdef debug3d30
	  std::cout << "!!!  nmathed= " << nmathed << " > numberYtracks= " << numberYtracks << std::endl;
#endif
	  }
	  else {
#ifdef debug3d30
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
#ifdef debug3d30
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
#ifdef debug3d
      std::cout << " for track tr= " << tr << " tx= " << tx << " ty= " << ty << std::endl;
      std::cout << " Ax= " << Ax[tx]   << " Ay= " << Ay[ty]   << std::endl;
      std::cout << " Bx= " << Bx[tx]   << " By= " << By[ty]   << std::endl;
      std::cout << " Cx= " << Cx[tx]   << " Cy= " << Cy[ty]   << std::endl;
      std::cout << " Mx= " << Mx[tx]   << " My= " << My[ty]   << std::endl;
      std::cout << " AxW= " << AxW[tx]   << " AyW= " << AyW[ty]   << std::endl;
      std::cout << " BxW= " << BxW[tx]   << " ByW= " << ByW[ty]   << std::endl;
      std::cout << " CxW= " << CxW[tx]   << " CyW= " << CyW[ty]   << std::endl;
      std::cout << " MxW= " << MxW[tx]   << " MyW= " << MyW[ty]   << std::endl;
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
    for (int tr=0; tr<numberYtracks; ++tr) {
      rhits.push_back( TrackFP420(AyW[ty],ByW[ty],CyW[ty],MyW[ty],Ay[ty],By[ty],Cy[ty],My[ty]) );
    }//for tr
    //============
  }
  // case X plane types are available only
  else if(zn0==2) {
    for (int tr=0; tr<numberXtracks; ++tr) {
      rhits.push_back( TrackFP420(Ax[tx],Bx[tx],Cx[tx],Mx[tx],AxW[tx],BxW[tx],CxW[tx],MxW[tx]) );
    }//for tr
  //============
  }




///////////////////////////////////////



  return rhits;
  //============
  
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



/*
// For tests:  For tests:   For tests:  For tests:  For tests:  For tests:  For tests:  For tests: For tests:
// For tests:  For tests:   For tests:  For tests:  For tests:  For tests:  For tests:  For tests: For tests:
// For tests:  For tests:   For tests:  For tests:  For tests:  For tests:  For tests:  For tests: For tests:
// For tests:  For tests:   For tests:  For tests:  For tests:  For tests:  For tests:  For tests: For tests:
// For tests:  For tests:   For tests:  For tests:  For tests:  For tests:  For tests:  For tests: For tests:
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<TrackFP420> TrackProducerFP420::trackFinderVar1(const ClusterCollectionFP420 input){
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<TrackFP420> rhits;
  int restracks = 10;// max # tracks
  rhits.reserve(restracks); 
  rhits.clear();
  double Ax[10]; double Bx[10]; double Cx[10]; int Mx[10];
  double Ay[10]; double By[10]; double Cy[10]; int My[10];
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int zbeg = 1, zmax=3;// XY
  if( zn0==1){
              zmax=2; // Y
  }
  else if( zn0==2){
    zbeg = 2, zmax=3; // X
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int reshits = 15;// max # cl for every X and Y plane
  int nX[30], nY[30];// 20 NUMBER OF PLANES; nX,nY - # cl for every X and Y plane
  int uX[30], uY[30];// 20 NUMBER OF PLANES; nX,nY - current # cl used for every X and Y plane
  double zX[15][30], xX[15][30], wX[15][30];
  double zY[15][30], yY[15][30], wY[15][30];
  bool qX[15][30], qY[15][30];
  //   .
  int tx = 0; int txs1 = 0; int txss = 0;
  int ty = 0; int tys1 = 0; int tyss = 0;
  //   .
  for (int zside=zbeg; zside<zmax; zside++) {
    for (int sector=1; sector < sn0; sector++) {
      for (int zmodule=1; zmodule<pn0; zmodule++) {

	// index is a continues numbering of 3D detector of FP420
	int sScale = 2*(pn0-1);
	int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	//	unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) + 1;
	unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) ; // 0-19   --> 20 items

	double kplane = -(pn0-1)/2+(zmodule-1); 


    double zdiststat = 0.;
    if(sector==2) zdiststat = zD2;
    if(sector==3) zdiststat = zD3;
    double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
    //double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + (sector-1)*zUnit;  

	double pitch=0;
	if(zside==1){
	  pitch=pitchY;
	  zcurrent += (ZGapLDet+ZSiDetL/2);
	}
	if(zside==2){
	  pitch=pitchX;
	  zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
	}
	  //   .
	  //   GET CLUSTER collection  !!!!
	  //   .
//============================================================================================================ put into currentclust
  std::vector<ClusterFP420> currentclust;
	currentclust.clear();
	ClusterCollectionFP420::Range outputRange;
	outputRange = input.get(iu);
  // fill output in currentclust vector (for may be sorting? or other checks)
  ClusterCollectionFP420::ContainerIterator sort_begin = outputRange.first;
  ClusterCollectionFP420::ContainerIterator sort_end = outputRange.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
  //  std::sort(currentclust.begin(),currentclust.end());
    currentclust.push_back(*sort_begin);
  } // for

#ifdef debugvar1
  std::cout << "TrackProducerFP420: currentclust.size = " << currentclust.size() << std::endl; 
#endif
//============================================================================================================

  vector<ClusterFP420>::const_iterator simHitIter = currentclust.begin();
  vector<ClusterFP420>::const_iterator simHitIterEnd = currentclust.end();
  
  nY[ii] = 0;// # cl in every Y plane (max is reshits)
  nX[ii] = 0;// # cl in every X plane (max is reshits)
  uX[ii] = 0;// current used # cl in every X plane 
  uY[ii] = 0;// current used # cl in every X plane 
  // loop in #clusters
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    const ClusterFP420 icluster = *simHitIter;
    
    // fill vectors for track reconstruction
    
    // local - global systems with possible shift of every second plate:
    float dYYcur = dYY;
    float dXXcur = dXX;
    if (UseHalfPitchShiftInY== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii != 0)  dYYcur -= pitch/2.;
    }
    if (UseHalfPitchShiftInX== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii == 0)  dXXcur += pitch/2.;
    }
    
    
    if(zside ==1){
      nY[ii]++;		
      if(nY[ii]>reshits){
	nY[ii]=reshits;
	std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << " zcurrent=" << zY[nY[ii]-1][ii] << " ii= "  << ii << std::endl;
      }
      zY[nY[ii]-1][ii] = zcurrent;
      yY[nY[ii]-1][ii] = icluster.barycenter()*pitch;
      // go to global system:
      yY[nY[ii]-1][ii] = yY[nY[ii]-1][ii] - dYYcur; 
      wY[nY[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wY[nY[ii]-1][ii] *= wY[nY[ii]-1][ii];//reciprocal of the variance for each datapoint in y
      qY[nY[ii]-1][ii] = true;
      if(nY[ii]==reshits) break;
    }
    // X:
    else if(zside ==2){
      nX[ii]++;	
      if(nX[ii]>reshits){
	std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << std::endl;
	nX[ii]=reshits;
      }
      zX[nX[ii]-1][ii] = zcurrent;
      xX[nX[ii]-1][ii] = icluster.barycenter()*pitch;
      // go to global system:
      xX[nX[ii]-1][ii] =-(xX[nX[ii]-1][ii]+dXXcur); 
      wX[nX[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wX[nX[ii]-1][ii] *= wX[nX[ii]-1][ii];//reciprocal of the variance for each datapoint in y
      qX[nX[ii]-1][ii] = true;
      if(nX[ii]==reshits) break;
    }
  

  } // for loop in #clusters

    // Y:
    if(zside ==1){
	  if(nY[ii] != 0) {  
	    ++ty; if(sector==1) ++tys1; if(sector==(sn0-1)) ++tyss;
	  }	  
    }
    // X:
    else if(zside ==2){
	  if(nX[ii] != 0) {  
	    ++tx; if(sector==1) ++txs1; if(sector==(sn0-1)) ++txss;
	  }	  
    }
  //================================== end of for loops in continuius number iu:
      }   // for
    }   // for
  }   // for
#ifdef debugvar1
  std::cout << "trackFinderVar1: ty= " << ty << " tys1 = " << tys1 << " tyss = " << tyss << std::endl;
  std::cout << "trackFinderVar1: tx= " << tx << " txs1 = " << txs1 << " txss = " << txss << std::endl;
  std::cout << "============================================================" << std::endl;
#endif


  // float chiCutX =3000.;
  int numberXtracks=0; double sigma;
  double tocollection =-999999.,tocollection0 =-999999.,tocollection1 =-999999.,tocollection2 =-999999.;

  int nitMax=5;

  //   int nsigma=6, msigma=5;//max # iterations 
  double sigman=0.1, ssigma = 1.0, sigmam=0.15;
//  int nsigma=5, msigma=3, nitMax=10;//max # iterations 
//  double ssigma = 2.5; 

  for (int zside=zbeg; zside<zmax; ++zside) {
    // Y:
    if(zside ==1){
    }

    // X:
    else if(zside ==2){
      
      double tg0= 2./(800.+20.); // for X: 2cm/...    
#ifdef debugvar1
		    std::cout << " tg0= " << tg0 << std::endl;
#endif
      int qXcl[30], qXii[30], fip=0, niteration = 0;
      int rx = 0, rxs1 = 0, rxss = 0;
      bool SelectTracks = true;
      numberXtracks=0;  
      do {
      double fxX[30], fzX[30], fwX[30];
      int px = 0, pxs1 = 0, pxss = 0;
      bool NewStation = false, px1first = false;
#ifdef debugvar1
	    std::cout << "Start do: px= " << px << std::endl;
#endif
	for (int sector=1; sector < sn0; ++sector) {
#ifdef debugvar1
	    std::cout << "sector= " << sector << std::endl;
#endif
	  double tav=0., t1=0., t2=0., t=0., sm;
	  int stattimes=0;
	  if( sector != 1 ) {
	    NewStation = true;  
	  }
	  for (int zmodule=1; zmodule<pn0; ++zmodule) {
#ifdef debugvar1
	    std::cout << "zmodule= " << zmodule << " px= " << px << std::endl;
#endif

	    int sScale = 2*(pn0-1); 
	    // unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) + 1;
	    unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) ;
	    
#ifdef debugvar1
	    std::cout << " ii= " << ii << " nX[ii]= " << nX[ii] << " uX[ii]= " << uX[ii] << std::endl;
#endif
	    if(nX[ii]!=0  && uX[ii]!= nX[ii]) { 
	      
	      ++px; if(sector==1) ++pxs1; if(sector==(sn0-1)) ++pxss;
#ifdef debugvar1
	      std::cout << "just added: px= " << px << std::endl;
#endif
	      if(px==2 && sector==1) { 
		double dxmin=9999999., df2; int cl2=-1;
		for (int cl=0; cl<nX[ii]; ++cl) {
#ifdef debugvar1
		  std::cout << " cl= " << cl << " qX[cl][ii]= " << qX[cl][ii] << std::endl;
#endif
		  if(qX[cl][ii]){
		    df2 = abs(fxX[fip]-xX[cl][ii]);
#ifdef debugvar1
		    std::cout << " df2= " << df2 << " abs= " << abs(fxX[fip]-xX[cl][ii]) << std::endl;
#endif
		    if(df2 < dxmin) {
		      dxmin = df2;
		      cl2=cl;
#ifdef debugvar1
		      std::cout << " dxmin= " << dxmin << " inside:cl2= " << cl2 << std::endl;
#endif
		    }//if(df2		
		  }//if(qX		
		}//for(cl
#ifdef debugvar1
		std::cout << " cl2= " << cl2 << std::endl;
#endif
		if(cl2!=-1){
		  t=(xX[cl2][ii]-fxX[fip])/(zX[cl2][ii]-fzX[fip]);
		  t1 = t*wX[cl2][ii];
		  t2 = wX[cl2][ii];
#ifdef debugvar1
		  std::cout << " fxX[fip]= " << fxX[fip] << " xX[cl2][ii]= " << xX[cl2][ii] << std::endl;
		  std::cout << " zX[cl2][ii]= " << zX[cl2][ii] << " fzX[fip]= " << fzX[fip] << std::endl;
		  std::cout << " t1= " << t1 << " t2= " << t2 << " t= " << t << std::endl;
#endif
		  if(abs(t)<tg0) { 
		    qX[cl2][ii] = false;//point is taken, mark it for not using again
		    fxX[px-1]=xX[cl2][ii];
		    fzX[px-1]=zX[cl2][ii];
		    fwX[px-1]=wX[cl2][ii];
		    qXcl[px-1] = cl2;
		    qXii[px-1] = ii;
		    ++uX[ii];
		    if(uX[ii]==nX[ii]){
#ifdef debugvar1
		      std::cout << "was last cluster cl2= " << cl2 << " for ii= " << ii << std::endl;
#endif
		      ++rx; if(sector==1) ++rxs1; if(sector==(sn0-1)) ++rxss;
		    }//if(uX
		  }//if abs
		  else{
		    px--; if(sector==1) pxs1--;  if(sector==(sn0-1)) pxss--;
		    t1 -= t*wX[cl2][ii]; t2 -= wX[cl2][ii];
#ifdef debugvar1
		      std::cout << "2 t1!!!: just reduced px= " << px << std::endl;
#endif
		  }//if(abs
		}//if(cl2!=-1
		else{
		  px--; if(sector==1) pxs1--;  if(sector==(sn0-1)) pxss--;
#ifdef debugvar1
		      std::cout << "2:just reduced  px= " << px << std::endl;
#endif
		}//if(cl2!=-1
	      }//if(px==2
	      else {
		int clcurr=-1;
		for (int cl=0; cl<nX[ii]; ++cl) {
		  if(qX[cl][ii]){
		    clcurr = cl;
		    if(px<3 ){
#ifdef debugvar1
		      std::cout << " px<3:cl= " << cl << " ii= " << ii << " px= " << px << std::endl;
#endif
		      if(px==1) { 
			px1first = true;
			fip=px-1;
			qX[cl][ii] = false;//point is taken, mark it for not using again
			fxX[px-1]=xX[cl][ii];
			fzX[px-1]=zX[cl][ii];
			fwX[px-1]=wX[cl][ii];
			qXcl[px-1] = cl;
			qXii[px-1] = ii;
			++uX[ii];
#ifdef debugvar1
		      std::cout << " qX[cl][ii]= " << qX[cl][ii] << " fxX[fip]= " << fxX[fip] << std::endl;
		      std::cout << " px1first= " << px1first << " fzX[fip]= " << fzX[fip] << std::endl;
#endif
		      }//if px=1
		      if(uX[ii]==nX[ii]){
#ifdef debugvar1
			std::cout << "was last cluster cl= " << cl << " for ii= " << ii << std::endl;
#endif
			++rx; if(sector==1) ++rxs1; if(sector==(sn0-1)) ++rxss;
		      }//if(uX
		    }//px<3
		    else {
#ifdef debugvar1
		      std::cout << " px>=3:cl= " << cl << " ii= " << ii << std::endl;
#endif
		      if(NewStation){
			sigma = ssigma/(sn0-sector);
			//if(stattimes==1 || sector==3 ) sigma = msigma * sqrt(1./wX[cl][ii]);
			if(stattimes==1 || sector==3 ) sigma = sigmam;
			
			double cov00, cov01, cov11, c0X, c1X, chisqX;
#ifdef debugvar1
			for (int tr=0; tr<px-1; ++tr) {
			  std::cout << " for point = " << tr   << std::endl;
			  std::cout << " fxX= " << fxX[tr]   << std::endl;
			  std::cout << " fzX= " << fzX[tr]   << std::endl;
			  std::cout << " fwX= " << fwX[tr]   << std::endl;
			}
#endif
			gsl_fit_wlinear (fzX, 1, fwX, 1, fxX, 1, px-1, 
					 &c0X, &c1X, &cov00, &cov01, &cov11, 
					 &chisqX);
			sm = c0X+ c1X*zX[cl][ii];


			// to collect info abou fit:
			//   t=c1X;
			//  t1 += t*wX[cl][ii];
			//  t2 += wX[cl][ii];
			
			//t=(xX[cl][ii]-fxX[fip])/(zX[cl][ii]-fzX[fip]);
			//t1 += t*wX[cl][ii];
			//t2 += wX[cl][ii];
			//tav=t1/t2;
			//sm = fxX[fip]+tav*(zX[cl][ii]-fzX[fip]);
			//	double sm0 = fxX[fip]+tav*(zX[cl][ii]-fzX[fip]);

		      }//NewStation 1
		      else{
#ifdef debugvar1
			std::cout << "before: t= " << t << " t1= " << t1 << " t2= " << t2 << std::endl;
#endif
			t=(xX[cl][ii]-fxX[fip])/(zX[cl][ii]-fzX[fip]);
			t1 += t*wX[cl][ii];
			t2 += wX[cl][ii];
			tav=t1/t2;
			sm = fxX[fip]+tav*(zX[cl][ii]-fzX[fip]);
			//sigma = nsigma * sqrt(1./wX[cl][ii]);
			sigma = sigman;
		      }
#ifdef debugvar1
		      std::cout << " (xX[cl][ii]-fxX[fip])= " << (xX[cl][ii]-fxX[fip]) << " (zX[cl][ii]-fzX[fip])= " << (zX[cl][ii]-fzX[fip]) << " t*wX[cl][ii]= " << t*wX[cl][ii] << std::endl;
		      std::cout << " t= " << t << " t1= " << t1 << " t2= " << t2 << std::endl;
		      std::cout << " tav= " << tav << " fxX[fip]= " << fxX[fip] << std::endl;
		      std::cout << " zX[cl][ii]= " << zX[cl][ii] << " fzX[fip]= " << fzX[fip] << std::endl;
		      std::cout << " sm= " << sm << " xX[cl][ii]= " << xX[cl][ii] << std::endl;
		      std::cout << " sm0= " << fxX[fip]+tav*(zX[cl][ii]-fzX[fip]) << std::endl;
		      std::cout << " wX[cl][ii]= " << wX[cl][ii] << std::endl;
		      std::cout << " abs(xX[cl][ii]-sm)= " << abs(xX[cl][ii]-sm) << std::endl;
		      std::cout << " sigma= " << sigma << std::endl;
#endif
		      double diffpo = xX[cl][ii]-sm;
		      //Start tests:
		      if(stattimes==0 && !NewStation && sector==1) {tocollection=diffpo;}//before fit
		      if(stattimes==0 && NewStation && sector==2){tocollection0=diffpo;}//fit 1 sec. 2

		      // for 3 Station configuration set up
		      if(stattimes==0 && NewStation && sector==3){tocollection1=diffpo;}//fit 1 sec. 3
		      if(stattimes==2 && sector==3){tocollection2=diffpo;}//after fit in last section

		      // for 2 Station configuration set up
		      //		      if(stattimes==1 && sector==2){tocollection1=diffpo;}//fit 2
		      //                if(stattimes==2 && sector==2){tocollection2=diffpo;}//after fit
		      // End test

		      if(abs(diffpo) < sigma ) {
			if(NewStation){
			  ++stattimes;
			  if(stattimes==1) {
			    fip=px-1;
			    t1 = 0; t2 = 0;
#ifdef debugvar1
			    std::cout << "stattimes==1: fip= " << fip << std::endl;
			    std::cout << "stattimes==1: xX[cl][ii]= " << xX[cl][ii] << std::endl;
			    std::cout << "stattimes==1: zX[cl][ii]= " << zX[cl][ii] << std::endl;
#endif
			  }
			  else if(stattimes==2){
			    NewStation = false; 
			    t=(xX[cl][ii]-fxX[fip])/(zX[cl][ii]-fzX[fip]);
			    //t1 += t*wX[cl][ii];
			    //t2 += wX[cl][ii];
			    t1 = t*wX[cl][ii];
			    t2 = wX[cl][ii];
#ifdef debugvar1
			    std::cout << "stattimes==2: fip= " << fip << std::endl;
			    std::cout << "stattimes==2: xX[cl][ii]= " << xX[cl][ii] << std::endl;
			    std::cout << "stattimes==2: zX[cl][ii]= " << zX[cl][ii] << std::endl;
			    std::cout << "stattimes==2: t = " << t << std::endl;
#endif
			  }//if(stattime
			}//if(NewStation 2
			fxX[px-1]=xX[cl][ii];
			fzX[px-1]=zX[cl][ii];
			fwX[px-1]=wX[cl][ii];
			qX[cl][ii] = false;//point is taken, mark it for not using again
			qXcl[px-1] = cl;
			qXii[px-1] = ii;
			++uX[ii];
#ifdef debugvar1
			std::cout << "point is taken: cl= " << cl << " ii= " << ii << std::endl;
#endif
			if(uX[ii]==nX[ii]){
#ifdef debugvar1
			  std::cout << "was last cluster cl= " << cl << " for ii= " << ii << std::endl;
#endif
			  ++rx; if(sector==1) ++rxs1; if(sector==(sn0-1)) ++rxss;
			}//if(cl==
			//  break; // to go on next plane
		      }//if abs
		      else{
			t1 -= t*wX[cl][ii]; t2 -= wX[cl][ii];
		      }//if abs
		    }// if px<3 and else px>3
#ifdef debugvar1
		    std::cout << "Do break if !qX[cl][ii]:    qX[cl][ii]= " << qX[cl][ii] << std::endl;
#endif

		     if(!qX[cl][ii]) break;// go on next plane if point is found among clusters of current plane;
		  }// if qX
		}// for cl     --  can be break and return to "for zmodule"
#ifdef debugvar1
		std::cout << "End of for cl:px= " << px << " px1first= " << px1first << " clcurr= " << clcurr << std::endl;
		if(clcurr != -1) std::cout << " qX[clcurr][ii]= " << qX[clcurr][ii] << std::endl;
#endif
		if( (px!=1 && clcurr != -1 && qX[clcurr][ii]) || (px==1 && !px1first)) { 
		  // if point is not found - continue natural loop, but reduce px 
		  px--; if(sector==1) pxs1--;  if(sector==(sn0-1)) pxss--;
#ifdef debugvar1
		  std::cout << "px= " << px << std::endl;
#endif
		}//if(px!=1
	      }//if(px==2 else 
	    }//if(nX !=0	   : inside  this if( -  ask  ++px
	  }// for zmodule
	}// for sector
	//============
	
	
	
	// apply criteria for track selection: 
	// do not take track if 
	//if( pxs1<4 || pxss<4 || px<8 ){
	  //	if( pxs1<3 || pxss<2 || px<4 ){
	  	if( pxs1<2 || pxss<1 || px<3 ){
#ifdef debugvar1
	  std::cout << "do not take track: pxs1= " << pxs1 << " pxss= " << pxss << " px= " << px << std::endl;
#endif
	}
	// do fit:
	else{
#ifdef debugvar1
	  std::cout << "Take track!!!: pxs1= " << pxs1 << " pxss= " << pxss << " px= " << px << std::endl;
#endif
	  double cov00, cov01, cov11;
	  double c0X, c1X, chisqX;
	  gsl_fit_wlinear (fzX, 1, fwX, 1, fxX, 1, px, 
			   &c0X, &c1X, &cov00, &cov01, &cov11, 
			   &chisqX);
	  float chindfx;
	  if(px>2) {
	    chindfx = chisqX/(px-2);
	  }
	  else{
	    chindfx = chisqX;
	  }//px
	  
#ifdef debugvar1
	  std::cout << " chindfx= " << chindfx << " chiCutX= " << chiCutX << std::endl;
#endif
	  if(chindfx < chiCutX ) {
	    ++numberXtracks;
	    Ax[numberXtracks-1] = c0X; 
	    Bx[numberXtracks-1] = c1X; 
	    Cx[numberXtracks-1] = chisqX; 
	    Mx[numberXtracks-1] = px;
	    
	  }//chindfx
	}//  if else
	
	  // do not select tracks anymore if
#ifdef debugvar1
	std::cout << " numberXtracks= " << numberXtracks << std::endl;
	std::cout << " pxs1= " << pxs1 << " pxss = " << pxss << " px = " << px << std::endl;
	std::cout << " txs1= " << txs1 << " txss = " << txss << " tx = " << tx << std::endl;
	std::cout << " rxs1= " << rxs1 << " rxss = " << rxss << " rx = " << rx << std::endl;
	std::cout << " txs1-rxs1= " << txs1-rxs1 << " txss-rxss = " << txss-rxss << " tx-rx = " << tx-rx << std::endl;
	std::cout << "---------------------------------------------------------- " << std::endl;
#endif
	// let's decide: do we continue track finder procedure
	if( txs1-rxs1<3 || txss-rxss<2 || tx-rx<4  ){
	  SelectTracks = false;
	}
	else{
	  ++niteration;
#ifdef debugvar1
	  if(niteration > nitMax-1){
	    std::cout << "Next Zside, niteration >= " << niteration << std::endl;
	    std::cout << " pxs1= " << pxs1 << " pxss = " << pxss << " px = " << px << std::endl;
	    std::cout << " txs1= " << txs1 << " txss = " << txss << " tx = " << tx << std::endl;
	    std::cout << " rxs1= " << rxs1 << " rxss = " << rxss << " rx = " << rx << std::endl;
	    std::cout << " txs1-rxs1= " << txs1-rxs1 << " txss-rxss = " << txss-rxss << " tx-rx = " << tx-rx << std::endl;
	    std::cout << "---------------------------------------------------------- " << std::endl;
	  }
#endif
	}
	
      } while(SelectTracks && niteration < nitMax );      
      
      //============
      
    }// if zside
  }// for zside
  
  //============================================================================================================
  // match selected X and Y tracks to each other: tgphi=By/Bx->phi=artg(By/Bx); tgtheta=Bx/cosphi=By/sinphi->
  // min of |Bx/cosphi-By/sinphi| 
  
#ifdef debugvar1
  std::cout << " numberXtracks= " << numberXtracks << std::endl;
#endif
  //   rhits.push_back( TrackFP420(c0X,c1X,chisqX,nhitplanesY,c0Y,c1Y,chisqY,nhitplanesY) );
  if(numberXtracks>restracks){
    std::cout << "WARNING-ERROR:TrackproducerFP420: numberXtracks= " << numberXtracks <<" bigger reservated number of tracks" << std::endl;
    numberXtracks=restracks;
  }
  for (int tr=0; tr<numberXtracks; ++tr) {
#ifdef debugvar1
    std::cout << " for track tr= " << tr << std::endl;
    std::cout << " Ax= " << Ax[tr]   << std::endl;
    std::cout << " Bx= " << Bx[tr]   << std::endl;
    std::cout << " Cx= " << Cx[tr]   << std::endl;
    std::cout << " Mx= " << Mx[tr]   << std::endl;
#endif
    //    rhits.push_back( TrackFP420(Ax[tr],Bx[tr],Cx[tr],Mx[tr],Ax[tr],Bx[tr],Cx[tr],Mx[tr]) );
    int ttt = tocollection2*1000000.;
    rhits.push_back( TrackFP420(Ax[tr],Bx[tr],Cx[tr],Mx[tr],tocollection,tocollection0,tocollection1,ttt) );
    tocollection  =-999999.;tocollection0 =-999999.;tocollection1 =-999999.;tocollection2 =-999999.;
  }
  //============================================================================================================
  
  //============
  return rhits;
  //============
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<TrackFP420> TrackProducerFP420::trackFinderVar2(const ClusterCollectionFP420 input){
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<TrackFP420> rhits;
  int restracks = 10;// max # tracks
  rhits.reserve(restracks); 
  rhits.clear();
  double Ax[10]; double Bx[10]; double Cx[10]; int Mx[10];
  double Ay[10]; double By[10]; double Cy[10]; int My[10];
  double finxb[10]; double finxe[10]; double finzb[10]; double finze[10]; 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //   .
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int zbeg = 1, zmax=3;// XY
  if( zn0==1){
              zmax=2; // Y
  }
  else if( zn0==2){
    zbeg = 2, zmax=3; // X
  }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int reshits = 15;// max # cl for every X and Y plane
  int nX[30], nY[30];// 20 NUMBER OF PLANES; nX, nY - # cl for every X and Y plane
  int uX[30], uY[30];// 20 NUMBER OF PLANES; nX, nY - current # cl used for every X and Y plane
  double zX[15][30], xX[15][30], wX[15][30];
  double zY[15][30], yY[15][30], wY[15][30];
  bool qX[15][30], qY[15][30];
  //   .
  int tx = 0; int txs1 = 0; int txss = 0;
  int ty = 0; int tys1 = 0; int tyss = 0;
  //   .
  for (int zside=zbeg; zside<zmax; zside++) {
    for (int sector=1; sector < sn0; sector++) {
      for (int zmodule=1; zmodule<pn0; zmodule++) {

	// index is a continues numbering of 3D detector of FP420
	int sScale = 2*(pn0-1);
	int zScale=2;  unsigned int iu = sScale*(sector - 1)+zScale*(zmodule - 1)+zside;
	//	unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) + 1;
	unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) ; // 0-19   --> 20 items

	double kplane = -(pn0-1)/2+(zmodule-1); 


    double zdiststat = 0.;
    if(sector==2) zdiststat = zD2;
    if(sector==3) zdiststat = zD3;
    double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + zdiststat;  
    //double zcurrent = zinibeg +(ZSiStep-ZSiPlane)/2  + kplane*ZSiStep + (sector-1)*zUnit;  

	double pitch=0;
	if(zside==1){
	  pitch=pitchY;
	  zcurrent += (ZGapLDet+ZSiDetL/2);
	}
	if(zside==2){
	  pitch=pitchX;
	  zcurrent += (ZGapLDet+ZSiDetR/2)+ZSiPlane/2;
	}
	  //   .
	  //   GET CLUSTER collection  !!!!
	  //   .
//============================================================================================================ put into currentclust
  std::vector<ClusterFP420> currentclust;
	currentclust.clear();
	ClusterCollectionFP420::Range outputRange;
	outputRange = input.get(iu);
  // fill output in currentclust vector (for may be sorting? or other checks)
  ClusterCollectionFP420::ContainerIterator sort_begin = outputRange.first;
  ClusterCollectionFP420::ContainerIterator sort_end = outputRange.second;
  for ( ;sort_begin != sort_end; ++sort_begin ) {
  //  std::sort(currentclust.begin(),currentclust.end());
    currentclust.push_back(*sort_begin);
  } // for

#ifdef debugvar2
  std::cout << "TrackProducerFP420: currentclust.size = " << currentclust.size() << std::endl; 
#endif
//============================================================================================================

  vector<ClusterFP420>::const_iterator simHitIter = currentclust.begin();
  vector<ClusterFP420>::const_iterator simHitIterEnd = currentclust.end();
  
    if(zside ==1){
      nY[ii] = 0;// # cl in every Y plane (max is reshits)
      uY[ii] = 0;// current used # cl in every X plane 
    }
    else if(zside ==2){
      nX[ii] = 0;// # cl in every X plane (max is reshits)
      uX[ii] = 0;// current used # cl in every X plane 
    }
  // loop in #clusters
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    const ClusterFP420 icluster = *simHitIter;
    
    // fill vectors for track reconstruction
    
    // local - global systems with possible shift of every second plate:
    float dYYcur = dYY;
    float dXXcur = dXX;
    if (UseHalfPitchShiftInY== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii != 0)  dYYcur -= pitch/2.;
    }
    if (UseHalfPitchShiftInX== true){
      int iii = zmodule - 2*int(zmodule/2.);//   zmodule = 1,2,3,...10   -------   iii = 1,0,1,...0 
      if( iii == 0)  dXXcur += pitch/2.;
    }
    
    
    //disentangle complicated pattern recognition of hits?
    // Y:
    if(zside ==1){
      nY[ii]++;		
      if(nY[ii]>reshits){
	nY[ii]=reshits;
	std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << " zcurrent=" << zY[nY[ii]-1][ii] << " ii= "  << ii << std::endl;
      }
      zY[nY[ii]-1][ii] = zcurrent;
      yY[nY[ii]-1][ii] = icluster.barycenter()*pitch;
      // go to global system:
      yY[nY[ii]-1][ii] = yY[nY[ii]-1][ii] - dYYcur; 
      wY[nY[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wY[nY[ii]-1][ii] *= wY[nY[ii]-1][ii];//reciprocal of the variance for each datapoint in y
      qY[nY[ii]-1][ii] = true;
      if(nY[ii]==reshits) break;
    }
    // X:
    else if(zside ==2){
      nX[ii]++;	
      if(nX[ii]>reshits){
	std::cout << "WARNING-ERROR:TrackproducerFP420: currentclust.size()= " << currentclust.size() <<" bigger reservated number of hits" << std::endl;
	nX[ii]=reshits;
      }
      zX[nX[ii]-1][ii] = zcurrent;
      xX[nX[ii]-1][ii] = icluster.barycenter()*pitch;
      // go to global system:
      xX[nX[ii]-1][ii] =-(xX[nX[ii]-1][ii]+dXXcur); 
      wX[nX[ii]-1][ii] = 1./(icluster.barycerror()*pitch);//reciprocal of the variance for each datapoint in y
      wX[nX[ii]-1][ii] *= wX[nX[ii]-1][ii];//reciprocal of the variance for each datapoint in y
      qX[nX[ii]-1][ii] = true;
      if(nX[ii]==reshits) break;
    }

  } // for loop in #clusters (can be breaked)

    // Y:
    if(zside ==1){
	  if(nY[ii] != 0) {  
	    ++ty; if(sector==1) ++tys1; if(sector==(sn0-1)) ++tyss;
	  }	  
    }
    // X:
    else if(zside ==2){
	  if(nX[ii] != 0) { 
	    ++tx; if(sector==1) ++txs1; if(sector==(sn0-1)) ++txss;
	  }	  
    }
  //================================== end of for loops in continuius number iu:
      }   // for zmodule
    }   // for sector
  }   // for zside
#ifdef debugvar2
  std::cout << "trackFinderVar2: ty= " << ty << " tys1 = " << tys1 << " tyss = " << tyss << std::endl;
  std::cout << "trackFinderVar2: tx= " << tx << " txs1 = " << txs1 << " txss = " << txss << std::endl;
  std::cout << "============================================================" << std::endl;
#endif

  //===========================================================================================================================

  int nitMax=5;

  // float chiCutX =3.,chiCutY =3.;
  //  float chiCutX =3000.,chiCutY =3000.;
  int numberXtracks=0, numberYtracks=0; double sigma;

  //   int nsigma=6, msigma=5;//max # iterations 
  //double sigman=0.18, ssigma = 1.6, sigmam=0.18;
  double sigman=0.1, ssigma = 1.0, sigmam=0.15;
  //double ssigma = 2.5; 
  //    int nsigma=6, msigma=6, nitMax=10;//max # iterations 
  //  double ssigma = 2.5; 

//  int nsigma=5, msigma=3, nitMax=10;//max # iterations 
//  double ssigma = 2.5; 

  for (int zside=zbeg; zside<zmax; ++zside) {
    // Y:
    if(zside ==1){
  //===========================================================================================================================
      
      double tg0= 3*1./(800.+20.); // for Y: 1cm/...   *3 - 3sigma range
#ifdef debugvar2
		    std::cout << " tg0= " << tg0 << std::endl;
#endif
      int qYcl[30], qYii[30], fip=0, niteration = 0;
      int ry = 0, rys1 = 0, ryss = 0;
      bool SelectTracks = true;
      numberYtracks=0;  
      do {
      double fyY[30], fzY[30], fwY[30];
      int py = 0, pys1 = 0, pyss = 0;
      bool NewStation = false, py1first = false;
#ifdef debugvar2
	    std::cout << "Start do: py= " << py << std::endl;
#endif
	for (int sector=1; sector < sn0; ++sector) {
#ifdef debugvar2
	    std::cout << "sector= " << sector << std::endl;
#endif
	  double tav=0., t1=0., t2=0., t=0., sm;
	  int stattimes=0;
	  if( sector != 1 ) {
	    NewStation = true;  
	  }
	  for (int zmodule=1; zmodule<pn0; ++zmodule) {
#ifdef debugvar2
	    std::cout << "zmodule= " << zmodule << " py= " << py << std::endl;
#endif

	    int sScale = 2*(pn0-1); 
	    // unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) + 1;
	    unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) ;
	    
#ifdef debugvar2
	    std::cout << " ii= " << ii << " nY[ii]= " << nY[ii] << " uY[ii]= " << uY[ii] << std::endl;
#endif
	    if(nY[ii]!=0  && uY[ii]!= nY[ii]) { 
	      
	      ++py; if(sector==1) ++pys1; if(sector==(sn0-1)) ++pyss;
#ifdef debugvar2
	      std::cout << "just added: py= " << py << std::endl;
#endif
	      if(py==2 && sector==1) { 
		double dymin=9999999., df2; int cl2=-1;
		for (int cl=0; cl<nY[ii]; ++cl) {
#ifdef debugvar2
		  std::cout << " cl= " << cl << " qY[cl][ii]= " << qY[cl][ii] << std::endl;
#endif
		  if(qY[cl][ii]){
		    df2 = abs(fyY[fip]-yY[cl][ii]);
#ifdef debugvar2
		    std::cout << " df2= " << df2 << " abs= " << abs(fyY[fip]-yY[cl][ii]) << std::endl;
#endif
		    if(df2 < dymin) {
		      dymin = df2;
		      cl2=cl;
#ifdef debugvar2
		      std::cout << " dymin= " << dymin << " inside:cl2= " << cl2 << std::endl;
#endif
		    }//if(df2		
		  }//if(qY		
		}//for(cl
#ifdef debugvar2
		std::cout << " cl2= " << cl2 << std::endl;
#endif
		if(cl2!=-1){
		  t=(yY[cl2][ii]-fyY[fip])/(zY[cl2][ii]-fzY[fip]);
		  t1 = t*wY[cl2][ii];
		  t2 = wY[cl2][ii];
#ifdef debugvar2
		  std::cout << " fyY[fip]= " << fyY[fip] << " yY[cl2][ii]= " << yY[cl2][ii] << std::endl;
		  std::cout << " zY[cl2][ii]= " << zY[cl2][ii] << " fzY[fip]= " << fzY[fip] << std::endl;
		  std::cout << " t1= " << t1 << " t2= " << t2 << " t= " << t << std::endl;
#endif
		  if(abs(t)<tg0) { 
		    qY[cl2][ii] = false;//point is taken, mark it for not using again
		    fyY[py-1]=yY[cl2][ii];
		    fzY[py-1]=zY[cl2][ii];
		    fwY[py-1]=wY[cl2][ii];
		    qYcl[py-1] = cl2;
		    qYii[py-1] = ii;
		    ++uY[ii];
		    if(uY[ii]==nY[ii]){
#ifdef debugvar2
		      std::cout << "was last cluster cl2= " << cl2 << " for ii= " << ii << std::endl;
#endif
		      ++ry; if(sector==1) ++rys1; if(sector==(sn0-1)) ++ryss;
		    }//if(uY
		  }//if abs
		  else{
		    py--; if(sector==1) pys1--;  if(sector==(sn0-1)) pyss--;
		    t1 -= t*wY[cl2][ii]; t2 -= wY[cl2][ii];
#ifdef debugvar2
		      std::cout << "2 t1!!!: just reduced py= " << py << std::endl;
#endif
		  }//if(abs
		}//if(cl2!=-1
		else{
		  py--; if(sector==1) pys1--;  if(sector==(sn0-1)) pyss--;
#ifdef debugvar2
		      std::cout << "2:just reduced  py= " << py << std::endl;
#endif
		}//if(cl2!=-1
	      }//if(py==2
	      else {
		int clcurr=-1;
		for (int cl=0; cl<nY[ii]; ++cl) {
		  if(qY[cl][ii]){
		    clcurr = cl;
		    if(py<3 ){
#ifdef debugvar2
		      std::cout << " py<3:cl= " << cl << " ii= " << ii << " py= " << py << std::endl;
#endif
		      if(py==1) { 
			py1first = true;
			fip=py-1;
			qY[cl][ii] = false;//point is taken, mark it for not using again
			fyY[py-1]=yY[cl][ii];
			fzY[py-1]=zY[cl][ii];
			fwY[py-1]=wY[cl][ii];
			qYcl[py-1] = cl;
			qYii[py-1] = ii;
			++uY[ii];
#ifdef debugvar2
		      std::cout << " qY[cl][ii]= " << qY[cl][ii] << " fyY[fip]= " << fyY[fip] << std::endl;
		      std::cout << " py1first= " << py1first << " fzY[fip]= " << fzY[fip] << std::endl;
#endif
		      }//if py=1
		      if(uY[ii]==nY[ii]){
#ifdef debugvar2
			std::cout << "was last cluster cl= " << cl << " for ii= " << ii << std::endl;
#endif
			++ry; if(sector==1) ++rys1; if(sector==(sn0-1)) ++ryss;
		      }//if(uY
		    }//py<3
		    else {
#ifdef debugvar2
		      std::cout << " py>=3:cl= " << cl << " ii= " << ii << std::endl;
#endif
		      if(NewStation){
			sigma = ssigma/(sn0-sector);
			//if(stattimes==1 || sector==3 ) sigma = msigma * sqrt(1./wY[cl][ii]);
			if(stattimes==1 || sector==3 ) sigma = sigmam;
			
			double cov00, cov01, cov11, c0Y, c1Y, chisqY;
#ifdef debugvar2
			for (int tr=0; tr<py-1; ++tr) {
			  std::cout << " for point = " << tr   << std::endl;
			  std::cout << " fyY= " << fyY[tr]   << std::endl;
			  std::cout << " fzY= " << fzY[tr]   << std::endl;
			  std::cout << " fwY= " << fwY[tr]   << std::endl;
			}
#endif
			gsl_fit_wlinear (fzY, 1, fwY, 1, fyY, 1, py-1, 
					 &c0Y, &c1Y, &cov00, &cov01, &cov11, 
					 &chisqY);
			sm = c0Y+ c1Y*zY[cl][ii];


		      }//NewStation 1
		      else{
#ifdef debugvar2
			std::cout << "before: t= " << t << " t1= " << t1 << " t2= " << t2 << std::endl;
#endif
			t=(yY[cl][ii]-fyY[fip])/(zY[cl][ii]-fzY[fip]);
			t1 += t*wY[cl][ii];
			t2 += wY[cl][ii];
			tav=t1/t2;
			sm = fyY[fip]+tav*(zY[cl][ii]-fzY[fip]);
			//sigma = nsigma * sqrt(1./wY[cl][ii]);
			sigma = sigman;
		      }
#ifdef debugvar2
		      std::cout << " (yY[cl][ii]-fyY[fip])= " << (yY[cl][ii]-fyY[fip]) << " (zY[cl][ii]-fzY[fip])= " << (zY[cl][ii]-fzY[fip]) << " t*wY[cl][ii]= " << t*wY[cl][ii] << std::endl;
		      std::cout << " t= " << t << " t1= " << t1 << " t2= " << t2 << std::endl;
		      std::cout << " tav= " << tav << " fyY[fip]= " << fyY[fip] << std::endl;
		      std::cout << " zY[cl][ii]= " << zY[cl][ii] << " fzY[fip]= " << fzY[fip] << std::endl;
		      std::cout << " sm= " << sm << " yY[cl][ii]= " << yY[cl][ii] << std::endl;
		      std::cout << " sm0= " << fyY[fip]+tav*(zY[cl][ii]-fzY[fip]) << std::endl;
		      std::cout << " wY[cl][ii]= " << wY[cl][ii] << std::endl;
		      std::cout << " abs(yY[cl][ii]-sm)= " << abs(yY[cl][ii]-sm) << std::endl;
		      std::cout << " sigma= " << sigma << std::endl;
#endif
		      double diffpo = yY[cl][ii]-sm;

		      if(abs(diffpo) < sigma ) {
			if(NewStation){
			  ++stattimes;
			  if(stattimes==1) {
			    fip=py-1;
			    t1 = 0; t2 = 0;
#ifdef debugvar2
			    std::cout << "stattimes==1: fip= " << fip << std::endl;
			    std::cout << "stattimes==1: yY[cl][ii]= " << yY[cl][ii] << std::endl;
			    std::cout << "stattimes==1: zY[cl][ii]= " << zY[cl][ii] << std::endl;
#endif
			  }
			  else if(stattimes==2){
			    NewStation = false; 
			    t=(yY[cl][ii]-fyY[fip])/(zY[cl][ii]-fzY[fip]);
			    //t1 += t*wY[cl][ii];
			    //t2 += wY[cl][ii];
			    t1 = t*wY[cl][ii];
			    t2 = wY[cl][ii];
#ifdef debugvar2
			    std::cout << "stattimes==2: fip= " << fip << std::endl;
			    std::cout << "stattimes==2: yY[cl][ii]= " << yY[cl][ii] << std::endl;
			    std::cout << "stattimes==2: zY[cl][ii]= " << zY[cl][ii] << std::endl;
			    std::cout << "stattimes==2: t = " << t << std::endl;
#endif
			  }//if(stattime
			}//if(NewStation 2
			fyY[py-1]=yY[cl][ii];
			fzY[py-1]=zY[cl][ii];
			fwY[py-1]=wY[cl][ii];
			qY[cl][ii] = false;//point is taken, mark it for not using again
			qYcl[py-1] = cl;
			qYii[py-1] = ii;
			++uY[ii];
#ifdef debugvar2
			std::cout << "point is taken: cl= " << cl << " ii= " << ii << std::endl;
#endif
			if(uY[ii]==nY[ii]){
#ifdef debugvar2
			  std::cout << "was last cluster cl= " << cl << " for ii= " << ii << std::endl;
#endif
			  ++ry; if(sector==1) ++rys1; if(sector==(sn0-1)) ++ryss;
			}//if(cl==
			//  break; // to go on neyt plane
		      }//if abs
		      else{
			t1 -= t*wY[cl][ii]; t2 -= wY[cl][ii];
		      }//if abs
		    }// if py<3 and else py>3
#ifdef debugvar2
		    std::cout << "Do break if !qY[cl][ii]:    qY[cl][ii]= " << qY[cl][ii] << std::endl;
#endif

		     if(!qY[cl][ii]) break;// go on neyt plane if point is found among clusters of current plane;
		  }// if qY
		}// for cl     --  can be break and return to "for zmodule"
#ifdef debugvar2
		std::cout << "End of for cl:py= " << py << " py1first= " << py1first << " clcurr= " << clcurr << std::endl;
		if(clcurr != -1) std::cout << " qY[clcurr][ii]= " << qY[clcurr][ii] << std::endl;
#endif
		if( (py!=1 && clcurr != -1 && qY[clcurr][ii]) || (py==1 && !py1first)) { 
		  // if point is not found - continue natural loop, but reduce py 
		  py--; if(sector==1) pys1--;  if(sector==(sn0-1)) pyss--;
#ifdef debugvar2
		  std::cout << "py= " << py << std::endl;
#endif
		}//if(py!=1
	      }//if(py==2 else 
	    }//if(nY !=0	   : inside  this if( -  ask  ++py
	  }// for zmodule
	}// for sector
	//============
	
	
	
	// apply criteria for track selection: 
	// do not take track if 
	if( pys1<4 || pyss<4 || py<8 ){
	  //	if( pys1<3 || pyss<2 || py<4 ){
#ifdef debugvar2
	  std::cout << "do not take track: pys1= " << pys1 << " pyss= " << pyss << " py= " << py << std::endl;
#endif
	}
	// do fit:
	else{
#ifdef debugvar2
	  std::cout << "Take track!!!: pys1= " << pys1 << " pyss= " << pyss << " py= " << py << std::endl;
#endif
	  double cov00, cov01, cov11;
	  double c0Y, c1Y, chisqY;
	  gsl_fit_wlinear (fzY, 1, fwY, 1, fyY, 1, py, 
			   &c0Y, &c1Y, &cov00, &cov01, &cov11, 
			   &chisqY);
	  float chindfy;
	  if(py>2) {
	    chindfy = chisqY/(py-2);
	  }
	  else{
	    chindfy = chisqY;
	  }//py
	  
#ifdef debugvar2
	  std::cout << " chindfy= " << chindfy << " chiCutY= " << chiCutY << std::endl;
#endif
	  if(chindfy < chiCutY ) {
	    ++numberYtracks;
	    Ay[numberYtracks-1] = c0Y; 
	    By[numberYtracks-1] = c1Y; 
	    Cy[numberYtracks-1] = chisqY; 
	    My[numberYtracks-1] = py;
	    
	  }//chindfy
	}//  if else
	
	  // do not select tracks anymore if
#ifdef debugvar2
	std::cout << " numberYtracks= " << numberYtracks << std::endl;
	std::cout << " pys1= " << pys1 << " pyss = " << pyss << " py = " << py << std::endl;
	std::cout << " tys1= " << tys1 << " tyss = " << tyss << " ty = " << ty << std::endl;
	std::cout << " rys1= " << rys1 << " ryss = " << ryss << " ry = " << ry << std::endl;
	std::cout << " tys1-rys1= " << tys1-rys1 << " tyss-ryss = " << tyss-ryss << " ty-ry = " << ty-ry << std::endl;
	std::cout << "---------------------------------------------------------- " << std::endl;
#endif
	// let's decide: do we continue track finder procedure
	if( tys1-rys1<3 || tyss-ryss<2 || ty-ry<4  ){
	  SelectTracks = false;
	}
	else{
	  ++niteration;
#ifdef debugvar2
	  if(niteration > nitMax-1){
	    std::cout << "Neyt iteration, niteration >= " << niteration << std::endl;
	    std::cout << " pys1= " << pys1 << " pyss = " << pyss << " py = " << py << std::endl;
	    std::cout << " tys1= " << tys1 << " tyss = " << tyss << " ty = " << ty << std::endl;
	    std::cout << " rys1= " << rys1 << " ryss = " << ryss << " ry = " << ry << std::endl;
	    std::cout << " tys1-rys1= " << tys1-rys1 << " tyss-ryss = " << tyss-ryss << " ty-ry = " << ty-ry << std::endl;
	    std::cout << "---------------------------------------------------------- " << std::endl;
	  }
#endif
	}
	
      } while(SelectTracks && niteration < nitMax );      
      
      //============
      
  //===========================================================================================================================
    }

    // X:
    else if(zside ==2){
  //===========================================================================================================================
      
      double tg0= 3*2./(800.+20.); // for X: 2cm/...   *3 - 3sigma range
# ifdef debugvar2
		    std::cout << " tg0= " << tg0 << std::endl;
#endif
      int qXcl[30], qXii[30], fip=0, niteration = 0;
      int rx = 0, rxs1 = 0, rxss = 0;
      bool SelectTracks = true;
      numberXtracks=0;  
      do {
      double fxX[30], fzX[30], fwX[30];
      int px = 0, pxs1 = 0, pxss = 0;
      bool NewStation = false, px1first = false;
#ifdef debugvar2
	    std::cout << "Start do: px= " << px << std::endl;
#endif
	for (int sector=1; sector < sn0; ++sector) {
#ifdef debugvar2
	    std::cout << "sector= " << sector << std::endl;
#endif
	  double tav=0., t1=0., t2=0., t=0., sm;
	  int stattimes=0;
	  if( sector != 1 ) {
	    NewStation = true;  
	  }
	  for (int zmodule=1; zmodule<pn0; ++zmodule) {
#ifdef debugvar2
	    std::cout << "zmodule= " << zmodule << " px= " << px << std::endl;
#endif

	    int sScale = 2*(pn0-1); 
	    // unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) + 1;
	    unsigned int ii = sScale*(sector - 1)/2 + (zmodule - 1) ;
	    
#ifdef debugvar2
	    std::cout << " ii= " << ii << " nX[ii]= " << nX[ii] << " uX[ii]= " << uX[ii] << std::endl;
#endif
	    if(nX[ii]!=0  && uX[ii]!= nX[ii]) { 
	      
	      ++px; if(sector==1) ++pxs1; if(sector==(sn0-1)) ++pxss;
#ifdef debugvar2
	      std::cout << "just added: px= " << px << std::endl;
#endif
	      if(px==2 && sector==1) { 
		double dxmin=9999999., df2; int cl2=-1;
		for (int cl=0; cl<nX[ii]; ++cl) {
#ifdef debugvar2
		  std::cout << " cl= " << cl << " qX[cl][ii]= " << qX[cl][ii] << std::endl;
#endif
		  if(qX[cl][ii]){
		    df2 = abs(fxX[fip]-xX[cl][ii]);
#ifdef debugvar2
		    std::cout << " df2= " << df2 << " abs= " << abs(fxX[fip]-xX[cl][ii]) << std::endl;
#endif
		    if(df2 < dxmin) {
		      dxmin = df2;
		      cl2=cl;
#ifdef debugvar2
		      std::cout << " dxmin= " << dxmin << " inside:cl2= " << cl2 << std::endl;
#endif
		    }//if(df2		
		  }//if(qX		
		}//for(cl
#ifdef debugvar2
		std::cout << " cl2= " << cl2 << std::endl;
#endif
		if(cl2!=-1){
		  t=(xX[cl2][ii]-fxX[fip])/(zX[cl2][ii]-fzX[fip]);
		  t1 = t*wX[cl2][ii];
		  t2 = wX[cl2][ii];
#ifdef debugvar2
		  std::cout << " fxX[fip]= " << fxX[fip] << " xX[cl2][ii]= " << xX[cl2][ii] << std::endl;
		  std::cout << " zX[cl2][ii]= " << zX[cl2][ii] << " fzX[fip]= " << fzX[fip] << std::endl;
		  std::cout << " t1= " << t1 << " t2= " << t2 << " t= " << t << std::endl;
#endif
		  if(abs(t)<tg0) { 
		    qX[cl2][ii] = false;//point is taken, mark it for not using again
		    fxX[px-1]=xX[cl2][ii];
		    fzX[px-1]=zX[cl2][ii];
		    fwX[px-1]=wX[cl2][ii];
		    qXcl[px-1] = cl2;
		    qXii[px-1] = ii;
		    ++uX[ii];
		    if(uX[ii]==nX[ii])
#ifdef debugvar2
		      std::cout << "was last cluster cl2= " << cl2 << " for ii= " << ii << std::endl;
#endif
		      ++rx; if(sector==1) ++rxs1; if(sector==(sn0-1)) ++rxss;
		    }//if(uX
		  }//if abs
		  else{
		    px--; if(sector==1) pxs1--;  if(sector==(sn0-1)) pxss--;
		    t1 -= t*wX[cl2][ii]; t2 -= wX[cl2][ii];
#ifdef debugvar2
		      std::cout << "2 t1!!!: just reduced px= " << px << std::endl;
#endif
		  }//if(abs
		}//if(cl2!=-1
		else{
		  px--; if(sector==1) pxs1--;  if(sector==(sn0-1)) pxss--;
#ifdef debugvar2
		      std::cout << "2:just reduced  px= " << px << std::endl;
#endif
		}//if(cl2!=-1
	      }//if(px==2
	      else {
		int clcurr=-1;
		for (int cl=0; cl<nX[ii]; ++cl) {
		  if(qX[cl][ii]){
		    clcurr = cl;
		    if(px<3 ){
#ifdef debugvar2
		      std::cout << " px<3:cl= " << cl << " ii= " << ii << " px= " << px << std::endl;
#endif
		      if(px==1) { 
			px1first = true;
			fip=px-1;
			qX[cl][ii] = false;//point is taken, mark it for not using again
			fxX[px-1]=xX[cl][ii];
			fzX[px-1]=zX[cl][ii];
			fwX[px-1]=wX[cl][ii];
			qXcl[px-1] = cl;
			qXii[px-1] = ii;
			++uX[ii];
#ifdef debugvar2
		      std::cout << " qX[cl][ii]= " << qX[cl][ii] << " fxX[fip]= " << fxX[fip] << std::endl;
		      std::cout << " px1first= " << px1first << " fzX[fip]= " << fzX[fip] << std::endl;
#endif
		      }//if px=1
		      if(uX[ii]==nX[ii]){
#ifdef debugvar2
			std::cout << "was last cluster cl= " << cl << " for ii= " << ii << std::endl;
#endif
			++rx; if(sector==1) ++rxs1; if(sector==(sn0-1)) ++rxss;
		      }//if(uX
		    }//px<3
		    else {
#ifdef debugvar2
		      std::cout << " px>=3:cl= " << cl << " ii= " << ii << std::endl;
#endif
		      if(NewStation){
			sigma = ssigma/(sn0-sector);
			//if(stattimes==1 || sector==3 ) sigma = msigma * sqrt(1./wX[cl][ii]);
			if(stattimes==1 || sector==3 ) sigma = sigmam;
			
			double cov00, cov01, cov11, c0X, c1X, chisqX;
#ifdef debugvar2
			for (int tr=0; tr<px-1; ++tr) {
			  std::cout << " for point = " << tr   << std::endl;
			  std::cout << " fxX= " << fxX[tr]   << std::endl;
			  std::cout << " fzX= " << fzX[tr]   << std::endl;
			  std::cout << " fwX= " << fwX[tr]   << std::endl;
			}
#endif
			gsl_fit_wlinear (fzX, 1, fwX, 1, fxX, 1, px-1, 
					 &c0X, &c1X, &cov00, &cov01, &cov11, 
					 &chisqX);
			sm = c0X+ c1X*zX[cl][ii];



		      }//NewStation 1
		      else{
#ifdef debugvar2
			std::cout << "before: t= " << t << " t1= " << t1 << " t2= " << t2 << std::endl;
#endif
			t=(xX[cl][ii]-fxX[fip])/(zX[cl][ii]-fzX[fip]);
			t1 += t*wX[cl][ii];
			t2 += wX[cl][ii];
			tav=t1/t2;
			sm = fxX[fip]+tav*(zX[cl][ii]-fzX[fip]);
			//sigma = nsigma * sqrt(1./wX[cl][ii]);
			sigma = sigman;
		      }
#ifdef debugvar2
		      std::cout << " (xX[cl][ii]-fxX[fip])= " << (xX[cl][ii]-fxX[fip]) << " (zX[cl][ii]-fzX[fip])= " << (zX[cl][ii]-fzX[fip]) << " t*wX[cl][ii]= " << t*wX[cl][ii] << std::endl;
		      std::cout << " t= " << t << " t1= " << t1 << " t2= " << t2 << std::endl;
		      std::cout << " tav= " << tav << " fxX[fip]= " << fxX[fip] << std::endl;
		      std::cout << " zX[cl][ii]= " << zX[cl][ii] << " fzX[fip]= " << fzX[fip] << std::endl;
		      std::cout << " sm= " << sm << " xX[cl][ii]= " << xX[cl][ii] << std::endl;
		      std::cout << " sm0= " << fxX[fip]+tav*(zX[cl][ii]-fzX[fip]) << std::endl;
		      std::cout << " wX[cl][ii]= " << wX[cl][ii] << std::endl;
		      std::cout << " abs(xX[cl][ii]-sm)= " << abs(xX[cl][ii]-sm) << std::endl;
		      std::cout << " sigma= " << sigma << std::endl;
#endif
		      double diffpo = xX[cl][ii]-sm;


		      if(abs(diffpo) < sigma ) {
			if(NewStation){
			  ++stattimes;
			  if(stattimes==1) {
			    fip=px-1;
			    t1 = 0; t2 = 0;
#ifdef debugvar2
			    std::cout << "stattimes==1: fip= " << fip << std::endl;
			    std::cout << "stattimes==1: xX[cl][ii]= " << xX[cl][ii] << std::endl;
			    std::cout << "stattimes==1: zX[cl][ii]= " << zX[cl][ii] << std::endl;
#endif
			  }
			  else if(stattimes==2){
			    NewStation = false; 
			    t=(xX[cl][ii]-fxX[fip])/(zX[cl][ii]-fzX[fip]);
			    //t1 += t*wX[cl][ii];
			    //t2 += wX[cl][ii];
			    t1 = t*wX[cl][ii];
			    t2 = wX[cl][ii];
#ifdef debugvar2
			    std::cout << "stattimes==2: fip= " << fip << std::endl;
			    std::cout << "stattimes==2: xX[cl][ii]= " << xX[cl][ii] << std::endl;
			    std::cout << "stattimes==2: zX[cl][ii]= " << zX[cl][ii] << std::endl;
			    std::cout << "stattimes==2: t = " << t << std::endl;
#endif
			  }//if(stattime
			}//if(NewStation 2
			fxX[px-1]=xX[cl][ii];
			fzX[px-1]=zX[cl][ii];
			fwX[px-1]=wX[cl][ii];
			qX[cl][ii] = false;//point is taken, mark it for not using again
			qXcl[px-1] = cl;
			qXii[px-1] = ii;
			++uX[ii];
#ifdef debugvar2
			std::cout << "point is taken: cl= " << cl << " ii= " << ii << std::endl;
#endif
			if(uX[ii]==nX[ii]){
#ifdef debugvar2
			  std::cout << "was last cluster cl= " << cl << " for ii= " << ii << std::endl;
#endif
			  ++rx; if(sector==1) ++rxs1; if(sector==(sn0-1)) ++rxss;
			}//if(cl==
			//  break; // to go on next plane
		      }//if abs
		      else{
			t1 -= t*wX[cl][ii]; t2 -= wX[cl][ii];
		      }//if abs
		    }// if px<3 and else px>3
#ifdef debugvar2
		    std::cout << "Do break if !qX[cl][ii]:    qX[cl][ii]= " << qX[cl][ii] << std::endl;
#endif

		     if(!qX[cl][ii]) break;// go on next plane if point is found among clusters of current plane;
		  }// if qX
		}// for cl     --  can be break and return to "for zmodule"
#ifdef debugvar2
		std::cout << "End of for cl:px= " << px << " px1first= " << px1first << " clcurr= " << clcurr << std::endl;
		if(clcurr != -1) std::cout << " qX[clcurr][ii]= " << qX[clcurr][ii] << std::endl;
#endif
		if( (px!=1 && clcurr != -1 && qX[clcurr][ii]) || (px==1 && !px1first)) { 
		  // if point is not found - continue natural loop, but reduce px 
		  px--; if(sector==1) pxs1--;  if(sector==(sn0-1)) pxss--;
#ifdef debugvar2
		  std::cout << "px= " << px << std::endl;
#endif
		}//if(px!=1
	      }//if(px==2 else 
	    }//if(nX !=0	   : inside  this if( -  ask  ++px
	  }// for zmodule
	}// for sector
	//============
	
	
	
	// apply criteria for track selection: 
	// do not take track if 
	if( pxs1<4 || pxss<4 || px<8 ){
	  //	if( pxs1<3 || pxss<2 || px<4 ){
#ifdef debugvar2
	  std::cout << "do not take track: pxs1= " << pxs1 << " pxss= " << pxss << " px= " << px << std::endl;
#endif
	}
	// do fit:
	else{
#ifdef debugvar2
	  std::cout << "Take track!!!: pxs1= " << pxs1 << " pxss= " << pxss << " px= " << px << std::endl;
#endif
	  double cov00, cov01, cov11;
	  double c0X, c1X, chisqX;
	  gsl_fit_wlinear (fzX, 1, fwX, 1, fxX, 1, px, 
			   &c0X, &c1X, &cov00, &cov01, &cov11, 
			   &chisqX);
	  float chindfx;
	  if(px>2) {
	    chindfx = chisqX/(px-2);
	  }
	  else{
	    chindfx = chisqX;
	  }//px
	  
#ifdef debugvar2
	  std::cout << " chindfx= " << chindfx << " chiCutX= " << chiCutX << std::endl;
#endif
	  if(chindfx < chiCutX ) {
	    ++numberXtracks;
	    Ax[numberXtracks-1] = c0X; 
	    Bx[numberXtracks-1] = c1X; 
	    Cx[numberXtracks-1] = chisqX; 
	    Mx[numberXtracks-1] = px;
	    finxb[numberXtracks-1]=fxX[0];
	    finxe[numberXtracks-1]=fxX[px-1];
	    finzb[numberXtracks-1]=fzX[0];
	    finze[numberXtracks-1]=fzX[px-1];
	    
	  }//chindfx
	}//  if else
	
	  // do not select tracks anymore if
#ifdef debugvar2
	std::cout << " numberXtracks= " << numberXtracks << std::endl;
	std::cout << " pxs1= " << pxs1 << " pxss = " << pxss << " px = " << px << std::endl;
	std::cout << " txs1= " << txs1 << " txss = " << txss << " tx = " << tx << std::endl;
	std::cout << " rxs1= " << rxs1 << " rxss = " << rxss << " rx = " << rx << std::endl;
	std::cout << " txs1-rxs1= " << txs1-rxs1 << " txss-rxss = " << txss-rxss << " tx-rx = " << tx-rx << std::endl;
	std::cout << "---------------------------------------------------------- " << std::endl;
#endif
	// let's decide: do we continue track finder procedure
	if( txs1-rxs1<3 || txss-rxss<2 || tx-rx<4  ){
	  SelectTracks = false;
	}
	else{
	  ++niteration;
#ifdef debugvar2
	  if(niteration > nitMax-1){
	    std::cout << "Next iteration, niteration >= " << niteration << std::endl;
	    std::cout << " pxs1= " << pxs1 << " pxss = " << pxss << " px = " << px << std::endl;
	    std::cout << " txs1= " << txs1 << " txss = " << txss << " tx = " << tx << std::endl;
	    std::cout << " rxs1= " << rxs1 << " rxss = " << rxss << " rx = " << rx << std::endl;
	    std::cout << " txs1-rxs1= " << txs1-rxs1 << " txss-rxss = " << txss-rxss << " tx-rx = " << tx-rx << std::endl;
	    std::cout << "---------------------------------------------------------- " << std::endl;
	  }
#endif
	}
	
      } while(SelectTracks && niteration < nitMax );      
      
      //============
      
  //===========================================================================================================================
    }// if zside
  }// for zside
  
#ifdef debugvar2
  std::cout << " numberXtracks= " << numberXtracks << " numberYtracks= " << numberYtracks << std::endl;
#endif
  //===========================================================================================================================
  //
  //===========================================================================================================================
  // match selected X and Y tracks to each other: tgphi=By/Bx->phi=artg(By/Bx); tgtheta=Bx/cosphi=By/sinphi->  ================
  //                min of |Bx/cosphi-By/sinphi|                                                               ================

  //  
#ifdef debugvar22
      std::cout << " numberXtracks= " << numberXtracks << " numberYtracks= " << numberYtracks << std::endl;
#endif
      double yyyvtx = 0.0, xxxvtx = -22;  //mm
      if(numberXtracks>0) {
	int newxnum[10], newynum[10];// max # tracks = restracks = 10
	int nmathed=0;
	do {
	  double dthmin= 999999.; 
	  int trminx=-1, trminy=-1;
	  for (int trx=0; trx<numberXtracks; ++trx) {
#ifdef debugvar22
	    std::cout << "----------- trx= " << trx << " nmathed= " << nmathed << std::endl;
#endif
	    for (int tr=0; tr<numberYtracks; ++tr) {
#ifdef debugvar22
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
		//--------------------------------------------------------------------	
		double yyyyyy = 999999.;
		//if(Bx[trx] != 0.) yyyyyy = Ay[tr]-Ax[trx]*By[tr]/Bx[trx]+xxxvtx*By[tr]/Bx[trx];
		if(Bx[trx] != 0.) yyyyyy = Ay[tr]-(Ax[trx]-xxxvtx)*By[tr]/Bx[trx];
		double xxxxxx = 999999.;
		//if(By[tr] != 0.) xxxxxx = Ax[trx]-Ay[tr]*Bx[trx]/By[tr]+yyyvtx*Bx[trx]/By[tr];
		if(By[tr] != 0.) xxxxxx = Ax[trx]-(Ay[tr]-yyyvtx)*Bx[trx]/By[tr];
			double  dthdif= abs(yyyyyy-yyyvtx) + abs(xxxxxx-xxxvtx);
			//	double  dthdif= abs(xxxxxx-xxxvtx);
#ifdef debugvar22
		  std::cout << " yyyyyy= " << yyyyyy << " xxxxxx= " << xxxxxx << " dthdif= " << dthdif << std::endl;
#endif
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
	  newxnum[nmathed-1] = trminx;
#ifdef debugvar22
	  std::cout << " trminx= " << trminx << std::endl;
#endif
	  if(nmathed>numberYtracks){
	    newynum[nmathed-1] = -1;
#ifdef debugvar22
	  std::cout << "!!!  nmathed= " << nmathed << " > numberYtracks= " << numberYtracks << std::endl;
#endif
	  }
	  else {
#ifdef debugvar22
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
#ifdef debugvar2
      std::cout << " for track tr= " << tr << " tx= " << tx << " ty= " << ty << std::endl;
      std::cout << " Ax= " << Ax[tx]   << " Ay= " << Ay[ty]   << std::endl;
      std::cout << " Bx= " << Bx[tx]   << " By= " << By[ty]   << std::endl;
      std::cout << " Cx= " << Cx[tx]   << " Cy= " << Cy[ty]   << std::endl;
      std::cout << " Mx= " << Mx[tx]   << " My= " << My[ty]   << std::endl;
#endif
#ifdef debugvar22
      std::cout << " for track tr= " << tr << " tx= " << tx << " ty= " << ty << std::endl;
      std::cout << " Ax= " << Ax[tx]   << " Ay= " << Ay[ty]   << std::endl;
      std::cout << " Bx= " << Bx[tx]   << " By= " << By[ty]   << std::endl;
      std::cout << " Cx= " << Cx[tx]   << " Cy= " << Cy[ty]   << std::endl;
      std::cout << " Mx= " << Mx[tx]   << " My= " << My[ty]   << std::endl;
#endif
      //   rhits.push_back( TrackFP420(c0X,c1X,chisqX,nhitplanesY,c0Y,c1Y,chisqY,nhitplanesY) );
      rhits.push_back( TrackFP420(Ax[tx],Bx[tx],Cx[tx],Mx[tx],Ay[ty],By[ty],Cy[ty],My[ty]) );
    }//for tr
    //============================================================================================================
  }//in  numberXtracks >0
  //============
  return rhits;
  //============
}
//============
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
*/
