/*
 *  SiPixelTemplateReco.cc
 *  
 *
 *  Created by Morris Swartz on 10/27/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#else
#include "SiPixelTemplateReco.h"
#endif

#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>


using namespace SiPixelTemplateReco;

// ******************************************************************************************
//! Reconstruct the best estimate of the hit position for pixel clusters.      
//! \param         id - (input) identifier of the template to use                                  
//! \param       fpix - (input) logical input indicating whether to use 
//!                     FPix templates (true) or Barrel templates (false)
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014) 
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)  
//! \param    cluster - (input) boost multi_array container of 7x21 array of pixel signals, 
//!           origin of local coords (0,0) at center of pixel cluster[0][0].                      
//! \param    ydouble - (input) STL vector of 21 element array to flag a double-pixel
//! \param    xdouble - (input) STL vector of 7 element array to flag a double-pixel
//! \param      templ - (input) the template used in the reconstruction
//! \param       yrec - (output) best estimate of y-coordinate of hit in microns
//! \param     sigmay - (output) best estimate of uncertainty on yrec in microns
//! \param       xrec - (output) best estimate of x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec in microns
// ******************************************************************************************
int SiPixelTemplateReco::PixelTempReco2D(int id, bool fpix, float cotalpha, float cotbeta, array_2d cluster, 
		    std::vector<bool> ydouble, std::vector<bool> xdouble, 
		    SiPixelTemplate& templ, 
		    float& yrec, float& sigmay, float& xrec, float& sigmax)
{
    // Local variables 
	int i, j, k, minbin, binl, binh, binq, midpix, nclusx, nclusy;
	std::vector<float> ysig2(25), xsig2(11);
	float sythr, sxthr, rnorm, delta, sigma, sigavg;
	float chimin, ss2, ssa, sa2, ssba, saba, sba2, rat, chi2, sigi, sigi2, sigi3, sigi4, fq;
	float originx, originy;
	const float ysize={150.}, xsize={100.};


	      
// First, interpolate the template needed to analyze this cluster     
   
    templ.interpolate(id, fpix, cotalpha, cotbeta);
	
// Next, copy the y- and x-templates to matrix containers     
   
	array_2d ytemp(boost::extents[41][25]);
    for(i=0; i<41; ++i) {
	   for(j=0; j<25; ++j) {
	      ytemp[i][j]=templ.ytemp(i,j);
	   }
	}
   
	array_2d xtemp(boost::extents[41][11]);
    for(i=0; i<41; ++i) {
	   for(j=0; j<11; ++j) {
	      xtemp[i][j]=templ.xtemp(i,j);
	   }
	}
    
// Check that the cluster container is (up to) a 7x21 matrix and matches the dimensions of the double pixel flags

	if(cluster.num_dimensions() != 2) {return 3;}
	nclusx = cluster.size();
	nclusy = cluster.num_elements()/nclusx;
	if(nclusx != xdouble.size()) {return 4;}
	if(nclusy != ydouble.size()) {return 5;}
	
// enforce maximum size	
	
	if(nclusx > 7) {nclusx = 7;}
	if(nclusy > 21) {nclusy = 21;}
	
	
// Next, make y-projection of the cluster and copy the double pixel flags into a 25 element container         

	float qtotal = 0.;
	std::vector<float> ysum(25, 0.);
	std::vector<bool> yd(25, false);
	k=0;
    for(i=0; i<nclusy; ++i) {
	   for(j=0; j<nclusx; ++j) {
		  ysum[k] += cluster[j][i];
		  qtotal += cluster[j][i];
	   }
    
// If this is a double pixel, put 1/2 of the charge in 2 consective single pixels  
   
	   if(ydouble[i]) {
	      ysum[k] /= 2.;
		  ysum[k+1] = ysum[k];
		  yd[k] = true;
		  yd[k+1] = false;
		  k=k+2;
	   } else {
		  yd[k] = false;
	      ++k;
	   }
	   if(k > 24) {break;}
	}
		 
// Next, make x-projection of the cluster and copy the double pixel flags into an 11 element container         

	std::vector<float> xsum(11, 0.);
	std::vector<bool> xd(11, false);
	k=0;
    for(j=0; j<nclusx; ++j) {
	   for(i=0; i<nclusy; ++i) {
		  xsum[k] += cluster[j][i];
	   }
    
// If this is a double pixel, put 1/2 of the charge in 2 consective single pixels  
   
	   if(xdouble[j]) {
	      xsum[k] /= 2.;
		  xsum[k+1] = xsum[k];
		  xd[k]=true;
		  xd[k+1]=false;
		  k=k+2;
	   } else {
		  xd[k]=false;
	      ++k;
	   }
	   if(k > 10) {break;}
	}
        
// next, identify the y-cluster ends, count total pixels, nypix, and logical pixels, logypx   

    int fypix=-1;
	int nypix=0;
	int lypix=0;
	int logypx=0;
	std::vector<float> ysort;
	for(i=0; i<25; ++i) {
	   if(ysum[i] > 0.) {
	      if(fypix == -1) {fypix = i;}
		  if(!yd[i]) {
		     ysort.push_back(ysum[i]);
			 ++logypx;
		  }
		  ++nypix;
		  lypix = i;
		}
	}
	if((lypix-fypix+1) != nypix) { return 1; }
        
// next, center the cluster on pixel 12 if necessary   

	midpix = (fypix+lypix)/2;
	int shifty = 12 - midpix;
	if(shifty > 0) {
	   for(i=lypix; i>=fypix; --i) {
	      ysum[i+shifty] = ysum[i];
		  ysum[i] = 0.;
		  yd[i+shifty] = yd[i];
		  yd[i] = false;
	   }
	} else if (shifty < 0) {
	   for(i=fypix; i<=lypix; ++i) {
	      ysum[i+shifty] = ysum[i];
		  ysum[i] = 0.;
		  yd[i+shifty] = yd[i];
		  yd[i] = false;
	   }
    }
	lypix +=shifty;
	fypix +=shifty;
	
// Add pseudo-pixels   
	
	ysum[fypix-1] = 0.2*templ.s50();
	ysum[lypix+1] = ysum[fypix-1];	
	ysum[fypix-2] = ysum[fypix-1];
	ysum[lypix+2] = ysum[fypix-1];
        
// finally, determine if pixel[0] is a double pixel and make an origin correction if it is   

    if(ydouble[0]) {
	   originy = -0.5;
	} else {
	   originy = 0.;
	}
        
// next, identify the x-cluster ends, count total pixels, nxpix, and logical pixels, logxpx   

    int fxpix=-1;
	int nxpix=0;
	int lxpix=0;
	int logxpx=0;
	std::vector<float> xsort;
	for(i=0; i<11; ++i) {
	   if(xsum[i] > 0.) {
	      if(fxpix == -1) {fxpix = i;}
		  if(!xd[i]) {
		     xsort.push_back(xsum[i]);
			 ++logxpx;
		  }
		  ++nxpix;
		  lxpix = i;
		}
	}
	if((lxpix-fxpix+1) != nxpix) { return 2; }
        
// next, center the cluster on pixel 5 if necessary   

	midpix = (fxpix+lxpix)/2;
	int shiftx = 5 - midpix;
	if(shiftx > 0) {
	   for(i=lxpix; i>=fxpix; --i) {
	      xsum[i+shiftx] = xsum[i];
		  xsum[i] = 0.;
	      xd[i+shiftx] = xd[i];
		  xd[i] = false;
	   }
	} else if (shiftx < 0) {
	   for(i=fxpix; i<=lxpix; ++i) {
	      xsum[i+shiftx] = xsum[i];
		  xsum[i] = 0.;
	      xd[i+shiftx] = xd[i];
		  xd[i] = false;
	   }
    }
	lxpix +=shiftx;
	fxpix +=shiftx;
	
// Add pseudo-pixels   
	
	xsum[fxpix-1] = 0.2*templ.s50();
	xsum[lxpix+1] = xsum[fxpix-1];
	xsum[fxpix-2] = xsum[fxpix-1];
	xsum[lxpix+2] = xsum[fxpix-1];
		
        
// finally, determine if pixel[0] is a double pixel and make an origin correction if it is   

    if(xdouble[0]) {
	   originx = -0.5;
	} else {
	   originx = 0.;
	}
	
// uncertainty and final corrections depend upon total charge bin 	   
	   
	fq = qtotal/templ.qavg();
	if(fq > 1.5) {
	   binq=0;
	} else {
	   if(fq > 1.0) {
	      binq=1;
	   } else {
		  if(fq > 0.85) {
			 binq=2;
		  } else {
			 binq=3;
		  }
	   }
	}
		
// Do the y-reconstruction first 
	
// Do not apply templates to 1-pixel clusters (use calibrated offset) 
	
	if(logypx == 1) {
	
	   if(nypix ==1) {
	      delta = templ.dyone();
		  sigma = templ.syone();
	   } else {
	      delta = templ.dytwo();
		  sigma = templ.sytwo();
	   }
	   
	   yrec = 0.5*(fypix+lypix-2*shifty+2.*originy)*ysize-delta;
	   
	   if(sigma <= 0.) {
	      sigmay = 43.3;
	   } else {
          sigmay = sigma;
	   }
	   
	} else {
			  		
// Apply the template algorithm 
			  
// Modify the template if double pixels are present   
	
	   if(nypix > logypx) {
	      i=fypix;
	      while(i < lypix) {
	         if(yd[i] && !yd[i+1]) {
		        for(j=0; j<41; ++j) {
		
// Sum the adjacent cells and put the average signal in both   

			       sigavg = (ytemp[j][i] +  ytemp[j][i+1])/2.;
				   ytemp[j][i] = sigavg;
				   ytemp[j][i+1] = sigavg;
			    }
			    i += 2;
		     } else {
		        ++i;
		     }
	      }
  	   }	
	     
// Define the maximum signal to allow before de-weighting a pixel 

       sythr = 1.1*(templ.symax());
			  
// Make sure that there will be at least two pixels that are not de-weighted 

       std::sort(ysort.begin(), ysort.end());
	   if (ysort[1] > sythr) { sythr = 1.01*ysort[1]; }
	   
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	   for(i=fypix-2; i<=lypix+2; ++i) {
		  if(i < fypix || i > lypix) {
			 ysig2[i] = (templ.s50())*(templ.s50());
		  } else {
			 if(ysum[i] < templ.symax()) {
				sigi = ysum[i];
			 } else {
				sigi = templ.symax();
			 }
			 sigi2 = sigi*sigi; sigi3 = sigi2*sigi; sigi4 = sigi3*sigi;
			 if(i <= 12) {
				ysig2[i] = (1.-(templ.yratio()))*
				(templ.yparl(0,0)+templ.yparl(0,1)*sigi+templ.yparl(0,2)*sigi2+templ.yparl(0,3)*sigi3+templ.yparl(0,4)*sigi4)
				+ (templ.yratio())*
				(templ.yparh(0,0)+templ.yparh(0,1)*sigi+templ.yparh(0,2)*sigi2+templ.yparh(0,3)*sigi3+templ.yparh(0,4)*sigi4);
			 } else {
				ysig2[i] = (1.-(templ.yratio()))*
				(templ.yparl(1,0)+templ.yparl(1,1)*sigi+templ.yparl(1,2)*sigi2+templ.yparl(1,3)*sigi3+templ.yparl(1,4)*sigi4)
				+ (templ.yratio())*
			    (templ.yparh(1,0)+templ.yparh(1,1)*sigi+templ.yparh(1,2)*sigi2+templ.yparh(1,3)*sigi3+templ.yparh(1,4)*sigi4);
			 }
		     if(ysum[i] > sythr) {ysig2[i] = 1.e8;}
//		     if(yd[i] != 0) {ysig2[i] = 1.e8;} 
	      }
	   }

			  
// Find the template bin that minimizes the Chi^2 

       chimin = 1.e15;
	   minbin = -1;
	   for(j=0; j<41; ++j) {
	      ss2 = 0.;
	      ssa = 0.;
	      sa2 = 0.;
	      for(i=fypix-2; i<=lypix+2; ++i) {
			 ss2 += ysum[i]*ysum[i]/ysig2[i];
			 ssa += ysum[i]*ytemp[j][i]/ysig2[i];
			 sa2 += ytemp[j][i]*ytemp[j][i]/ysig2[i];
		  }
		  rat=ssa/ss2;
		  chi2=rat*rat*ss2-2.*rat*ssa+sa2;
          if(chi2 < chimin) {
		     chimin = chi2;
			 minbin = j;
		  }
	   }
	   
// Now make the second, interpolating pass with the templates 

       binl = minbin - 1;
	   binh = binl + 2;
	   if(binl < 0) { binl = 0;}
	   if(binh > 40) { binh = 40;}	  
	   ss2 = 0.;
	   ssa = 0.;
	   ssba = 0.;
	   saba = 0.;
	   sba2 = 0.;
	   for(i=fypix-2; i<=lypix+2; ++i) {
		  ss2 += ysum[i]*ysum[i]/ysig2[i];
		  ssa += ysum[i]*ytemp[binl][i]/ysig2[i];
		  ssba += ysum[i]*(ytemp[binh][i] - ytemp[binl][i])/ysig2[i];
		  saba += ytemp[binl][i]*(ytemp[binh][i] - ytemp[binl][i])/ysig2[i];
		  sba2 += (ytemp[binh][i] - ytemp[binl][i])*(ytemp[binh][i] - ytemp[binl][i])/ysig2[i];
	   }
	   
// rat is the fraction of the "distance" from template a to template b 	   
	   
	   rat=(ssba*ssa-ss2*saba)/(ss2*sba2-ssba*ssba);
	   if(rat < 0.) {rat=0.;}
	   if(rat > 1.) {rat=1.0;}
	   rnorm = (ssa+rat*ssba)/ss2;
	   
// uncertainty and final correction depend upon charge bin 	   
	   
	   yrec = (0.125*binl+9.5+rat*(binh-binl)*0.125-(float)shifty+originy)*ysize - templ.yavg(binq);
	   sigmay = templ.yrms(binq);
	}
	
// Do the x-reconstruction next 

// Do not apply templates to 1-pixel clusters (use calibrated offset) 
	
	if(logxpx == 1) {
	
	   if(nxpix ==1) {
	      delta = templ.dxone();
		  sigma = templ.sxone();
	   } else {
	      delta = templ.dxtwo();
		  sigma = templ.sxtwo();
	   }
	   xrec = 0.5*(fxpix+lxpix-2*shiftx+2.*originx)*xsize-delta;
	   if(sigma <= 0.) {
	      sigmax = 28.9;
	   } else {
          sigmax = sigma;
	   }
	   
	} else {
			  
// Apply the template algorithm 

// Modify the template if double pixels are present 

	   if(nxpix > logxpx) {
	      i=fxpix;
	      while(i < lxpix) {
	         if(xd[i] && !xd[i+1]) {
		        for(j=0; j<41; ++j) {
		
// Sum the adjacent cells and put the average signal in both   

			       sigavg = (xtemp[j][i] +  xtemp[j][i+1])/2.;
				   xtemp[j][i] = sigavg;
				   xtemp[j][i+1] = sigavg;
			    }
			    i += 2;
		     } else {
		        ++i;
		     }
	      }
	   }	  
				  
// Define the maximum signal to allow before de-weighting a pixel 

       sxthr = 1.1*templ.sxmax();
			  
// Make sure that there will be at least two pixels that are not de-weighted 

       std::sort(xsort.begin(), xsort.end());
	   if (xsort[1] > sxthr) { sxthr = 1.01*xsort[1]; }
	   
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	   for(i=fxpix-2; i<=lxpix+2; ++i) {
		  if(i < fxpix || i > lxpix) {
			 xsig2[i] = templ.s50()*templ.s50();
		  } else {
			 if(xsum[i] < templ.sxparmax()) {
				sigi = xsum[i];
			 } else {
				sigi = templ.sxparmax();
			 }
			 sigi2 = sigi*sigi; sigi3 = sigi2*sigi; sigi4 = sigi3*sigi;
			 if(i <= 5) {
				xsig2[i] = (1.-(templ.xratio()))*
				(templ.xparl(0,0)+templ.xparl(0,1)*sigi+templ.xparl(0,2)*sigi2+templ.xparl(0,3)*sigi3+templ.xparl(0,4)*sigi4)
				+ (templ.xratio())*
				(templ.xparh(0,0)+templ.xparh(0,1)*sigi+templ.xparh(0,2)*sigi2+templ.xparh(0,3)*sigi3+templ.xparh(0,4)*sigi4);
			 } else {
				xsig2[i] = (1.-(templ.xratio()))*
				(templ.xparl(1,0)+templ.xparl(1,1)*sigi+templ.xparl(1,2)*sigi2+templ.xparl(1,3)*sigi3+templ.xparl(1,4)*sigi4)
				+ (templ.xratio())*
			    (templ.xparh(1,0)+templ.xparh(1,1)*sigi+templ.xparh(1,2)*sigi2+templ.xparh(1,3)*sigi3+templ.xparh(1,4)*sigi4);
			 }
		     if(xsum[i] > sxthr) {xsig2[i] = 1.e8;}
//		     if(xd[i] != 0) {xsig2[i] = 1.e8;} 
	      }
	   }

			  
// Find the template bin that minimizes the Chi^2 

       chimin = 1.e15;
	   minbin = -1;
	   for(j=0; j<41; ++j) {
	      ss2 = 0.;
	      ssa = 0.;
	      sa2 = 0.;
	      for(i=fxpix-2; i<=lxpix+2; ++i) {
			 ss2 += xsum[i]*xsum[i]/xsig2[i];
			 ssa += xsum[i]*xtemp[j][i]/xsig2[i];
			 sa2 += xtemp[j][i]*xtemp[j][i]/xsig2[i];
		  }
		  rat=ssa/ss2;
		  chi2=rat*rat*ss2-2.*rat*ssa+sa2;
          if(chi2 < chimin) {
		     chimin = chi2;
			 minbin = j;
		  }
	   }
	   
// Now make the second, interpolating pass with the templates 

       binl = minbin - 1;
	   binh = binl + 2;
	   if(binl < 0) { binl = 0;}
	   if(binh > 40) { binh = 40;}	  
	   ss2 = 0.;
	   ssa = 0.;
	   ssba = 0.;
	   saba = 0.;
	   sba2 = 0.;
	   for(i=fxpix-2; i<=lxpix+2; ++i) {
		  ss2 += xsum[i]*xsum[i]/xsig2[i];
		  ssa += xsum[i]*xtemp[binl][i]/xsig2[i];
		  ssba += xsum[i]*(xtemp[binh][i] - xtemp[binl][i])/xsig2[i];
		  saba += xtemp[binl][i]*(xtemp[binh][i] - xtemp[binl][i])/xsig2[i];
		  sba2 += (xtemp[binh][i] - xtemp[binl][i])*(xtemp[binh][i] - xtemp[binl][i])/xsig2[i];
	   }
	   
// rat is the fraction of the "distance" from template a to template b 	   
	   
	   rat=(ssba*ssa-ss2*saba)/(ss2*sba2-ssba*ssba);
	   if(rat < 0.) {rat=0.;}
	   if(rat > 1.) {rat=1.0;}
	   rnorm = (ssa+rat*ssba)/ss2;
	   
// uncertainty and final correction depend upon charge bin 	   
	   
	   xrec = (0.125*binl+2.5+rat*(binh-binl)*0.125-(float)shiftx+originx)*xsize - templ.xavg(binq);
	   sigmax = templ.xrms(binq);
	}

    return 0;
} // TempRecon2D 
