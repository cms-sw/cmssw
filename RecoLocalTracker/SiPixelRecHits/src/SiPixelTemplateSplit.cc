//
//  SiPixelTemplateSplit.cc (Version 0.1)
//
//  Procedure to fit two templates (same angle hypotheses) to a single cluster
//  Return two x- and two y-coordinates for the cluster
//
//  Created by Morris Swartz on 04/10/08.
//  Copyright 2008 __TheJohnsHopkinsUniversity__. All rights reserved.
//
//

#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>
// ROOT::Math has a c++ function that does the probability calc, but only in v5.12 and later
//#include "Math/DistFunc.h"
#include "TMath.h"
// Use current version of gsl instead of ROOT::Math
//#include <gsl/gsl_cdf.h>

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateSplit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) edm::LogError(x)
#define LOGDEBUG(x) LogDebug(x)
#define ENDL " "
#else
#include "SiPixelTemplateSplit.h"
//static int theVerboseLevel = {2};
#define LOGERROR(x) std::cout << x << ": "
#define LOGDEBUG(x) std::cout << x << ": "
#define ENDL std::endl
#endif

static int theVerboseLevel = {2};

using namespace SiPixelTemplateReco;

// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit positions for pixel clusters.      
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
//! \param      yrec1 - (output) best estimate of first y-coordinate of hit in microns
//! \param      yrec2 - (output) best estimate of second y-coordinate of hit in microns
//! \param     sigmay - (output) best estimate of uncertainty on yrec in microns
//! \param      proby - (output) probability describing goodness-of-fit for y-reco
//! \param      xrec1 - (output) best estimate of first x-coordinate of hit in microns
//! \param      xrec2 - (output) best estimate of second x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec in microns
//! \param      probx - (output) probability describing goodness-of-fit for x-reco
//! \param       qbin - (output) index (0-4) describing the charge of the cluster
//!                     [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
// *************************************************************************************************************************************
int SiPixelTemplateReco::PixelTempSplit(int id, bool fpix, float cotalpha, float cotbeta, array_2d cluster, 
		    std::vector<bool> ydouble, std::vector<bool> xdouble, 
		    SiPixelTemplate& templ, 
		    float& yrec1, float& yrec2, float& sigmay, float& proby,
			float& xrec1, float& xrec2, float& sigmax, float& probx, int& qbin)
			
{
    // Local variables 
	static int i, j, k, minbin, binl, binh, binq, midpix, fypix, nypix, lypix, logypx;
    static int fxpix, nxpix, lxpix, logxpx, shifty, shiftx;
	static unsigned int nclusx, nclusy;
	static int nybin, ycbin, nxbin, xcbin, minbinj, minbink;
	static float sythr, sxthr, rnorm, delta, sigma, sigavg, pseudopix, qscale;
	static float ss2, ssa, sa2, ssba, saba, sba2, rat, fq, qtotal;
	static float originx, originy, qfy, qly, qfx, qlx, bias, err, maxpix;
	static double chi2x, meanx, chi2y, meany, chi2ymin, chi2xmin, chi2;
	static double hchi2, hndof;
	static float ysum[25], xsum[11], ysort[25], xsort[11];
	static float ysig2[25], xsig2[11];
	static bool yd[25], xd[11], anyyd, anyxd;
	const float ysize={150.}, xsize={100.}, sqrt2={2.};
//  const float sqrt2={1.41421356};
	
// The minimum chi2 for a valid one pixel cluster = pseudopixel contribution only

	const double mean1pix={0.100}, chi21min={0.160};
		      
// First, interpolate the template needed to analyze this cluster     
   
    templ.interpolate(id, fpix, cotalpha, cotbeta);
	
// Define size of pseudopixel

    pseudopix = templ.s50();
	
// Get charge scaling factor

	qscale = templ.qscale();
    
// Check that the cluster container is (up to) a 7x21 matrix and matches the dimensions of the double pixel flags

	if(cluster.num_dimensions() != 2) {
	   LOGERROR("SiPixelTemplateReco") << "input cluster container (BOOST Multiarray) has wrong number of dimensions" << ENDL;	
	   return 3;
	}
	nclusx = cluster.size();
	nclusy = cluster.num_elements()/nclusx;
	if(nclusx != xdouble.size()) {
	   LOGERROR("SiPixelTemplateReco") << "input cluster container x-size is not equal to double pixel flag container size" << ENDL;	
	   return 4;
	}
	if(nclusy != ydouble.size()) {
	   LOGERROR("SiPixelTemplateReco") << "input cluster container y-size is not equal to double pixel flag container size" << ENDL;	
	   return 5;
	}
	
// enforce maximum size	
	
	if(nclusx > 7) {nclusx = 7;}
	if(nclusy > 21) {nclusy = 21;}
	
// First, rescale all pixel charges       

    for(i=0; i<nclusy; ++i) {
	   for(j=0; j<nclusx; ++j) {
		  if(cluster[j][i] > 0) {cluster[j][i] *= qscale;}
	   }
	}
	
// Next, sum the total charge and "decapitate" big pixels         

	qtotal = 0.;
    for(i=0; i<nclusy; ++i) {
	   maxpix = 2.*templ.symax();
	   if(ydouble[i]) {maxpix *=2.;}
	   for(j=0; j<nclusx; ++j) {
		  qtotal += cluster[j][i];
		  if(cluster[j][i] > maxpix) {cluster[j][i] = maxpix;}
	   }
	}
	
// Next, make y-projection of the cluster and copy the double pixel flags into a 25 element container         

    for(i=0; i<25; ++i) { ysum[i] = 0.; yd[i] = false;}
	k=0;
	anyyd = false;
    for(i=0; i<nclusy; ++i) {
	   for(j=0; j<nclusx; ++j) {
		  ysum[k] += cluster[j][i];
	   }
    
// If this is a double pixel, put 1/2 of the charge in 2 consective single pixels  
   
	   if(ydouble[i]) {
	      ysum[k] /= 2.;
		  ysum[k+1] = ysum[k];
		  yd[k] = true;
		  yd[k+1] = false;
		  k=k+2;
		  anyyd = true;
	   } else {
		  yd[k] = false;
	      ++k;
	   }
	   if(k > 24) {break;}
	}
		 
// Next, make x-projection of the cluster and copy the double pixel flags into an 11 element container         

    for(i=0; i<11; ++i) { xsum[i] = 0.; xd[i] = false;}
	k=0;
	anyxd = false;
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
		  anyxd = true;
	   } else {
		  xd[k]=false;
	      ++k;
	   }
	   if(k > 10) {break;}
	}
        
// next, identify the y-cluster ends, count total pixels, nypix, and logical pixels, logypx   

    fypix=-1;
	nypix=0;
	lypix=0;
	logypx=0;
	for(i=0; i<25; ++i) {
	   if(ysum[i] > 0.) {
	      if(fypix == -1) {fypix = i;}
		  if(!yd[i]) {
		     ysort[logypx] = ysum[i];
			 ++logypx;
		  }
		  ++nypix;
		  lypix = i;
		}
	}
	
// Make sure cluster is continuous

	if((lypix-fypix+1) != nypix || nypix == 0) { 
	   LOGDEBUG("SiPixelTemplateReco") << "y-length of pixel cluster doesn't agree with number of pixels above threshold" << ENDL;
	   if (theVerboseLevel > 2) {
          LOGDEBUG("SiPixelTemplateReco") <<
           "ysum[0-9] = " << ysum[0] << ", " << ysum[1] << ", " << ysum[2] << ", " << ysum[3] << ", " << ysum[4] << ", "
		                  << ysum[5] << ", " << ysum[6] << ", " << ysum[7] << ", " << ysum[8] << ", " << ysum[9] << ENDL;
          LOGDEBUG("SiPixelTemplateReco") <<
           "ysum[10-19] = " << ysum[10] << ", " << ysum[11] << ", " << ysum[12] << ", " << ysum[13] << ", " << ysum[14] << ", "
		                  << ysum[15] << ", " << ysum[16] << ", " << ysum[17] << ", " << ysum[18] << ", " << ysum[19] << ENDL;
          LOGDEBUG("SiPixelTemplateReco") <<
           "ysum[20-24] = " << ysum[20] << ", " << ysum[21] << ", " << ysum[22] << ", " << ysum[23] << ", " << ysum[24] << ENDL;
       }
	
	   return 1; 
	}
	
// If cluster is longer than max template size, technique fails

	if(nypix > 21) { 
	   LOGDEBUG("SiPixelTemplateReco") << "y-length of pixel cluster is larger than maximum template size" << ENDL;
	   if (theVerboseLevel > 2) {
          LOGDEBUG("SiPixelTemplateReco") <<
           "ysum[0-9] = " << ysum[0] << ", " << ysum[1] << ", " << ysum[2] << ", " << ysum[3] << ", " << ysum[4] << ", "
		                  << ysum[5] << ", " << ysum[6] << ", " << ysum[7] << ", " << ysum[8] << ", " << ysum[9] << ENDL;
          LOGDEBUG("SiPixelTemplateReco") <<
           "ysum[10-19] = " << ysum[10] << ", " << ysum[11] << ", " << ysum[12] << ", " << ysum[13] << ", " << ysum[14] << ", "
		                  << ysum[15] << ", " << ysum[16] << ", " << ysum[17] << ", " << ysum[18] << ", " << ysum[19] << ENDL;
          LOGDEBUG("SiPixelTemplateReco") <<
           "ysum[20-24] = " << ysum[20] << ", " << ysum[21] << ", " << ysum[22] << ", " << ysum[23] << ", " << ysum[24] << ENDL;
       }
	
	   return 6; 
	}
	
// next, center the cluster on pixel 12 if necessary   

	midpix = (fypix+lypix)/2;
	shifty = 12 - midpix;
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
	
// If the cluster boundaries are OK, add pesudopixels, otherwise quit
	
	if(fypix > 1 && fypix < 23) {
	   ysum[fypix-1] = pseudopix;
	   ysum[fypix-2] = 0.2*pseudopix;
	} else {return 8;}
	if(lypix > 1 && lypix < 23) {
	   ysum[lypix+1] = pseudopix;	
	   ysum[lypix+2] = 0.2*pseudopix;
	} else {return 8;}
        
// finally, determine if pixel[0] is a double pixel and make an origin correction if it is   

    if(ydouble[0]) {
	   originy = -0.5;
	} else {
	   originy = 0.;
	}
        
// next, identify the x-cluster ends, count total pixels, nxpix, and logical pixels, logxpx   

    fxpix=-1;
	nxpix=0;
	lxpix=0;
	logxpx=0;
	for(i=0; i<11; ++i) {
	   if(xsum[i] > 0.) {
	      if(fxpix == -1) {fxpix = i;}
		  if(!xd[i]) {
		     xsort[logxpx] = xsum[i];
			 ++logxpx;
		  }
		  ++nxpix;
		  lxpix = i;
		}
	}
	
// Make sure cluster is continuous

	if((lxpix-fxpix+1) != nxpix) { 
	
	   LOGDEBUG("SiPixelTemplateReco") << "x-length of pixel cluster doesn't agree with number of pixels above threshold" << ENDL;
	   if (theVerboseLevel > 2) {
          LOGDEBUG("SiPixelTemplateReco") <<
           "xsum[0-10] = " << xsum[0] << ", " << xsum[1] << ", " << xsum[2] << ", " << xsum[3] << ", " << xsum[4] << ", "
		                  << xsum[5] << ", " << xsum[6] << ", " << xsum[7] << ", " << xsum[8] << ", " << xsum[9] << ", " << xsum[10] << ENDL;
       }

	   return 2; 
	}

// If cluster is longer than max template size, technique fails

	if(nxpix > 7) { 
	
	   LOGDEBUG("SiPixelTemplateReco") << "x-length of pixel cluster is larger than maximum template size" << ENDL;
	   if (theVerboseLevel > 2) {
          LOGDEBUG("SiPixelTemplateReco") <<
           "xsum[0-10] = " << xsum[0] << ", " << xsum[1] << ", " << xsum[2] << ", " << xsum[3] << ", " << xsum[4] << ", "
		                  << xsum[5] << ", " << xsum[6] << ", " << xsum[7] << ", " << xsum[8] << ", " << xsum[9] << ", " << xsum[10] << ENDL;
       }

	   return 7; 
	}
        
// next, center the cluster on pixel 5 if necessary   

	midpix = (fxpix+lxpix)/2;
	shiftx = 5 - midpix;
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
	
// If the cluster boundaries are OK, add pesudopixels, otherwise quit
	
	if(fxpix > 1 && fxpix < 9) {
	   xsum[fxpix-1] = pseudopix;
	   xsum[fxpix-2] = 0.2*pseudopix;
	} else {return 9;}
	if(lxpix > 1 && lxpix < 9) {
	   xsum[lxpix+1] = pseudopix;
	   xsum[lxpix+2] = 0.2*pseudopix;
	} else {return 9;}
		        
// finally, determine if pixel[0] is a double pixel and make an origin correction if it is   

    if(xdouble[0]) {
	   originx = -0.5;
	} else {
	   originx = 0.;
	}
	
// uncertainty and final corrections depend upon total charge bin 	   
	   
	fq = qtotal/templ.qavg();
	if(fq > 3.0) {
	   binq=0;
	} else {
	   if(fq > 2.0) {
	      binq=1;
	   } else {
		  if(fq > 1.70) {
			 binq=2;
		  } else {
			 binq=3;
		  }
	   }
	}
	
// Return the charge bin via the parameter list unless the charge is too small (then flag it)
	
	qbin = binq;
	if(qtotal < 0.95*templ.qmin()) {qbin = 4;}
	
	if (theVerboseLevel > 9) {
       LOGDEBUG("SiPixelTemplateReco") <<
        "ID = " << id << " FPix = " << fpix << 
         " cot(alpha) = " << cotalpha << " cot(beta) = " << cotbeta << 
         " nclusx = " << nclusx << " nclusy = " << nclusy << ENDL;
    }

	
// Next, generate the 3d y- and x-templates
   
    array_3d ytemp;    	
	templ.ytemp3d(nypix, ytemp);
   
    array_3d xtemp;
	templ.xtemp3d(nxpix, xtemp);
	
// Do the y-reconstruction first 
			  		
// retrieve the number of y-bins 

    nybin = ytemp.shape()[0]; ycbin = nybin/2;
			  
// Modify the template if double pixels are present   
	
	if(nypix > logypx) {
		i=fypix;
		while(i < lypix) {
		   if(yd[i] && !yd[i+1]) {
			  for(j=0; j<nybin; ++j) {
			     for(k=0; k<=j; ++k) {
		
// Sum the adjacent cells and put the average signal in both   

				    sigavg = (ytemp[j][k][i] +  ytemp[j][k][i+1])/2.;
				    ytemp[j][k][i] = sigavg;
				    ytemp[j][k][i+1] = sigavg;
				  }
			   }
			   i += 2;
			} else {
			   ++i;
			}
		 }
	}	
	     
// Define the maximum signal to allow before de-weighting a pixel 

	sythr = 2.1*(templ.symax());
			  
// Make sure that there will be at least two pixels that are not de-weighted 

	std::sort(&ysort[0], &ysort[logypx]);
	if(logypx == 1) {sythr = 1.01*ysort[0];} else {
	   if (ysort[1] > sythr) { sythr = 1.01*ysort[1]; }
	}
	
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	for(i=0; i<25; ++i) { ysig2[i] = 0.;}
	templ.ysigma2(fypix, lypix, sythr, ysum, ysig2);
			  
// Find the template bin that minimizes the Chi^2 

	chi2ymin = 1.e15;
	minbinj = -1; 
	minbink = -1;
	for(j=0; j<nybin; ++j) {
	   for(k=0; k<=j; ++k) {
	      ss2 = 0.;
		  ssa = 0.;
		  sa2 = 0.;
		  for(i=fypix-2; i<=lypix+2; ++i) {
			 ss2 += ysum[i]*ysum[i]/ysig2[i];
			 ssa += ysum[i]*ytemp[j][k][i]/ysig2[i];
			 sa2 += ytemp[j][k][i]*ytemp[j][k][i]/ysig2[i];
		  }
		  rat=ssa/ss2;
		  if(rat <= 0.) {LOGERROR("SiPixelTemplateReco") << "illegal chi2ymin normalization = " << rat << ENDL; rat = 1.;}
		  chi2y=ss2-2.*ssa/rat+sa2/(rat*rat);
		  if(chi2y < chi2ymin) {
			  chi2ymin = chi2y;
			  minbinj = j;
			  minbink = k;
		  }
	   } 
	}
	
	if (theVerboseLevel > 9) {
       LOGDEBUG("SiPixelTemplateReco") <<
        "minbins " << minbinj << "," << minbink << " chi2ymin = " << chi2ymin << ENDL;
    }
	
// Do not apply final template pass to 1-pixel clusters (use calibrated offset) 
	
	if(logypx == 1) {
	
	   if(nypix ==1) {
	      delta = templ.dyone();
		  sigma = templ.syone();
	   } else {
	      delta = templ.dytwo();
		  sigma = templ.sytwo();
	   }
	   
	   yrec1 = 0.5*(fypix+lypix-2*shifty+2.*originy)*ysize-delta;
	   yrec2 = yrec1;
	   
	   if(sigma <= 0.) {
	      sigmay = 43.3;
	   } else {
          sigmay = sigma;
	   }
	   
// Do probability calculation for one-pixel clusters

       chi2ymin -=chi21min;
	   if(chi2ymin < 0.) {chi2ymin = 0.;}
//	   proby = gsl_cdf_chisq_Q(chi2ymin, mean1pix);
       hchi2 = chi2ymin/2.; hndof = mean1pix/2.;
	   proby = 1. - TMath::Gamma(hndof, hchi2);
	   
	} else {
	   
// For cluster > 1 pix, use chi^2 minimm to recontruct the two y-positions

// uncertainty and final correction depend upon charge bin 	   
	   
	   bias = templ.yavgc2m(binq);
	   yrec1 = (0.125*(minbink-ycbin)+12.0-(float)shifty+originy)*ysize - bias;
	   yrec2 = (0.125*(minbinj-ycbin)+12.0-(float)shifty+originy)*ysize - bias;
	   sigmay = sqrt2*templ.yrmsc2m(binq);
	   
// Do goodness of fit test in y  
	   
	   if(chi2ymin < 0.0) {chi2ymin = 0.0;}
	   meany = 2.*templ.chi2yavg(binq);
	   if(meany < 0.01) {meany = 0.01;}
// gsl function that calculates the chi^2 tail prob for non-integral dof
//	   proby = gsl_cdf_chisq_Q(chi2y, meany);
//	   proby = ROOT::Math::chisquared_cdf_c(chi2y, meany);
       hchi2 = chi2ymin/2.; hndof = meany/2.;
	   proby = 1. - TMath::Gamma(hndof, hchi2);
	}
	
// Do the x-reconstruction next 
			  
// retrieve the number of x-bins 

    nxbin = xtemp.shape()[0]; xcbin = nxbin/2;
			  
// Modify the template if double pixels are present   
	
	if(nxpix > logxpx) {
		i=fxpix;
		while(i < lxpix) {
		   if(xd[i] && !xd[i+1]) {
			  for(j=0; j<nxbin; ++j) {
			     for(k=0; k<=j; ++k) {
		
// Sum the adjacent cells and put the average signal in both   

				    sigavg = (xtemp[j][k][i] +  xtemp[j][k][i+1])/2.;
				    xtemp[j][k][i] = sigavg;
				    xtemp[j][k][i+1] = sigavg;
				  }
			   }
			   i += 2;
			} else {
			   ++i;
			}
		 }
	}	
				  
// Define the maximum signal to allow before de-weighting a pixel 

	sxthr = 2.1*templ.sxmax();
			  
// Make sure that there will be at least two pixels that are not de-weighted 
	std::sort(&xsort[0], &xsort[logxpx]);
	if(logxpx == 1) {sxthr = 1.01*xsort[0];} else {
	   if (xsort[1] > sxthr) { sxthr = 1.01*xsort[1]; }
	}
	   
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	for(i=0; i<11; ++i) { xsig2[i] = 0.; }
	templ.xsigma2(fxpix, lxpix, sxthr, xsum, xsig2);
			  
// Find the template bin that minimizes the Chi^2 

	chi2xmin = 1.e15;
	minbinj = -1; 
	minbink = -1;
	for(j=0; j<nxbin; ++j) {
	   for(k=0; k<=j; ++k) {
	      ss2 = 0.;
		  ssa = 0.;
		  sa2 = 0.;
		  for(i=fxpix-2; i<=lxpix+2; ++i) {
			 ss2 += xsum[i]*xsum[i]/xsig2[i];
			 ssa += xsum[i]*xtemp[j][k][i]/xsig2[i];
			 sa2 += xtemp[j][k][i]*xtemp[j][k][i]/xsig2[i];
		  }
		  rat=ssa/ss2;
		  if(rat <= 0.) {LOGERROR("SiPixelTemplateReco") << "illegal chi2ymin normalization = " << rat << ENDL; rat = 1.;}
		  chi2x=ss2-2.*ssa/rat+sa2/(rat*rat);
		  if(chi2x < chi2xmin) {
			  chi2xmin = chi2x;
			  minbinj = j;
			  minbink = k;
		  }
	   } 
	}
	
	if (theVerboseLevel > 9) {
       LOGDEBUG("SiPixelTemplateReco") <<
        "minbin " << minbin << " chi2xmin = " << chi2xmin << ENDL;
    }

// Do not apply final template pass to 1-pixel clusters (use calibrated offset)
	
	if(logxpx == 1) {
	
	   if(nxpix ==1) {
	      delta = templ.dxone();
		  sigma = templ.sxone();
	   } else {
	      delta = templ.dxtwo();
		  sigma = templ.sxtwo();
	   }
	   xrec1 = 0.5*(fxpix+lxpix-2*shiftx+2.*originx)*xsize-delta;
	   xrec2 = xrec1;
	   if(sigma <= 0.) {
	      sigmax = 28.9;
	   } else {
          sigmax = sigma;
	   }
	   
// Do probability calculation for one-pixel clusters

       chi2xmin -=chi21min;
	   if(chi2xmin < 0.) {chi2xmin = 0.;}
//	   probx = gsl_cdf_chisq_Q(chi2xmin, mean1pix);
       hchi2 = chi2xmin/2.; hndof = mean1pix/2.;
	   probx = 1. - TMath::Gamma(hndof, hchi2);
	   
	} else {
	   
// For cluster > 1 pix, use chi^2 minimm to recontruct the two x-positions

// uncertainty and final correction depend upon charge bin 	   
	   
	   bias = templ.xavgc2m(binq);
	   k = std::min(minbink, minbinj);
	   j = std::max(minbink, minbinj);
       xrec1 = (0.125*(minbink-xcbin)+5.0-(float)shiftx+originx)*xsize - bias;
       xrec2 = (0.125*(minbinj-xcbin)+5.0-(float)shiftx+originx)*xsize - bias;
	   sigmax = sqrt2*templ.xrmsc2m(binq);
	   
// Do goodness of fit test in y  
	   
	   if(chi2xmin < 0.0) {chi2xmin = 0.0;}
	   meanx = 2.*templ.chi2xavg(binq);
	   if(meanx < 0.01) {meanx = 0.01;}
       hchi2 = chi2xmin/2.; hndof = meanx/2.;
	   probx = 1. - TMath::Gamma(hndof, hchi2);
	}
	
    return 0;
} // PixelTempSplit 

