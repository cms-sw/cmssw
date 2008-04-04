//
//  SiPixelTemplateReco.cc (Version 3.40)
//
//  Add goodness-of-fit to algorithm, include single pixel clusters in chi2 calculation
//  Try "decapitation" of large single pixels
//  Add correction for (Q_F-Q_L)/(Q_F+Q_L) bias
//  Add cot(beta) reflection to reduce y-entries and more sophisticated x-interpolation
//  Fix small double pixel bug with decapitation (2.41 5-Mar-2007).
//  Fix pseudopixel bug causing possible memory overwrite (2.42 12-Mar-2007)
//  Adjust template binning to span 3 (or 4) central pixels and implement improved (faster) chi2min search
//  Replace internal containers with static arrays
//  Add external threshold to calls to ysigma2 and xsigma2, use sorted signal heights to guarrantee min clust size = 2
//  Use denser search over larger bin range for clusters with big pixels.
//  Use single calls to template object to load template arrays (had been many)
//  Add speed switch to trade-off speed and robustness
//  Add qmin and re-define qbin to flag low-q clusters
//  Add qscale to match charge scales
//
//  Created by Morris Swartz on 10/27/06.
//  Copyright 2006 __TheJohnsHopkinsUniversity__. All rights reserved.
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
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#else
#include "SiPixelTemplateReco.h"
#endif

static int theVerboseLevel = {2};
#define LogDebug(x) std::cout << x << ": "

using namespace SiPixelTemplateReco;

// *************************************************************************************************************************************
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
//! \param      proby - (output) probability describing goodness-of-fit for y-reco
//! \param       xrec - (output) best estimate of x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec in microns
//! \param      probx - (output) probability describing goodness-of-fit for x-reco
//! \param       qbin - (output) index (0-4) describing the charge of the cluster
//!                     [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
//! \param      speed - (input) switch (0-4) trading speed vs robustness
//!                      0       totally bombproof, searches the entire 41 bin range at full density (equiv to V2_4)
//!                      1       faster, searches reduced 25 bin range (no big pix) + 33 bins (big pix at ends) at full density
//!                      2       faster yet, searches same range as 1 but at 1/2 density
//!                      3       fastest, searches same range as 1 but at 1/4 density (no big pix) and 1/2 density (big pix in cluster)
// *************************************************************************************************************************************
int SiPixelTemplateReco::PixelTempReco2D(int id, bool fpix, float cotalpha, float cotbeta, array_2d cluster, 
		    std::vector<bool> ydouble, std::vector<bool> xdouble, 
		    SiPixelTemplate& templ, 
		    float& yrec, float& sigmay, float& proby, float& xrec, float& sigmax, float& probx, int& qbin, int speed)
			
{
    // Local variables 
	static int i, j, k, minbin, binl, binh, binq, midpix, fypix, nypix, lypix, logypx;
    static int fxpix, nxpix, lxpix, logxpx, shifty, shiftx;
	static unsigned int nclusx, nclusy;
	static int deltaj, jmin, jmax, fxbin, lxbin, fybin, lybin, djy, djx;
	static float sythr, sxthr, rnorm, delta, sigma, sigavg, pseudopix, qscale;
	static float ss2, ssa, sa2, ssba, saba, sba2, rat, fq, qtotal;
	static float originx, originy, qfy, qly, qfx, qlx, bias, err, maxpix;
	static double chi2x, meanx, chi2y, meany, chi2ymin, chi2xmin, chi2;
	static double hchi2, hndof;
	static float ytemp[41][25], xtemp[41][11], ysum[25], xsum[11], ysort[25], xsort[11];
	static float chi2ybin[41], chi2xbin[41], ysig2[25], xsig2[11];
	static bool yd[25], xd[11], anyyd, anyxd;
	const float ysize={150.}, xsize={100.};
	
// The minimum chi2 for a valid one pixel cluster = pseudopixel contribution only

	const double mean1pix={0.100}, chi21min={0.160};
		      
// First, interpolate the template needed to analyze this cluster     
   
    templ.interpolate(id, fpix, cotalpha, cotbeta);
	
// Define size of pseudopixel

    pseudopix = 0.2*templ.s50();
	
// Get charge scaling factor

	qscale = templ.qscale();
    
// Check that the cluster container is (up to) a 7x21 matrix and matches the dimensions of the double pixel flags

	if(cluster.num_dimensions() != 2) {return 3;}
	nclusx = cluster.size();
	nclusy = cluster.num_elements()/nclusx;
	if(nclusx != xdouble.size()) {return 4;}
	if(nclusy != ydouble.size()) {return 5;}
	
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
	   maxpix = templ.symax();
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

	if((lypix-fypix+1) != nypix) { 
	   if (theVerboseLevel > 1) {
          LogDebug("SiPixelTemplateReco") <<
           "ysum[0-9] = " << ysum[0] << ", " << ysum[1] << ", " << ysum[2] << ", " << ysum[3] << ", " << ysum[4] << ", "
		                  << ysum[5] << ", " << ysum[6] << ", " << ysum[7] << ", " << ysum[8] << ", " << ysum[9] << std::endl;
          LogDebug("SiPixelTemplateReco") <<
           "ysum[10-19] = " << ysum[10] << ", " << ysum[11] << ", " << ysum[12] << ", " << ysum[13] << ", " << ysum[14] << ", "
		                  << ysum[15] << ", " << ysum[16] << ", " << ysum[17] << ", " << ysum[18] << ", " << ysum[19] << std::endl;
          LogDebug("SiPixelTemplateReco") <<
           "ysum[20-24] = " << ysum[20] << ", " << ysum[21] << ", " << ysum[22] << ", " << ysum[23] << ", " << ysum[24] << std::endl;
       }
	
	   return 1; 
	}
	
// If cluster is longer than max template size, technique fails

	if(nypix > 21) { 
	   if (theVerboseLevel > 1) {
          LogDebug("SiPixelTemplateReco") <<
           "ysum[0-9] = " << ysum[0] << ", " << ysum[1] << ", " << ysum[2] << ", " << ysum[3] << ", " << ysum[4] << ", "
		                  << ysum[5] << ", " << ysum[6] << ", " << ysum[7] << ", " << ysum[8] << ", " << ysum[9] << std::endl;
          LogDebug("SiPixelTemplateReco") <<
           "ysum[10-19] = " << ysum[10] << ", " << ysum[11] << ", " << ysum[12] << ", " << ysum[13] << ", " << ysum[14] << ", "
		                  << ysum[15] << ", " << ysum[16] << ", " << ysum[17] << ", " << ysum[18] << ", " << ysum[19] << std::endl;
          LogDebug("SiPixelTemplateReco") <<
           "ysum[20-24] = " << ysum[20] << ", " << ysum[21] << ", " << ysum[22] << ", " << ysum[23] << ", " << ysum[24] << std::endl;
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
	   ysum[fypix-2] = pseudopix;
	} else {return 8;}
	if(lypix > 1 && lypix < 23) {
	   ysum[lypix+1] = pseudopix;	
	   ysum[lypix+2] = pseudopix;
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
	
	   if (theVerboseLevel > 1) {
          LogDebug("SiPixelTemplateReco") <<
           "xsum[0-10] = " << xsum[0] << ", " << xsum[1] << ", " << xsum[2] << ", " << xsum[3] << ", " << xsum[4] << ", "
		                  << xsum[5] << ", " << xsum[6] << ", " << xsum[7] << ", " << xsum[8] << ", " << xsum[9] << ", " << xsum[10] << std::endl;
       }

	   return 2; 
	}

// If cluster is longer than max template size, technique fails

	if(nxpix > 7) { 
	
	   if (theVerboseLevel > 1) {
          LogDebug("SiPixelTemplateReco") <<
           "xsum[0-10] = " << xsum[0] << ", " << xsum[1] << ", " << xsum[2] << ", " << xsum[3] << ", " << xsum[4] << ", "
		                  << xsum[5] << ", " << xsum[6] << ", " << xsum[7] << ", " << xsum[8] << ", " << xsum[9] << ", " << xsum[10] << std::endl;
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
	   xsum[fxpix-2] = pseudopix;
	} else {return 9;}
	if(lxpix > 1 && lxpix < 9) {
	   xsum[lxpix+1] = pseudopix;
	   xsum[lxpix+2] = pseudopix;
	} else {return 9;}
		        
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
	
// Return the charge bin via the parameter list unless the charge is too small (then flag it)
	
	qbin = binq;
	if(qtotal < 0.95*templ.qmin()) {qbin = 4;}
	
	if (theVerboseLevel > 9) {
       LogDebug("SiPixelTemplateReco") <<
        "ID = " << id << " FPix = " << fpix << 
         " cot(alpha) = " << cotalpha << " cot(beta) = " << cotbeta << 
         " nclusx = " << nclusx << " nclusy = " << nclusy << std::endl;
       LogDebug("SiPixelTemplateReco") <<
        "ID = " << id << " FPix = " << fpix << 
         " cot(alpha) = " << cotalpha << " cot(beta) = " << cotbeta << 
         " nclusx = " << nclusx << " nclusy = " << nclusy << std::endl;
    }

	
// Next, copy the y- and x-templates to local arrays
   
// First, decide on chi^2 min search parameters
    
    assert(speed >= 0 && speed < 4);
	fybin = 0; lybin = 40; fxbin = 0; lxbin = 40; djy = 1; djx = 1;
    if(speed > 0) {
       fybin = 8; lybin = 32;
       if(yd[fypix]) {fybin = 4; lybin = 36;}
	   if(lypix > fypix) {
	      if(yd[lypix-1]) {fybin = 4; lybin = 36;}
	   }
       fxbin = 8; lxbin = 32;
       if(xd[fxpix]) {fxbin = 4; lxbin = 36;}
	   if(lxpix > fxpix) {
	      if(xd[lxpix-1]) {fxbin = 4; lxbin = 36;}
	   }
	}
	
	if(speed > 1) { 
	   djy = 2; djx = 2;
	   if(speed > 2) {
	      if(!anyyd) {djy = 4;}
		  if(!anyxd) {djx = 4;}
	   }
	}
       	
// Now do the copies

	templ.ytemp(fybin, lybin, ytemp);
   
	templ.xtemp(fxbin, lxbin, xtemp);
	
// Do the y-reconstruction first 
			  		
// Apply the first-pass template algorithm to all clusters
			  
// Modify the template if double pixels are present   
	
	if(nypix > logypx) {
		i=fypix;
		while(i < lypix) {
		   if(yd[i] && !yd[i+1]) {
			  for(j=fybin; j<=lybin; ++j) {
		
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

	std::sort(&ysort[0], &ysort[logypx]);
	if(logypx == 1) {sythr = 1.01*ysort[0];} else {
	   if (ysort[1] > sythr) { sythr = 1.01*ysort[1]; }
	}
	
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	for(i=0; i<25; ++i) { ysig2[i] = 0.;}
	templ.ysigma2(fypix, lypix, sythr, ysum, ysig2);
			  
// Find the template bin that minimizes the Chi^2 

	chi2ymin = 1.e15;
	for(i=fybin; i<=lybin; ++i) { chi2ybin[i] = -1.e15;}
	minbin = -1;
	deltaj = djy;
	jmin = fybin;
	jmax = lybin;
	while(deltaj > 0) {
	   for(j=jmin; j<=jmax; j+=deltaj) {
	      if(chi2ybin[j] < -100.) {
		     ss2 = 0.;
		     ssa = 0.;
		     sa2 = 0.;
		     for(i=fypix-2; i<=lypix+2; ++i) {
			     ss2 += ysum[i]*ysum[i]/ysig2[i];
			     ssa += ysum[i]*ytemp[j][i]/ysig2[i];
			     sa2 += ytemp[j][i]*ytemp[j][i]/ysig2[i];
		     }
		     rat=ssa/ss2;
		     if(rat <= 0.) {std::cout << "illegal chi2ymin normalization = " << rat << std::endl; rat = 1.;}
		     chi2ybin[j]=ss2-2.*ssa/rat+sa2/(rat*rat);
		  }
		  if(chi2ybin[j] < chi2ymin) {
			  chi2ymin = chi2ybin[j];
			  minbin = j;
		  }
	   } 
	   deltaj /= 2;
	   if(minbin > fybin) {jmin = minbin - deltaj;} else {jmin = fybin;}
	   if(minbin < lybin) {jmax = minbin + deltaj;} else {jmax = lybin;}
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
	   
	   yrec = 0.5*(fypix+lypix-2*shifty+2.*originy)*ysize-delta;
	   
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
	   
// For cluster > 1 pix, make the second, interpolating pass with the templates 

       binl = minbin - 1;
	   binh = binl + 2;
	   if(binl < fybin) { binl = fybin;}
	   if(binh > lybin) { binh = lybin;}	  
	   ss2 = 0.;
	   ssa = 0.;
	   sa2 = 0.;
	   ssba = 0.;
	   saba = 0.;
	   sba2 = 0.;
	   for(i=fypix-2; i<=lypix+2; ++i) {
		  ss2 += ysum[i]*ysum[i]/ysig2[i];
		  ssa += ysum[i]*ytemp[binl][i]/ysig2[i];
		  sa2 += ytemp[binl][i]*ytemp[binl][i]/ysig2[i];
		  ssba += ysum[i]*(ytemp[binh][i] - ytemp[binl][i])/ysig2[i];
		  saba += ytemp[binl][i]*(ytemp[binh][i] - ytemp[binl][i])/ysig2[i];
		  sba2 += (ytemp[binh][i] - ytemp[binl][i])*(ytemp[binh][i] - ytemp[binl][i])/ysig2[i];
	   }
	   
// rat is the fraction of the "distance" from template a to template b 	   
	   
	   rat=(ssba*ssa-ss2*saba)/(ss2*sba2-ssba*ssba);
	   if(rat < 0.) {rat=0.;}
	   if(rat > 1.) {rat=1.0;}
	   rnorm = (ssa+rat*ssba)/ss2;
	
// Calculate the charges in the first and last pixels

       qfy = ysum[fypix];
       if(yd[fypix]) {qfy+=ysum[fypix+1];}
       if(logypx > 1) {
           qly=ysum[lypix];
	       if(yd[lypix-1]) {qly+=ysum[lypix-1];}
	    } else {
	       qly = qfy;
	    }
		
//  Now calculate the mean bias correction and uncertainties

        float qyfrac = (qfy-qly)/(qfy+qly);
		bias = templ.yflcorr(binq,qyfrac)+templ.yavg(binq);
		   	   
// uncertainty and final correction depend upon charge bin 	   
	   
	   yrec = (0.125*binl+9.5+rat*(binh-binl)*0.125-(float)shifty+originy)*ysize - bias;
	   sigmay = templ.yrms(binq);
	   
// Do goodness of fit test in y  
	   
	   if(rnorm <= 0.) {std::cout << "illegal chi2y normalization = " << rnorm << std::endl; rnorm = 1.;}
	   chi2y=ss2-2./rnorm*ssa-2./rnorm*rat*ssba+(sa2+2.*rat*saba+rat*rat*sba2)/(rnorm*rnorm)-templ.chi2ymin(binq);
	   if(chi2y < 0.0) {chi2y = 0.0;}
	   meany = templ.chi2yavg(binq);
	   if(meany < 0.01) {meany = 0.01;}
// gsl function that calculates the chi^2 tail prob for non-integral dof
//	   proby = gsl_cdf_chisq_Q(chi2y, meany);
//	   proby = ROOT::Math::chisquared_cdf_c(chi2y, meany);
       hchi2 = chi2y/2.; hndof = meany/2.;
	   proby = 1. - TMath::Gamma(hndof, hchi2);
	}
	
// Do the x-reconstruction next 
			  
// Apply the first-pass template algorithm to all clusters

// Modify the template if double pixels are present 

	if(nxpix > logxpx) {
		i=fxpix;
		while(i < lxpix) {
		   if(xd[i] && !xd[i+1]) {
			  for(j=fxbin; j<=lxbin; ++j) {
		
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
	std::sort(&xsort[0], &xsort[logxpx]);
	if(logxpx == 1) {sxthr = 1.01*xsort[0];} else {
	   if (xsort[1] > sxthr) { sxthr = 1.01*xsort[1]; }
	}
	   
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

	for(i=0; i<11; ++i) { xsig2[i] = 0.; }
	templ.xsigma2(fxpix, lxpix, sxthr, xsum, xsig2);
			  
// Find the template bin that minimizes the Chi^2 

	chi2xmin = 1.e15;
	for(i=fxbin; i<=lxbin; ++i) { chi2xbin[i] = -1.e15;}
	minbin = -1;
	deltaj = djx;
	jmin = fxbin;
	jmax = lxbin;
	while(deltaj > 0) {
	   for(j=jmin; j<=jmax; j+=deltaj) {
	      if(chi2xbin[j] < -100.) {
		     ss2 = 0.;
		     ssa = 0.;
		     sa2 = 0.;
		     for(i=fxpix-2; i<=lxpix+2; ++i) {
			     ss2 += xsum[i]*xsum[i]/xsig2[i];
			     ssa += xsum[i]*xtemp[j][i]/xsig2[i];
			     sa2 += xtemp[j][i]*xtemp[j][i]/xsig2[i];
			 }
		     rat=ssa/ss2;
		     if(rat <= 0.) {std::cout << "illegal chi2ymin normalization = " << rat << std::endl; rat = 1.;}
		     chi2xbin[j]=ss2-2.*ssa/rat+sa2/(rat*rat);
		  }
		  if(chi2xbin[j] < chi2xmin) {
			  chi2xmin = chi2xbin[j];
			  minbin = j;
		  }
	   } 
	   deltaj /= 2;
	   if(minbin > fxbin) {jmin = minbin - deltaj;} else {jmin = fxbin;}
	   if(minbin < lxbin) {jmax = minbin + deltaj;} else {jmax = lxbin;}
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
	   xrec = 0.5*(fxpix+lxpix-2*shiftx+2.*originx)*xsize-delta;
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
	   
// Now make the second, interpolating pass with the templates 

       binl = minbin - 1;
	   binh = binl + 2;
	   if(binl < fxbin) { binl = fxbin;}
	   if(binh > lxbin) { binh = lxbin;}	  
	   ss2 = 0.;
	   ssa = 0.;
	   sa2 = 0.;
	   ssba = 0.;
	   saba = 0.;
	   sba2 = 0.;
	   for(i=fxpix-2; i<=lxpix+2; ++i) {
		  ss2 += xsum[i]*xsum[i]/xsig2[i];
		  ssa += xsum[i]*xtemp[binl][i]/xsig2[i];
		  sa2 += xtemp[binl][i]*xtemp[binl][i]/xsig2[i];
		  ssba += xsum[i]*(xtemp[binh][i] - xtemp[binl][i])/xsig2[i];
		  saba += xtemp[binl][i]*(xtemp[binh][i] - xtemp[binl][i])/xsig2[i];
		  sba2 += (xtemp[binh][i] - xtemp[binl][i])*(xtemp[binh][i] - xtemp[binl][i])/xsig2[i];
	   }
	   
// rat is the fraction of the "distance" from template a to template b 	   
	   
	   rat=(ssba*ssa-ss2*saba)/(ss2*sba2-ssba*ssba);
	   if(rat < 0.) {rat=0.;}
	   if(rat > 1.) {rat=1.0;}
	   rnorm = (ssa+rat*ssba)/ss2;
	
// Calculate the charges in the first and last pixels

       qfx = xsum[fxpix];
       if(xd[fxpix]) {qfx+=xsum[fxpix+1];}
       if(logxpx > 1) {
           qlx=xsum[lxpix];
	       if(xd[lxpix-1]) {qlx+=xsum[lxpix-1];}
	    } else {
	       qlx = qfx;
	    }
		
//  Now calculate the mean bias correction and uncertainties

        float qxfrac = (qfx-qlx)/(qfx+qlx);
		bias = templ.xflcorr(binq,qxfrac)+templ.xavg(binq);
	   
// uncertainty and final correction depend upon charge bin 	   
	   
	   xrec = (0.125*binl+2.5+rat*(binh-binl)*0.125-(float)shiftx+originx)*xsize - bias;
	   sigmax = templ.xrms(binq);
	   
// Do goodness of fit test in x  
	   
	   if(rnorm <= 0.) {std::cout << "illegal chi2x normalization = " << rnorm << std::endl; rnorm = 1.;}
	   chi2x=ss2-2./rnorm*ssa-2./rnorm*rat*ssba+(sa2+2.*rat*saba+rat*rat*sba2)/(rnorm*rnorm)-templ.chi2xmin(binq);
	   if(chi2x < 0.0) {chi2x = 0.0;}
	   meanx = templ.chi2xavg(binq);
	   if(meanx < 0.01) {meanx = 0.01;}
// gsl function that calculates the chi^2 tail prob for non-integral dof
//	   probx = gsl_cdf_chisq_Q(chi2x, meanx);
//	   probx = ROOT::Math::chisquared_cdf_c(chi2x, meanx, trx0);
       hchi2 = chi2x/2.; hndof = meanx/2.;
	   probx = 1. - TMath::Gamma(hndof, hchi2);
	}
	
    return 0;
} // TempRecon2D 

