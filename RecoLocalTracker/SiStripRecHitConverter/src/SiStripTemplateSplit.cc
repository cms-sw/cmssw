//
//  SiStripTemplateSplit.cc
//
//  Procedure to fit two templates (same angle hypotheses) to a single cluster
//
//  Version 1.00 [based on SiPixelTemplateSplit.cc Version 2.30]
//  Version 1.01 [improve error estimation for qbin=3 events]
//  Version 1.05 [Incorporate VI-like speed improvements]
//  Version 1.06 Clean-up irrelevant (since truncation) sorting
//  Version 2.10 Clone speed improvements from the pixels (eliminate 3-d multi-arays, improve seach algorithm)
//
//  Created by Morris Swartz on 04/10/08.
//
//

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
//#include <cmath.h>
#else
#include <math.h>
#endif
#include <algorithm>
#include <vector>
#include <list>
#include <iostream>
// ROOT::Math has a c++ function that does the probability calc, but only in v5.12 and later
#include "Math/DistFunc.h"
#include "TMath.h"
// Use current version of gsl instead of ROOT::Math
//#include <gsl/gsl_cdf.h>

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplateSplit.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/VVIObj.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) edm::LogError(x)
#define LOGDEBUG(x) LogDebug(x)
constexpr int theVerboseLevel = 2;
#define ENDL " "
#else
#include "SiStripTemplateSplit.h"
#include "VVIObj.h"
//#include "SiStripTemplate2D.h"
//#include "SimpleTemplate2D.h"
//static int theVerboseLevel = {2};
#define LOGERROR(x) std::cout << x << ": "
#define LOGDEBUG(x) std::cout << x << ": "
#define ENDL std::endl
#endif

using namespace SiStripTemplateSplit;

// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit positions for pixel clusters.      
//! \param         id - (input) identifier of the template to use                                  
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014) 
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)  
//! \param      locBy - (input) the sign of the local B_y field to specify the Lorentz drift direction 
//! \param      speed - (input) switch (-1->2) trading speed vs robustness
//!                     -1       totally bombproof, searches the entire ranges of template bins, 
//!                              calculates Q probability w/ VVIObj
//!                      0       totally bombproof, searches the entire template bin range at full density (no Qprob)
//!                      1       faster, searches same range as 0 but at 1/2 density
//!                      2       fastest, searches same range as 1 but at 1/4 density (no big pix) and 1/2 density (big pix in cluster)
//! \param    cluster - (input) boost multi_array container of 7x21 array of pixel signals, 
//!           origin of local coords (0,0) at center of pixel cluster[0][0].                      
//! \param      templ - (input) the template used in the reconstruction
//! \param      xrec1 - (output) best estimate of first x-coordinate of hit in microns
//! \param      xrec2 - (output) best estimate of second x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec1 and xrec2 in microns
//! \param     prob2x - (output) probability describing goodness-of-fit to a merged cluster hypothesis for x-reco
//! \param      q2bin - (output) index (0-4) describing the charge of the cluster assuming a merged 2-hit cluster hypothesis
//!                     [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
//! \param     prob2Q - (output) probability that the cluster charge is compatible with a 2-hit merging
// *************************************************************************************************************************************
int SiStripTemplateSplit::StripTempSplit(int id, float cotalpha, float cotbeta, float locBy, int speed, std::vector<float>& cluster, 
		    SiStripTemplate& templ, 
			float& xrec1, float& xrec2, float& sigmax, float& prob2x, int& q2bin, float& prob2Q)
			
{
    // Local variables 
	int i, j, k, binq, binqerr, midpix;
	int fxpix, nxpix, lxpix, logxpx, shiftx;
	int nclusx;
	int nxbin, xcbin, minbinj, minbink;
	int deltaj, jmin, jmax, kmin, kmax, km, fxbin, lxbin, djx;
	float sxthr, delta, sigma, pseudopix, xsize, qscale;
	float ss2, ssa, sa2, rat, fq, qtotal, qavg;
	float originx, bias, maxpix;
	double chi2x, meanx, chi2xmin, chi21max;
	double hchi2, hndof;
	double prvav, mpv, sigmaQ, kappa, xvav, beta2;
	float xsum[BSXSIZE];
	float xsig2[BSXSIZE];
	float xw2[BSXSIZE], xsw[BSXSIZE];
	const float sqrt2x={2.00000};
	const float sqrt12={3.4641};
	const float probmin={1.110223e-16};
	const float prob2Qmin={1.e-5};
	
//	bool SiStripTemplateSplit::SimpleTemplate2D(float cotalpha, float cotbeta, float xhit, float yhit, float thick, float lorxwidth, float lorywidth, 
//						  float qavg, std::vector<bool> ydouble, std::vector<bool> xdouble, float template2d[BSXM2][BYM2]);
	
// The minimum chi2 for a valid one pixel cluster = pseudopixel contribution only

	const double mean1pix={0.100}, chi21min={0.160};
		      
// First, interpolate the template needed to analyze this cluster     
// check to see of the track direction is in the physical range of the loaded template

	if(!templ.interpolate(id, cotalpha, cotbeta, locBy)) {
	   LOGDEBUG("SiStripTemplateReco") << "input cluster direction cot(alpha) = " << cotalpha << ", cot(beta) = " << cotbeta 
	   << " is not within the acceptance of template ID = " << id << ", no reconstruction performed" << ENDL;	
	   return 20;
	}
		      
	// Get pixel dimensions from the template (to allow multiple detectors in the future)
	
	xsize = templ.xsize();
   
	// Define size of pseudopixel
	
	pseudopix = templ.s50();
	
	// Get charge scaling factor
	
	qscale = templ.qscale();
	
	// enforce maximum size	
	
	nclusx = (int)cluster.size();
	
	if(nclusx > TSXSIZE) {nclusx = TSXSIZE;}
	
	// First, rescale all strip charges, sum them and trunate the strip charges      
	
	qtotal = 0.f;
	for(i=0; i<BSXSIZE; ++i) {xsum[i] = 0.f;}
	maxpix = 2.f*templ.sxmax();
	for(j=0; j<nclusx; ++j) {
		xsum[j] = qscale*cluster[j];
		qtotal += xsum[j];
	   if(xsum[j] > maxpix) {xsum[j] = maxpix;}
   }
	
	// next, identify the x-cluster ends, count total pixels, nxpix, and logical pixels, logxpx   
	
	fxpix = -1;
	nxpix=0;
	lxpix=0;
	logxpx=0;
	for(i=0; i<BSXSIZE; ++i) {
	   if(xsum[i] > 0.f) {
	      if(fxpix == -1) {fxpix = i;}
			++logxpx;
			++nxpix;
			lxpix = i;
		}
	}
	
	
	//	dlengthx = (float)nxpix - templ.clslenx();
	
	// Make sure cluster is continuous
	
	if((lxpix-fxpix+1) != nxpix) { 
		
	   LOGDEBUG("SiStripTemplateReco") << "x-length of pixel cluster doesn't agree with number of pixels above threshold" << ENDL;
	   if (theVerboseLevel > 2) {
			LOGDEBUG("SiStripTemplateReco") << "xsum[] = ";
			for(i=0; i<BSXSIZE-1; ++i) {LOGDEBUG("SiStripTemplateReco") << xsum[i] << ", ";}           
			LOGDEBUG("SiStripTemplateReco") << ENDL;
		}
		
	   return 2; 
	}
	
	// If cluster is longer than max template size, technique fails
	
	if(nxpix > TSXSIZE) { 
		
	   LOGDEBUG("SiStripTemplateReco") << "x-length of pixel cluster is larger than maximum template size" << ENDL;
	   if (theVerboseLevel > 2) {
			LOGDEBUG("SiStripTemplateReco") << "xsum[] = ";
			for(i=0; i<BSXSIZE-1; ++i) {LOGDEBUG("SiStripTemplateReco") << xsum[i] << ", ";}           
			LOGDEBUG("SiStripTemplateReco") << ENDL;
		}
		
	   return 7; 
	}
	
	// next, center the cluster on template center if necessary   
	
	midpix = (fxpix+lxpix)/2;
	shiftx = templ.cxtemp() - midpix;
	if(shiftx > 0) {
	   for(i=lxpix; i>=fxpix; --i) {
		   xsum[i+shiftx] = xsum[i];
		   xsum[i] = 0.f;
	   }
	} else if (shiftx < 0) {
	   for(i=fxpix; i<=lxpix; ++i) {
	      xsum[i+shiftx] = xsum[i];
		   xsum[i] = 0.f;
	   }
	}
	lxpix +=shiftx;
	fxpix +=shiftx;
	
	// If the cluster boundaries are OK, add pesudopixels, otherwise quit
	
	if(fxpix > 1 && fxpix <BSXM2) {
	   xsum[fxpix-1] = pseudopix;
	   xsum[fxpix-2] = 0.2f*pseudopix;
	} else {return 9;}
	if(lxpix > 1 && lxpix < BSXM2) {
	   xsum[lxpix+1] = pseudopix;
	   xsum[lxpix+2] = 0.2f*pseudopix;
	} else {return 9;}
	
	// finally, determine if pixel[0] is a double pixel and make an origin correction if it is   
	
	originx = 0.f;
        	
// uncertainty and final corrections depend upon total charge bin 	   
	   
	qavg = templ.qavg();
	fq = qtotal/qavg;
	if(fq > 3.0f) {
	   binq=0;
	} else {
	   if(fq > 2.0f) {
	      binq=1;
	   } else {
		  if(fq > 1.70f) {
			 binq=2;
		  } else {
			 binq=3;
		  }
	   }
	}
	
	// Return the charge bin via the parameter list unless the charge is too small (then flag it)
	
	q2bin = binq;
	if(qtotal < 1.9f*templ.qmin()) {q2bin = 5;} else {
		if(qtotal < 1.9f*templ.qmin(1)) {q2bin = 4;}
	}
	if (theVerboseLevel > 9) {
		LOGDEBUG("SiStripTemplateReco") <<
		"ID = " << id <<  
		" cot(alpha) = " << cotalpha << " cot(beta) = " << cotbeta << 
		" nclusx = " << nclusx << ENDL;
	}
	
// binqerr is the charge bin for error estimation
	
	binqerr = binq;
	if(binqerr > 2) binqerr = 2;	
		
// Calculate the Vavilov probability that the cluster charge is consistent with a merged cluster
		
	if(speed < 0) {
	   templ.vavilov2_pars(mpv, sigmaQ, kappa);
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	   if((sigmaQ <=0.) || (mpv <= 0.) || (kappa < 0.01) || (kappa > 9.9)) {
			throw cms::Exception("DataCorrupt") << "SiStripTemplateSplit::Vavilov parameters mpv/sigmaQ/kappa = " << mpv << "/" << sigmaQ << "/" << kappa << std::endl;
	   }
#else
	   assert((sigmaQ > 0.) && (mpv > 0.) && (kappa > 0.01) && (kappa < 10.));
#endif
	   xvav = ((double)qtotal-mpv)/sigmaQ;
	   beta2 = 1.;
//  VVIObj is a private port of CERNLIB VVIDIS
	   sistripvvi::VVIObj vvidist(kappa, beta2, 1);
	   prvav = vvidist.fcn(xvav);			
	   prob2Q = 1. - prvav;
	   if(prob2Q < prob2Qmin) {prob2Q = prob2Qmin;}
	} else {
		prob2Q = -1.f;
	}
	
	
// Next, generate the 3d x-template
   
	templ.xtemp3d_int(nxpix, nxbin);
	
// retrieve the number of x-bins 
	
	xcbin = nxbin/2;
	
// Next, decide on chi^2 min search parameters
	
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
	if(speed < -1 || speed > 2) {
		throw cms::Exception("DataCorrupt") << "SiStripTemplateReco::PixelTempReco2D called with illegal speed = " << speed << std::endl;
	}
#else
	assert(speed >= -1 && speed < 3);
#endif
	fxbin = 0; lxbin = nxbin-1; djx = 1;
	if(speed > 0) {
        djx = 2;
        if(speed > 1) {djx = 4;}
	}
			  				  
// Define the maximum signal to allow before de-weighting a pixel 

	sxthr = 1.1f*maxpix;
			  	   
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

//	for(i=0; i<BYSIZE; ++i) { xsig2[i] = 0.; }
	templ.xsigma2(fxpix, lxpix, sxthr, xsum, xsig2);
			  
// Find the template bin that minimizes the Chi^2 

	chi2xmin = 1.e15;
	ss2 = 0.f;
	for(i=fxpix-2; i<=lxpix+2; ++i) {
		xw2[i] = 1.f/xsig2[i];
		xsw[i] = xsum[i]*xw2[i];
		ss2 += xsum[i]*xsw[i];
	}
	minbinj = -1; 
	minbink = -1;
	deltaj = djx;
	jmin = fxbin;
	jmax = lxbin;
	kmin = fxbin;
	kmax = lxbin;
    std::vector<float> xtemp(BSXSIZE);
	while(deltaj > 0) {
        for(j=jmin; j<jmax; j+=deltaj) {
			km = std::min(kmax, j);
            for(k=kmin; k<=km; k+=deltaj) {
                
                // Get the template for this set of indices
                
                templ.xtemp3d(j, k, xtemp);
                
                ssa = 0.f;
                sa2 = 0.f;
                for(i=fxpix-2; i<=lxpix+2; ++i) {
                    ssa += xsw[i]*xtemp[i];
                    sa2 += xtemp[i]*xtemp[i]*xw2[i];
                }
                rat=ssa/ss2;
                if(rat <= 0.f) {LOGERROR("SiStripTemplateSplit") << "illegal chi2xmin normalization = " << rat << ENDL; rat = 1.;}
                chi2x=ss2-2.f*ssa/rat+sa2/(rat*rat);
                if(chi2x < chi2xmin) {
                    chi2xmin = chi2x;
                    minbinj = j;
                    minbink = k;
				}
            }
        } 
        deltaj /= 2;
        if(minbinj > fxbin) {jmin = minbinj - deltaj;} else {jmin = fxbin;}
        if(minbinj < lxbin) {jmax = minbinj + deltaj;} else {jmax = lxbin;}
        if(minbink > fxbin) {kmin = minbink - deltaj;} else {kmin = fxbin;}
        if(minbink < lxbin) {kmax = minbink + deltaj;} else {kmax = lxbin;}		
	}
	
	if (theVerboseLevel > 9) {
		LOGDEBUG("SiStripTemplateReco") <<
		"minbinj/minbink " << minbinj<< "/" << minbink << " chi2xmin = " << chi2xmin << ENDL;
	}

// Do not apply final template pass to 1-pixel clusters (use calibrated offset)
	
	if(logxpx == 1) {
	
		delta = templ.dxone();
		sigma = templ.sxone();
		xrec1 = 0.5f*(fxpix+lxpix-2*shiftx+2.f*originx)*xsize-delta;
	   xrec2 = xrec1;
	   if(sigma <= 0.f) {
	      sigmax = xsize/sqrt12;
	   } else {
          sigmax = sigma;
	   }
	   
// Do probability calculation for one-pixel clusters

		chi21max = fmax(chi21min, (double)templ.chi2xminone());
		chi2xmin -=chi21max;
		if(chi2xmin < 0.) {chi2xmin = 0.;}
		meanx = fmax(mean1pix, (double)templ.chi2xavgone());
		hchi2 = chi2xmin/2.; hndof = meanx/2.;
		prob2x = 1. - TMath::Gamma(hndof, hchi2);
	   
	} else {
	   
			
// uncertainty and final correction depend upon charge bin 	   
	   
		bias = templ.xavgc2m(binq);
		k = std::min(minbink, minbinj);
		j = std::max(minbink, minbinj);
		xrec1 = (0.125f*(minbink-xcbin)+BSHX-(float)shiftx+originx)*xsize - bias;
		xrec2 = (0.125f*(minbinj-xcbin)+BSHX-(float)shiftx+originx)*xsize - bias;
		sigmax = sqrt2x*templ.xrmsc2m(binqerr);
			
	   
// Do goodness of fit test in y  
	   
	   if(chi2xmin < 0.0) {chi2xmin = 0.0;}
	   meanx = templ.chi2xavgc2m(binq);
	   if(meanx < 0.01) {meanx = 0.01;}
       hchi2 = chi2xmin/2.; hndof = meanx/2.;
	   prob2x = 1. - TMath::Gamma(hndof, hchi2);
   }
	
	//  Don't return exact zeros for the probability
	
	if(prob2x < probmin) {prob2x = probmin;}
	
	
    return 0;
} // StripTempSplit 


// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit positions for pixel clusters.      
//! \param         id - (input) identifier of the template to use                                  
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014) 
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)  
//! \param      locBy - (input) the sign of the local B_y field to specify the Lorentz drift direction 
//! \param    cluster - (input) boost multi_array container of 7x21 array of pixel signals, 
//!           origin of local coords (0,0) at center of pixel cluster[0][0].                      
//! \param      templ - (input) the template used in the reconstruction
//! \param      xrec1 - (output) best estimate of first x-coordinate of hit in microns
//! \param      xrec2 - (output) best estimate of second x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec1 and xrec2 in microns
//! \param     prob2x - (output) probability describing goodness-of-fit to a merged cluster hypothesis for x-reco
//! \param      q2bin - (output) index (0-4) describing the charge of the cluster assuming a merged 2-hit cluster hypothesis
//!                     [0: 1.5<Q/Qavg, 1: 1<Q/Qavg<1.5, 2: 0.85<Q/Qavg<1, 3: 0.95Qmin<Q<0.85Qavg, 4: Q<0.95Qmin]
//! \param     prob2Q - (output) probability that the cluster charge is compatible with a 2-hit merging
// *************************************************************************************************************************************
int SiStripTemplateSplit::StripTempSplit(int id, float cotalpha, float cotbeta, float locBy, std::vector<float>& cluster, 
                                         SiStripTemplate& templ, 
                                         float& xrec1, float& xrec2, float& sigmax, float& prob2x, int& q2bin, float& prob2Q)

{
    // Local variables 
	const int speed = 1;

    return SiStripTemplateSplit::StripTempSplit(id, cotalpha, cotbeta, locBy, speed, cluster, templ, xrec1, xrec2, sigmax, prob2x, q2bin, prob2Q);
} // StripTempSplit
