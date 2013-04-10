//
//  SiStripTemplateReco.cc Version 2.01 (V1.00 based on SiPixelTemplateReco Version 8.20)
//
//  V1.05 - add VI optimizations from pixel reco
//  V1.07 - Improve cluster centering
//  V2.00 - Change to chi2min estimator and choose barycenter when errors/bias are large
//        - Increase buffer size to avoid truncation
//        - Change pseudo-strip weighting to accommodate asymmetric clusters (deco mode)
//        - Remove obsolete sorting of signal for weighting (truncation makes it worthless)
//  V2.01 - Add barycenter bias correction
//  
//
//
//  Created by Morris Swartz on 10/27/06.
//
//

#include <math.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <iostream>
// ROOT::Math has a c++ function that does the probability calc, but only in v5.12 and later
#include "TMath.h"
#include "Math/DistFunc.h"
// Use current version of gsl instead of ROOT::Math
//#include <gsl/gsl_cdf.h>

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplateReco.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/VVIObj.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) edm::LogError(x)
#define LOGDEBUG(x) LogDebug(x)
constexpr int theVerboseLevel = 2;
#define ENDL " "
#include "FWCore/Utilities/interface/Exception.h"
#else
#include "SiStripTemplateReco.h"
#include "VVIObj.h"
//static int theVerboseLevel = {2};
#define LOGERROR(x) std::cout << x << ": "
#define LOGDEBUG(x) std::cout << x << ": "
#define ENDL std::endl
#endif

using namespace SiStripTemplateReco;

// *************************************************************************************************************************************
//! Reconstruct the best estimate of the hit position for strip clusters,
//! includes autoswitching to barycenter when that technique is more accurate.
//! \param         id - (input) identifier of the template to use                                  
//! \param   cotalpha - (input) the cotangent of the alpha track angle (see CMS IN 2004/014) 
//! \param    cotbeta - (input) the cotangent of the beta track angle (see CMS IN 2004/014)  
//! \param      locBy - (input) the sign of the local B_y field to specify the Lorentz drift direction 
//! \param    cluster - (input) boost multi_array container array of 13 pixel signals, 
//!           origin of local coords (0,0) at center of pixel cluster[0].                    
//! \param      templ - (input) the template used in the reconstruction
//! \param       xrec - (output) best estimate of x-coordinate of hit in microns
//! \param     sigmax - (output) best estimate of uncertainty on xrec in microns
//! \param      probx - (output) probability describing goodness-of-fit for x-reco
//! \param       qbin - (output) index (0-4) describing the charge of the cluster
//!                       qbin = 0        Q/Q_avg > 1.5   [few % of all hits]
//!                              1  1.5 > Q/Q_avg > 1.0   [~30% of all hits]
//!                              2  1.0 > Q/Q_avg > 0.85  [~30% of all hits]
//!                              3 0.85 > Q/Q_avg > min1  [~30% of all hits]
//!                              4 min1 > Q/Q_avg > min2  [~0.1% of all hits]
//!                              5 min2 > Q/Q_avg         [~0.1% of all hits]
//! \param      speed - (input) switch (-2->5) trading speed vs robustness
//!                     -2       totally bombproof, searches the entire 41 bin range at full density (equiv to V2_4), 
//!                              calculates Q probability w/ VVIObj (better but slower)
//!                     -1       totally bombproof, searches the entire 41 bin range at full density (equiv to V2_4), 
//!                              calculates Q probability w/ TMath::VavilovI (poorer but faster)
//!                      0       totally bombproof, searches the entire 41 bin range at full density (equiv to V2_4)
//!                      1       faster, searches reduced 25 bin range (no big pix) + 33 bins (big pix at ends) at full density
//!                      2       faster yet, searches same range as 1 but at 1/2 density
//!                      3       fastest, searches same range as 1 but at 1/4 density (no big pix) and 1/2 density (big pix in cluster)
//!                      4       fastest w/ Q prob, searches same range as 1 but at 1/4 density (no big pix) and 1/2 density (big pix in cluster), 
//!                              calculates Q probability w/ VVIObj (better but slower)
//!                      5       fastest w/ Q prob, searches same range as 1 but at 1/4 density (no big pix) and 1/2 density (big pix in cluster),
//!                              calculates Q probability w/ TMath::VavilovI (poorer but faster)
//! \param      probQ - (output) the Vavilov-distribution-based cluster charge probability
// *************************************************************************************************************************************
int SiStripTemplateReco::StripTempReco1D(int id, float cotalpha, float cotbeta, float locBy, std::vector<float>& cluster, 
		    SiStripTemplate& templ, 
		    float& xrec, float& sigmax, float& probx, int& qbin, int speed, float& probQ)
			
{
    // Local variables 
	int i, j, minbin, binl, binh, binq, midpix;
	int fxpix, nxpix, lxpix, logxpx, shiftx, ftpix, ltpix;
	int nclusx;
	int deltaj, jmin, jmax, fxbin, lxbin, djx;
	float sxthr, rnorm, delta, sigma, pseudopix, qscale, q50, q100;
	float ss2, ssa, sa2, ssba, saba, sba2, rat, fq, qtotal, barycenter, sigmaxbcn;
	float originx, qfx, qlx, bias, biasbcn, maxpix;
	double chi2x, meanx, chi2xmin, chi21max;
	double hchi2, hndof, prvav, mpv, sigmaQ, kappa, xvav, beta2;
	float xtemp[41][BSXSIZE], xsum[BSXSIZE];
	float chi2xbin[41], xsig2[BSXSIZE];
	float xw2[BSXSIZE],  xsw[BSXSIZE];
	bool calc_probQ, use_VVIObj;
	float xsize;
	const float probmin={1.110223e-16};
	const float probQmin={1.e-5};
	
// The minimum chi2 for a valid one pixel cluster = pseudopixel contribution only

	const double mean1pix={0.100}, chi21min={0.160};
		      
// First, interpolate the template needed to analyze this cluster     
// check to see of the track direction is in the physical range of the loaded template

	if(!templ.interpolate(id, cotalpha, cotbeta, locBy)) {
	   if (theVerboseLevel > 2) {LOGDEBUG("SiStripTemplateReco") << "input cluster direction cot(alpha) = " << cotalpha << ", cot(beta) = " << cotbeta << ", local B_y = " << locBy << ", template ID = " << id << ", no reconstruction performed" << ENDL;}	
	   return 20;
	}
	
// Check to see if Q probability is selected
	
	calc_probQ = false;
	use_VVIObj = false;
	if(speed < 0) {
		calc_probQ = true;
		if(speed < -1) use_VVIObj = true;
		speed = 0;
	}
	
	if(speed > 3) {
		calc_probQ = true;
		if(speed < 5) use_VVIObj = true;
		speed = 3;
	}
	
// Get pixel dimensions from the template (to allow multiple detectors in the future)
	
	xsize = templ.xsize();
   
// Define size of pseudopixel
	
	q50 = templ.s50();
	q100 = 2.f * q50;
	pseudopix = q50;
	
// Get charge scaling factor

	qscale = templ.qscale();
    	
// enforce maximum size	
	
	nclusx = (int)cluster.size();
	
	if(nclusx > TSXSIZE) {nclusx = TSXSIZE;}
	
// First, rescale all strip charges, sum them and trunate the strip charges      

	qtotal = 0.;
	for(i=0; i<BSXSIZE; ++i) {xsum[i] = 0.f;}
	maxpix = templ.sxmax();
	barycenter = 0.f;
	for(j=0; j<nclusx; ++j) {
		xsum[j] = qscale*cluster[j];
		qtotal += xsum[j];
		barycenter += j*xsize*xsum[j];
	   if(xsum[j] > maxpix) {xsum[j] = maxpix;}
   }
	
	barycenter = barycenter/qtotal - 0.5f*templ.lorxwidth();
	        
// next, identify the x-cluster ends, count total pixels, nxpix, and logical pixels, logxpx   

	fxpix = -1;
	ftpix = -1;
	nxpix=0;
	lxpix=0;
	ltpix=0;
	logxpx=0;
	for(i=0; i<BSXSIZE; ++i) {
	   if(xsum[i] > 0.f) {
	      if(fxpix == -1) {fxpix = i;}
		  ++logxpx;
		  ++nxpix;
		  lxpix = i;
			if(xsum[i] > q100) {
				if(ftpix == -1) {ftpix = i;}
				ltpix = i;
         }				
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

	midpix = (ftpix+ltpix)/2;
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
	
	if(fxpix > 1 && fxpix < BSXM2) {
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
	   
	fq = qtotal/templ.qavg();
	if(fq > 1.5f) {
	   binq=0;
	} else {
	   if(fq > 1.0f) {
	      binq=1;
	   } else {
		  if(fq > 0.85f) {
			 binq=2;
		  } else {
			 binq=3;
		  }
	   }
	}
	
// Return the charge bin via the parameter list unless the charge is too small (then flag it)
	
	qbin = binq;
	if(qtotal < 0.95f*templ.qmin()) {qbin = 5;} else {
		if(qtotal < 0.95f*templ.qmin(1)) {qbin = 4;}
	}
	if (theVerboseLevel > 9) {
       LOGDEBUG("SiStripTemplateReco") <<
        "ID = " << id <<  
         " cot(alpha) = " << cotalpha << " cot(beta) = " << cotbeta << 
         " nclusx = " << nclusx << ENDL;
    }

	
// Next, copy the y- and x-templates to local arrays
   
// First, decide on chi^2 min search parameters
    
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
    if(speed < 0 || speed > 3) {
		throw cms::Exception("DataCorrupt") << "SiStripTemplateReco::StripTempReco2D called with illegal speed = " << speed << std::endl;
	}
#else
    assert(speed >= 0 && speed < 4);
#endif
	fxbin = 2; lxbin = 38; djx = 1;
    if(speed > 0) {
       fxbin = 8; lxbin = 32;
	 }
	
	if(speed > 1) { 
	   djx = 2;
	   if(speed > 2) {
		  djx = 4;
	   }
	}
	
	if (theVerboseLevel > 9) {
       LOGDEBUG("SiStripTemplateReco") <<
        "fxpix " << fxpix << " lxpix = " << lxpix << 
         " fxbin = " << fxbin << " lxbin = " << lxbin << 
         " djx = " << djx << " logxpx = " << logxpx << ENDL;
    }
       	
// Now do the copies
		
	templ.xtemp(fxbin, lxbin, xtemp);
	
// Do the x-reconstruction next 
			  
// Apply the first-pass template algorithm to all clusters

// Modify the template if double pixels are present 
				  
// Define the maximum signal to allow before de-weighting a pixel 

	sxthr = 1.1f*maxpix;
			  
// Evaluate pixel-by-pixel uncertainties (weights) for the templ analysis 

//	for(i=0; i<BSXSIZE; ++i) { xsig2[i] = 0.; }
	templ.xsigma2(fxpix, lxpix, sxthr, xsum, xsig2);
			  
// Find the template bin that minimizes the Chi^2 

	chi2xmin = 1.e15;
	for(i=fxbin; i<=lxbin; ++i) { chi2xbin[i] = -1.e15f;}
	ss2 = 0.f;
	for(i=fxpix-2; i<=lxpix+2; ++i) {
		xw2[i] = 1.f/xsig2[i];
		xsw[i] = xsum[i]*xw2[i];
		ss2 += xsum[i]*xsw[i];
	}
	minbin = -1;
	deltaj = djx;
	jmin = fxbin;
	jmax = lxbin;
	while(deltaj > 0) {
	   for(j=jmin; j<=jmax; j+=deltaj) {
	      if(chi2xbin[j] < -100.f) {
				ssa = 0.f;
				sa2 = 0.f;
				for(i=fxpix-2; i<=lxpix+2; ++i) {
					ssa += xsw[i]*xtemp[j][i];
					sa2 += xtemp[j][i]*xtemp[j][i]*xw2[i];
				}
		     rat=ssa/ss2;
		     if(rat <= 0.f) {LOGERROR("SiStripTemplateReco") << "illegal chi2xmin normalization (1) = " << rat << ENDL; rat = 1.;}
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
	
	if (theVerboseLevel > 9) {
       LOGDEBUG("SiStripTemplateReco") <<
        "minbin " << minbin << " chi2xmin = " << chi2xmin << ENDL;
    }

// Do not apply final template pass to 1-pixel clusters (use calibrated offset)
	
	if(nxpix == 1) {
	
		delta = templ.dxone();
		sigma = templ.sxone();
	   xrec = 0.5f*(fxpix+lxpix-2*shiftx+2.f*originx)*xsize-delta;
	   if(sigma <= 0.f) {
	      sigmax = 28.9f;
	   } else {
          sigmax = sigma;
	   }
	   
// Do probability calculation for one-pixel clusters

		chi21max = fmax(chi21min, (double)templ.chi2xminone());
		chi2xmin -=chi21max;
		if(chi2xmin < 0.) {chi2xmin = 0.;}
		meanx = fmax(mean1pix, (double)templ.chi2xavgone());
		hchi2 = chi2xmin/2.; hndof = meanx/2.;
		probx = 1. - TMath::Gamma(hndof, hchi2);
	   
	} else {
		
// Now make the second, interpolating pass with the templates 
        
		binl = minbin - 1;
        binh = binl + 2;
        if(binl < fxbin) { binl = fxbin;}
        if(binh > lxbin) { binh = lxbin;}	  
        ssa = 0.;
        sa2 = 0.;
        ssba = 0.;
        saba = 0.;
        sba2 = 0.;
        for(i=fxpix-2; i<=lxpix+2; ++i) {
            ssa += xsw[i]*xtemp[binl][i];
            sa2 += xtemp[binl][i]*xtemp[binl][i]*xw2[i];
            ssba += xsw[i]*(xtemp[binh][i] - xtemp[binl][i]);
			saba += xtemp[binl][i]*(xtemp[binh][i] - xtemp[binl][i])*xw2[i];
			sba2 += (xtemp[binh][i] - xtemp[binl][i])*(xtemp[binh][i] - xtemp[binl][i])*xw2[i];
        }
        
// rat is the fraction of the "distance" from template a to template b 	   
        
        rat=(ssba*ssa-ss2*saba)/(ss2*sba2-ssba*ssba);
        if(rat < 0.f) {rat=0.f;}
        if(rat > 1.f) {rat=1.0f;}
        rnorm = (ssa+rat*ssba)/ss2;
        
// Calculate the charges in the first and last pixels
        
        qfx = xsum[fxpix];
        if(logxpx > 1) {
            qlx=xsum[lxpix];
	    } else {
            qlx = qfx;
	    }
		
//  Now calculate the mean bias correction and uncertainties
        
        float qxfrac = (qfx-qlx)/(qfx+qlx);
		bias = templ.xflcorr(binq,qxfrac)+templ.xavg(binq);
        
// uncertainty and final correction depend upon charge bin 	   
        
        xrec = (0.125f*binl+BSHX-2.5f+rat*(binh-binl)*0.125f-(float)shiftx+originx)*xsize - bias;
        sigmax = templ.xrms(binq);
        
// Do goodness of fit test in x  
        
        if(rnorm <= 0.f) {LOGERROR("SiStripTemplateReco") << "illegal chi2x normalization (2) = " << rnorm << ENDL; rnorm = 1.;}
        chi2x=ss2-2.f/rnorm*ssa-2.f/rnorm*rat*ssba+(sa2+2.f*rat*saba+rat*rat*sba2)/(rnorm*rnorm)-templ.chi2xmin(binq);
        if(chi2x < 0.0) {chi2x = 0.0;}
        meanx = templ.chi2xavg(binq);
        if(meanx < 0.01) {meanx = 0.01;}
        // gsl function that calculates the chi^2 tail prob for non-integral dof
        //	   probx = gsl_cdf_chisq_Q(chi2x, meanx);
        //	   probx = ROOT::Math::chisquared_cdf_c(chi2x, meanx, trx0);
        hchi2 = chi2x/2.; hndof = meanx/2.;
        probx = 1. - TMath::Gamma(hndof, hchi2);
			   
// Now choose the better result

        bias = templ.xavg(binq);
	    biasbcn = templ.xavgbcn(binq);
	    sigmaxbcn = templ.xrmsbcn(binq);
		
		if((bias*bias+sigmax*sigmax) > (biasbcn*biasbcn+sigmaxbcn*sigmaxbcn)) {
	   			
			xrec = barycenter - biasbcn;
			sigmax = sigmaxbcn;
            
		}	
	   
	}
	
//  Don't return exact zeros for the probability
	
	if(probx < probmin) {probx = probmin;}
	
//  Decide whether to generate a cluster charge probability
	
	if(calc_probQ) {
		
// Calculate the Vavilov probability that the cluster charge is OK
	
		templ.vavilov_pars(mpv, sigmaQ, kappa);
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
		if((sigmaQ <=0.) || (mpv <= 0.) || (kappa < 0.01) || (kappa > 9.9)) {
			throw cms::Exception("DataCorrupt") << "SiStripTemplateReco::Vavilov parameters mpv/sigmaQ/kappa = " << mpv << "/" << sigmaQ << "/" << kappa << std::endl;
		}
#else
		assert((sigmaQ > 0.) && (mpv > 0.) && (kappa > 0.01) && (kappa < 10.));
#endif
		xvav = ((double)qtotal-mpv)/sigmaQ;
		beta2 = 1.;
		if(use_VVIObj) {			
//  VVIObj is a private port of CERNLIB VVIDIS
                  sistripvvi::VVIObj vvidist(kappa, beta2, 1);
		   prvav = vvidist.fcn(xvav);			
		} else {
//  Use faster but less accurate TMath Vavilov distribution function
			prvav = TMath::VavilovI(xvav, kappa, beta2);
      }
//  Change to upper tail probability
//		if(prvav > 0.5) prvav = 1. - prvav;
//		probQ = (float)(2.*prvav);
		probQ = 1. - prvav;
		if(probQ < probQmin) {probQ = probQmin;}
	} else {
		probQ = -1;
	}
	
	return 0;
} // StripTempReco2D 

