//
//  SiPixelTemplateReco.cc (Version 6.01)
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
//  Return error if no pixels in cluster
//  Replace 4 cout's with LogError's
//  Add LogDebug I/O to report various common errors
//  Incorporate "cluster repair" to handle dead pixels
//  Take truncation size from new pixmax information
//  Change to allow template sizes to be changed at compile time
//  Move interpolation range error to LogDebug
//  Add qbin = 5 and change 1-pixel probability to use new template info
//  Add floor for probabilities (no exact zeros)
//  Replace asserts with exceptions in CMSSW
//  Change calling sequence to handle cot(beta)<0 for FPix cosmics
//
//  Created by Morris Swartz on 10/27/06.
//  Copyright 2006 __TheJohnsHopkinsUniversity__. All rights reserved.
//
//
 
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#else
#include "SiPixelTemplate.h"
#endif

#define TYSIZE 21
#define BYSIZE TYSIZE+4
#define BHY 12 // = BYSIZE/2
#define BYM1 TYSIZE+3
#define BYM2 TYSIZE+2
#define BYM3 TYSIZE+1
#define TXSIZE 13
#define BXSIZE TXSIZE+4
#define BHX 8 // = BXSIZE/2
#define BXM1 TXSIZE+3
#define BXM2 TXSIZE+2
#define BXM3 TXSIZE+1

#include <vector>
#include "boost/multi_array.hpp"


namespace SiPixelTemplateReco 
 {
 
    typedef boost::multi_array<float, 2> array_2d;

	int PixelTempReco2D(int id, bool fpix, float cotalpha, float cotbeta, float locBz, array_2d cluster, 
				std::vector<bool> ydouble, std::vector<bool> xdouble, 
				SiPixelTemplate& templ, 
				float& yrec, float& sigmay, float& proby, float& xrec, float& sigmax, float& probx, int& qbin, int speed, bool deadpix, std::vector<std::pair<int, int> > zeropix);

	int PixelTempReco2D(int id, bool fpix, float cotalpha, float cotbeta, float locBz, array_2d cluster, 
				std::vector<bool> ydouble, std::vector<bool> xdouble, 
				SiPixelTemplate& templ, 
				float& yrec, float& sigmay, float& proby, float& xrec, float& sigmax, float& probx, int& qbin, int speed);
	 
	 int PixelTempReco2D(int id, bool fpix, float cotalpha, float cotbeta, array_2d cluster, 
						 std::vector<bool> ydouble, std::vector<bool> xdouble, 
						 SiPixelTemplate& templ, 
						 float& yrec, float& sigmay, float& proby, float& xrec, float& sigmax, float& probx, int& qbin, int speed);
 }
				

