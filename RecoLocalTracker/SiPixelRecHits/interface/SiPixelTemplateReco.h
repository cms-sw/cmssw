//
//  SiPixelTemplateReco.cc (Version 8.25)
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
//  V7.00 - Decouple BPix and FPix information into separate templates
//  Pass all containers by alias to prevent excessive cpu-usage (v7.01)
//  Slightly modify search bin range to avoid problem with single pixel clusters + large Lorentz drift (V7.02)
//
//  V8.00 - Add 2D probabilities, take pixel sizes from the template
//  V8.05 - Shift 2-D cluster to center on the buffer
//  V8.06 - Add locBz to the 2-D template (causes failover to the simple template when the cotbeta-locBz correlation is incorrect ... ie for non-IP tracks).
//        - include minimum value for prob2D (1.e-30)
//  V8.07 - Tune 2-d probability: consider only pixels above threshold and use threshold value for zero signal pixels (non-zero template)
//  V8.10 - Remove 2-d probability for ineffectiveness and replace with simple cluster charge probability
//  V8.11 - Change probQ to upper tail probability always (rather than two-sided tail probability)
//  V8.20 - Use template cytemp/cxtemp methods to center the data cluster in the right place when the template becomes asymmetric after irradiation
//  V8.25 - Incorporate VIs speed improvements
//
//
//
//  Created by Morris Swartz on 10/27/06.
//
//

#ifndef SiPixelTemplateReco_h
#define SiPixelTemplateReco_h 1

#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateDefs.h"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#else
#include "SiPixelTemplate.h"
#endif

#define N2D 500

#include <vector>
#include "boost/multi_array.hpp"


namespace SiPixelTemplateReco 
 {
 
    typedef boost::multi_array<float, 2> array_2d;

	int PixelTempReco2D(int id, float cotalpha, float cotbeta, float locBz, array_2d& cluster, 
				std::vector<bool>& ydouble, std::vector<bool>& xdouble, 
				SiPixelTemplate& templ, 
				float& yrec, float& sigmay, float& proby, float& xrec, float& sigmax, float& probx, int& qbin, int speed, bool deadpix, std::vector<std::pair<int, int> >& zeropix,
				float& probQ);

	int PixelTempReco2D(int id, float cotalpha, float cotbeta, float locBz, array_2d& cluster, 
				std::vector<bool>& ydouble, std::vector<bool>& xdouble, 
				SiPixelTemplate& templ, 
				float& yrec, float& sigmay, float& proby, float& xrec, float& sigmax, float& probx, int& qbin, int speed,
				float& probQ);
	 
	 int PixelTempReco2D(int id, float cotalpha, float cotbeta, array_2d& cluster, 
						 std::vector<bool>& ydouble, std::vector<bool>& xdouble, 
						 SiPixelTemplate& templ, 
						 float& yrec, float& sigmay, float& proby, float& xrec, float& sigmax, float& probx, int& qbin, int speed,
						 float& probQ);
	 
	 int PixelTempReco2D(int id, float cotalpha, float cotbeta, array_2d& cluster, 
								std::vector<bool>& ydouble, std::vector<bool>& xdouble, 
								SiPixelTemplate& templ, 
								float& yrec, float& sigmay, float& proby, float& xrec, float& sigmax, float& probx, int& qbin, int speed);
 }
				
#endif
