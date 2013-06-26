//
//  SiStripTemplateReco.h Version 2.01 (V1.00 based on SiPixelTemplateReco Version 8.20)
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

#ifndef SiStripTemplateReco_h
#define SiStripTemplateReco_h 1

#include "SiStripTemplateDefs.h"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplate.h"
#else
#include "SiStripTemplate.h"
#endif

#include <vector>


namespace SiStripTemplateReco 
 {

	int StripTempReco1D(int id, float cotalpha, float cotbeta, float locBy, std::vector<float>& cluster, 
				SiStripTemplate& templ, float& xrec, float& sigmax, float& probx, int& qbin, int speed, float& probQ);

 }
				
#endif
