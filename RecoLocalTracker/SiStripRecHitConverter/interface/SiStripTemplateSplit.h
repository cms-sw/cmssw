//
//  SiStripTemplateSplit.h
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
 
#ifndef SiStripTemplateSplit_h
#define SiStripTemplateSplit_h 1

#include "SiStripTemplateDefs.h"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripTemplate.h"
#else
#include "SiStripTemplate.h"
#endif

#include <vector>
#include "boost/multi_array.hpp"



namespace SiStripTemplateSplit
 {
 
	 
	 int StripTempSplit(int id, float cotalpha, float cotbeta, float locBy, int speed, std::vector<float>& cluster, 
				SiStripTemplate& templ, 
							 float& xrec1, float& xrec2, float& sigmax, float& prob2x, int& q2bin, float& prob2Q);

	 
	 int StripTempSplit(int id, float cotalpha, float cotbeta, float locBy, std::vector<float>& cluster, 
                        SiStripTemplate& templ, 
                        float& xrec1, float& xrec2, float& sigmax, float& prob2x, int& q2bin, float& prob2Q);
	 
}
				
#endif
