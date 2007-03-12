//
//   SiPixelTemplateReco.h (v2.42)
//
//  Add goodness-of-fit to algorithm, include single pixel clusters in chi2 calculation
//  Try "decapitation" of large single pixels
//  Add correction for (Q_F-Q_L)/(Q_F+Q_L) bias
//  Add cot(beta) reflection to reduce y-entries and more sophisticated x-interpolation
//  Fix small double pixel bug with decapitation (2.41 5-Mar-2007).
//  Fix pseudopixel bug causing possible memory overwrite (2.42 12-Mar-2007)
//
//    Created by Morris Swartz on 10/27/06.
//  Copyright 2006 __TheJohnsHopkinsUniversity__. All rights reserved.
//
//
 
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplate.h"
#else
#include "SiPixelTemplate.h"
#endif

#include <vector>
#include "boost/multi_array.hpp"


namespace SiPixelTemplateReco 
 {
 
    typedef boost::multi_array<float, 2> array_2d;

	int PixelTempReco2D(int id, bool fpix, float cotalpha, float cotbeta, array_2d cluster, 
				std::vector<bool> ydouble, std::vector<bool> xdouble, 
				SiPixelTemplate& templ, 
				float& yrec, float& sigmay, float& proby, float& xrec, float& sigmax, float& probx, int& qbin);

}
				

