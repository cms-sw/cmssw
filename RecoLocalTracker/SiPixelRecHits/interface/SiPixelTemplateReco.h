/*
 *  SiPixelTemplateReco.h
 *  
 *
 *  Created by Morris Swartz on 10/27/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */
 
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
				float& yrec, float& sigmay, float& xrec, float& sigmax);
				
}
				

