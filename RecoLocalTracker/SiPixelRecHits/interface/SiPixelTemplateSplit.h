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
    typedef boost::multi_array<float, 3> array_3d;

	int PixelTempSplit(int id, bool fpix, float cotalpha, float cotbeta, array_2d cluster, 
				std::vector<bool> ydouble, std::vector<bool> xdouble, 
				SiPixelTemplate& templ, 
				float& yrec1, float& yrec2, float& sigmay, float& proby, 
				float& xrec1, float& xrec2, float& sigmax, float& probx, int& qbin);

}
				

