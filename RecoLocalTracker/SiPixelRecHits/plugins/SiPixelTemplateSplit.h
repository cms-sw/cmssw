//
//  SiPixelTemplateSplit.cc (Version 1.05)
//
//  Procedure to fit two templates (same angle hypotheses) to a single cluster
//  Return two x- and two y-coordinates for the cluster
//
//  Created by Morris Swartz on 04/10/08.
//  Copyright 2008 __TheJohnsHopkinsUniversity__. All rights reserved.
//
//  Incorporate "cluster repair" to handle dead pixels
//  Take truncation size from new pixmax information
//  Change to allow template sizes to be changed at compile time
//  Move interpolation range error to LogDebug
//  Add qbin = 5 and change 1-pixel probability to use new template info
//  Add floor for probabilities (no exact zeros)
//
 
#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "SiPixelTemplate.h"
#else
#include "SiPixelTemplate.h"
#endif

//#define TYSIZE 21
//#define BYSIZE TYSIZE+4
//#define BHY 12 // = BYSIZE/2
//#define BYM1 TYSIZE+3
//#define BYM2 TYSIZE+2
//#define BYM3 TYSIZE+1
//#define TXSIZE 7
//#define BXSIZE TXSIZE+4
//#define BHX 5 // = BXSIZE/2
//#define BXM1 TXSIZE+3
//#define BXM2 TXSIZE+2
//#define BXM3 TXSIZE+1

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
				float& xrec1, float& xrec2, float& sigmax, float& probx, int& qbin, bool deadpix, std::vector<std::pair<int, int> > zeropix);

	int PixelTempSplit(int id, bool fpix, float cotalpha, float cotbeta, array_2d cluster, 
				std::vector<bool> ydouble, std::vector<bool> xdouble, 
				SiPixelTemplate& templ, 
				float& yrec1, float& yrec2, float& sigmay, float& proby, 
				float& xrec1, float& xrec2, float& sigmax, float& probx, int& qbin);

}
				

