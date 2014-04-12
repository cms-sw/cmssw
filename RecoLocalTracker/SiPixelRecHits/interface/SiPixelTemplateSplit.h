//
//  SiPixelTemplateSplit.cc (Version 2.30)
//
//  Procedure to fit two templates (same angle hypotheses) to a single cluster
//  Return two x- and two y-coordinates for the cluster
//
//  Created by Morris Swartz on 04/10/08.
//
//  Incorporate "cluster repair" to handle dead pixels
//  Take truncation size from new pixmax information
//  Change to allow template sizes to be changed at compile time
//  Move interpolation range error to LogDebug
//  Add q2bin = 5 and change 1-pixel probability to use new template info
//  Add floor for probabilities (no exact zeros)
//  Add ambiguity resolution with crude 2-D templates (v2.00)
//  Pass all containers by alias to prevent excessive cpu-usage (v2.01)
//  Add ambiguity resolution with fancy 2-D templates (v2.10)
//  Make small change to indices for ambiguity resolution (v2.11)
//  Tune x and y errors separately (v2.12)
//  Use template cytemp/cxtemp methods to center the data cluster in the right place when the templates become asymmetric after irradiation (v2.20)
//  Add charge probability to the splitter [tests consistency with a two-hit merged cluster hypothesis]  (v2.20)
//  Improve likelihood normalization slightly (v2.21)
//  Replace hardwired pixel size derived errors with ones from templated pixel sizes (v2.22)
//  Add shape and charge probabilities for the merged cluster hypothesis (v2.23)
//  Incorporate VI-like speed improvements (v2.25)
//  Improve speed by eliminating the triple index boost::multiarray objects and add speed switch to optimize the algorithm (v2.30)
//
 
#ifndef SiPixelTemplateSplit_h
#define SiPixelTemplateSplit_h 1

#include "SiPixelTemplateDefs.h"

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate2D.h"
#else
#include "SiPixelTemplate.h"
#include "SiPixelTemplate2D.h"
#endif

#include <vector>
#include "boost/multi_array.hpp"



namespace SiPixelTemplateSplit
 {
 
    typedef boost::multi_array<float, 2> array_2d;
    typedef boost::multi_array<bool, 2> array_2d_bool;
    typedef boost::multi_array<float, 3> array_3d;
	 
	 	 
	 
	int PixelTempSplit(int id, float cotalpha, float cotbeta, array_2d& cluster, 
				std::vector<bool>& ydouble, std::vector<bool>& xdouble, 
				SiPixelTemplate& templ, 
				float& yrec1, float& yrec2, float& sigmay, float& prob2y, 
							 float& xrec1, float& xrec2, float& sigmax, float& prob2x, int& q2bin, float& prob2Q, bool resolve, int speed, float& dchisq, bool deadpix, 
							 std::vector<std::pair<int, int> >& zeropix, SiPixelTemplate2D& templ2D);
	 
	 int PixelTempSplit(int id, float cotalpha, float cotbeta, array_2d& cluster, 
							  std::vector<bool>& ydouble, std::vector<bool>& xdouble, 
							  SiPixelTemplate& templ, 
							  float& yrec1, float& yrec2, float& sigmay, float& prob2y, 
							  float& xrec1, float& xrec2, float& sigmax, float& prob2x, int& q2bin, float& prob2Q, bool resolve, int speed, float& dchisq, SiPixelTemplate2D& templ2D);
	 
	int PixelTempSplit(int id, float cotalpha, float cotbeta, array_2d& cluster, 
				std::vector<bool>& ydouble, std::vector<bool>& xdouble, 
				SiPixelTemplate& templ, 
				float& yrec1, float& yrec2, float& sigmay, float& prob2y, 
				float& xrec1, float& xrec2, float& sigmax, float& prob2x, int& q2bin, float& prob2Q, bool resolve, float& dchisq, SiPixelTemplate2D& templ2D);
	 
	 int PixelTempSplit(int id, float cotalpha, float cotbeta, array_2d& cluster, 
						std::vector<bool>& ydouble, std::vector<bool>& xdouble, 
						SiPixelTemplate& templ, 
						float& yrec1, float& yrec2, float& sigmay, float& prob2y, 
						float& xrec1, float& xrec2, float& sigmax, float& prob2x, int& q2bin, float& prob2Q, SiPixelTemplate2D& templ2D);

	 int PixelTempSplit(int id, float cotalpha, float cotbeta, array_2d& cluster, 
							  std::vector<bool>& ydouble, std::vector<bool>& xdouble, 
							  SiPixelTemplate& templ, 
							  float& yrec1, float& yrec2, float& sigmay, float& prob2y, 
							  float& xrec1, float& xrec2, float& sigmax, float& prob2x, int& q2bin, SiPixelTemplate2D& templ2D);
	 
}
				
#endif
