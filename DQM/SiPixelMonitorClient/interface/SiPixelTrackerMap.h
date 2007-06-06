// -*- C++ -*-
//
// Package:    SiPixelMonitorClient
// Class:      SiPixelTrackerMap
// 
/**\class 

 Description: Pixel DQM source used to generate an svgmap.xml file with the topology of 
 the Pixel Detector (used by web-based clients)

 Implementation:
     This class inherits from the Common/Tools/TrackerMap.cc and reimplements the methods used 
     to create a reduced-set map with Pixel Detectors only
*/
//
// Original Author:  Dario Menasce
//         Created:  
// $Id: SiPixelTrackerMap.cc,v 1.0 2007/05/18 20:24:44 menasce Exp $
//
//

#include "CommonTools/TrackerMap/interface/TrackerMap.h"

using namespace std ;

class SiPixelTrackerMap : public TrackerMap
{
 public :
 
          SiPixelTrackerMap(string   s           = " ",
	                    int      xsize1      = 340,
			    int      ysize1      = 200  ) ;
         ~SiPixelTrackerMap(void) {;} 
	 
	  void drawModule(TmModule * mod, 
	                  int        key,
			  int        nlay, 
			  bool       print_total        ) ;
          void print(     bool       print_total = true,
	                  float      minval      = 0., 
			  float      maxval      = 0.   ) ;
 
 private :
 
          int    dummy ;
	  string title ;
} ;
