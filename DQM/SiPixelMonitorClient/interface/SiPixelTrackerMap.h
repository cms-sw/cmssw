#include "CommonTools/TrackerMap/interface/TrackerMap.h"

using namespace std ;

class SiPixelTrackerMap : public TrackerMap
{
 public :
 
          SiPixelTrackerMap(string   s           = " ",
	                    int      xsize1      = 340,
			    int      ysize1      = 200  ) ;
         ~SiPixelTrackerMap(void) {cout << "[~SiPixelTrackerMap]" << endl ;} 
	 
	  void drawModule(TmModule * mod, 
	                  int        key,
			  int        nlay, 
			  bool       print_total        ) ;
          void print(     bool       print_total = true,
	                  float      minval      = 0., 
			  float      maxval      = 0.   ) ;
 
 private :
 
          int dummy ;
} ;
