#include "CommonTools/TrackerMap/interface/TrackerMap.h"

using namespace std ;

class SiStripTrackerMap : public TrackerMap
{
 public :
 
          SiStripTrackerMap(string     s           = " ",
	                    int        xsize1      = 340,
			    int        ysize1      = 200  ) ;
         ~SiStripTrackerMap(void) {;} 
	 
          void printonline();
          void printlayers(bool print_total=true,float minval=0., float maxval=0.,std::string s="layer");
 
 private :
 
          int    dummy ;
          bool firsttime;
	  string title ;
} ;

