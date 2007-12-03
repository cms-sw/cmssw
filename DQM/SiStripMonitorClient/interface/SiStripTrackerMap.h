#ifndef _SiStripTrackerMap_h_
#define _SiStripTrackerMap_h_
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace std ;

class SiStripTrackerMap : public TrackerMap
{
 public :
 
          SiStripTrackerMap(const edm::ParameterSet & tkmapPset, 
                            const edm::ESHandle<SiStripFedCabling> tkFed);
         ~SiStripTrackerMap(void) {;} 
	 
          void printonline();
          void fedprintonline();
          void printlayers(bool print_total=true,float minval=0., float maxval=0.,std::string s="layer");
 
 private :
 
          int    dummy ;
          bool firsttime;
          bool fedfirsttime;
	  string title ;
} ;
#endif
