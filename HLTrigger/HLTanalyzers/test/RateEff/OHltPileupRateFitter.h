//////////////////////////////////////////////////////////
//
// Class to refit rates with non-linear dependence for pileup
//
//////////////////////////////////////////////////////////

#ifndef OHltPileupRateFitter_h
#define OHltPileupRateFitter_h

#include <vector>
#include "OHltMenu.h"
#include "OHltConfig.h"
#include "OHltTree.h"

class OHltPileupRateFitter
{
public:

   OHltPileupRateFitter() {}

   virtual ~OHltPileupRateFitter() {}

   void fitForPileup(
		     OHltConfig *thecfg, 
		     OHltMenu *themenu,		     
		     std::vector< std::vector<double> > RatePerLS,
		     std::vector<double> tTotalRatePerLS,
		     std::vector<double> tLumiPerLS,
		     std::vector< std::vector<double> > tCountPerLS,
		     std::vector<double> ttotalCountPerLS,
		     TFile *histogramfile);

   std::vector< std::vector<double> > RatePerLS;
   std::vector<double> totalRatePerLS;
   std::vector<double> LumiPerLS;
   std::vector< std::vector<double> > CountPerLS;
   std::vector<double> totalCountPerLS;

};

#endif
