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
		     std::vector< std::vector<float> > RatePerLS,
		     std::vector<float> tTotalRatePerLS,
		     std::vector<double> tLumiPerLS,
		     std::vector< std::vector<int> > tCountPerLS,
		     std::vector<int> ttotalCountPerLS,
		     TFile *histogramfile);

   std::vector< std::vector<float> > RatePerLS;
   std::vector<float> totalRatePerLS;
   std::vector<double> LumiPerLS;
   std::vector< std::vector<int> > CountPerLS;
   std::vector<int> totalCountPerLS;

};

#endif
