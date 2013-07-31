//////////////////////////////////////////////////////////
//
// Class to print rates
//
//////////////////////////////////////////////////////////

#ifndef OHltRatePrinter_h
#define OHltRatePrinter_h

#include <vector>
#include "OHltMenu.h"
#include "OHltConfig.h"
#include "OHltTree.h"

class OHltRatePrinter
{
public:

   constexpr static double fTwo=2.;

   OHltRatePrinter() {}

   virtual ~OHltRatePrinter() {}

   void SetupAll(
         std::vector<double> Rate,
         std::vector<double> RateErr,
         std::vector<double> spureRate,
         std::vector<double> spureRateErr,
         std::vector<double> pureRate,
         std::vector<double> pureRateErr,
         std::vector< std::vector<double> >coMa,
         std::vector< std::vector<double> > RatePerLS,
         std::vector<int> tRunID,
         std::vector<int> tLumiSection,
         std::vector<double> tTotalRatePerLS,
         std::vector< std::vector<int> > tRefPrescalePerLS,
         std::vector< std::vector<int> > tRefL1PrescalePerLS,
         std::vector<double> averageRefPrescaleHLT,
         std::vector<double> averageRefPrescaleL1,
	 std::vector< std::vector<double> > CountPerLS,
	 std::vector<double> tTotalCountPerLS,
	 std::vector<double> tLumiPerLS);

   void ReorderRunLS();

   void printRatesASCII(OHltConfig *cfg, OHltMenu *menu);

   void printCorrelationASCII();

   void printRatesTex(OHltConfig *cfg, OHltMenu *menu);

   void printHltRatesTex(OHltConfig *cfg, OHltMenu *menu);

   void printL1RatesTex(OHltConfig *cfg, OHltMenu *menu);

   void printRatesTwiki(OHltConfig *cfg, OHltMenu *menu);

   void printHltRatesTwiki(OHltConfig *cfg, OHltMenu *menu);

   void printL1RatesTwiki(OHltConfig *cfg, OHltMenu *menu);

   void printHltRatesBocci(OHltConfig *cfg, OHltMenu *menu);

   void writeHistos(OHltConfig *cfg, OHltMenu *menu, int Nevents);

   void fitRatesForPileup(OHltConfig *cfg, OHltMenu *menu);

   TString GetFileName(OHltConfig *cfg, OHltMenu *menu);

   void printPrescalesCfg(OHltConfig *cfg, OHltMenu *menu);

   void printHLTDatasets(
         OHltConfig *cfg,
         OHltMenu *menu,
         HLTDatasets &hltDatasets,
         TString &fullPathTableName,
         const Int_t significantDigits);

   int ivecMax(std::vector<int> ivec);

   int ivecMin(std::vector<int> ivec);
   
   std::vector<double> Rate;
   std::vector<double> RateErr;
   std::vector<double> spureRate;
   std::vector<double> spureRateErr;
   std::vector<double> pureRate;
   std::vector<double> pureRateErr;
   std::vector< std::vector<double> > coMa;
   std::vector< std::vector<double> > CountPerLS;
   std::vector<double> totalCountPerLS;

   std::vector< std::vector<double> > RatePerLS;
   std::vector<double> totalRatePerLS;
   std::vector< std::vector<int> > prescaleRefPerLS;
   std::vector< std::vector<int> > prescaleL1RefPerLS;
   std::vector<int> runID;
   std::vector<int> lumiSection;
   std::vector<double> averageRefPrescaleHLT;
   std::vector<double> averageRefPrescaleL1;
   std::vector<double> LumiPerLS;
};

#endif
