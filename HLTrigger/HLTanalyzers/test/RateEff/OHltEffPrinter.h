//////////////////////////////////////////////////////////
//
// Class to print effs
//
//////////////////////////////////////////////////////////

#ifndef OHltEffPrinter_h
#define OHltEffPrinter_h

#include <vector>
#include "OHltMenu.h"
#include "OHltConfig.h"
#include "OHltTree.h"

class OHltEffPrinter
{
public:

   OHltEffPrinter() {}

   virtual ~OHltEffPrinter() {}

   void SetupAll(
         std::vector<double> Eff,
         std::vector<double> EffErr,
         std::vector<double> spureEff,
         std::vector<double> spureEffErr,
         std::vector<double> pureEff,
         std::vector<double> pureEffErr,
         std::vector< std::vector<double> >coMa,
         double DenEff);

   void printEffASCII(OHltConfig *cfg, OHltMenu *menu);

   std::vector<double> Eff;
   std::vector<double> EffErr;
   std::vector<double> spureEff;
   std::vector<double> spureEffErr;
   std::vector<double> pureEff;
   std::vector<double> pureEffErr;
   std::vector< std::vector<double> > coMa;
   double DenEff;

};

#endif
