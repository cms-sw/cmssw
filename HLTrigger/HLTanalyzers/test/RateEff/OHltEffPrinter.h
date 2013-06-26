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
         std::vector<float> Eff,
         std::vector<float> EffErr,
         std::vector<float> spureEff,
         std::vector<float> spureEffErr,
         std::vector<float> pureEff,
         std::vector<float> pureEffErr,
         std::vector< std::vector<float> >coMa,
         float DenEff);

   void printEffASCII(OHltConfig *cfg, OHltMenu *menu);

   std::vector<float> Eff;
   std::vector<float> EffErr;
   std::vector<float> spureEff;
   std::vector<float> spureEffErr;
   std::vector<float> pureEff;
   std::vector<float> pureEffErr;
   std::vector< std::vector<float> > coMa;
   float DenEff;

};

#endif
