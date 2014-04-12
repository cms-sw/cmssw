#include "OHltRateCounter.h"

using namespace std;
using namespace libconfig;

OHltRateCounter::OHltRateCounter(unsigned int size, unsigned int l1size)
{
   vector<int> itmp;
   for (unsigned int i=0; i<size; i++)
   {
      iCount.push_back(0);
      sPureCount.push_back(0);
      pureCount.push_back(0);
      prescaleCount.push_back(0);

      itmp.push_back(0);
   }
   for (unsigned int j=0; j<size; j++)
   {
      overlapCount.push_back(itmp);
   }
   for (unsigned int k=0; k<l1size; k++)
   {
      prescaleCountL1.push_back(0);
      iL1Count.push_back(0);
   }
}

bool OHltRateCounter::isNewRunLS(int Run, int LumiBlock)
{
   for (unsigned int i=0; i<runID.size(); i++)
   {
      if (Run==runID[i] && LumiBlock==lumiSection[i])
         return false;
   }
   return true;
}

void OHltRateCounter::addRunLS(int Run, int LumiBlock, double AvgInstDelLumi)
{
   runID.push_back(Run);
   lumiSection.push_back(LumiBlock);
   perLumiSectionLumi.push_back(AvgInstDelLumi);

   vector< int> vtmp;
   vector< int> vtmpl1;
   for (unsigned int i=0; i<iCount.size(); i++)
   {
      vtmp.push_back(0);
   }
   perLumiSectionCount.push_back(vtmp);
   perLumiSectionTotCount.push_back(0);
   perLumiSectionRefPrescale.push_back(vtmp);

   for (unsigned int i=0; i<iL1Count.size(); i++)
   {
      vtmpl1.push_back(0);
   }
   perLumiSectionRefL1Prescale.push_back(vtmpl1);
}

int OHltRateCounter::getIDofRunLSCounter(int Run, int LumiBlock)
{
   for (unsigned int i=0; i<runID.size(); i++)
   {
      if (Run==runID[i] && LumiBlock==lumiSection[i])
         return i;
   }
   return -999;
}

void OHltRateCounter::incrRunLSCount(int Run, int LumiBlock, int iTrig, int incr)
{
   int id = getIDofRunLSCounter(Run, LumiBlock);
   if (id>-1)
   {
      perLumiSectionCount[id][iTrig] = perLumiSectionCount[id][iTrig] + incr;
   }
}

void OHltRateCounter::incrRunLSTotCount(int Run, int LumiBlock, int incr)
{
   int id = getIDofRunLSCounter(Run, LumiBlock);
   if (id>-1)
   {
      perLumiSectionTotCount[id] = perLumiSectionTotCount[id] + incr;
   }
}

void OHltRateCounter::updateRunLSRefPrescale(
      int Run,
      int LumiBlock,
      int iTrig,
      int refprescale)
{
   int id = getIDofRunLSCounter(Run, LumiBlock);
   if (id>-1)
   {
      perLumiSectionRefPrescale[id][iTrig] = refprescale;
   }
}

void OHltRateCounter::updateRunLSRefL1Prescale(
      int Run,
      int LumiBlock,
      int iL1Trig,
      int refl1prescale)
{
   int id = getIDofRunLSCounter(Run, LumiBlock);
   if (id>-1)
   {
      perLumiSectionRefL1Prescale[id][iL1Trig] = refl1prescale;
   }
}
