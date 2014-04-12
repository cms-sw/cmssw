//////////////////////////////////////////////////////////
//
// Class to store  and process rate counts
//
//////////////////////////////////////////////////////////

#ifndef OHltRateCounter_h
#define OHltRateCounter_h

#include <vector>
#include <libconfig.h++>
#include <TMath.h>

class OHltRateCounter
{
public:

   OHltRateCounter(unsigned int size, unsigned int l1size);
   
   virtual ~OHltRateCounter() {}

   bool isNewRunLS(int Run, int LumiBlock);
   
   void addRunLS(int Run, int LumiBlock, double AvgInstDelLumi);
   
   void incrRunLSCount(int Run, int LumiBlock, int iTrig, int incr=1);
   
   void incrRunLSTotCount(int Run, int LumiBlock, int incr=1);
   
   void updateRunLSRefPrescale(
         int Run,
         int LumiBlock,
         int iTrig,
         int refprescale);
   
   void updateRunLSRefL1Prescale(
         int Run,
         int LumiBlock,
         int iL1Trig,
         int refl1prescale);
   
   int getIDofRunLSCounter(int Run, int LumiBlock);

   // Helper functions
   static inline float eff(int a, int b)
   {
      if (b==0.)
      {
         return -1.;
      }
      float af = float(a), bf = float(b), effi = af/bf;
      return effi;
   }
   
   static inline float effErr(int a, int b)
   {
      if (b==0.)
      {
         return -1.;
      }
      float af = float(a), bf = float(b), r = af/bf;
      float unc = sqrt(af + (r*r*bf))/bf;
      return unc;
   }
   
   static inline float effErrb(int a, int b)
   {
      if (b==0.)
      {
         return -1.;
      }
      float af = float(a), bf = float(b), r = af/bf;
      float unc = sqrt(af - (r*r*bf))/bf;
      return unc;
   }
   
   static inline float eff(float a, float b)
   {
      if (b==0.)
      {
         return -1.;
      }
      float af = float(a), bf = float(b), effi = af/bf;
      return effi;
   }
   
   static inline float effErr(float a, float b)
   {
      if (b==0.)
      {
         return -1.;
      }
      float af = float(a), bf = float(b), r = af/bf;
      float unc = sqrt(af + (r*r*bf))/bf;
      return unc;
   }
   
   static inline float effErrb(float a, float b)
   {
      if (b==0.)
      {
         return -1.;
      }
      float af = float(a), bf = float(b), r = af/bf;
      float unc = sqrt(af - (r*r*bf))/bf;
      return unc;
   }

   static inline float errRate2(float a, float b)
   {
      if (b==0.)
      {
         return -1.;
      }
      float af = float(a), bf = float(b);
      float unc = af/(bf*bf);

      //float unc = sqrt(af + (r*r*bf) )/bf;
      return unc;
   }
   
   static inline float errRate2(int a, int b)
   {
      if (b==0.)
      {
         return -1.;
      }
      float af = float(a), bf = float(b);
      float unc = af/(bf*bf);

      //float unc = sqrt(af + (r*r*bf) )/bf;
      return unc;
   }
   
   static inline float errRate2(int a, float b)
   {
      if (b==0.)
      {
         return -1.;
      }
      float af = float(a), bf = float(b);
      float unc = af/(bf*bf);

      //float unc = sqrt(af + (r*r*bf) )/bf;
      return unc;
   }

   // Data
   std::vector<int> iCount;
   std::vector<int> iL1Count;
   std::vector<int> sPureCount;
   std::vector<int> pureCount;
   std::vector< std::vector<int> > overlapCount;
   std::vector<int> prescaleCount;
   std::vector<int> prescaleCountL1;

   std::vector< std::vector<int> > perLumiSectionCount;
   std::vector<int> perLumiSectionTotCount;
   std::vector< std::vector<int> > perLumiSectionRefPrescale;
   std::vector< std::vector<int> > perLumiSectionRefL1Prescale;
   std::vector<int> runID;
   std::vector<int> lumiSection;
   std::vector<double> perLumiSectionLumi;

};

#endif
