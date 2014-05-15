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

   OHltRateCounter(double size, double l1size);
   
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
   static inline double eff(int a, int b)
   {
      if (b==0.)
      {
         return -1.;
      }
      double af = double(a), bf = double(b), effi = af/bf;
      return effi;
   }
   
   static inline double effErr(int a, int b)
   {
      if (b==0.)
      {
         return -1.;
      }
      double af = double(a), bf = double(b), r = af/bf;
      double unc = sqrt(af + (r*r*bf))/bf;
      return unc;
   }
   
   static inline double effErrb(int a, int b)
   {
      if (b==0.)
      {
         return -1.;
      }
      double af = double(a), bf = double(b), r = af/bf;
      double unc = sqrt(af - (r*r*bf))/bf;
      return unc;
   }
   
   static inline double eff(double a, double b)
   {
      if (b==0.)
      {
         return -1.;
      }
      double af = double(a), bf = double(b), effi = af/bf;
      return effi;
   }
   
   static inline double effErr(double a, double b)
   {
      if (b==0.)
      {
         return -1.;
      }
      double af = double(a), bf = double(b), r = af/bf;
      double unc = sqrt(af + (r*r*bf))/bf;
      return unc;
   }
   
   static inline double effErrb(double a, double b)
   {
      if (b==0.)
      {
         return -1.;
      }
      double af = double(a), bf = double(b), r = af/bf;
      double unc = sqrt(af - (r*r*bf))/bf;
      return unc;
   }

   static inline double errRate2(double a, double b)
   {
      if (b==0.)
      {
         return -1.;
      }
      double af = double(a), bf = double(b);
      double unc = af/(bf*bf);

      //double unc = sqrt(af + (r*r*bf) )/bf;
      return unc;
   }
   
   static inline double errRate2(int a, int b)
   {
      if (b==0.)
      {
         return -1.;
      }
      double af = double(a), bf = double(b);
      double unc = af/(bf*bf);

      //double unc = sqrt(af + (r*r*bf) )/bf;
      return unc;
   }
   
   static inline double errRate2(int a, double b)
   {
      if (b==0.)
      {
         return -1.;
      }
      double af = double(a), bf = double(b);
      double unc = af/(bf*bf);

      //double unc = sqrt(af + (r*r*bf) )/bf;
      return unc;
   }

   // Data
   std::vector<double> iCount;
   std::vector<double> iL1Count;
   std::vector<double> sPureCount;
   std::vector<double> pureCount;
   std::vector< std::vector<double> > overlapCount;
   std::vector<double> prescaleCount;
   std::vector<double> prescaleCountL1;

   std::vector< std::vector<int> > perLumiSectionCount;
   std::vector<int> perLumiSectionTotCount;
   std::vector< std::vector<int> > perLumiSectionRefPrescale;
   std::vector< std::vector<int> > perLumiSectionRefL1Prescale;
   std::vector<int> runID;
   std::vector<int> lumiSection;
   std::vector<double> perLumiSectionLumi;

};

#endif
