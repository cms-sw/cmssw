//
// Original Author:  Georgios Daskalakis , Georgios.Daskalakis@cern.ch
//         Created:  Fri Mar 30 18:15:12 CET 2007
//         
//
//
//


#ifndef ExponCorrector_h
#define ExponCorrector_h

class ExponCorrector { 
public:
   ExponCorrector() {}
   ~ExponCorrector() {}
   double value(int DeadCrystal, int DeadCrystalEta, int estimE, int subRegion, double estimX, double estimY);
private:
   double EXPAN[4];
   double LOWLIM,HIGLIM;
   int theCrystal;
   int theSubR;

   void ExpE_50();
};

#endif // ExponCorrector_h

