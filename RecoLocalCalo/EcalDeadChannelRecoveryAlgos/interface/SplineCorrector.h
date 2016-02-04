//
// Original Author:  Georgios Daskalakis , Georgios.Daskalakis@cern.ch
//         Created:  Fri Mar 30 18:15:12 CET 2007
//         
//
//
//

#ifndef SplineCorrector_h
#define SplineCorrector_h

class SplineCorrector { 
public:
   SplineCorrector() {}
   ~SplineCorrector() {}
   double value(int DeadCrystal, int DeadCrystalEta, int estimE, double estimX, double estimY);
private:
   double constants[25];
   int index;

   void SplE_50();


};

#endif // SplineCorrector_h

