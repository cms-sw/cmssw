//
// Original Author:  Georgios Daskalakis , Georgios.Daskalakis@cern.ch
//         Created:  Fri Mar 30 18:15:12 CET 2007
//         
//
//
//

#ifndef PositionCorrector_h
#define PositionCorrector_h

class PositionCorrector { 
public:
   PositionCorrector() {}
   ~PositionCorrector() {}
   double CORRX(int DeadCrystal, int DeadCrystalEta, int estimE, double estimX);
   double CORRY(int DeadCrystal, int DeadCrystalEta, int estimE, double estimY);
private:
   int index;
   double a0, a1, a2, a3;

/*    void correction_20(); */
/*    void correction_30(); */
   void correction_50();
/*    void correction_80(); */
/*    void correction_120(); */
/*    void correction_150(); */
/*    void correction_180(); */

};

#endif // PositionCorrector_h

