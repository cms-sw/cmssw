#ifndef MEEEGeom_hh
#define MEEEGeom_hh

//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay 
//

#include <vector>
#include <map>
#include <list>

#include <TPolyLine.h>
#include <TGraph.h>

class MEEEGeom
{
  // static functions
public:

  enum EEUnit { iEcalEndCap=0, iDee, iQuadrant, iSector, iDCC, iLMRegion, iLMModule, iSuperCrystal, iCrystal };

  typedef int SuperCrysCoord;
  typedef int CrysCoord;

  typedef std::pair<float,float> EtaPhiPoint;

  static int        dee( SuperCrysCoord iX, SuperCrysCoord iY, int iz );
  static int   quadrant( SuperCrysCoord iX, SuperCrysCoord iY         );
  static int     sector( SuperCrysCoord iX, SuperCrysCoord iY         );
  static int         sm( SuperCrysCoord iX, SuperCrysCoord iY, int iz );
  static int      lmmod( SuperCrysCoord iX, SuperCrysCoord iY         );
  static int sc_in_quad( SuperCrysCoord iX, SuperCrysCoord iY         );
  static int         sc( SuperCrysCoord iX, SuperCrysCoord iY         );
  static int        dcc( SuperCrysCoord iX, SuperCrysCoord iY, int iz );
  static int        lmr( SuperCrysCoord iX, SuperCrysCoord iY, int iz );
  static int       side( SuperCrysCoord iX, SuperCrysCoord iY, int iz );

  static int crystal_in_sc( CrysCoord ix, CrysCoord iy );
  static int       crystal( CrysCoord ix, CrysCoord iy );

  static int   dee( int ilmr );
  static bool near( int ilmr );

  static TString smName( int ism  );
  static int smFromDcc(  int idcc );
  static int dccFromSm(  int ism  );

  static bool pnTheory; // if true: theoretical PN cabling for all dees
  static std::pair<int, int> pn( int dee, int ilmod );
  static std::pair<int,int> memFromLmr( int ilmr );
  static std::vector<int> lmmodFromLmr( int ilmr );
  static int deeFromMem( int imem );
  static int apdRefTower(  int ilmr, int ilmmod );
  static std::vector<int> apdRefChannels( int ilmmod );

  static TGraph* getGraphBoundary( int type, int num, int iz=-1, int xside=0 );
  static void getBoundary( std::list< std::pair< float, float > >& l, int type, int num, int iz=-1, int xside=0 );

  static int sc_type(     SuperCrysCoord iX, SuperCrysCoord iY         );

  virtual ~MEEEGeom() {}

protected:

  //GHM  ClassDef(MEEEGeom,0) // MEEEGeom -- Monitoring utility for survey of Ecal

};

#endif

