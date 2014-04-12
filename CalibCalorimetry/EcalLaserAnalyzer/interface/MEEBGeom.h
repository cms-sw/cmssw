#ifndef MEEBGeom_hh
#define MEEBGeom_hh

//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay 
//

#include <vector>
#include <map>

#include <TPolyLine.h>
#include <TGraph.h>
#include <TString.h>

class MEEBGeom
{
  // static functions
public:

  enum EBUnit { iEcalBarrel, iSuperModule, iDCC, iSide, iLMRegion, iLMModule, iTriggerTower, iCrystal };

  typedef int EBLocalCoord;    // ix from 0 to 84, iy from 0 to 19  (note ix=etaSM-1, iy=phiSM-1)
  typedef int EBGlobalCoord;   // ieta from 1 to 85 (EB+) or -85 to -1 (EB-) and iphi from 1 to 360 (or -9 to 350)
  typedef int EBTTLocalCoord;  // iX=ix/5 from 0 to 3, iY=iy/5 from 0 to 16
  typedef std::pair<float,float> EtaPhiPoint;
  typedef std::pair<EBGlobalCoord, EBGlobalCoord> EtaPhiCoord;
  typedef std::pair<EBLocalCoord, EBLocalCoord> XYCoord;

  static int     barrel( EBGlobalCoord ieta, EBGlobalCoord iphi );
  static int         sm( EBGlobalCoord ieta, EBGlobalCoord iphi );
  static int        dcc( EBGlobalCoord ieta, EBGlobalCoord iphi );
  static int       side( EBGlobalCoord ieta, EBGlobalCoord iphi );
  static int        lmr( EBGlobalCoord ieta, EBGlobalCoord iphi );
  static int      lmmod( EBGlobalCoord ieta, EBGlobalCoord iphi );
  static int         tt( EBGlobalCoord ieta, EBGlobalCoord iphi );
  static int    crystal( EBGlobalCoord ieta, EBGlobalCoord iphi );

  static TString  smName( int ism );
  static int   smFromDcc( int idcc );
  static int   dccFromSm( int ism );

  static std::pair<int, int> pn( int ilmmod );
  static std::pair<int,int> memFromLmr( int ilmr );
  static std::vector<int> lmmodFromLmr( int ilmr );

  static int apdRefTower( int ilmmod );
  static std::vector<int> apdRefChannels( int ilmmod );

  // get local from crystal number
  static XYCoord   localCoord( int icr );

  // get local from global 
  static XYCoord   localCoord( EBGlobalCoord ieta, EBGlobalCoord iphi );

  // get global from local 
  static EtaPhiCoord globalCoord( int ism, EBLocalCoord ix, EBLocalCoord iy );
  static EtaPhiPoint globalCoord( int ism, float x, float y );
  static EtaPhiCoord globalCoord( int ism, int icrystal );


  static TGraph* getGraphBoundary( int type, int num, bool global=false );
  static int    crystal_channel( EBLocalCoord ix, EBLocalCoord iy );
  static int electronic_channel( EBLocalCoord ix, EBLocalCoord iy );

  static int tt_type(    EBTTLocalCoord iX, EBTTLocalCoord iY );
  static int hv_channel( EBTTLocalCoord iX, EBTTLocalCoord iY );
  static int lv_channel( EBTTLocalCoord iX, EBTTLocalCoord iY );
  static int lm_channel( EBTTLocalCoord iX, EBTTLocalCoord iY );
  static int tt_channel( EBTTLocalCoord iX, EBTTLocalCoord iY );

  virtual ~MEEBGeom() {}

private:

  //GHM  ClassDef(MEEBGeom,0) // MEEBGeom -- Monitoring utility for survey of Ecal
};

#endif

