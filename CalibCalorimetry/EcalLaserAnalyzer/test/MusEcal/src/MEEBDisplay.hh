#ifndef MEEBDisplay_hh
#define MEEBDisplay_hh

//
// Authors  : Gautier Hamel de Monchenault and Julie Malcles, Saclay 
//

#include <list>
#include <map>

#include "../../interface/MEEBGeom.h"

class MEEBDisplay
{
  // static functions
public:

  static MEEBGeom::EtaPhiPoint getNode( int iSM, 
					MEEBGeom::EBTTLocalCoord iX, 
					MEEBGeom::EBTTLocalCoord iY, 
					int jx, int jy );
  static void drawEB();
  static void drawSM(  int iSM, float shift=0 );
  static void drawTT(  int iSM, 
		       MEEBGeom::EBTTLocalCoord iX, 
		       MEEBGeom::EBTTLocalCoord iY, float shift=0 );
  static void drawXtal( MEEBGeom::EBGlobalCoord ieta, 
			MEEBGeom::EBGlobalCoord iphi, 
			int color=kBlue, float shift=0 );

  static void drawEBGlobal();
  static void drawEBLocal();

  static TPolyLine* getXtalPolyLine( MEEBGeom::EBGlobalCoord ieta, 
				     MEEBGeom::EBGlobalCoord iphi, 
				     float shift=0 );    
  static TPolyLine* getTTPolyLine( int iSM, 
				   MEEBGeom::EBTTLocalCoord iX, 
				   MEEBGeom::EBTTLocalCoord iY, 
				   float shift=0 ); 
  static TPolyLine* getSMPolyLine( int iSM, float shift=0 );
  static void drawRz();
  static int bkgColor;
  static int lineColor;
  static int lineWidth;

  static void refresh();

  virtual ~MEEBDisplay() {}

private:

  static void setPhiLimits( int iSM,  
			    MEEBGeom::EBLocalCoord iy, 
			    MEEBGeom::EBGlobalCoord iphi, 
			    float phioverpi_0, float phioverpi_1 );
  static void setEtaLimits( int iSM,  
			    MEEBGeom::EBLocalCoord ix, 
			    MEEBGeom::EBGlobalCoord ieta, 
			    float eta_0, float eta_1 );
  static void setSM_2_and_20();

  static std::map< int, std::pair<float,float>  > _phiLimits;
  static std::map< int, std::pair<float,float>  > _etaLimits;

  static void setRzXtals();
  static std::map< int, TPolyLine* > _rzXtals;

  static list<TObject*> _list;
  static void registerTObject( TObject* );
  
  ClassDef(MEEBDisplay,0) // MEEBDisplay -- Monitoring utility for survey of Ecal
};

#endif

