
//////////////////////////////////////////////////////////////////////////////
// File: DDEcalEndcapAlgo.cc
// Description: Geometry factory class for Ecal Barrel
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "Geometry/EcalCommonData/interface/DDEcalEndcapAlgo.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <CLHEP/Geometry/Transform3D.h>

// Header files for endcap supercrystal geometry
#include "Geometry/EcalCommonData/interface/DDEcalEndcapTrap.h"


namespace std{} using namespace std;

DDEcalEndcapAlgo::DDEcalEndcapAlgo() :
   m_idNameSpace   ( "" ),
   m_EEMat         ( "" ),
   m_EEzOff        ( 0 ),
   m_EEQuaName     ( "" ),
   m_EEQuaMat      ( "" ),
   m_EECrysMat     ( "" ),
   m_EEWallMat     ( "" ),
   m_EECrysLength  ( 0 ) ,
   m_EECrysRear    ( 0 ) ,
   m_EECrysFront   ( 0 ) ,
   m_EESCELength   ( 0 ) ,
   m_EESCERear     ( 0 ) ,
   m_EESCEFront    ( 0 ) ,
   m_EESCALength   ( 0 ) ,
   m_EESCARear     ( 0 ) ,
   m_EESCAFront    ( 0 ) ,
   m_EESCAWall     ( 0 ) ,
   m_EESCHLength   ( 0 ) ,
   m_EESCHSide     ( 0 ) ,
   m_EEnSCTypes    ( 0 ) ,
   m_vecEESCProf   (),
   m_EEnColumns    ( 0 ),
   m_vecEEShape    (),
   m_EEnSCCutaway  ( 0 ) ,
   m_vecEESCCutaway (),
   m_EEnSCquad     ( 0 ) ,
   m_vecEESCCtrs(),
   m_EEnCRSC       ( 0 ) ,
   m_vecEECRCtrs(),
   m_cutParms      ( nullptr ),
   m_cutBoxName    ( "" ),
   m_envName    ( "" ),
   m_alvName    ( "" ),
   m_intName    ( "" ),
   m_cryName    ( "" ),
   m_PFhalf     ( 0 ) ,
   m_PFfifth    ( 0 ) ,
   m_PF45       ( 0 ) ,
   m_vecEESCLims (),
   m_iLength    ( 0 ) ,
   m_iXYOff     ( 0 ) ,
   m_cryZOff    ( 0 ) ,
   m_zFront     ( 0 )
{
   LogDebug("EcalGeom") << "DDEcalEndcapAlgo info: Creating an instance" ;
}

DDEcalEndcapAlgo::~DDEcalEndcapAlgo() {}




void DDEcalEndcapAlgo::initialize(const DDNumericArguments      & nArgs,
				  const DDVectorArguments       & vArgs,
				  const DDMapArguments          & /*mArgs*/,
				  const DDStringArguments       & sArgs,
				  const DDStringVectorArguments & /*vsArgs*/)
{
  DDCurrentNamespace ns;
   m_idNameSpace = *ns;
   // TRICK!
   m_idNameSpace = parent().name().ns();
   // barrel parent volume
   m_EEMat      = sArgs["EEMat"  ] ;
   m_EEzOff     = nArgs["EEzOff"   ] ;

   m_EEQuaName  = sArgs["EEQuaName" ] ;
   m_EEQuaMat   = sArgs["EEQuaMat"  ] ;
   m_EECrysMat  = sArgs["EECrysMat" ] ;
   m_EEWallMat  = sArgs["EEWallMat" ] ;
   m_EECrysLength = nArgs["EECrysLength" ] ;
   m_EECrysRear   = nArgs["EECrysRear" ] ;
   m_EECrysFront  = nArgs["EECrysFront" ] ;
   m_EESCELength = nArgs["EESCELength" ] ;
   m_EESCERear   = nArgs["EESCERear" ] ;
   m_EESCEFront  = nArgs["EESCEFront" ] ;
   m_EESCALength = nArgs["EESCALength" ] ;
   m_EESCARear   = nArgs["EESCARear" ] ;
   m_EESCAFront  = nArgs["EESCAFront" ] ;
   m_EESCAWall   = nArgs["EESCAWall" ] ;
   m_EESCHLength = nArgs["EESCHLength" ] ;
   m_EESCHSide   = nArgs["EESCHSide" ] ;
   m_EEnSCTypes  = nArgs["EEnSCTypes"];
   m_EEnColumns  = nArgs["EEnColumns"];
   m_EEnSCCutaway  = nArgs["EEnSCCutaway"];
   m_EEnSCquad  = nArgs["EEnSCquad"];
   m_EEnCRSC    = nArgs["EEnCRSC"];
   m_vecEESCProf = vArgs["EESCProf"];
   m_vecEEShape = vArgs["EEShape"];
   m_vecEESCCutaway = vArgs["EESCCutaway"];
   m_vecEESCCtrs = vArgs["EESCCtrs"];
   m_vecEECRCtrs = vArgs["EECRCtrs"];

   m_cutBoxName  = sArgs["EECutBoxName" ] ;

   m_envName  = sArgs["EEEnvName" ] ;
   m_alvName  = sArgs["EEAlvName" ] ;
   m_intName  = sArgs["EEIntName" ] ;
   m_cryName  = sArgs["EECryName" ] ;
   
   m_PFhalf   = nArgs["EEPFHalf" ] ;
   m_PFfifth  = nArgs["EEPFFifth" ] ;
   m_PF45     = nArgs["EEPF45" ] ;

   m_vecEESCLims = vArgs["EESCLims"];

   m_iLength = nArgs["EEiLength" ] ;

   m_iXYOff  = nArgs["EEiXYOff" ] ;

   m_cryZOff = nArgs["EECryZOff"] ;

   m_zFront  = nArgs["EEzFront"] ;
}

////////////////////////////////////////////////////////////////////
// DDEcalEndcapAlgo methods...
////////////////////////////////////////////////////////////////////


DDRotation
DDEcalEndcapAlgo::myrot( const std::string&      s,
			 const DDRotationMatrix& r ) const 
{
  return DDrot( ddname( m_idNameSpace + ":" + s ), std::make_unique<DDRotationMatrix>( r ) ) ; 
}
 
DDMaterial
DDEcalEndcapAlgo::ddmat( const std::string& s ) const
{
   return DDMaterial( ddname( s ) ) ; 
}

DDName
DDEcalEndcapAlgo::ddname( const std::string& s ) const
{ 
   const pair<std::string,std::string> temp ( DDSplit(s) ) ;
   if ( temp.second.empty() ) {
     return DDName( temp.first,
		    m_idNameSpace ) ;
   } else {
     return DDName( temp.first, temp.second );
   } 
}

//-------------------- Endcap SC geometry methods ---------------------

void 
DDEcalEndcapAlgo::execute( DDCompactView& cpv ) 
{
//  Position supercrystals in EE Quadrant
//  Version:    1.00
//  Created:    30 July 2007
//  Last Mod:   
//---------------------------------------------------------------------

//********************************* cutbox for trimming edge SCs
   const double cutWid ( eeSCERear()/sqrt(2.) ) ;
   const DDSolid eeCutBox ( DDSolidFactory::box(
			       cutBoxName(), 
			       cutWid,  
			       cutWid,  
			       eeSCELength()/sqrt(2.) ) ) ;
   m_cutParms = &eeCutBox.parameters() ;
//**************************************************************

   const double zFix ( m_zFront - 3172*mm ) ; // fix for changing z offset

//** fill supercrystal front and rear center positions from xml input
   for( unsigned int iC ( 0 ) ; iC != (unsigned int) eenSCquad() ; ++iC )
   {
      const unsigned int iOff ( 8*iC ) ;
      const unsigned int ix ( (unsigned int) eevecEESCCtrs()[ iOff + 0 ] ) ;
      const unsigned int iy ( (unsigned int) eevecEESCCtrs()[ iOff + 1 ] ) ;

      assert( ix > 0 && ix < 11 && iy >0 && iy < 11 ) ;

      m_scrFCtr[ ix - 1 ][ iy - 1 ] = DDTranslation( eevecEESCCtrs()[ iOff + 2 ] ,
						     eevecEESCCtrs()[ iOff + 4 ] ,
						     eevecEESCCtrs()[ iOff + 6 ]  + zFix ) ;

      m_scrRCtr[ ix - 1 ][ iy - 1 ] = DDTranslation( eevecEESCCtrs()[ iOff + 3 ] ,
						     eevecEESCCtrs()[ iOff + 5 ] ,
						     eevecEESCCtrs()[ iOff + 7 ] + zFix ) ;
   }

//** fill crystal front and rear center positions from xml input
   for( unsigned int iC ( 0 ) ; iC != 25 ; ++iC )
   {
      const unsigned int iOff ( 8*iC ) ;
      const unsigned int ix ( (unsigned int) eevecEECRCtrs()[ iOff + 0 ] ) ;
      const unsigned int iy ( (unsigned int) eevecEECRCtrs()[ iOff + 1 ] ) ;

      assert( ix > 0 && ix < 6 && iy >0 && iy < 6 ) ;

      m_cryFCtr[ ix - 1 ][ iy - 1 ] = DDTranslation( eevecEECRCtrs()[ iOff + 2 ] ,
						     eevecEECRCtrs()[ iOff + 4 ] ,
						     eevecEECRCtrs()[ iOff + 6 ]  ) ;

      m_cryRCtr[ ix - 1 ][ iy - 1 ] = DDTranslation( eevecEECRCtrs()[ iOff + 3 ] ,
						     eevecEECRCtrs()[ iOff + 5 ] ,
						     eevecEECRCtrs()[ iOff + 7 ]  ) ;
   }

   EECreateCR() ; // make a single crystal just once here

   for( unsigned int isc ( 0 ); isc<eenSCTypes() ; ++isc ) 
   {
      EECreateSC( isc+1, cpv );
   }

   const std::vector<double>& colLimits ( eevecEEShape() );
//** Loop over endcap columns
   for( int icol = 1; icol<=int(eenColumns()); icol++ )
   {
//**  Loop over SCs in column, using limits from xml input
      for( int irow = int(colLimits[2*icol-2]);
	   irow <= int(colLimits[2*icol-1]) ; ++irow )
      {
	 if( vecEESCLims()[0] <= icol &&
	     vecEESCLims()[1] >= icol &&
	     vecEESCLims()[2] <= irow &&
	     vecEESCLims()[3] >= irow    )
	 {
	   // Find SC type (complete or partial) for this location
	    const unsigned int isctype ( EEGetSCType( icol, irow ) );

	    // Create SC as a DDEcalEndcapTrap object and calculate rotation and
	    // translation required to position it in the endcap.
	    DDEcalEndcapTrap scrys( 1, eeSCEFront(), eeSCERear(), eeSCELength() ) ;

	    scrys.moveto( scrFCtr( icol, irow ),
			  scrRCtr( icol, irow ) );
	    scrys.translate( DDTranslation( 0., 0., -eezOff() ) ) ;

	    DDName rname ( envName( isctype ).name()
			   + std::to_string( icol ) + "R" + std::to_string( irow ) ) ;
/*
         edm::LogInfo("EcalGeom") << "Quadrant, SC col/row " 
				  << eeQuaName() << " " << icol << " / " << irow << std::endl
                                  << "   Limits " << int(colLimits[2*icol-2]) << "->" << int(colLimits[2*icol-1]) << std::endl
                                  << "   SC type = " << isctype << std::endl
                                  << "   Zoff, Scz = " << eezOff() << " " << sc1.z() << std::endl
                                  << "   Rotation " << rname << " " << scrys.rotation() << std::endl
                                  << "   Position " << sccentre << std::endl;
*/
            // Position SC in endcap
	    cpv.position( envName( isctype ), 
			  eeQuaName(),
			  100*isctype + 10*(icol-1) + (irow-1),
			  scrys.centrePos(),
			  myrot( rname.fullname(), scrys.rotation() ) ) ;
	 }
      }
   }
}


void
DDEcalEndcapAlgo::EECreateSC( const unsigned int iSCType ,
			      DDCompactView&     cpv       )
{ //  EECreateSCType   Create SC logical volume of the given type

   DDRotation noRot ;

   DDLogicalPart eeSCELog;
   DDLogicalPart eeSCALog;
   DDLogicalPart eeSCILog;
   
//   edm::LogInfo("EcalGeom") << "EECreateSC: Creating SC envelope" << std::endl;

   const string anum ( std::to_string(iSCType) ) ;

   const double eFront ( 0.5*eeSCEFront() ) ;
   const double eRear  ( 0.5*eeSCERear()  ) ;
   const double eAng   ( atan((eeSCERear()-eeSCEFront())/(sqrt(2.)*eeSCELength())) ) ;
   const double ffived ( 45*deg ) ;
   const double zerod  (  0*deg ) ;
   DDSolid eeSCEnv   ( DDSolidFactory::trap(
			  ( 1 == iSCType ? envName( iSCType ) : addTmp( envName( iSCType ) ) ),
			  0.5*eeSCELength(),
			  eAng,
			  ffived,
			  eFront,
			  eFront,
			  eFront,
			  zerod,
			  eRear, 
			  eRear, 
			  eRear, 
			  zerod                 ) );

   const double aFront ( 0.5*eeSCAFront() ) ;
   const double aRear  ( 0.5*eeSCARear()  ) ;
   const double aAng   ( atan((eeSCARear()-eeSCAFront())/(sqrt(2.)*eeSCALength())) ) ;
   const DDSolid eeSCAlv ( DDSolidFactory::trap(
			      ( 1== iSCType ? alvName( iSCType ) : addTmp( alvName( iSCType ) ) ),
			      0.5*eeSCALength(),
			      aAng,
			      ffived,
			      aFront,
			      aFront,
			      aFront,
			      zerod,
			      aRear,
			      aRear,
			      aRear,
			      zerod             	) );

   const double dwall   ( eeSCAWall()    ) ;
   const double iFront  ( aFront - dwall ) ;
   const double iRear   ( iFront ) ; //aRear  - dwall ) ;
   const double iLen    ( iLength() ) ; //0.075*eeSCALength() ) ;
   const DDSolid eeSCInt ( DDSolidFactory::trap(
			      ( 1==iSCType ? intName( iSCType ) : addTmp( intName( iSCType ) ) ),
			      iLen/2.,
			      atan((eeSCARear()-eeSCAFront())/(sqrt(2.)*eeSCALength())),
			      ffived,
			      iFront,
			      iFront,
			      iFront,
			      zerod,
			      iRear,
			      iRear,
			      iRear,
			      zerod                     ) );

   const double dz  ( -0.5*( eeSCELength() - eeSCALength() ) ) ;
   const double dxy (  0.5* dz * (eeSCERear() - eeSCEFront())/eeSCELength() ) ;
   const double zIOff (  -( eeSCALength() - iLen )/2. ) ;
   const double xyIOff ( iXYOff() ) ;

   if( 1 == iSCType ) // standard SC in this block
   {
      eeSCELog = DDLogicalPart( envName( iSCType ), eeMat(),     eeSCEnv   );
      eeSCALog = DDLogicalPart( alvName( iSCType ), eeWallMat(), eeSCAlv   );
      eeSCILog = DDLogicalPart( intName( iSCType ), eeMat(),     eeSCInt   );
   }
   else // partial SCs this block: create subtraction volumes as appropriate
   {
      const double half  ( (*m_cutParms)[0] - eePFHalf() *eeCrysRear() ) ;
      const double fifth ( (*m_cutParms)[0] + eePFFifth()*eeCrysRear() ) ;
      const double fac   ( eePF45() ) ;

      const double zmm ( 0*mm ) ;
      
      DDTranslation cutTra ( 2 == iSCType ?       DDTranslation(        zmm,       half, zmm ) :
			     ( 3 == iSCType ?     DDTranslation(       half,        zmm, zmm ) :
			       ( 4 == iSCType ?   DDTranslation(        zmm,     -fifth, zmm ) :
				 ( 5 == iSCType ? DDTranslation(  -half*fac,  -half*fac, zmm ) :
				   DDTranslation(                    -fifth,        zmm,  zmm ) ) ) ) ) ;

      const CLHEP::HepRotationZ cutm ( ffived ) ;

      DDRotation cutRot ( 5 != iSCType ? noRot : myrot( "EECry5Rot", 
							DDRotationMatrix( cutm.xx(), cutm.xy(), cutm.xz(),
									  cutm.yx(), cutm.yy(), cutm.yz(),
									  cutm.zx(), cutm.zy(), cutm.zz() ) ) ) ;

      DDSolid eeCutEnv   ( DDSolidFactory::subtraction( envName( iSCType ),
							addTmp( envName( iSCType ) ),
							cutBoxName(), 
							cutTra,
							cutRot ) ) ;

      const DDTranslation extra ( dxy, dxy, dz ) ;

      DDSolid eeCutAlv   ( DDSolidFactory::subtraction( alvName( iSCType ),
							addTmp( alvName( iSCType ) ),
							cutBoxName(), 
							cutTra - extra,
							cutRot ) ) ;

      const double mySign ( iSCType < 4 ? +1. : -1. ) ;
      
      const DDTranslation extraI ( xyIOff + mySign*2*mm, 
				   xyIOff + mySign*2*mm, zIOff ) ;

      DDSolid eeCutInt   ( DDSolidFactory::subtraction( intName( iSCType ),
							addTmp( intName( iSCType ) ),
							cutBoxName(), 
							cutTra - extraI,
							cutRot ) ) ;
      
      eeSCELog = DDLogicalPart( envName( iSCType ), eeMat()    , eeCutEnv ) ;
      eeSCALog = DDLogicalPart( alvName( iSCType ), eeWallMat(), eeCutAlv   ) ;
      eeSCILog = DDLogicalPart( intName( iSCType ), eeMat()    , eeCutInt   ) ;
   }


   cpv.position( eeSCALog, envName( iSCType ), iSCType*100 + 1, DDTranslation( dxy,    dxy,    dz   ), noRot );
   cpv.position( eeSCILog, alvName( iSCType ), iSCType*100 + 1, DDTranslation( xyIOff, xyIOff, zIOff), noRot );

   DDTranslation croffset( 0., 0., 0.) ;
   EEPositionCRs( alvName( iSCType ), croffset, iSCType, cpv ) ;

}

unsigned int 
DDEcalEndcapAlgo::EEGetSCType( const unsigned int iCol, 
			       const unsigned int iRow )
{
   unsigned int iType = 1;
   for( unsigned int ii = 0; ii < (unsigned int)( eenSCCutaway() ) ; ++ii ) 
   {
      if( ( eevecEESCCutaway()[ 3*ii     ] == iCol ) &&
	  ( eevecEESCCutaway()[ 3*ii + 1 ] == iRow )    )
      {
	 iType = int(eevecEESCCutaway()[3*ii+2]);
//	 edm::LogInfo("EcalGeom") << "EEGetSCType: col, row, type = " 
//				  << iCol << " " << iRow << " " << iType << std::endl;
      }
   }
   return iType;
}

void
DDEcalEndcapAlgo::EECreateCR() 
{
   //  EECreateCR   Create endcap crystal logical volume

//   edm::LogInfo("EcalGeom") << "EECreateCR:  = " << std::endl;

   DDSolid EECRSolid (DDSolidFactory::trap(
			 cryName(),
			 0.5*eeCrysLength(),
			 atan((eeCrysRear()-eeCrysFront())/(sqrt(2.)*eeCrysLength())),
			 45.*deg,
			 0.5*eeCrysFront(),0.5*eeCrysFront(),0.5*eeCrysFront(),0.*deg,
			 0.5*eeCrysRear(), 0.5*eeCrysRear(), 0.5*eeCrysRear(),0.*deg     ) );

   DDLogicalPart part ( cryName(), eeCrysMat(), EECRSolid ) ;
}

void 
DDEcalEndcapAlgo::EEPositionCRs( const DDName&        pName, 
				 const DDTranslation& /*offset*/, 
				 const int           iSCType,
				 DDCompactView&      cpv      ) 
{
  //  EEPositionCRs Position crystals within parent supercrystal interior volume

//   edm::LogInfo("EcalGeom") << "EEPositionCRs called " << std::endl;

   static const unsigned int ncol ( 5 ) ;

   if( iSCType > 0             &&
       iSCType <= eenSCTypes()    ) 
   {
      const unsigned int icoffset ( ( iSCType - 1 )*ncol - 1 ) ;
      
      // Loop over columns of SC
      for( unsigned int icol ( 1 ); icol <= ncol ; ++icol ) 
      {
	// Get column limits for this SC type from xml input
	 const int ncrcol ( (int) eevecEESCProf()[ icoffset + icol ] ) ;

	 const int imin ( 0 < ncrcol ?      1 : ( 0 > ncrcol ? ncol + ncrcol + 1 : 0 ) ) ;
	 const int imax ( 0 < ncrcol ? ncrcol : ( 0 > ncrcol ? ncol              : 0 ) ) ;

	 if( imax>0 ) 
	 {
	   // Loop over crystals in this row
	    for( int irow ( imin ); irow <= imax ; ++irow ) 
	    {
//	       edm::LogInfo("EcalGeom") << " type, col, row " << iSCType 
//					<< " " << icol << " " << irow << std::endl;

  	       // Create crystal as a DDEcalEndcapTrap object and calculate rotation and
	       // translation required to position it in the SC.
	       DDEcalEndcapTrap crystal( 1, eeCrysFront(), eeCrysRear(), eeCrysLength() ) ;

	       crystal.moveto( cryFCtr( icol, irow ) ,
			       cryRCtr( icol, irow )   );

	       DDName rname ( "EECrRoC" + std::to_string( icol ) + "R" + std::to_string( irow ) ) ;

	       cpv.position( cryName(),
			     pName,
			     100*iSCType + 10*( icol - 1 ) + ( irow - 1 ),
			     crystal.centrePos() - DDTranslation(0,0,m_cryZOff),
			     myrot( rname.fullname(), crystal.rotation() ) ) ;
	    }
	 }
      }
   }
}

