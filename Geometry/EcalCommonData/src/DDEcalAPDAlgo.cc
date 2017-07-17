#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "Geometry/EcalCommonData/interface/DDEcalAPDAlgo.h"

DDEcalAPDAlgo::DDEcalAPDAlgo() :
  m_vecCerPos    ( ),
  m_APDHere      (0),

  m_capName      (""),
  m_capMat       (""),
  m_capXSize     (0),
  m_capYSize     (0),
  m_capThick     (0),

  m_CERName      (""),
  m_CERMat       (""),
  m_CERXSize     (0),
  m_CERYSize     (0),
  m_CERThick     (0),

  m_BSiName      (""),
  m_BSiMat       (""),
  m_BSiXSize     (0),
  m_BSiYSize     (0),
  m_BSiThick     (0),

  m_APDName      (""),
  m_APDMat       (""),
  m_APDSide      (0),
  m_APDThick     (0),
  m_APDZ         (0),
  m_APDX1        (0),
  m_APDX2        (0),

  m_ATJName      (""),
  m_ATJMat       (""),
  m_ATJThick     (0),

  m_SGLName      (""),
  m_SGLMat       (""),
  m_SGLThick     (0),

  m_AGLName      (""),
  m_AGLMat       (""),
  m_AGLThick     (0),

  m_ANDName      (""),
  m_ANDMat       (""),
  m_ANDThick     (0) {

  LogDebug("EcalGeom") << "DDEcalAPDAlgo info: Creating an instance" ;
}

DDEcalAPDAlgo::~DDEcalAPDAlgo() {}

void DDEcalAPDAlgo::initialize(const DDNumericArguments      & nArgs,
			       const DDVectorArguments       & vArgs,
			       const DDMapArguments          & mArgs,
			       const DDStringArguments       & sArgs,
			       const DDStringVectorArguments & vsArgs) {

  LogDebug("EcalGeom") << "DDEcalAPDAlgo info: Initialize" ;

  m_idNameSpace = parent().name().ns();

  m_vecCerPos= vArgs["CerPos" ] ;
  m_APDHere  = (int)(nArgs["APDHere"]) ;

  m_capName  = sArgs["CapName"] ;
  m_capMat   = sArgs["CapMat"] ;
  m_capXSize = nArgs["CapXSize"] ;
  m_capYSize = nArgs["CapYSize"] ;
  m_capThick = nArgs["CapThick"] ;

  m_CERName  = sArgs["CerName"] ;
  m_CERMat   = sArgs["CerMat"] ;
  m_CERXSize = nArgs["CerXSize"] ;
  m_CERYSize = nArgs["CerYSize"] ;
  m_CERThick = nArgs["CerThick"] ;

  m_BSiName  = sArgs["BSiName"] ;
  m_BSiMat   = sArgs["BSiMat"] ;
  m_BSiXSize = nArgs["BSiXSize"] ;
  m_BSiYSize = nArgs["BSiYSize"] ;
  m_BSiThick = nArgs["BSiThick"] ;

  m_APDName  = sArgs["APDName"] ;
  m_APDMat   = sArgs["APDMat"] ;
  m_APDSide  = nArgs["APDSide"] ;
  m_APDThick = nArgs["APDThick"] ;
  m_APDZ     = nArgs["APDZ"] ;
  m_APDX1    = nArgs["APDX1"] ;
  m_APDX2    = nArgs["APDX2"] ;

  m_ATJName  = sArgs["ATJName"] ;
  m_ATJMat   = sArgs["ATJMat"] ;
  m_ATJThick = nArgs["ATJThick"] ;

  m_SGLName  = sArgs["SGLName"] ;
  m_SGLMat   = sArgs["SGLMat"] ;
  m_SGLThick = nArgs["SGLThick"] ;

  m_AGLName  = sArgs["AGLName"] ;
  m_AGLMat   = sArgs["AGLMat"] ;
  m_AGLThick = nArgs["AGLThick"] ;

  m_ANDName  = sArgs["ANDName"] ;
  m_ANDMat   = sArgs["ANDMat"] ;
  m_ANDThick = nArgs["ANDThick"] ;
   
  LogDebug("EcalGeom") << "DDEcalAPDAlgo info: end initialize" ;
}

////////////////////////////////////////////////////////////////////
// DDEcalBarrelAlgo methods...
////////////////////////////////////////////////////////////////////

void DDEcalAPDAlgo::execute(DDCompactView& cpv) {

  LogDebug("EcalGeom") << "******** DDEcalAPDAlgo execute!" << std::endl ;

//++++++++++++++++++++++++++++++++++  APD ++++++++++++++++++++++++++++++++++
  const DDName        capDDName (capName().name()) ;

  DDSolid capSolid ( DDSolidFactory::box( capDDName, capXSize()/2.,
					  capYSize()/2., capThick()/2. ) ) ;
	 
  const unsigned int copyCAP ( 1 ) ;
	 
  const DDLogicalPart capLog ( capDDName, capMat(), capSolid ) ;

  const DDName        sglDDName ( sglName().name()) ;

  DDSolid sglSolid ( DDSolidFactory::box( sglDDName, capXSize()/2.,
					  capYSize()/2., sglThick()/2. ) ) ;

  const DDLogicalPart sglLog ( sglDDName, sglMat(), sglSolid ) ;
	 
  const unsigned int copySGL ( 1 ) ;

  const DDName        cerDDName ( cerName().name() ) ;

  DDSolid cerSolid ( DDSolidFactory::box( cerDDName, cerXSize()/2.,
					  cerYSize()/2., cerThick()/2. ) ) ;

  const DDLogicalPart cerLog ( cerDDName, cerMat(), cerSolid ) ;
	 
  const unsigned int copyCER ( 1 ) ;

  const DDName        bsiDDName ( bsiName().name() ) ;

  DDSolid bsiSolid ( DDSolidFactory::box( bsiDDName, bsiXSize()/2.,
					  bsiYSize()/2., bsiThick()/2. ) ) ;

  const DDLogicalPart bsiLog ( bsiDDName, bsiMat(), bsiSolid ) ;
	 
  const unsigned int copyBSi ( 1 ) ;

  const DDName        atjDDName ( atjName().name() ) ;

  DDSolid atjSolid ( DDSolidFactory::box( atjDDName, apdSide()/2.,
					  apdSide()/2., atjThick()/2. ) ) ;

  const DDLogicalPart atjLog ( atjDDName, atjMat(), atjSolid ) ;
	 
  const unsigned int copyATJ ( 1 ) ;

  const DDName        aglDDName ( aglName().name() ) ;

  DDSolid aglSolid ( DDSolidFactory::box( aglDDName, bsiXSize()/2.,
					  bsiYSize()/2., aglThick()/2. ) ) ;

  const DDLogicalPart aglLog ( aglDDName, aglMat(), aglSolid ) ;
	 
  const unsigned int copyAGL ( 1 ) ;

  const DDName        andDDName ( andName().name() ) ;

  DDSolid andSolid ( DDSolidFactory::box( andDDName, apdSide()/2.,
					  apdSide()/2., andThick()/2. ) ) ;

  const DDLogicalPart andLog ( andDDName, andMat(), andSolid ) ;
  
  const unsigned int copyAND ( 1 ) ;
  
  const DDName        apdDDName ( apdName().name() ) ;

  DDSolid apdSolid ( DDSolidFactory::box( apdDDName, apdSide()/2.,
					 apdSide()/2., apdThick()/2. ) ) ;

  const DDLogicalPart apdLog ( apdDDName, apdMat(), apdSolid ) ;
	 
  const unsigned int copyAPD ( 1 ) ;
 
  if ( 0 != apdHere() ) { 
    cpv.position(aglLog, bsiLog, copyAGL, 
		 DDTranslation(0,0,-aglThick()/2.+bsiThick()/2.),DDRotation());

    cpv.position(andLog, bsiLog, copyAND, 
		 DDTranslation(0, 0, -andThick()/2.-aglThick()+bsiThick()/2.),
		 DDRotation() ) ;

    cpv.position(apdLog, bsiLog, copyAPD, 
		 DDTranslation(0,0,-apdThick()/2.-andThick()-aglThick()+bsiThick()/2.),
		 DDRotation() ) ;

    cpv.position(atjLog, bsiLog, copyATJ, 
		 DDTranslation(0,0,-atjThick()/2.-apdThick()-andThick()-aglThick()+bsiThick()/2. ),
		 DDRotation() ) ;

    cpv.position(bsiLog, cerLog, copyBSi, 
		 DDTranslation(0,0,-bsiThick()/2.+cerThick()/2.),DDRotation());

    cpv.position(sglLog, capLog, copySGL, 
		 DDTranslation(0,0,-sglThick()/2.+capThick()/2.),DDRotation());

    cpv.position(cerLog, capLog, copyCER, 
		 DDTranslation(0,0,-sglThick()-cerThick()/2.+capThick()/2.),
		 DDRotation() ) ;

    cpv.position(capLog, parent().name(), copyCAP, 
		 DDTranslation(vecCerPos()[0],vecCerPos()[1],vecCerPos()[2]),
		 DDRotation() ) ;
  }

  LogDebug("EcalGeom") << "******** DDEcalAPDAlgo test: end it..." ;
}

DDName DDEcalAPDAlgo::ddname( const std::string& s ) const { 

  const std::pair<std::string,std::string> temp ( DDSplit(s) ) ;
  if ( temp.second == "" ) {
    return DDName( temp.first, m_idNameSpace ) ;
  } else {
    return DDName( temp.first, temp.second );
  } 
}  
