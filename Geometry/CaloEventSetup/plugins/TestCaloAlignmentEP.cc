// -*- C++ -*-
//
// Package:    TestCaloAlignmentEP
// Class:      TestCaloAlignmentEP
// 
/**\class TestCaloAlignmentEP TestCaloAlignmentEP.h Alignment/TestCaloAlignmentEP/interface/TestCaloAlignmentEP.h

Description: Producer of fake alignment data for calo geometries

Implementation: 
The alignment objects are filled with fixed alignments.
*/
//
// Original Author:  Brian Heltsley
//
//


// System
#include <memory>

// Framework
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CastorAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CastorAlignmentErrorExtendedRcd.h"

class TestCaloAlignmentEP : public edm::ESProducer 
{
   public:

      typedef boost::shared_ptr<Alignments>      ReturnAli    ;
      typedef boost::shared_ptr<AlignmentErrors> ReturnAliErr ;

      typedef AlignTransform::Translation Trl ;
      typedef AlignTransform::Rotation    Rot ;

      TestCaloAlignmentEP(const edm::ParameterSet&)
      {
	 setWhatProduced( this, &TestCaloAlignmentEP::produceEBAli    ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceEBAliErr ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceEEAli    ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceEEAliErr ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceESAli    ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceESAliErr ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceHBAli    ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceHBAliErr ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceHEAli    ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceHEAliErr ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceHOAli    ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceHOAliErr ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceHFAli    ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceHFAliErr ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceZdcAli    ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceZdcAliErr ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceCastorAli    ) ;
	 setWhatProduced( this, &TestCaloAlignmentEP::produceCastorAliErr ) ;
      }

      ~TestCaloAlignmentEP() {}

//-------------------------------------------------------------------
 
      ReturnAli    produceEBAli( const EBAlignmentRcd& /*iRecord*/ ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( EcalBarrelGeometry::numberOfAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const EBDetId id ( EcalBarrelGeometry::detIdFromLocalAlignmentIndex( i ) ) ;
	    vtr.push_back( AlignTransform( ( 1==id.ism() ? Trl( 0, 0, 0 ) : //-0.3 ) :
					     Trl(0,0,0 ) ) , 
					   Rot(),
					   id              ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceEBAliErr( const EBAlignmentErrorExtendedRcd& /*iRecord*/ ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceEEAli( const EEAlignmentRcd& /*iRecord*/ ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( EcalEndcapGeometry::numberOfAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const EEDetId id (  EcalEndcapGeometry::detIdFromLocalAlignmentIndex( i ) ) ;
	    vtr.push_back( AlignTransform(  ( 2 > i ? Trl( -0.02, -0.81, -0.94 ) :
					      Trl( +0.52, -0.81, +0.81 ) ) ,
					   Rot(),
					   id              ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceEEAliErr( const EEAlignmentErrorExtendedRcd& /*iRecord*/ ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceESAli( const ESAlignmentRcd& /*iRecord*/ ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( EcalPreshowerGeometry::numberOfAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const ESDetId id ( EcalPreshowerGeometry::detIdFromLocalAlignmentIndex( i ) ) ;
	    vtr.push_back( AlignTransform( ( 4 > i ? Trl( -0.02, -0.81, -0.94 ) :
					     Trl( +0.52, -0.81, +0.81 ) ) ,  
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceESAliErr( const ESAlignmentErrorExtendedRcd& /*iRecord*/ ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceHBAli( const HBAlignmentRcd& /*iRecord*/ ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( HcalGeometry::numberOfBarrelAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalDetId id ( HcalGeometry::detIdFromBarrelAlignmentIndex( i ) ) ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceHBAliErr( const HBAlignmentErrorExtendedRcd& /*iRecord*/ ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceHEAli( const HEAlignmentRcd& /*iRecord*/ ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( HcalGeometry::numberOfEndcapAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalDetId id ( HcalGeometry::detIdFromEndcapAlignmentIndex( i ) ) ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceHEAliErr( const HEAlignmentErrorExtendedRcd& /*iRecord*/ ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceHOAli( const HOAlignmentRcd& /*iRecord*/ ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( HcalGeometry::numberOfOuterAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalDetId id ( HcalGeometry::detIdFromOuterAlignmentIndex( i ) ) ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceHOAliErr( const HOAlignmentErrorExtendedRcd& /*iRecord*/ ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceHFAli( const HFAlignmentRcd& /*iRecord*/ ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( HcalGeometry::numberOfForwardAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalDetId id ( HcalGeometry::detIdFromForwardAlignmentIndex( i ) ) ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceHFAliErr( const HFAlignmentErrorExtendedRcd& /*iRecord*/ ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceZdcAli( const ZDCAlignmentRcd& /*iRecord*/ ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( ZdcGeometry::numberOfAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalZDCDetId id ( HcalZDCDetId::EM, false, 1 ) ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceZdcAliErr( const ZDCAlignmentErrorExtendedRcd& /*iRecord*/ ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceCastorAli( const CastorAlignmentRcd& /*iRecord*/ ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( CastorGeometry::numberOfAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalCastorDetId id ( HcalCastorDetId::EM, false, 1, 1 ) ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceCastorAliErr( const CastorAlignmentErrorExtendedRcd& /*iRecord*/ ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
};


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(TestCaloAlignmentEP);
