// -*- C++ -*-
//
// Package:    FakeCaloAlignmentEP
// Class:      FakeCaloAlignmentEP
// 
/**\class FakeCaloAlignmentEP FakeCaloAlignmentEP.h Alignment/FakeCaloAlignmentEP/interface/FakeCaloAlignmentEP.h

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
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentErrorRcd.h"

class FakeCaloAlignmentEP : public edm::ESProducer 
{
   public:

      typedef boost::shared_ptr<Alignments>      ReturnAli    ;
      typedef boost::shared_ptr<AlignmentErrors> ReturnAliErr ;

      typedef AlignTransform::Translation Trl ;
      typedef AlignTransform::Rotation    Rot ;

      FakeCaloAlignmentEP(const edm::ParameterSet&)
      {
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceEBAli    ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceEBAliErr ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceEEAli    ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceEEAliErr ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceESAli    ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceESAliErr ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceHBAli    ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceHBAliErr ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceHEAli    ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceHEAliErr ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceHOAli    ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceHOAliErr ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceHFAli    ) ;
	 setWhatProduced( this, &FakeCaloAlignmentEP::produceHFAliErr ) ;
      }

      ~FakeCaloAlignmentEP() {}

//-------------------------------------------------------------------
 
      ReturnAli    produceEBAli( const EBAlignmentRcd& iRecord ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( EcalBarrelGeometry::numberOfAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const EBDetId id ( i+1, 1, EBDetId::SMCRYSTALMODE ) ; // numbered by SM
	    vtr.push_back( AlignTransform( ( 1==id.ism() ? Trl( 0, 0, 0 ) : //-0.3 ) :
					     Trl(0,0,0) ) , 
					   Rot(),
					   id              ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceEBAliErr( const EBAlignmentErrorRcd& iRecord ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceEEAli( const EEAlignmentRcd& iRecord ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( EcalEndcapGeometry::numberOfAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const EEDetId id ( (i%2)==0 ? 1 : 100, 50, (i<2?-1:1) ) ; // numbered by Dee
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id              ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceEEAliErr( const EEAlignmentErrorRcd& iRecord ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceESAli( const ESAlignmentRcd& iRecord ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( EcalPreshowerGeometry::numberOfAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const ESDetId id ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceESAliErr( const ESAlignmentErrorRcd& iRecord ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceHBAli( const HBAlignmentRcd& iRecord ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( 10 ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalDetId id ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceHBAliErr( const HBAlignmentErrorRcd& iRecord ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceHEAli( const HEAlignmentRcd& iRecord ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( 10 ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalDetId id ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceHEAliErr( const HEAlignmentErrorRcd& iRecord ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceHOAli( const HOAlignmentRcd& iRecord ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( 10 ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalDetId id ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceHOAliErr( const HOAlignmentErrorRcd& iRecord ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceHFAli( const HFAlignmentRcd& iRecord ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( 10 ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalDetId id ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceHFAliErr( const HFAlignmentErrorRcd& iRecord ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
//-------------------------------------------------------------------

      ReturnAli    produceZDCAli( const ZDCAlignmentRcd& iRecord ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( 10 ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
	    const HcalZDCDetId id ;
	    vtr.push_back( AlignTransform( Trl( 0, 0, 0 ), 
					   Rot(),
					   id           ) ) ;
	 }
	 return ali ;
      }

      ReturnAliErr produceZDCAliErr( const ZDCAlignmentErrorRcd& iRecord ) 
      { 
	 ReturnAliErr aliErr ( new AlignmentErrors ); 
	 return aliErr ;
      }
};


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(FakeCaloAlignmentEP);
