// -*- C++ -*-
//
// Package:    HcalAlignmentEP
// Class:      HcalAlignmentEP
// 
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
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentErrorRcd.h"

class HcalAlignmentEP : public edm::ESProducer 
{
   public:

      typedef boost::shared_ptr<Alignments>      ReturnAli    ;
      typedef boost::shared_ptr<AlignmentErrors> ReturnAliErr ;

      typedef AlignTransform::Translation Trl ;
      typedef AlignTransform::Rotation    Rot ;

      HcalAlignmentEP(const edm::ParameterSet&)
      {
	 setWhatProduced( this, &HcalAlignmentEP::produceHcalAli    ) ;
      }

      ~HcalAlignmentEP() {}

//-------------------------------------------------------------------
 
      ReturnAli    produceHcalAli( const HcalAlignmentRcd& iRecord ) 
      {
	 ReturnAli ali ( new Alignments ) ;
	 std::vector<AlignTransform>& vtr ( ali->m_align ) ;
	 const unsigned int nA ( HcalGeometry::numberOfAlignments() ) ; 
	 vtr.reserve( nA ) ;
	 for( unsigned int i ( 0 ) ; i != nA ; ++i )
	 {
//	    const EBDetId id ( i+1, 1, EBDetId::SMCRYSTALMODE ) ; // numbered by SM
//	    vtr.push_back( AlignTransform( ( 1==id.ism() ? Trl( 0, 0, 0 ) : //-0.3 ) :
//					     Trl(0,0,0) ) , 
//					   Rot(),
//					   id              ) ) ;
	 }
	 return ali ;
      }
};


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalAlignmentEP);
