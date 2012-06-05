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
#include "FWCore/Framework/interface/ESHandle.h"

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
	 vtr.resize( nA ) ;

	 edm::ESHandle<Alignments> hb ;
	 edm::ESHandle<Alignments> he ;
	 edm::ESHandle<Alignments> hf ;
	 edm::ESHandle<Alignments> ho ;
	 iRecord.getRecord<HBAlignmentRcd>().get( hb ) ;
	 iRecord.getRecord<HEAlignmentRcd>().get( he ) ;
	 iRecord.getRecord<HFAlignmentRcd>().get( hf ) ;
	 iRecord.getRecord<HOAlignmentRcd>().get( ho ) ;

	 assert( hb.isValid() && // require valid alignments and expected size
		 ( hb->m_align.size() == HcalGeometry::numberOfBarrelAlignments() ) ) ;
	 assert( he.isValid() && // require valid alignments and expected size
		 ( he->m_align.size() == HcalGeometry::numberOfEndcapAlignments() ) ) ;
	 assert( ho.isValid() && // require valid alignments and expected size
		 ( ho->m_align.size() == HcalGeometry::numberOfForwardAlignments() ) ) ;
	 assert( ho.isValid() && // require valid alignments and expected size
		 ( ho->m_align.size() == HcalGeometry::numberOfOuterAlignments() ) ) ;
	 const std::vector<AlignTransform>& hbt = hb->m_align ;
	 const std::vector<AlignTransform>& het = he->m_align ;
	 const std::vector<AlignTransform>& hft = hf->m_align ;
	 const std::vector<AlignTransform>& hot = ho->m_align ;

	 copy( hbt.begin(), hbt.end(), vtr.begin() ) ;
	 copy( het.begin(), het.end(), vtr.begin()+hbt.size() ) ;
	 copy( hft.begin(), hft.end(), vtr.begin()+hbt.size()+het.size() ) ;
	 copy( hot.begin(), hot.end(), vtr.begin()+hbt.size()+het.size()+hft.size() ) ;

	 return ali ;
      }
};


//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalAlignmentEP);
