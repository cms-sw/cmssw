// -*- C++ -*-
//
// Package:    HcalAlignmentEP
// Class:      HcalAlignmentEP
// 
//
// Original Author:  Brian Heltsley
//
//

#include "Geometry/HcalEventSetup/interface/HcalAlignmentEP.h"

HcalAlignmentEP::HcalAlignmentEP(const edm::ParameterSet&) {
  setWhatProduced( this, &HcalAlignmentEP::produceHcalAli    ) ;
}

HcalAlignmentEP::~HcalAlignmentEP() {}
 
HcalAlignmentEP::ReturnAli HcalAlignmentEP::produceHcalAli( const HcalAlignmentRcd& iRecord ) {

  auto ali = std::make_unique<Alignments>();

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
  assert( hf.isValid() && // require valid alignments and expected size
	  ( hf->m_align.size() == HcalGeometry::numberOfForwardAlignments() ) ) ;
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
