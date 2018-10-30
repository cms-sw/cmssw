#ifndef GEOMETRY_HCALEVENTSETUP_HCALALIGNMENTEP_H
#define GEOMETRY_HCALEVENTSETUP_HCALALIGNMENTEP_H 1

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
#include "CondFormats/AlignmentRecord/interface/HcalAlignmentErrorExtendedRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

class HcalAlignmentEP : public edm::ESProducer {

public:

  using ReturnAli = std::unique_ptr<Alignments>;

  typedef AlignTransform::Translation Trl ;
  typedef AlignTransform::Rotation    Rot ;

  HcalAlignmentEP(const edm::ParameterSet&);
  ~HcalAlignmentEP() override;

//-------------------------------------------------------------------
 
  ReturnAli    produceHcalAli( const HcalAlignmentRcd& iRecord );
};

#endif
