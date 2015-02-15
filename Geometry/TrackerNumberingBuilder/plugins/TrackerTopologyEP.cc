// -*- C++ -*-
//
// Package:    TrackerTopologyEP
// Class:      TrackerTopologyEP
// 

#include "TrackerTopologyEP.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"

TrackerTopologyEP::TrackerTopologyEP(const edm::ParameterSet& conf)
{
  edm::LogInfo("TRACKER") << "TrackerTopologyIdealEP::TrackerTopologyIdealEP";

  setWhatProduced(this);
}

TrackerTopologyEP::~TrackerTopologyEP()
{ 
}

void
TrackerTopologyEP::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) 
{
  edm::ParameterSetDescription ttc;
  descriptions.add( "trackerTopologyConstants", ttc );
}

TrackerTopologyEP::ReturnType
TrackerTopologyEP::produce(const TrackerTopologyRcd& iRecord)
{
  edm::LogInfo("TrackerTopologyEP") <<  "TrackerTopologyIdealEP::produce(const IdealGeometryRecord& iRecord)";

  edm::ESHandle<PTrackerParameters> ptp;
  iRecord.getRecord<PTrackerParametersRcd>().get( ptp );

  pxbVals_.layerStartBit_ = ptp->pxb.layerStartBit; // 16
  pxbVals_.ladderStartBit_ = ptp->pxb.ladderStartBit; // 8
  pxbVals_.moduleStartBit_ = ptp->pxb.moduleStartBit; // 2
  pxbVals_.layerMask_ = ptp->pxb.layerMask; // 0xF
  pxbVals_.ladderMask_ = ptp->pxb.ladderMask; // 0xFF
  pxbVals_.moduleMask_ = ptp->pxb.moduleMask; // 0x3F

  pxfVals_.sideStartBit_ = ptp->pxf.sideStartBit;
  pxfVals_.diskStartBit_ = ptp->pxf.diskStartBit;
  pxfVals_.bladeStartBit_ = ptp->pxf.bladeStartBit;
  pxfVals_.panelStartBit_ = ptp->pxf.panelStartBit;
  pxfVals_.moduleStartBit_ = ptp->pxf.moduleStartBit;
  pxfVals_.sideMask_ = ptp->pxf.sideMask;
  pxfVals_.diskMask_ = ptp->pxf.diskMask;
  pxfVals_.bladeMask_ = ptp->pxf.bladeMask;
  pxfVals_.panelMask_ = ptp->pxf.panelMask;
  pxfVals_.moduleMask_ = ptp->pxf.moduleMask;
 
  tecVals_.sideStartBit_ = ptp->tec.sideStartBit;
  tecVals_.wheelStartBit_ = ptp->tec.wheelStartBit;
  tecVals_.petal_fw_bwStartBit_ = ptp->tec.petal_fw_bwStartBit;
  tecVals_.petalStartBit_ = ptp->tec.petalStartBit;
  tecVals_.ringStartBit_ = ptp->tec.ringStartBit;
  tecVals_.moduleStartBit_ = ptp->tec.moduleStartBit;
  tecVals_.sterStartBit_ = ptp->tec.sterStartBit;
  tecVals_.sideMask_ = ptp->tec.sideMask;
  tecVals_.wheelMask_ = ptp->tec.wheelMask;
  tecVals_.petal_fw_bwMask_ = ptp->tec.petal_fw_bwMask;
  tecVals_.petalMask_ = ptp->tec.petalMask;
  tecVals_.ringMask_ = ptp->tec.ringMask;
  tecVals_.moduleMask_ = ptp->tec.moduleMask;
  tecVals_.sterMask_ = ptp->tec.sterMask;
 
  tibVals_.layerStartBit_ = ptp->tib.layerStartBit;
  tibVals_.str_fw_bwStartBit_ = ptp->tib.str_fw_bwStartBit;
  tibVals_.str_int_extStartBit_ = ptp->tib.str_int_extStartBit;
  tibVals_.strStartBit_ = ptp->tib.strStartBit;
  tibVals_.moduleStartBit_ = ptp->tib.moduleStartBit;
  tibVals_.sterStartBit_ = ptp->tib.sterStartBit;
  tibVals_.layerMask_ = ptp->tib.layerMask;
  tibVals_.str_fw_bwMask_ = ptp->tib.str_fw_bwMask;
  tibVals_.str_int_extMask_ = ptp->tib.str_int_extMask;
  tibVals_.strMask_ = ptp->tib.strMask;
  tibVals_.moduleMask_ = ptp->tib.moduleMask;
  tibVals_.sterMask_ = ptp->tib.sterMask;
  
  tidVals_.sideStartBit_= ptp->tid.sideStartBit;
  tidVals_.wheelStartBit_= ptp->tid.wheelStartBit;
  tidVals_.ringStartBit_= ptp->tid.ringStartBit;
  tidVals_.module_fw_bwStartBit_= ptp->tid.module_fw_bwStartBit;
  tidVals_.moduleStartBit_= ptp->tid.moduleStartBit;
  tidVals_.sterStartBit_= ptp->tid.sterStartBit;
  tidVals_.sideMask_= ptp->tid.sideMask;
  tidVals_.wheelMask_= ptp->tid.wheelMask;
  tidVals_.ringMask_= ptp->tid.ringMask;
  tidVals_.module_fw_bwMask_= ptp->tid.module_fw_bwMask;
  tidVals_.moduleMask_= ptp->tid.moduleMask;
  tidVals_.sterMask_= ptp->tid.sterMask;

  tobVals_.layerStartBit_ = ptp->tob.layerStartBit;
  tobVals_.rod_fw_bwStartBit_= ptp->tob.rod_fw_bwStartBit;
  tobVals_.rodStartBit_= ptp->tob.rodStartBit;
  tobVals_.moduleStartBit_= ptp->tob.moduleStartBit;
  tobVals_.sterStartBit_= ptp->tob.sterStartBit;
  tobVals_.layerMask_= ptp->tob.layerMask;
  tobVals_.rod_fw_bwMask_= ptp->tob.rod_fw_bwMask;
  tobVals_.rodMask_= ptp->tob.rodMask;
  tobVals_.moduleMask_= ptp->tob.moduleMask;
  tobVals_.sterMask_= ptp->tob.sterMask;

  ReturnType myTopo(new TrackerTopology(pxbVals_, pxfVals_, tecVals_, tibVals_, tidVals_, tobVals_));

  return myTopo ;
}

DEFINE_FWK_EVENTSETUP_MODULE( TrackerTopologyEP);

