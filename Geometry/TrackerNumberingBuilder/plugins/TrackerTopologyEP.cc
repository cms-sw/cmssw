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

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackerTopologyEP::TrackerTopologyEP(const edm::ParameterSet& conf)
{
  edm::LogInfo("TRACKER") << "TrackerTopologyIdealEP::TrackerTopologyIdealEP";

  pxbVals_.layerStartBit_=conf.getParameter<unsigned int>("pxb_layerStartBit");
  pxbVals_.ladderStartBit_=conf.getParameter<unsigned int>("pxb_ladderStartBit");
  pxbVals_.moduleStartBit_=conf.getParameter<unsigned int>("pxb_moduleStartBit");
  pxbVals_.layerMask_=conf.getParameter<unsigned int>("pxb_layerMask");
  pxbVals_.ladderMask_=conf.getParameter<unsigned int>("pxb_ladderMask");
  pxbVals_.moduleMask_=conf.getParameter<unsigned int>("pxb_moduleMask");
  pxfVals_.sideStartBit_=conf.getParameter<unsigned int>("pxf_sideStartBit");
  pxfVals_.diskStartBit_=conf.getParameter<unsigned int>("pxf_diskStartBit");
  pxfVals_.bladeStartBit_=conf.getParameter<unsigned int>("pxf_bladeStartBit");
  pxfVals_.panelStartBit_=conf.getParameter<unsigned int>("pxf_panelStartBit");
  pxfVals_.moduleStartBit_=conf.getParameter<unsigned int>("pxf_moduleStartBit");
  pxfVals_.sideMask_=conf.getParameter<unsigned int>("pxf_sideMask");
  pxfVals_.diskMask_=conf.getParameter<unsigned int>("pxf_diskMask");
  pxfVals_.bladeMask_=conf.getParameter<unsigned int>("pxf_bladeMask");
  pxfVals_.panelMask_=conf.getParameter<unsigned int>("pxf_panelMask");
  pxfVals_.moduleMask_=conf.getParameter<unsigned int>("pxf_moduleMask");
  tecVals_.sideStartBit_=conf.getParameter<unsigned int>("tec_sideStartBit");
  tecVals_.wheelStartBit_=conf.getParameter<unsigned int>("tec_wheelStartBit");
  tecVals_.petal_fw_bwStartBit_=conf.getParameter<unsigned int>("tec_petal_fw_bwStartBit");
  tecVals_.petalStartBit_=conf.getParameter<unsigned int>("tec_petalStartBit");
  tecVals_.ringStartBit_=conf.getParameter<unsigned int>("tec_ringStartBit");
  tecVals_.moduleStartBit_=conf.getParameter<unsigned int>("tec_moduleStartBit");
  tecVals_.sterStartBit_=conf.getParameter<unsigned int>("tec_sterStartBit");
  tecVals_.sideMask_=conf.getParameter<unsigned int>("tec_sideMask");
  tecVals_.wheelMask_=conf.getParameter<unsigned int>("tec_wheelMask");
  tecVals_.petal_fw_bwMask_=conf.getParameter<unsigned int>("tec_petal_fw_bwMask");
  tecVals_.petalMask_=conf.getParameter<unsigned int>("tec_petalMask");
  tecVals_.ringMask_=conf.getParameter<unsigned int>("tec_ringMask");
  tecVals_.moduleMask_=conf.getParameter<unsigned int>("tec_moduleMask");
  tecVals_.sterMask_=conf.getParameter<unsigned int>("tec_sterMask");
  tibVals_.layerStartBit_=conf.getParameter<unsigned int>("tib_layerStartBit");
  tibVals_.str_fw_bwStartBit_=conf.getParameter<unsigned int>("tib_str_fw_bwStartBit");
  tibVals_.str_int_extStartBit_=conf.getParameter<unsigned int>("tib_str_int_extStartBit");
  tibVals_.strStartBit_=conf.getParameter<unsigned int>("tib_strStartBit");
  tibVals_.moduleStartBit_=conf.getParameter<unsigned int>("tib_moduleStartBit");
  tibVals_.sterStartBit_=conf.getParameter<unsigned int>("tib_sterStartBit");
  tibVals_.layerMask_=conf.getParameter<unsigned int>("tib_layerMask");
  tibVals_.str_fw_bwMask_=conf.getParameter<unsigned int>("tib_str_fw_bwMask");
  tibVals_.str_int_extMask_=conf.getParameter<unsigned int>("tib_str_int_extMask");
  tibVals_.strMask_=conf.getParameter<unsigned int>("tib_strMask");
  tibVals_.moduleMask_=conf.getParameter<unsigned int>("tib_moduleMask");
  tibVals_.sterMask_=conf.getParameter<unsigned int>("tib_sterMask");
  tidVals_.sideStartBit_=conf.getParameter<unsigned int>("tid_sideStartBit");
  tidVals_.wheelStartBit_=conf.getParameter<unsigned int>("tid_wheelStartBit");
  tidVals_.ringStartBit_=conf.getParameter<unsigned int>("tid_ringStartBit");
  tidVals_.module_fw_bwStartBit_=conf.getParameter<unsigned int>("tid_module_fw_bwStartBit");
  tidVals_.moduleStartBit_=conf.getParameter<unsigned int>("tid_moduleStartBit");
  tidVals_.sterStartBit_=conf.getParameter<unsigned int>("tid_sterStartBit");
  tidVals_.sideMask_=conf.getParameter<unsigned int>("tid_sideMask");
  tidVals_.wheelMask_=conf.getParameter<unsigned int>("tid_wheelMask");
  tidVals_.ringMask_=conf.getParameter<unsigned int>("tid_ringMask");
  tidVals_.module_fw_bwMask_=conf.getParameter<unsigned int>("tid_module_fw_bwMask");
  tidVals_.moduleMask_=conf.getParameter<unsigned int>("tid_moduleMask");
  tidVals_.sterMask_=conf.getParameter<unsigned int>("tid_sterMask");
  tobVals_.layerStartBit_=conf.getParameter<unsigned int>("tob_layerStartBit");
  tobVals_.rod_fw_bwStartBit_=conf.getParameter<unsigned int>("tob_rod_fw_bwStartBit");
  tobVals_.rodStartBit_=conf.getParameter<unsigned int>("tob_rodStartBit");
  tobVals_.moduleStartBit_=conf.getParameter<unsigned int>("tob_moduleStartBit");
  tobVals_.sterStartBit_=conf.getParameter<unsigned int>("tob_sterStartBit");
  tobVals_.layerMask_=conf.getParameter<unsigned int>("tob_layerMask");
  tobVals_.rod_fw_bwMask_=conf.getParameter<unsigned int>("tob_rod_fw_bwMask");
  tobVals_.rodMask_=conf.getParameter<unsigned int>("tob_rodMask");
  tobVals_.moduleMask_=conf.getParameter<unsigned int>("tob_moduleMask");
  tobVals_.sterMask_=conf.getParameter<unsigned int>("tob_sterMask");
  
  setWhatProduced(this);
}


TrackerTopologyEP::~TrackerTopologyEP()
{ 
}

void
TrackerTopologyEP::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) 
{
  edm::ParameterSetDescription ttc;
  ttc.add<unsigned int>("pxb_layerStartBit",16);
  ttc.add<unsigned int>("pxb_ladderStartBit",8);
  ttc.add<unsigned int>("pxb_moduleStartBit",2);
  ttc.add<unsigned int>("pxb_layerMask",0xF);
  ttc.add<unsigned int>("pxb_ladderMask",0xFF);
  ttc.add<unsigned int>("pxb_moduleMask",0x3F);
  
  ttc.add<unsigned int>("pxf_sideStartBit",23);
  ttc.add<unsigned int>("pxf_diskStartBit",16);
  ttc.add<unsigned int>("pxf_bladeStartBit",10); 
  ttc.add<unsigned int>("pxf_panelStartBit",8);
  ttc.add<unsigned int>("pxf_moduleStartBit",2);
  ttc.add<unsigned int>("pxf_sideMask",0x3);
  ttc.add<unsigned int>("pxf_diskMask",0xF);
  ttc.add<unsigned int>("pxf_bladeMask",0x3F);
  ttc.add<unsigned int>("pxf_panelMask",0x3);
  ttc.add<unsigned int>("pxf_moduleMask",0x3F);
  
  ttc.add<unsigned int>("tec_sideStartBit",18);
  ttc.add<unsigned int>("tec_wheelStartBit",14);
  ttc.add<unsigned int>("tec_petal_fw_bwStartBit",12);
  ttc.add<unsigned int>("tec_petalStartBit",8);
  ttc.add<unsigned int>("tec_ringStartBit",5);
  ttc.add<unsigned int>("tec_moduleStartBit",2);
  ttc.add<unsigned int>("tec_sterStartBit",0);  
  ttc.add<unsigned int>("tec_sideMask",0x3);
  ttc.add<unsigned int>("tec_wheelMask",0xF);
  ttc.add<unsigned int>("tec_petal_fw_bwMask",0x3);
  ttc.add<unsigned int>("tec_petalMask",0xF);
  ttc.add<unsigned int>("tec_ringMask",0x7);  
  ttc.add<unsigned int>("tec_moduleMask",0x7);
  ttc.add<unsigned int>("tec_sterMask",0x3);
  
  ttc.add<unsigned int>("tib_layerStartBit",14);
  ttc.add<unsigned int>("tib_str_fw_bwStartBit",12);
  ttc.add<unsigned int>("tib_str_int_extStartBit",10);
  ttc.add<unsigned int>("tib_strStartBit",4);
  ttc.add<unsigned int>("tib_moduleStartBit",2);
  ttc.add<unsigned int>("tib_sterStartBit",0);
  ttc.add<unsigned int>("tib_layerMask",0x7);
  ttc.add<unsigned int>("tib_str_fw_bwMask",0x3);
  ttc.add<unsigned int>("tib_str_int_extMask",0x3);
  ttc.add<unsigned int>("tib_strMask",0x3F);
  ttc.add<unsigned int>("tib_moduleMask",0x3);
  ttc.add<unsigned int>("tib_sterMask",0x3);
  
  ttc.add<unsigned int>("tid_sideStartBit",13); 
  ttc.add<unsigned int>("tid_wheelStartBit",11);
  ttc.add<unsigned int>("tid_ringStartBit",9);
  ttc.add<unsigned int>("tid_module_fw_bwStartBit",7);
  ttc.add<unsigned int>("tid_moduleStartBit",2);
  ttc.add<unsigned int>("tid_sterStartBit",0);
  ttc.add<unsigned int>("tid_sideMask",0x3);
  ttc.add<unsigned int>("tid_wheelMask",0x3);
  ttc.add<unsigned int>("tid_ringMask",0x3);
  ttc.add<unsigned int>("tid_module_fw_bwMask",0x3);
  ttc.add<unsigned int>("tid_moduleMask",0x1F);
  ttc.add<unsigned int>("tid_sterMask",0x3);
  
  ttc.add<unsigned int>("tob_layerStartBit",14);
  ttc.add<unsigned int>("tob_rod_fw_bwStartBit",12);
  ttc.add<unsigned int>("tob_rodStartBit",5);  
  ttc.add<unsigned int>("tob_moduleStartBit",2);
  ttc.add<unsigned int>("tob_sterStartBit",0);
  ttc.add<unsigned int>("tob_layerMask",0x7);
  ttc.add<unsigned int>("tob_rod_fw_bwMask",0x3);
  ttc.add<unsigned int>("tob_rodMask",0x7F);
  ttc.add<unsigned int>("tob_moduleMask",0x7);
  ttc.add<unsigned int>("tob_sterMask",0x3);
  
  descriptions.add( "trackerTopologyConstants", ttc );
}

//
// member functions
//

// ------------ method called to produce the data  ------------
TrackerTopologyEP::ReturnType
TrackerTopologyEP::produce(const IdealGeometryRecord& iRecord)
{
  edm::LogInfo("TrackerTopologyEP") <<  "TrackerTopologyIdealEP::produce(const IdealGeometryRecord& iRecord)";
    
  ReturnType myTopo(new TrackerTopology(pxbVals_, pxfVals_, tecVals_, tibVals_, tidVals_, tobVals_));

  return myTopo ;
}


DEFINE_FWK_EVENTSETUP_MODULE( TrackerTopologyEP);

