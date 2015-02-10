#include "Geometry/TrackerGeometryBuilder/interface/TrackerParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Base/interface/DDutils.h"

bool
TrackerParametersFromDD::build( const DDCompactView* cvp,
				PTrackerParameters& ptp)
{
  std::vector<int> pxbPars = dbl_to_int( DDVectorGetter::get( "pxbPars" ));
  assert( pxbPars.size() == 6 );
  
  ptp.pxb.layerStartBit = pxbPars.at(0); // 16
  ptp.pxb.ladderStartBit = pxbPars.at(1); // 8
  ptp.pxb.moduleStartBit = pxbPars.at(2); // 2
  ptp.pxb.layerMask = pxbPars.at(3); // 0xF
  ptp.pxb.ladderMask = pxbPars.at(4); // 0xFF
  ptp.pxb.moduleMask = pxbPars.at(5); // 0x3F

  std::vector<int> pxfPars = dbl_to_int( DDVectorGetter::get( "pxfPars" ));
  assert( pxfPars.size() == 10 );
  
  ptp.pxf.sideStartBit = pxfPars.at(0); // 23
  ptp.pxf.diskStartBit = pxfPars.at(1); // 16
  ptp.pxf.bladeStartBit = pxfPars.at(2); // 10
  ptp.pxf.panelStartBit = pxfPars.at(3); // 8
  ptp.pxf.moduleStartBit = pxfPars.at(4); // 2
  ptp.pxf.sideMask = pxfPars.at(5); // 0x3
  ptp.pxf.diskMask = pxfPars.at(6); // 0xF
  ptp.pxf.bladeMask = pxfPars.at(7); // 0x3F
  ptp.pxf.panelMask = pxfPars.at(8); // 0x3
  ptp.pxf.moduleMask = pxfPars.at(9); // 0x3F

  std::vector<int> tecPars = dbl_to_int( DDVectorGetter::get( "tecPars" ));
  assert( tecPars.size() == 14 );

  ptp.tec.sideStartBit = tecPars.at(0); // 18
  ptp.tec.wheelStartBit = tecPars.at(1); // 14
  ptp.tec.petal_fw_bwStartBit = tecPars.at(2); // 12
  ptp.tec.petalStartBit = tecPars.at(3); // 8
  ptp.tec.ringStartBit = tecPars.at(4); // 5
  ptp.tec.moduleStartBit = tecPars.at(5); // 2
  ptp.tec.sterStartBit = tecPars.at(6); // 0
  ptp.tec.sideMask = tecPars.at(7); // 0x3
  ptp.tec.wheelMask = tecPars.at(8); // 0xF
  ptp.tec.petal_fw_bwMask = tecPars.at(9); // 0x3
  ptp.tec.petalMask = tecPars.at(10); // 0xF
  ptp.tec.ringMask = tecPars.at(11); // 0x7
  ptp.tec.moduleMask = tecPars.at(12); // 0x7
  ptp.tec.sterMask = tecPars.at(13); // 0x3

  std::vector<int> tibPars = dbl_to_int( DDVectorGetter::get( "tibPars" ));
  assert( tibPars.size() == 12 );

  ptp.tib.layerStartBit = tibPars.at(0); // 14
  ptp.tib.str_fw_bwStartBit = tibPars.at(1); // 12
  ptp.tib.str_int_extStartBit = tibPars.at(2); // 10
  ptp.tib.strStartBit = tibPars.at(3); // 4
  ptp.tib.moduleStartBit = tibPars.at(4); // 2
  ptp.tib.sterStartBit = tibPars.at(5); // 0
  ptp.tib.layerMask = tibPars.at(6); // 0x7
  ptp.tib.str_fw_bwMask = tibPars.at(7); // 0x3
  ptp.tib.str_int_extMask = tibPars.at(8); // 0x3
  ptp.tib.strMask = tibPars.at(9); // 0x3F
  ptp.tib.moduleMask = tibPars.at(10); // 0x3
  ptp.tib.sterMask = tibPars.at(11); // 0x3

  std::vector<int> tidPars = dbl_to_int( DDVectorGetter::get( "tidPars" ));
  assert( tidPars.size() == 12 );
  
  ptp.tid.sideStartBit = tidPars.at(0); // 13
  ptp.tid.wheelStartBit = tidPars.at(1); // 11
  ptp.tid.ringStartBit = tidPars.at(2); // 9
  ptp.tid.module_fw_bwStartBit = tidPars.at(3); // 7
  ptp.tid.moduleStartBit = tidPars.at(4); // 2
  ptp.tid.sterStartBit = tidPars.at(5); // 0
  ptp.tid.sideMask = tidPars.at(6); // 0x3
  ptp.tid.wheelMask = tidPars.at(7); // 0x3
  ptp.tid.ringMask = tidPars.at(8); // 0x3
  ptp.tid.module_fw_bwMask = tidPars.at(9); // 0x3
  ptp.tid.moduleMask = tidPars.at(10); // 0x1F
  ptp.tid.sterMask = tidPars.at(11); // 0x3

  std::vector<int> tobPars = dbl_to_int( DDVectorGetter::get( "tobPars" ));
  assert( tobPars.size() == 10 );
  
  ptp.tob.layerStartBit = tobPars.at(0); // 14
  ptp.tob.rod_fw_bwStartBit = tobPars.at(1); // 12
  ptp.tob.rodStartBit = tobPars.at(2); // 5
  ptp.tob.moduleStartBit = tobPars.at(3); // 2
  ptp.tob.sterStartBit = tobPars.at(4); // 0
  ptp.tob.layerMask = tobPars.at(5); // 0x7
  ptp.tob.rod_fw_bwMask = tobPars.at(6); // 0x3
  ptp.tob.rodMask = tobPars.at(7); // 0x7F
  ptp.tob.moduleMask = tobPars.at(8); // 0x7
  ptp.tob.sterMask = tobPars.at(9); // 0x3

  ptp.topology = dbl_to_int( DDVectorGetter::get( "topologyPars" ));

  return true;
}
