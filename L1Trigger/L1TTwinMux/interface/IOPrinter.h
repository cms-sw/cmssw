//-------------------------------------------------
//
//   Class: IOPrinter
//
//   IOPrinter
//
//
//   Author :
//   G. Flouris               U Ioannina    Feb. 2015
//--------------------------------------------------
#ifndef L1T_TwinMux_IOPrinter_H
#define L1T_TwinMux_IOPrinter_H


#include <iostream>
#include <iomanip>
#include <iterator>

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "L1Trigger/L1TTwinMux/interface/AlignTrackSegments.h"
#include "L1Trigger/L1TTwinMux/interface/RPCtoDTTranslator.h"
#include "L1Trigger/L1TTwinMux/interface/DTRPCBxCorrection.h"
#include "L1Trigger/L1TTwinMux/interface/RPCHitCleaner.h"

class IOPrinter{
public:
  IOPrinter() {};
  ~IOPrinter() {};
  void run(edm::Handle<L1MuDTChambPhContainer>, const L1MuDTChambPhContainer &,  edm::Handle<RPCDigiCollection>,
            const edm::EventSetup& );
  void run(L1MuDTChambPhContainer const*, const L1MuDTChambPhContainer & , RPCDigiCollection const*,
            const edm::EventSetup& );

};

#endif
