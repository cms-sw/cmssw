// this file is generated automatically by /afs/fnal.gov/files/home/room1/ratnikov/bin/makeNewClass.sh
// name: ratnikov, date: Mon Sep 26 17:02:30 CDT 2005
#ifndef HcalGainsRcd_H
#define HcalGainsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
class HcalGainsRcd
    : public edm::eventsetup::
          DependentRecordImplementation<HcalGainsRcd, edm::mpl::Vector<HcalRecNumberingRecord, IdealGeometryRecord> > {
};
#endif
