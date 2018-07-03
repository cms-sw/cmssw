// this file is generated automatically by /afs/fnal.gov/files/home/room1/ratnikov/bin/makeNewClass.sh
// name: ratnikov, date: Mon Sep 26 17:02:41 CDT 2005
#ifndef HcalPedestalsRcd_H
#define HcalPedestalsRcd_H
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalPedestalsRcd : public edm::eventsetup::DependentRecordImplementation<HcalPedestalsRcd, boost::mpl::vector<HcalRecNumberingRecord,IdealGeometryRecord> > {};
#endif
