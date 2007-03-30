#ifndef RecoLocalTracker_Records_TkStripCPERecord_h
#define RecoLocalTracker_Records_TkStripCPERecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"                
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "boost/mpl/vector.hpp"

class  TkStripCPERecord: public edm::eventsetup::DependentRecordImplementation<TkStripCPERecord,
  boost::mpl::vector<TrackerDigiGeometryRecord,IdealMagneticFieldRecord,SiStripLorentzAngleRcd> > {};

#endif 

