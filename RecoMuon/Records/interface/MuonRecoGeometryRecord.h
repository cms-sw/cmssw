#ifndef RecoMuon_Record_MuonRecoGeometryRecord_h
#define RecoMuon_Record_MuonRecoGeometryRecord_h

/** \class MuonRecoGeometryRecord
 *
 *  Record to hold muon reconstruction geometries.
 *
 *  \author N. Amapane - CERN
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class MuonRecoGeometryRecord
    : public edm::eventsetup::DependentRecordImplementation<MuonRecoGeometryRecord,
                                                            edm::mpl::Vector<MuonGeometryRecord> > {};

#endif
