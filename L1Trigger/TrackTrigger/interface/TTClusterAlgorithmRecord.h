/*! \class   TTClusterAlgorithmRecord
 *  \brief   Class to store the TTClusterAlgorithm used
 *           in TTClusterBuilder
 *
 *  \author Andrew W. Rose
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ALGO_RECORD_H
#define L1_TRACK_TRIGGER_CLUSTER_ALGO_RECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "FWCore/Utilities/interface/mplVector.h"

class TTClusterAlgorithmRecord
    : public edm::eventsetup::DependentRecordImplementation<TTClusterAlgorithmRecord,
                                                            edm::mpl::Vector<IdealMagneticFieldRecord> > {};
#endif
