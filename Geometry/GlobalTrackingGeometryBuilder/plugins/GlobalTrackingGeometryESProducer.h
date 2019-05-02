#ifndef GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h
#define GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h

/** \class GlobalTrackingGeometry
 * 
 *  ESProducer for GlobalTrackingGeometry in MuonGeometryRecord
 *
 *  \author Matteo Sani
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include <memory>
#include <string>

class GlobalTrackingGeometry;
class TrackerGeometry;
class MTDGeometry;
class DTGeometry;
class CSCGeometry;
class RPCGeometry;
class GEMGeometry;
class ME0Geometry;
class TrackerDigiGeometryRecord;
class MTDDigiGeometryRecord;
class MuonGeometryRecord;

class GlobalTrackingGeometryESProducer : public edm::ESProducer {
public:

  /// Constructor
  GlobalTrackingGeometryESProducer(const edm::ParameterSet & p);

  /// Destructor
  ~GlobalTrackingGeometryESProducer() override;

  /// Produce GlobalTrackingGeometry
  std::unique_ptr<GlobalTrackingGeometry> produce(const GlobalTrackingGeometryRecord& record);

private:  

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerToken_;
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdToken_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtToken_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscToken_;
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemToken_;
  edm::ESGetToken<ME0Geometry, MuonGeometryRecord> me0Token_;
};
#endif






