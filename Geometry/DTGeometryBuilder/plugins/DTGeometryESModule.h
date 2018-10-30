#ifndef DTGeometryBuilder_DTGeometryESModule_h
#define DTGeometryBuilder_DTGeometryESModule_h

/** \class DTGeometryESModule
 * 
 *  ESProducer for DTGeometry in MuonGeometryRecord
 *
 *  \author N. Amapane - CERN
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>

#include <memory>
#include <string>

class DTGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  DTGeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  ~DTGeometryESModule() override;

  /// Produce DTGeometry.
  std::shared_ptr<DTGeometry> produce(const MuonGeometryRecord& record);

private:

  using HostType = edm::ESProductHost<DTGeometry,
                                      MuonNumberingRecord,
                                      DTRecoGeometryRcd>;

  void setupGeometry(MuonNumberingRecord const&, std::shared_ptr<HostType>&);
  void setupDBGeometry(DTRecoGeometryRcd const&, std::shared_ptr<HostType>&);

  edm::ReusableObjectHolder<HostType> holder_;

  bool applyAlignment_; // Switch to apply alignment corrections
  const std::string alignmentsLabel_;
  const std::string myLabel_;
  bool fromDDD_;
};
#endif
