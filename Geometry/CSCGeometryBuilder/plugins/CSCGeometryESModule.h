#ifndef CSCGeometryBuilder_CSCGeometryESModule_h
#define CSCGeometryBuilder_CSCGeometryESModule_h

/** \class CSCGeometryESModule
 * 
 *  ESProducer for CSCGeometry in MuonGeometryRecord
 *
 *  \author Tim Cox
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include "FWCore/Framework/interface/ESProductHost.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <memory>
#include <string>

class CSCGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  CSCGeometryESModule(const edm::ParameterSet& p);

  /// Destructor
  ~CSCGeometryESModule() override;

  /// Produce CSCGeometry
  std::shared_ptr<CSCGeometry> produce(const MuonGeometryRecord& record);

private:  


  using HostType = edm::ESProductHost<CSCGeometry,
                                      MuonNumberingRecord,
                                      CSCRecoGeometryRcd,
                                      CSCRecoDigiParametersRcd>;

  void initCSCGeometry_(const MuonGeometryRecord&, std::shared_ptr<HostType>& host);

  edm::ReusableObjectHolder<HostType> holder_;

  // Flags for controlling geometry modelling during build of CSCGeometry
  bool useRealWireGeometry;
  bool useOnlyWiresInME1a;
  bool useGangedStripsInME1a;
  bool useCentreTIOffsets;
  bool debugV;
  bool applyAlignment_; // Switch to apply alignment corrections
  bool useDDD_; // whether to build from DDD or DB
  const std::string alignmentsLabel_;
  const std::string myLabel_;
};
#endif
