#ifndef CSCGeometryBuilder_CSCGeometryESModule_h
#define CSCGeometryBuilder_CSCGeometryESModule_h

/** \class CSCGeometryESModule
 * 
 *  ESProducer for CSCGeometry in MuonGeometryRecord
 *
 *  \author Tim Cox
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <boost/shared_ptr.hpp>

#include <string>

class CSCGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  CSCGeometryESModule(const edm::ParameterSet& p);

  /// Destructor
  virtual ~CSCGeometryESModule();

  /// Produce CSCGeometry
  boost::shared_ptr<CSCGeometry> produce(const MuonGeometryRecord& record);

private:  

  /// Called when geometry description changes
  void muonNumberingChanged_( const MuonNumberingRecord& );
  void cscRecoGeometryChanged_( const CSCRecoGeometryRcd& );
  void cscRecoDigiParametersChanged_( const CSCRecoDigiParametersRcd& );

  void initCSCGeometry_(const MuonGeometryRecord& );
  boost::shared_ptr<CSCGeometry> cscGeometry;
  bool recreateGeometry_;

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






