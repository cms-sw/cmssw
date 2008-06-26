#ifndef DTGeometryBuilder_DTGeometryESModule_h
#define DTGeometryBuilder_DTGeometryESModule_h

/** \class DTGeometryESModule
 * 
 *  ESProducer for DTGeometry in MuonGeometryRecord
 *
 *  $Date: 2007/05/02 15:49:01 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <boost/shared_ptr.hpp>

#include <string>

class DTGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  DTGeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  virtual ~DTGeometryESModule();

  /// Produce DTGeometry.
  boost::shared_ptr<DTGeometry>  produce(const MuonGeometryRecord & record);

private:  
  /// Called when geometry description changes
  void geometryCallback_( const MuonNumberingRecord& );
  boost::shared_ptr<DTGeometry> _dtGeometry;
  bool applyAlignment_; // Switch to apply alignment corrections

  const std::string alignmentsLabel_;
  const std::string myLabel_;
};
#endif






