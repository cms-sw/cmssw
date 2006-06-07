#ifndef Alignment_MisalignedMuonESProducer_MisalignedMuonESProducerESProducer_h
#define Alignment_MisalignedMuonESProducer_MisalignedMuonESProducerESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

#include <boost/shared_ptr.hpp>


///
/// An ESProducer that fills the MuonDigiGeometryRcd with a misaligned Muon
/// 
/// 
/// FIXME: configuration file, output POOL-ORA object?
///
class MisalignedMuonESProducer: public edm::ESProducer
{
public:

  /// Constructor
  MisalignedMuonESProducer( const edm::ParameterSet & p );
  
  /// Destructor
  virtual ~MisalignedMuonESProducer(); 
  
  /// Produce the misaligned Muon geometry and store it
//  boost::shared_ptr<DTGeometry> produce( const MuonGeometryRecord& , boost::shared_ptr<DTGeometry> , boost::shared_ptr<CSCGeometry> );

  boost::shared_ptr<DTGeometry> produce( const MuonGeometryRecord&  );

private:

  edm::ParameterSet theParameterSet;
  
  boost::shared_ptr<DTGeometry> theDTGeometry;
  boost::shared_ptr<CSCGeometry> theCSCGeometry;

};


#endif




