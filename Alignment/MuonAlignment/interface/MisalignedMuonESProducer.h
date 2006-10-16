#ifndef Alignment_MisalignedMuonESProducer_MisalignedMuonESProducerESProducer_h
#define Alignment_MisalignedMuonESProducer_MisalignedMuonESProducerESProducer_h

/** \class MisalignedMuonESProducer
 *  The misaligned muon ES producer.
 *
 *  $Date: 2006/08/04 20:18:50 $
 *  $Revision: 1.4 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

#include <boost/shared_ptr.hpp>


///
/// An ESProducer that fills the MuonDigiGeometryRcd with a misaligned Muon
/// 
/// This should replace the standard DTGeometry and CSCGeometry producers 
/// when producing Misalignment scenarios.
///
class MisalignedMuonESProducer: public edm::ESProducer
{
public:

  /// Constructor
  MisalignedMuonESProducer( const edm::ParameterSet & p );
  
  /// Destructor
  virtual ~MisalignedMuonESProducer(); 
  
  /// Produce the misaligned Muon geometry and store it
  edm::ESProducts< boost::shared_ptr<DTGeometry>,
 				   boost::shared_ptr<CSCGeometry> > produce( const MuonGeometryRecord&  );

  /// Save alignemnts and error to database
  void saveToDB();
  
private:

  edm::ParameterSet theParameterSet;
  
  boost::shared_ptr<DTGeometry> theDTGeometry;
  boost::shared_ptr<CSCGeometry> theCSCGeometry;

  Alignments*      dtAlignments;
  AlignmentErrors* dtAlignmentErrors; 
  Alignments*      cscAlignments;
  AlignmentErrors* cscAlignmentErrors;  

};


#endif




