#ifndef DetLayers_MuonDeLayerGeometryESProducer_h
#define DetLayers_MuonDeLayerGeometryESProducer_h

/** \class MuonDeLayerGeometryESProducer
 *
 *  ESProducer for MuonDeLayerGeometry in MuonRecoGeometryRecord.
 *
 *  \author N. Amapane - CERN
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <RecoMuon/Records/interface/MuonRecoGeometryRecord.h>
#include <RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h>
#include <boost/shared_ptr.hpp>


class  MuonDetLayerGeometryESProducer: public edm::ESProducer{
 public:
  /// Constructor
  MuonDetLayerGeometryESProducer(const edm::ParameterSet & p);

  /// Destructor
  virtual ~MuonDetLayerGeometryESProducer(); 

  /// Produce MuonDeLayerGeometry.
  boost::shared_ptr<MuonDetLayerGeometry> produce(const MuonRecoGeometryRecord & record);

 private:
};


#endif
