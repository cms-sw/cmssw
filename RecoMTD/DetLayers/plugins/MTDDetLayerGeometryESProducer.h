#ifndef RecoMTD_DetLayers_MTDDetLayerGeometryESProducer_h
#define RecoMTD_DetLayers_MTDDetLayerGeometryESProducer_h

/** \class MTDDetLayerGeometryESProducer
 *
 *  ESProducer for MTDDetLayerGeometry in RecoMTD/DetLayers
 *
 *  \author L. Gray - FNAL
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <RecoMTD/Records/interface/MTDRecoGeometryRecord.h>
#include <RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h>
#include <memory>


class  MTDDetLayerGeometryESProducer: public edm::ESProducer{
 public:
  /// Constructor
  MTDDetLayerGeometryESProducer(const edm::ParameterSet & p);

  /// Destructor
  ~MTDDetLayerGeometryESProducer() override; 

  /// Produce MuonDeLayerGeometry.
  std::shared_ptr<MTDDetLayerGeometry> produce(const MTDRecoGeometryRecord & record);

 private:
};


#endif
