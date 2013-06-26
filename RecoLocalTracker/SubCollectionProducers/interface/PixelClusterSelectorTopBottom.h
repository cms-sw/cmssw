#ifndef RecoSelectors_PixelClusterSelectorTopBottom_h
#define RecoSelectors_PixelClusterSelectorTopBottom_h

/* \class PixelClusterSelectorTopBottom
*
* \author Giuseppe Cerati, INFN
*
*  $Date: 2011/01/14 01:24:51 $
*  $Revision: 1.2 $
*
*/

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/InputTag.h"

class PixelClusterSelectorTopBottom : public edm::EDProducer {

 public:
  explicit PixelClusterSelectorTopBottom( const edm::ParameterSet& cfg) :
    label_( cfg.getParameter<edm::InputTag>( "label" ) ),
    y_( cfg.getParameter<double>( "y" ) ) { produces<SiPixelClusterCollectionNew>(); }
  
  void produce( edm::Event& event, const edm::EventSetup& setup);
  
 private:
  edm::InputTag label_;
  double y_;
};

#endif
