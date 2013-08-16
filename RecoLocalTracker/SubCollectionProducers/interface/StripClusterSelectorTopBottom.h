#ifndef RecoSelectors_StripClusterSelectorTopBottom_h
#define RecoSelectors_StripClusterSelectorTopBottom_h

/* \class StripClusterSelectorTopBottom
*
* \author Giuseppe Cerati, INFN
*
*
*/

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

class StripClusterSelectorTopBottom : public edm::EDProducer {

 public:
  explicit StripClusterSelectorTopBottom( const edm::ParameterSet& cfg) :
    label_( cfg.getParameter<edm::InputTag>( "label" ) ),
    y_( cfg.getParameter<double>( "y" ) ) { produces<edmNew::DetSetVector<SiStripCluster> >(); }
  
  void produce( edm::Event& event, const edm::EventSetup& setup);
  
 private:
  edm::InputTag label_;
  double y_;
};

#endif
