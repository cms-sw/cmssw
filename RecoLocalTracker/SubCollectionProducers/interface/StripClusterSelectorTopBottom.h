#ifndef RecoSelectors_StripClusterSelectorTopBottom_h
#define RecoSelectors_StripClusterSelectorTopBottom_h

/* \class StripClusterSelectorTopBottom
*
* \author Giuseppe Cerati, INFN
*
*
*/

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
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

class StripClusterSelectorTopBottom : public edm::global::EDProducer<> {

 public:
  explicit StripClusterSelectorTopBottom( const edm::ParameterSet& cfg) :
    token_( consumes<edmNew::DetSetVector<SiStripCluster>>(cfg.getParameter<edm::InputTag>( "label" ) )),
    y_( cfg.getParameter<double>( "y" ) ) { produces<edmNew::DetSetVector<SiStripCluster> >(); }
  
  void produce( edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const override;
  
 private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> token_;
  double y_;
};

#endif
