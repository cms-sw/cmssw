#ifndef SiStripClusterizer_h
#define SiStripClusterizer_h

/** \class SiStripClusterizer
 *
 * SiStripClusterizer is the EDProducer subclass which clusters
 * SiStripDigi/interface/StripDigi.h to SiStripCluster/interface/SiStripCluster.h
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Aug. 01, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgorithm.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"


namespace cms
{
  class SiStripClusterizer : public edm::EDProducer
  {
  public:

    explicit SiStripClusterizer(const edm::ParameterSet& conf);

    virtual ~SiStripClusterizer();

    virtual void beginJob( const edm::EventSetup& );

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    SiStripClusterizerAlgorithm siStripClusterizerAlgorithm_;
    edm::ParameterSet conf_;
    edm::ESHandle<SiStripNoises> noise;
    edm::ESHandle<TrackingGeometry> pDD;
    bool UseNoiseBadStripFlagFromDB_;
  };
}


#endif
