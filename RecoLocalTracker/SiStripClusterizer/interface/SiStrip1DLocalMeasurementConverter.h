#ifndef SiStrip1DLocalMeasurementConverter_h
#define SiStrip1DLocalMeasurementConverter_h

/** \class SiStrip1DLocalMeasurementConverter
 *
 * SiStrip1DLocalMeasurementConverter is the EDProducer subclass which converts
 * SiStripClusters/interface/SiStripCluster.h to SiStripClusters/interface/SiStrip1DLocalMeasurements.h
 *
 * \author Oliver Gutsche, Fermilab
 *
 * \version   1st Version Aug. 01, 2005  

 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/EDProduct/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStrip1DLocalMeasurementConverterAlgorithm.h"

namespace cms
{
  class SiStrip1DLocalMeasurementConverter : public edm::EDProducer
  {
  public:

    explicit SiStrip1DLocalMeasurementConverter(const edm::ParameterSet& conf);

    virtual ~SiStrip1DLocalMeasurementConverter();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    SiStrip1DLocalMeasurementConverterAlgorithm siStrip1DLocalMeasurementConverterAlgorithm_;
    edm::ParameterSet conf_;

  };
}


#endif
