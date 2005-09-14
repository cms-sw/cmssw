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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/EDProduct/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgorithm.h"

namespace cms
{
  class SiStripClusterizer : public edm::EDProducer
  {
  public:

    explicit SiStripClusterizer(const edm::ParameterSet& conf);

    virtual ~SiStripClusterizer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    SiStripClusterizerAlgorithm siStripClusterizerAlgorithm_;
    edm::ParameterSet conf_;

  };
}


#endif
