#ifndef RecoLocalTracker_SiStripClusterizer_SiStripClusterizer_h
#define RecoLocalTracker_SiStripClusterizer_SiStripClusterizer_h

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

//edm
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
//Clusterizer
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgorithm.h"

#include <iostream> 
#include <memory>
#include <string>

class SiStripQuality;

namespace cms
{
  class SiStripClusterizer : public edm::EDProducer
  {
  public:

    explicit SiStripClusterizer(const edm::ParameterSet& conf);

    virtual ~SiStripClusterizer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ParameterSet conf_;
    SiStripClusterizerAlgorithm SiStripClusterizerAlgorithm_;

    SiStripQuality emptyQuality;
  };
}
#endif
