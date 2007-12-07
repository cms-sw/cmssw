#ifndef SiStripRecHitConverter_h
#define SiStripRecHitConverter_h

/** \class SiStripRecHitConverter
 *
 * SiStripRecHitConverter is the EDProducer subclass which convets clusters into rechits
 *
 * \author C. Genta
 *
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverterAlgorithm.h"

namespace cms
{
  class SiStripRecHitConverter : public edm::EDProducer
  {
  public:

    explicit SiStripRecHitConverter(const edm::ParameterSet& conf);

    virtual ~SiStripRecHitConverter();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    SiStripRecHitConverterAlgorithm recHitConverterAlgorithm_;
    edm::ParameterSet conf_;
    std::string matchedRecHitsTag_, rphiRecHitsTag_, stereoRecHitsTag_;
  };
}


#endif
