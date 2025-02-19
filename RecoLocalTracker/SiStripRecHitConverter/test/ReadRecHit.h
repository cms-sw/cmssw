#ifndef ReadRecHit_h
#define ReadRecHit_h

/** \class ReadRecHit
 *
 * ReadRecHit is a analyzer which reads rechits
 *
 * \author C. Genta
 *
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/SiStripRecHitConverter/test/ReadRecHitAlgorithm.h"

namespace cms
{
  class ReadRecHit : public edm::EDAnalyzer
  {
  public:

    explicit ReadRecHit(const edm::ParameterSet& conf);

    virtual ~ReadRecHit();

    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    ReadRecHitAlgorithm readRecHitAlgorithm_;
    edm::ParameterSet conf_;

  };
}


#endif
