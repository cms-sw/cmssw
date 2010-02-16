#ifndef DTCLUSTERER_H
#define DTCLUSTERER_H

/** \class DTClusterer
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 17/04/2008 14:56:40 CEST $
 *
 * Modification:
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/EDProducer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

/* Collaborating Class Declarations */
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
class DTSuperLayer;

/* C++ Headers */
#include <vector>
#include <utility>

/* ====================================================================== */

/* Class DTClusterer Interface */

class DTClusterer : public edm::EDProducer {

  public:

    /* Constructor */ 
    DTClusterer(const edm::ParameterSet&) ;

    /* Destructor */ 
    virtual ~DTClusterer() ;

    /* Operations */ 
    virtual void produce(edm::Event& event, const edm::EventSetup& setup);

  private:
    // build clusters from hits
    std::vector<DTSLRecCluster> buildClusters(const DTSuperLayer* sl,
                                              std::vector<DTRecHit1DPair>& pairs);

    std::vector<std::pair<float, DTRecHit1DPair> > initHits(const DTSuperLayer* sl,
                                                            std::vector<DTRecHit1DPair>& pairs);

    unsigned int differentLayers(std::vector<DTRecHit1DPair>& hits);

  private:
    // to sort hits by x
    struct sortClusterByX {
      bool operator()(const std::pair<float, DTRecHit1DPair>& lhs, 
                      const std::pair<float, DTRecHit1DPair>& rhs) {
        return lhs.first < rhs.first;
      }
    };

  private:
    // Switch on verbosity
    bool debug;

    unsigned int theMinHits; // min number of hits to build a cluster
    unsigned int theMinLayers; // min number of layers to build a cluster
    edm::InputTag theRecHits1DLabel;
  protected:

};
#endif // DTCLUSTERER_H

