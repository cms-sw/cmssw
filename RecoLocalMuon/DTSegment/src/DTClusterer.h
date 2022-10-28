#ifndef DTCLUSTERER_H
#define DTCLUSTERER_H

/** \class DTClusterer
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/global/EDProducer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

/* Collaborating Class Declarations */
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/DTRecHit/interface/DTSLRecCluster.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
class DTSuperLayer;

/* C++ Headers */
#include <vector>
#include <utility>

/* ====================================================================== */

/* Class DTClusterer Interface */

class DTClusterer : public edm::global::EDProducer<> {
public:
  /* Constructor */
  DTClusterer(const edm::ParameterSet&);

  /* Destructor */
  ~DTClusterer() override;

  /* Operations */
  void produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const override;

private:
  // build clusters from hits
  std::vector<DTSLRecCluster> buildClusters(const DTSuperLayer* sl, std::vector<DTRecHit1DPair>& pairs) const;

  std::vector<std::pair<float, DTRecHit1DPair> > initHits(const DTSuperLayer* sl,
                                                          std::vector<DTRecHit1DPair>& pairs) const;

  unsigned int differentLayers(std::vector<DTRecHit1DPair>& hits) const;

private:
  // to sort hits by x
  struct sortClusterByX {
    bool operator()(const std::pair<float, DTRecHit1DPair>& lhs, const std::pair<float, DTRecHit1DPair>& rhs) const {
      return lhs.first < rhs.first;
    }
  };

private:
  // Switch on verbosity
  bool debug;

  unsigned int theMinHits;    // min number of hits to build a cluster
  unsigned int theMinLayers;  // min number of layers to build a cluster
  edm::EDGetTokenT<DTRecHitCollection> recHits1DToken_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;

protected:
};
#endif  // DTCLUSTERER_H
