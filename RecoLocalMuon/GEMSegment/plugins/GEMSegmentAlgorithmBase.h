#ifndef GEMRecHit_GEMSegmentAlgorithmBase_h
#define GEMRecHit_GEMSegmentAlgorithmBase_h

/** \class GEMSegmentAlgo derived from CSC
 * An abstract base class for algorithmic classes used to
 * build segments in one ensemble of GEM detector 
 *
 * Implementation notes: <BR>
 * For example, GEMSegmentAlgorithm inherits from this class,
 *
 * \author Piet Verwilligen
 *
 */

#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegment.h"
#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <map>
#include <vector>

class GEMSegmentAlgorithmBase {
public:
  typedef std::pair<const GEMSuperChamber*, std::map<uint32_t, const GEMEtaPartition*> > GEMEnsemble;

  /// Constructor
  explicit GEMSegmentAlgorithmBase(const edm::ParameterSet&){};
  /// Destructor
  virtual ~GEMSegmentAlgorithmBase(){};

  /** Run the algorithm = build the segments in this chamber
    */
  virtual std::vector<GEMSegment> run(const GEMEnsemble& ensemble, const std::vector<const GEMRecHit*>& rechits) = 0;

private:
};

#endif
