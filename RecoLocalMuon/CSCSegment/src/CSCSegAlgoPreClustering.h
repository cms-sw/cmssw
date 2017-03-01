#ifndef CSCSegment_CSCSegAlgoPreClustering_h
#define CSCSegment_CSCSegAlgoPreClustering_h
/**
 * \file CSCSegAlgoPreClustering.h
 *
 *  \authors: S. Stoynev  - NU
 *            I. Bloch    - FNAL
 *            E. James    - FNAL
 *
 * See header file for description.
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <vector>

class CSCChamber;

class CSCSegAlgoPreClustering {

 public:

  typedef std::vector<const CSCRecHit2D*> ChamberHitContainer;

  /// constructor
  explicit CSCSegAlgoPreClustering(const edm::ParameterSet& ps);

  /// destructor
  ~CSCSegAlgoPreClustering();

  /// clusterize
  std::vector< std::vector<const CSCRecHit2D*> > clusterHits( const CSCChamber* aChamber, const ChamberHitContainer& rechits);
 private:

  bool    debug;
  double  dXclusBoxMax;
  double  dYclusBoxMax;

  float mean_x, mean_y, err_x, err_y;
  const CSCChamber* theChamber;

};
#endif
