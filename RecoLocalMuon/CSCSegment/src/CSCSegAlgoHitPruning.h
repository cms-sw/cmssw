#ifndef CSCSegment_CSCSegAlgoHitPruning_h
#define CSCSegAlgoHitPruning_h
/**
 * \file CSCSegAlgoHitPruning.h
 *
 *  \authors: S. Stoynev  - NU
 *            I. Bloch    - FNAL
 *            E. James    - FNAL
 *            D. Fortin   - UC Riverside
 *
 * Go through segments and clean up selection of hits according
 * to changes in quality of segment (chi2/d.o.f. probability)
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>

#include <vector>

class CSCChamber;

class CSCSegAlgoHitPruning {

 public:

  typedef std::vector<const CSCRecHit2D*> ChamberHitContainer;

  /// constructor
  explicit CSCSegAlgoHitPruning(const edm::ParameterSet& ps);

  /// destructor
  ~CSCSegAlgoHitPruning();

  /// clusterize
  std::vector<CSCSegment> pruneBadHits(const CSCChamber* aChamber, const std::vector<CSCSegment>& segments);

 private:
  void fitSlopes(void);
  void fillChiSquared(void);
  void fillLocalDirection(void);
  CLHEP::HepMatrix derivativeMatrix(void) const;
  AlgebraicSymMatrix weightMatrix(void) const;
  AlgebraicSymMatrix calculateError(void) const;
  void flipErrors(AlgebraicSymMatrix&) const;

  const CSCChamber* theChamber;

  ChamberHitContainer protoSegment;
  float       protoSlope_u;
  float       protoSlope_v;
  LocalPoint  protoIntercept;		
  double      protoChi2;
  LocalVector protoDirection;

  bool    BrutePruning;
};
#endif
