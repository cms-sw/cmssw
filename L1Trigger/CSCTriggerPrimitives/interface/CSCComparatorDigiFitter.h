#ifndef L1Trigger_CSCTriggerPrimitives_CSCComparatorDigiFitter_h
#define L1Trigger_CSCTriggerPrimitives_CSCComparatorDigiFitter_h

/*
 * class CSCComparatorDigiFitter
 *
 * Fits a straight line to the comparator digis beloning to a stub
 *
 * This is a helper class, to be used in the first prototype
 * implementation of the displaced muon trigger. In due time,
 * the fitting procedure will be integrated in the
 * CSCCathodeLCTProcessor.
 *
 * authors: Sven Dildick (TAMU), Tao Huang (TAMU)
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include <vector>

typedef std::vector<CSCComparatorDigi> CSCComparatorDigiContainer;
typedef std::vector<std::pair<CSCDetId, CSCComparatorDigiContainer> > CSCComparatorDigiContainerIds;

class CSCComparatorDigiFitter {
public:
  CSCComparatorDigiFitter() {}
  ~CSCComparatorDigiFitter() {}

  /* CSC trigger geometry */
  void setGeometry(const CSCGeometry* csc_g) { cscGeometry_ = csc_g; }

  /* option to discretize the fitted stub phi */
  void setStripBits(int bits) { nStripBits_ = bits; }

  /* use key layer radius */
  void useKeyRadius(bool useKeyRadius) { useKeyRadius_ = useKeyRadius; }

  /* fit a straight line to the digis */
  void fit(const CSCDetId& ch_id,
           const CSCCorrelatedLCTDigi&,
           const CSCComparatorDigiCollection&,
           std::vector<float>& fit_phi_layers,
           std::vector<float>& fit_z_layers,
           float& keyRadius);

private:
  /* collect the comparator digis that match the LCT pattern
     from the comparator digi collection */
  void matchingComparatorDigisLCT(const CSCDetId& ch_id,
                                  const CSCCorrelatedLCTDigi&,
                                  const CSCComparatorDigiCollection&);

  /* collect the coordinates of comparators */
  void getComparatorDigiCoordinates(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi& stub);

  /* is this comparator in the LCT pattern? */
  bool comparatorInLCTPattern(int keyStrip, int pattern, int layer, int halfStrip) const;

  // calculate slope and intercept of fit
  void calculateSlopeIntercept(float& alpha, float& beta);

  /* width of the CSC half strips in this detId  */
  float cscHalfStripWidth(const CSCDetId& id) const;

  /* pointer to the CSC geometry */
  const CSCGeometry* cscGeometry_;

  /* number of bits allocated to the strip number after fit */
  int nStripBits_;

  CSCComparatorDigiContainerIds compDigisIds_;

  /* coordinates of the comparators */
  std::vector<float> phis_;
  std::vector<float> zs_;
  std::vector<float> ephis_;
  std::vector<float> ezs_;
  float radius_;
  bool useKeyRadius_;

  // number of strips and chamber width for each chamber type
  // ME1a ME1b ME12 ME13 ME21 ME22 ME31 ME32 ME41 ME42
  const std::vector<int> strips_ = {48, 64, 80, 64, 80, 80, 80, 80, 80, 80};
  const std::vector<float> degrees_ = {10., 10., 10., 10., 20., 10., 20., 10., 20., 10.};
};

#endif
