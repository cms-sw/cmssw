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

class CSCComparatorDigiFitter
{
 public:

  CSCComparatorDigiFitter() {}
  ~CSCComparatorDigiFitter() {}

  /* CSC trigger geometry */
  void setGeometry(const CSCGeometry* csc_g) {cscGeometry_= csc_g;}

  /* option to discretize the fitted stub phi */
  void setStripBits(int bits) {nStripBits_ = bits;}

  /* use key layer radius */
  void useKeyRadius(bool useKeyRadius) {useKeyRadius_ = useKeyRadius;}
  
  /* fit a straight line to the digis */
  void fit(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi&, const CSCComparatorDigiCollection&);
  void getFitResults(std::vector<float>& fit_phis, std::vector<float>& fit_zs, float keyRadius);
  
 private:
  
  /* collect the comparator digis that match the pattern */
  void matchingComparatorDigisLCT(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi&, const CSCComparatorDigiCollection&);
  
  /* collect the coordinates of comparators */
  void getComparatorDigiCoordinates(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi& stub);
  
  /* is this comparator in the LCT pattern? */
  bool comparatorInLCTPattern(int keyStrip, int pattern, int layer, int halfStrip) const;

  // calculate slope and intercept of fit
  void calculateSlopeIntercept(float& alpha, float& beta);
  
  /* width of the CSC half strips in this detId  */
  float cscHalfStripWidth(const CSCDetId& id) const;

  const CSCGeometry* cscGeometry_;
  int nStripBits_;

  CSCComparatorDigiContainerIds compDigisIds_;
  std::vector<float> phis_;
  std::vector<float> zs_;
  std::vector<float> ephis_;
  std::vector<float> ezs_;
  float radius_;
  bool useKeyRadius_;
};

#endif
