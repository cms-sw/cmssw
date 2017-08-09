#include "L1Trigger/CSCTriggerPrimitives/src/CSCComparatorDigiFitter.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

namespace
{
  // CSC LCT patterns
  // the number quotes the distance to the center

  // pid=0: no pattern found
  std::vector<std::vector<int> > pat0delta(CSCConstants::NUM_LAYERS);
 
  // pid=1: layer-OR trigger
  std::vector<std::vector<int> > pat1delta {
    {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5},
      {-2, -1, 0, 1, 2},
	{0},
	  {-2, -1, 0, 1, 2},
	    {-4, -3, -2, -1, 0, 1, 2, 3, 4},
	      {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
  };
  
  // pid=2: right-bending (large)
  std::vector<std::vector<int> > pat2delta {
    {3, 4, 5},
      {1, 2},
	{0},
	  {-2, -1, 0},
	    {-4, -3, -2},
	      {-5, -4, -3}
  };

  // pid=3: left-bending (large)
  std::vector<std::vector<int> > pat3delta {
    {-5, -4, -3},
      {-2, -1},
	{0},
	  {0, 1, 2},
	    {2, 3, 4},
	      {3, 4, 5}
  }; 

  // pid=4: right-bending (medium)
  std::vector<std::vector<int> > pat4delta {
    {2, 3, 4},
      {1, 2},
	{0},
	  {-2, -1},
	    {-4, -3, -2},
	      {-4, -3, -2}
            
  };

  // pid=5: left-bending (medium)
  std::vector<std::vector<int> > pat5delta {
    {-4, -3, -2},
      {-2, -1},
	{0},
	  {1, 2},
	    {2, 3, 4},
	      {2, 3, 4}
            
  };

  // pid=6: right-bending (medium)
  std::vector<std::vector<int> > pat6delta {
    {1, 2, 3},
      {0, 1},
	{0},
	  {-1, 0},
	    {-2, -1},
	      {-3, -2, -1}
  };

  // pid=7: left-bending (medium)
  std::vector<std::vector<int> > pat7delta {
    {-3, -2, -1},
      {-1, 0},
	{0},
	  {0, 1},
	    {1, 2},
	      {1, 2, 3}
  };

  // pid=8: right-bending (small)
  std::vector<std::vector<int> > pat8delta {
    {0, 1, 2},
      {0, 1},
	{0},
	  {-1, 0},
	    {-2, -1, 0},
	      {-2, -1, 0}
  };

  // pid=9: left-bending (small)
  std::vector<std::vector<int> > pat9delta {
    {-2, -1, 0},
      {-1, 0},
	{0},
	  {0, 1},
	    {0, 1, 2},
	      {0, 1, 2}
  };

  // pid=A: straight-through
  std::vector<std::vector<int> > patAdelta {
    {-1, 0, 1},
      {0},
	{0},
	  {0},
	    {-1, 0, 1},
	      {-1, 0, 1}
  };

  std::vector< std::vector<std::vector<int> > > patIndexToPatternDelta {
    pat0delta, pat1delta, pat2delta, pat3delta, pat4delta, pat5delta, pat6delta, pat7delta, pat8delta, pat9delta, patAdelta 
  };
}

void CSCComparatorDigiFitter::matchingComparatorDigisLCT(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi& stub, const CSCComparatorDigiCollection& hCSCComparators)
{
  // fetch the CSC comparator digis in this chamber
  for (int iLayer=1; iLayer<=CSCConstants::NUM_LAYERS; ++iLayer) {
    const CSCDetId layerId(ch_id.endcap(), ch_id.station(), ch_id.ring(), ch_id.chamber(), iLayer);
    
    // get the digis per layer
    const auto& compRange = hCSCComparators.get(layerId);
    CSCComparatorDigiContainer compDigis;
    
    for (auto compDigiItr = compRange.first; compDigiItr != compRange.second; compDigiItr++) {
      const auto& compDigi = *compDigiItr;
      
      //if (stub.getTimeBin() < 4 or stub.getTimeBin() > 8) continue;
      const int stubHalfStrip(compDigi.getHalfStrip());
      
      // these comparator digis never fit the pattern anyway!
      if (std::abs(stubHalfStrip-stub.getStrip())>5) continue;
      
      // check if this comparator digi fits the pattern
      if (comparatorInLCTPattern(stub.getStrip(), stub.getPattern(), iLayer, stubHalfStrip)) {
        compDigis.push_back(compDigi);
      }
    }
    compDigisIds_.emplace_back(layerId, compDigis);
  }
}

void CSCComparatorDigiFitter::getComparatorDigiCoordinates(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi& stub)
{
  const auto& cscChamber = cscGeometry_->chamber(ch_id);
  
  // get the z and phi positions of the comparator digis
  float radius_ = 0.0;

  // loop on all matching digis
  for (const auto& p: compDigisIds_) {
    const auto& detId = p.first;

    float phi_tmp = 0.0;
    float radius_tmp = 0.0;
    float z_tmp = 0.0;

    // ignore layers with no digis
    if (p.second.size()==0) continue;

    // loop on all matching digis in this layer
    for (const auto& hit: p.second) {
      const float fractional_strip = hit.getFractionalStrip();
      const auto& layer_geo = cscChamber->layer(detId.layer())->geometry();
      const float wire = layer_geo->middleWireOfGroup(stub.getKeyWG() + 1);

      // get the phi of each comparator digi
      const LocalPoint& csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
      const GlobalPoint& csc_gp = cscGeometry_->idToDet(detId)->surface().toGlobal(csc_intersect);
      const float gpphi = csc_gp.phi();

      // normalize phi values according to first one
      if (phis_.size()>0 and gpphi>0 and phis_[0]<0 and  (gpphi-phis_[0])>M_PI)
        phi_tmp += (gpphi-2*M_PI);
      else if (phis_.size()>0 and gpphi<0 and phis_[0]>0 and (gpphi-phis_[0])<-M_PI)
        phi_tmp += (gpphi+2*M_PI);
      else
        phi_tmp += (csc_gp.phi());

      z_tmp = csc_gp.z();
      radius_tmp += csc_gp.perp();
    }

    //in case there are more than one comparator digis in one layer
    radius_tmp = radius_tmp/(p.second).size();
    radius_ += radius_tmp;

    zs_.push_back(z_tmp);
    ezs_.push_back(0);

    phi_tmp = phi_tmp/(p.second).size();
    phis_.push_back(phi_tmp);
    // assume that the charge is flat distributed across the half-strip
    // this is only approximately valid, but good enough for now
    ephis_.push_back(cscHalfStripWidth(detId)/sqrt(12));
  }
}

void CSCComparatorDigiFitter::fit(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi& stub,
				  const CSCComparatorDigiCollection& hCSCComparators,
				  std::vector<float>& fit_phi_layers,
				  std::vector<float>& fit_z_layers, float& keyRadius)
{
  // clear fit results
  fit_phi_layers.clear();
  fit_z_layers.clear();
  keyRadius = 0;

  // first, match the comparator digis to the LCT
  matchingComparatorDigisLCT(ch_id, stub, hCSCComparators);

  // second, get the coordinates
  getComparatorDigiCoordinates(ch_id, stub);

  // get radius of the stub from key layer
  const CSCDetId key_id(ch_id.endcap(), ch_id.station(), ch_id.ring(), ch_id.chamber(), CSCConstants::KEY_CLCT_LAYER);
  const float fractional_strip = stub.getFractionalStrip();

  const auto& cscChamber = cscGeometry_->chamber(ch_id);
  const auto& layer_geo = cscChamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();

  // LCT::getKeyWG() also starts from 0
  const float wire = layer_geo->middleWireOfGroup(stub.getKeyWG() + 1);
  const LocalPoint& csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
  const GlobalPoint& csc_gp = cscGeometry_->idToDet(key_id)->surface().toGlobal(csc_intersect);

  // get radius from key layer
  if (useKeyRadius_)
    radius_ = radius_/phis_.size();
  else
    radius_ = csc_gp.perp();
  
  float alpha = -99., beta = 0.;
  // do a fit to the comparator digis
  calculateSlopeIntercept(alpha, beta);
  if (phis_ .size() <= 2 or std::abs(alpha)>=99){
    alpha = csc_gp.phi();
    beta = 0.0;
  }

  // determine the pitch of the chamber
  // option to discretize the pitch
  float stripPhiPitch = layer_geo->stripPhiPitch();
  if (nStripBits_)
    stripPhiPitch = stripPhiPitch/nStripBits_;

  // get the fit results
  // option to discretize the fitted phi value
  keyRadius = radius_;

  for (int i=0; i<CSCConstants::NUM_LAYERS; i++) {
    const float fit_z = cscChamber->layer(i+1)->centerOfStrip(20).z();
    const float fit_phi = normalizedPhi(alpha + beta * fit_z);
    fit_z_layers.push_back(fit_z);
    fit_phi_layers.push_back(fit_phi);
    if (nStripBits_)
      fit_phi_layers.push_back((std::floor(fit_phi/stripPhiPitch) + 0.5)*stripPhiPitch);
  }
}

void CSCComparatorDigiFitter::calculateSlopeIntercept(float& alpha, float& beta)
{
  // if there are at least 3 hits in the chamber, do a linear fit to the
  // comparator digi positions with the chi2 method
  if (phis_.size()>=3) {
  
    float Sxx = 0, Sxy = 0, Sx = 0, Sy = 0, S = 0;
    for (unsigned i = 0; i<phis_.size(); ++i){
      float sigma2_inv = 1./ephis_[i]*ephis_[i];
      Sxx += zs_[i]*zs_[i] * sigma2_inv;
      Sxy += zs_[i]*phis_[i] * sigma2_inv;
      Sx += zs_[i]*zs_[i] * sigma2_inv;
      Sy += phis_[i] * sigma2_inv;
      S += sigma2_inv;
    }
    float delta = S * Sxx - Sx * Sx;
    alpha = (Sxx * Sy - Sx * Sxy) / delta;
    beta = (S * Sxy - Sx * Sy) / delta;
  }
  else {
    alpha = -99; beta= 0.0;
  }
}

float
CSCComparatorDigiFitter::cscHalfStripWidth(const CSCDetId& id) const
{
  // what is the chamber type?
  int index = id.iChamberType()-1;

  // calculate the half strip width of this chamber
  return degrees_[index] * M_PI/180. / (2. * strips_[index]);
}

bool 
CSCComparatorDigiFitter::comparatorInLCTPattern(int keyStrip, int pattern, int layer, int halfStrip) const
{
  // get the (sub)pattern
  const std::vector<int>& subpat = patIndexToPatternDelta[pattern].at(layer-1);

  // due to comparator digi time extension in the CLCT processor we need to  
  // search a bigger region around the key HS. +/-1, 0 should be sufficient
  const int halfStripDelta = halfStrip - keyStrip;
  return ( std::find(subpat.begin(), subpat.end(), halfStripDelta+1) != subpat.end() or
	   std::find(subpat.begin(), subpat.end(), halfStripDelta)   != subpat.end() or
	   std::find(subpat.begin(), subpat.end(), halfStripDelta-1) != subpat.end() );
}
