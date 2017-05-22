#include "L1Trigger/CSCTriggerPrimitives/src/CSCComparatorDigiFitter.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

#include "TF1.h"
#include "TGraphErrors.h"

namespace
{
  // CSC LCT patterns
  // the number quotes the distance to the center
  // 999 is invalid

  std::vector<std::vector<int> > pat0delta {  
    { 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999}, 
      { 999, 999, 999, 999, 999}, 
	{999},            // pid=999: no pattern found 
	  {999, 999, 999, 999, 999}, 
	    {999, 999, 999, 999, 999, 999, 999, 999, 999}, 
	      {999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999} 
  }; 
  
  std::vector<std::vector<int> > pat1delta {  
    { -5,   -4,   -3,   -2,   -1,   0,   1,   2,   3,   4,   5}, 
      {-2,   -1,   0,   1,   2}, 
	{0},             // pid=1: layer-OR trigger 
	  {-2,   -1,   0,   1,   2}, 
	    { -4,   -3,   -2,   -1,   0,   1,   2,   3,   4}, 
	      { -5,   -4,   -3,   -2,   -1,   0,   1,   2,   3,   4,   5}
  }; 
  
  std::vector<std::vector<int> > pat2delta {  
    { 999, 999, 999, 999, 999, 999, 999, 999,   3,   4,   5}, 
      {999, 999, 999,   1,   2}, 
	{0},             // pid=2: right-bending (large) 
	  {-2,   -1,   0, 999, 999}, 
	    {-4,   -3,   -2, 999, 999, 999, 999, 999, 999}, 
	      {-5,   -4,   -3, 999, 999, 999, 999, 999, 999, 999, 999} 
  }; 

  std::vector<std::vector<int> > pat3delta { 
    {   -5,   -4,   -3, 999, 999, 999, 999, 999, 999, 999, 999}, 
      {-2,   -1, 999, 999, 999}, 
	{0},             // pid=3: left-bending (large) 
	  {999, 999,   0,   1,   2}, 
	    {999, 999, 999, 999, 999, 999,   2,   3,   4}, 
	      {999, 999, 999, 999, 999, 999, 999, 999,   3,   4,   5} 
  };  

  std::vector<std::vector<int> > pat4delta { 
    { 999, 999, 999, 999, 999, 999, 999,   2,   3,   4, 999}, 
      {999, 999, 999,   1,   2}, 
	{0},             // pid=4: right-bending (medium) 
	  {-2,   -1, 999, 999, 999}, 
	    {-4,   -3,   -2, 999, 999, 999, 999, 999, 999}, 
	      {999,   -4,   -3,   -2, 999, 999, 999, 999, 999, 999, 999} 
            
  }; 

  std::vector<std::vector<int> > pat5delta { 
    { 999,   -4,   -3,   -2, 999, 999, 999, 999, 999, 999, 999}, 
      {-2,   -1, 999, 999, 999}, 
	{0},             // pid=5: left-bending (medium) 
	  {999, 999, 999,   1,   2}, 
	    {999, 999, 999, 999, 999, 999,   2,   3,   4}, 
	      {999, 999, 999, 999, 999, 999, 999,   2,   3,   4, 999} 
            
  }; 

  std::vector<std::vector<int> > pat6delta { 
    { 999, 999, 999, 999, 999, 999,   1,   2,   3, 999, 999}, 
      {999, 999,   0,   1, 999}, 
	{0},             // pid=6: right-bending (medium) 
	  {999,   -1,   0, 999, 999}, 
	    {999, 999,   -2,   -1, 999, 999, 999, 999, 999}, 
	      {999, 999,   -3,   -2,   -1, 999, 999, 999, 999, 999, 999} 
  }; 

  std::vector<std::vector<int> > pat7delta { 
    { 999, 999,   -3,   -2,   -1, 999, 999, 999, 999, 999, 999}, 
      {999,   -1,   0, 999, 999}, 
	{0},             // pid=7: left-bending (medium) 
	  {999, 999,   0,   1, 999}, 
	    {999, 999, 999, 999, 999,   1,   2, 999, 999}, 
	      {999, 999, 999, 999, 999, 999,  1,  2,   3, 999, 999} 
  }; 

  std::vector<std::vector<int> > pat8delta { 
    { 999, 999, 999, 999, 999,   0,   1,   2, 999, 999, 999}, 
      {999, 999,   0,   1, 999}, 
	{0},             // pid=8: right-bending (small) 
	  {999,   -1,   0, 999, 999}, 
	    {999, 999,   -2,   -1,   0, 999, 999, 999, 999}, 
	      {999, 999, 999,   -2,   -1,   0, 999, 999, 999, 999, 999} 
  }; 

  std::vector<std::vector<int> > pat9delta { 
    { 999, 999, 999,   -2,   -1,   0, 999, 999, 999, 999, 999}, 
      {999,   -1,   0, 999, 999}, 
	{0},             // pid=9: left-bending (small) 
	  {999, 999,   0,   1, 999}, 
	    {999, 999, 999, 999,   0,   1,   2, 999, 999}, 
	      {999, 999, 999, 999, 999,   0,   1,   2, 999, 999, 999} 
  }; 

  std::vector<std::vector<int> > patAdelta { 
    { 999, 999, 999, 999,   -1,   0,   1, 999, 999, 999, 999}, 
      {999, 999,   0, 999, 999}, 
	{0},             // pid=A: straight-through 
	  {999, 999,   0, 999, 999}, 
	    {999, 999, 999,   -1,   0,   1, 999, 999, 999}, 
	      {999, 999, 999, 999,   -1,   0,   1, 999, 999, 999, 999} 
  }; 

  std::vector< std::vector<std::vector<int> > > patIndexToPatternDelta { 
    pat0delta, pat1delta, pat2delta, pat3delta, pat4delta, pat5delta, pat6delta, pat7delta, pat8delta, pat9delta, patAdelta 
  }; 
}

void CSCComparatorDigiFitter::matchingComparatorDigisLCT(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi& stub, const CSCComparatorDigiCollection& hCSCComparators)
{
  // fetch the CSC comparator digis in this chamber
  for (int iLayer=1; iLayer<=6; ++iLayer) {
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
    ephis_.push_back(cscHalfStripWidth(detId)/sqrt(12));
  }
}

void CSCComparatorDigiFitter::fit(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi& stub, const CSCComparatorDigiCollection& hCSCComparators)
{
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

  float stripPhiPitch = layer_geo->stripPhiPitch();
  if (nStripBits_)
    stripPhiPitch = stripPhiPitch/nStripBits_;

  int fit_z_layers[6];
  int fit_phi_layers[6];
  for (int i=0; i<6; i++){
    fit_z_layers[i] = cscChamber->layer(i+1)->centerOfStrip(20).z();
    fit_phi_layers[i] = normalizedPhi(alpha + beta * fit_z_layers[i]);
    if (nStripBits_)
      fit_phi_layers[i] = (std::floor(fit_phi_layers[i]/stripPhiPitch) + 0.5)*stripPhiPitch;
  }
}

void CSCComparatorDigiFitter::calculateSlopeIntercept(float& alpha, float& beta)
{
  if (phis_.size()>=3) {
  
    const bool isFront(zs_.front() < zs_.back());
    const float zmin = isFront ? zs_.front() : zs_.back();
    const float zmax = isFront ? zs_.back() : zs_.front();

    // do the fit
    std::unique_ptr<TF1> fit1(new TF1("fit1","pol1",zmin,zmax)); 
    std::unique_ptr<TGraphErrors> gr(new TGraphErrors(phis_.size(),&(zs_[0]),&(phis_[0]),&(ezs_[0]),&(ephis_[0])));
    gr->SetMinimum(ephis_[2]-5*0.002);
    gr->SetMaximum(ephis_[2]+5*0.002);
    gr->Fit(fit1.get(),"RQ"); 
    alpha = fit1.get()->GetParameter(0);
    beta  = fit1.get()->GetParameter(1);
  }
  else {
    alpha = -99; beta= 0.0;
  }
}

float
CSCComparatorDigiFitter::cscHalfStripWidth(const CSCDetId& id) const
{
  // number of strips and chamber width for each chamber type
  // ME1a ME1b ME12 ME13 ME21 ME22 ME31 ME32 ME41 ME42
  const std::vector<int> strips = {48,64,80,64, 80,80,80,80,80,80};
  const std::vector<float> degrees = {10.,10.,10.,10.,20.,10.,20.,10.,20.,10.};
  int index = id.iChamberType()-1;

  // half strip width
  return degrees[index] * M_PI/180. / (2. * strips[index]);
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
