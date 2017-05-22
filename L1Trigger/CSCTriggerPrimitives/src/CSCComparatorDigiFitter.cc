#include "L1Trigger/CSCTriggerPrimitives/src/CSCComparatorDigiFitter.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

#include "TF1.h"
#include "TFitResultPtr.h"
#include "TGraphErrors.h"

typedef std::vector<CSCComparatorDigi> CSCComparatorDigiContainer;
typedef std::vector<std::pair<unsigned int, CSCComparatorDigi> > CSCComparatorDigiDetIdContainer;
typedef std::vector<std::pair<CSCDetId, CSCComparatorDigiContainer> > CSCComparatorDigiContainerIds;
namespace{

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
	      { -5,   -4,   -3,   -2,   -1,   0,   1,   2,   3,   4,   5}, 
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

void CSCComparatorDigiFitter::fit(const CSCDetId& ch_id, const CSCCorrelatedLCTDigi& stub, const CSCComparatorDigiCollection& hCSCComparators)
{
  const auto& cscChamber = cscGeometry_->chamber(ch_id);

  CSCComparatorDigiContainerIds compDigisIds;

  // fetch the CSC comparator digis in this chamber
  for (int iLayer=1; iLayer<=6; ++iLayer){
    CSCDetId layerId(ch_id.endcap(), ch_id.station(), ch_id.ring(), ch_id.chamber(), iLayer);

    // get the digis per layer
    auto compRange = hCSCComparators.get(layerId);
    CSCComparatorDigiContainer compDigis;

    for (auto compDigiItr = compRange.first; compDigiItr != compRange.second; compDigiItr++) {
      const auto& compDigi = *compDigiItr;

      //if (stub.getTimeBin() < 4 or stub.getTimeBin() > 8) continue;
      int stubHalfStrip(compDigi.getHalfStrip());

      // these comparator digis never fit the pattern anyway!
      if (std::abs(stubHalfStrip-stub.getStrip())>5) continue;

      // check if this comparator digi fits the pattern
      if (comparatorInLCTPattern(stub.getStrip(), stub.getPattern(), iLayer, stubHalfStrip)) {
        compDigis.push_back(compDigi);
      }
    }
    compDigisIds.push_back(std::make_pair(layerId, compDigis));
  }

  // get the z and phi positions
  float perp = 0.0;
  std::vector<float> phis;
  std::vector<float> zs;
  std::vector<float> ephis;
  std::vector<float> ezs;
  for (auto p: compDigisIds){
    auto detId = p.first;
    float phi_tmp = 0.0;
    float perp_tmp = 0.0;
    float z_tmp = 0.0;
    if (p.second.size()==0) continue;
    for (auto hit: p.second){
      float fractional_strip = hit.getFractionalStrip();
      const auto& layer_geo = cscChamber->layer(detId.layer())->geometry();
      float wire = layer_geo->middleWireOfGroup(stub.getKeyWG() + 1);
      LocalPoint csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
      GlobalPoint csc_gp = cscGeometry_->idToDet(detId)->surface().toGlobal(csc_intersect);
      float gpphi = csc_gp.phi();

      if (phis.size()>0 and gpphi>0 and phis[0]<0 and  (gpphi-phis[0])>3.1416)
        phi_tmp += (gpphi-2*3.1415926);
      else if (phis.size()>0 and gpphi<0 and phis[0]>0 and (gpphi-phis[0])<-3.1416)
        phi_tmp += (gpphi+2*3.1415926);
      else
        phi_tmp += (csc_gp.phi());

      z_tmp = csc_gp.z();
      perp_tmp += csc_gp.perp();
    }
    //in case there are more than one comparator digis in one layer
    perp_tmp = perp_tmp/(p.second).size();
    phi_tmp = phi_tmp/(p.second).size();
    perp += perp_tmp;
    phis.push_back(phi_tmp);
    zs.push_back(z_tmp);
    ezs.push_back(0);

    // phis.push_back(csc_gp.phi());
    ephis.push_back(cscHalfStripWidth(detId)/sqrt(12));
  }


  CSCDetId key_id(ch_id.endcap(), ch_id.station(), ch_id.ring(), ch_id.chamber(), CSCConstants::KEY_CLCT_LAYER);
  float fractional_strip = 0.5 * (stub.getStrip() + 1) - 0.25;
  auto layer_geo = cscChamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();
  // LCT::getKeyWG() also starts from 0
  float wire = layer_geo->middleWireOfGroup(stub.getKeyWG() + 1);
  LocalPoint csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
  GlobalPoint csc_gp = cscGeometry_->idToDet(key_id)->surface().toGlobal(csc_intersect);
  perp = csc_gp.perp();
  float stripPhiPitch = layer_geo->stripPhiPitch();
  int stripBits = 0;
  if (stripBits > 0)
    stripPhiPitch = stripPhiPitch/stripBits;

  // use average perp
  //perp = perp/phis.size();
  // do a fit to the comparator digis
  float alpha = -99., beta = 0.;
  calculateAlphaBeta(zs, phis, ezs, ephis, alpha, beta);
  if (phis.size() <= 2 or fabs(alpha)>=99){
    alpha = csc_gp.phi();
    beta = 0.0;
  }
  int fit_z_layers[6];
  int fit_phi_layers[6];
  for (int i=0; i<6; i++){
    fit_z_layers[i] = cscChamber->layer(i+1)->centerOfStrip(20).z();
    fit_phi_layers[i] = normalizedPhi(alpha + beta * fit_z_layers[i]);
    if (stripBits > 0)
      fit_phi_layers[i] = (std::floor(fit_phi_layers[i]/stripPhiPitch) + 0.5)*stripPhiPitch;
  }

}

void CSCComparatorDigiFitter::calculateAlphaBeta(const std::vector<float>& v, 
						 const std::vector<float>& w, 
						 const std::vector<float>& ev, 
						 const std::vector<float>& ew, 
						 float& alpha, float& beta)
{
  if (v.size()>=3) {
  
    float zmin;
    float zmax;
    if (v.front() < v.back()){
      zmin = v.front();
      zmax = v.back();
    }
    else{
      zmin = v.back();
      zmax = v.front();
    }

    TF1 *fit1 = new TF1("fit1","pol1",zmin,zmax); 
    //where 0 = x-axis_lowest and 48 = x_axis_highest 
    TGraphErrors* gr = new TGraphErrors(v.size(),&(v[0]),&(w[0]),&(ev[0]),&(ew[0]));
    gr->SetMinimum(w[2]-5*0.002);
    gr->SetMaximum(w[2]+5*0.002);
 
    gr->Fit(fit1,"RQ"); 
  
    alpha = fit1->GetParameter(0); //value of 0th parameter
    beta  = fit1->GetParameter(1); //value of 1st parameter

    delete fit1;
    delete gr;
  }
  else {alpha = -99; beta= 0.0;}
}

float
CSCComparatorDigiFitter::cscHalfStripWidth(const CSCDetId& id)
{
  // ME1a ME1b ME12 ME13 ME21 ME22 ME31 ME32 ME41 ME42
  const std::vector<int> strips = {48,64,80,64, 80,80, 80,80, 80,80};
  const std::vector<int> degrees = {10,10,10,10, 20,10, 20,10, 20,10};
  int index = id.iChamberType()-1;
  const float width = float(degrees[index]) * 3.14159265358979323846/180. / float(2 * strips[index]);
  return width;
}

bool 
CSCComparatorDigiFitter::comparatorInLCTPattern(int keyStrip, int pattern, int layer, int halfStrip)
{
  bool debug = false;
  // first, get the (sub)pattern
  bool returnValue = false;
  const std::vector<int>& subpat = patIndexToPatternDelta[pattern].at(layer-1);
  if (debug) for (const auto& p: subpat) std::cout << "\t" << p << std::endl;
  if (debug) std::cout << "\tkeyStrip pattern layer halfstrip " << keyStrip << " " <<pattern << " " <<layer << " " <<halfStrip <<std::endl <<std::endl;
  // due to comparator digi time extension in the CLCT processor we need to  
  // search a bigger region around the key HS. +/-1, 0 should be sufficient
  int halfStripDelta = halfStrip - keyStrip;
  returnValue = ( std::find(subpat.begin(), subpat.end(), halfStripDelta+1) != subpat.end() or
                  std::find(subpat.begin(), subpat.end(), halfStripDelta)   != subpat.end() or
                  std::find(subpat.begin(), subpat.end(), halfStripDelta-1) != subpat.end() );
  return returnValue;
}
