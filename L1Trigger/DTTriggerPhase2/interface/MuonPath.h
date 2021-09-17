#ifndef L1Trigger_DTTriggerPhase2_MuonPath_h
#define L1Trigger_DTTriggerPhase2_MuonPath_h
#include <iostream>
#include <memory>

#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"

class MuonPath {
public:
  MuonPath();
  MuonPath(DTPrimitivePtrs &ptrPrimitive, int prup = 0, int prdw = 0);
  MuonPath(DTPrimitives &ptrPrimitive, int prup = 0, int prdw = 0);
  MuonPath(std::shared_ptr<MuonPath> &ptr);
  virtual ~MuonPath() {}

  // setter methods
  void setPrimitive(DTPrimitivePtr &ptr, int layer);
  void setNPrimitives(short nprim) { nprimitives_ = nprim; }
  void setNPrimitivesUp(short nprim) { nprimitivesUp_ = nprim; }
  void setNPrimitivesDown(short nprim) { nprimitivesDown_ = nprim; }
  void setCellHorizontalLayout(int layout[4]);
  void setCellHorizontalLayout(const int *layout);
  void setBaseChannelId(int bch) { baseChannelId_ = bch; }
  void setQuality(cmsdt::MP_QUALITY qty) { quality_ = qty; }
  void setBxTimeValue(int time);
  void setLateralComb(cmsdt::LATERAL_CASES latComb[4]);
  void setLateralComb(const cmsdt::LATERAL_CASES *latComb);
  void setLateralCombFromPrimitives(void);

  void setHorizPos(float pos) { horizPos_ = pos; }
  void setTanPhi(float tanPhi) { tanPhi_ = tanPhi; }
  void setChiSquare(float chi) { chiSquare_ = chi; }
  void setPhi(float phi) { phi_ = phi; }
  void setPhiB(float phib) { phiB_ = phib; }
  void setPhiCMSSW(float phi_cmssw) { phicmssw_ = phi_cmssw; }
  void setPhiBCMSSW(float phib_cmssw) { phiBcmssw_ = phib_cmssw; }
  void setXCoorCell(float x, int cell) { xCoorCell_[cell] = x; }
  void setDriftDistance(float dx, int cell) { xDriftDistance_[cell] = dx; }
  void setXWirePos(float x, int cell) { xWirePos_[cell] = x; }
  void setZWirePos(float z, int cell) { zWirePos_[cell] = z; }
  void setTWireTDC(float t, int cell) { tWireTDC_[cell] = t; }
  void setRawId(uint32_t id) { rawId_ = id; }

  // getter methods
  DTPrimitivePtr primitive(int layer) const { return prim_[layer]; }
  short nprimitives() const { return nprimitives_; }
  short nprimitivesDown() const { return nprimitivesDown_; }
  short nprimitivesUp() const { return nprimitivesUp_; }
  const int *cellLayout() const { return cellLayout_; }
  int baseChannelId() const { return baseChannelId_; }
  cmsdt::MP_QUALITY quality() const { return quality_; }
  int bxTimeValue() const { return bxTimeValue_; }
  int bxNumId() const { return bxNumId_; }
  float tanPhi() const { return tanPhi_; }
  const cmsdt::LATERAL_CASES *lateralComb() const { return (lateralComb_); }
  float horizPos() const { return horizPos_; }
  float chiSquare() const { return chiSquare_; }
  float phi() const { return phi_; }
  float phiB() const { return phiB_; }
  float phi_cmssw() const { return phicmssw_; }
  float phiB_cmssw() const { return phiBcmssw_; }
  float xCoorCell(int cell) const { return xCoorCell_[cell]; }
  float xDriftDistance(int cell) const { return xDriftDistance_[cell]; }
  float xWirePos(int cell) const { return xWirePos_[cell]; }
  float zWirePos(int cell) const { return zWirePos_[cell]; }
  float tWireTDC(int cell) const { return tWireTDC_[cell]; }
  uint32_t rawId() const { return rawId_; }

  // Other methods
  bool isEqualTo(MuonPath *ptr);
  bool isAnalyzable();
  bool completeMP();

private:
  //------------------------------------------------------------------
  //---  MuonPath's data
  //------------------------------------------------------------------
  /*
      Primitives that make up the path. The 0th position holds the channel ID of 
      the lower layer. The order is critical. 
  */
  DTPrimitivePtrs prim_;  //ENSURE that there are no more than 4-8 prims
  short nprimitives_;
  short nprimitivesUp_;
  short nprimitivesDown_;

  /* Horizontal position of each cell (one per layer), in half-cell units,
     with respect of the lower layer (layer 0). 
  */
  int cellLayout_[cmsdt::NUM_LAYERS];
  int baseChannelId_;

  //------------------------------------------------------------------
  //--- Fit results:
  //------------------------------------------------------------------
  /* path quality */
  cmsdt::MP_QUALITY quality_;

  /* Lateral combination    */
  cmsdt::LATERAL_CASES lateralComb_[cmsdt::NUM_LAYERS];

  /* BX time value with respect to BX0 of the orbit  */
  int bxTimeValue_;

  /* BX number in the orbit   */
  int bxNumId_;

  /* Cell parameters   */
  float xCoorCell_[cmsdt::NUM_LAYERS_2SL];       // Horizontal position of the hit in each cell
  float xDriftDistance_[cmsdt::NUM_LAYERS_2SL];  // Drift distance on the cell (absolute value)
  float xWirePos_[cmsdt::NUM_LAYERS_2SL];
  float zWirePos_[cmsdt::NUM_LAYERS_2SL];
  float tWireTDC_[cmsdt::NUM_LAYERS_2SL];

  float tanPhi_;
  float horizPos_;
  float chiSquare_;
  float phi_;
  float phiB_;
  float phicmssw_;
  float phiBcmssw_;

  uint32_t rawId_;
};

typedef std::vector<MuonPath> MuonPaths;
typedef std::shared_ptr<MuonPath> MuonPathPtr;
typedef std::vector<MuonPathPtr> MuonPathPtrs;

#endif
