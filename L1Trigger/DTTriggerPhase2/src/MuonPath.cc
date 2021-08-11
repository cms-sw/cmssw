#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"

#include <cmath>
#include <iostream>
#include <memory>

using namespace cmsdt;

MuonPath::MuonPath() {
  quality_ = NOPATH;
  baseChannelId_ = -1;

  for (int i = 0; i < NUM_LAYERS_2SL; i++) {
    prim_.push_back(std::make_shared<DTPrimitive>());
  }

  nprimitives_ = NUM_LAYERS;
  bxTimeValue_ = -1;
  bxNumId_ = -1;
  tanPhi_ = 0;
  horizPos_ = 0;
  chiSquare_ = 0;
  for (int i = 0; i < NUM_LAYERS; i++) {
    lateralComb_[i] = LEFT;
    setXCoorCell(0, i);
    setDriftDistance(0, i);
  }
}

MuonPath::MuonPath(DTPrimitivePtrs &ptrPrimitive, int nprimUp, int nprimDown) {
  if (nprimUp > 0 || nprimDown > 0)
    nprimitives_ = NUM_LAYERS_2SL;  //Instead of nprimUp + nprimDown;
  else {
    nprimitives_ = NUM_LAYERS;
  }
  nprimitivesUp_ = nprimUp;
  nprimitivesDown_ = nprimDown;
  rawId_ = 0;
  quality_ = NOPATH;
  baseChannelId_ = -1;
  bxTimeValue_ = -1;
  bxNumId_ = -1;
  tanPhi_ = 0;
  horizPos_ = 0;
  chiSquare_ = 0;
  phi_ = 0;
  phiB_ = 0;
  phicmssw_ = 0;
  phiBcmssw_ = 0;

  for (short i = 0; i < nprimitives_; i++) {
    lateralComb_[i] = LEFT;
    prim_.push_back(std::make_shared<DTPrimitive>(ptrPrimitive[i]));

    setXCoorCell(0, i);
    setDriftDistance(0, i);
    setXWirePos(0, i);
    setZWirePos(0, i);
    setTWireTDC(0, i);
  }
}

MuonPath::MuonPath(DTPrimitives &ptrPrimitive, int nprimUp, int nprimDown) {
  if (nprimUp > 0 || nprimDown > 0)
    nprimitives_ = NUM_LAYERS_2SL;  //Instead of nprimUp + nprimDown;
  else {
    nprimitives_ = NUM_LAYERS;
  }
  nprimitivesUp_ = nprimUp;
  nprimitivesDown_ = nprimDown;
  rawId_ = 0;
  quality_ = NOPATH;
  baseChannelId_ = -1;
  bxTimeValue_ = -1;
  bxNumId_ = -1;
  tanPhi_ = 0;
  horizPos_ = 0;
  chiSquare_ = 0;
  phi_ = 0;
  phiB_ = 0;
  phicmssw_ = 0;
  phiBcmssw_ = 0;

  for (short i = 0; i < nprimitives_; i++) {
    lateralComb_[i] = LEFT;
    prim_.push_back(std::make_shared<DTPrimitive>(ptrPrimitive[i]));

    setXCoorCell(0, i);
    setDriftDistance(0, i);
    setXWirePos(0, i);
    setZWirePos(0, i);
    setTWireTDC(0, i);
  }
}

MuonPath::MuonPath(MuonPathPtr &ptr) {
  setRawId(ptr->rawId());
  setPhi(ptr->phi());
  setPhiB(ptr->phiB());
  setPhiCMSSW(ptr->phi_cmssw());
  setPhiBCMSSW(ptr->phiB_cmssw());
  setQuality(ptr->quality());
  setBaseChannelId(ptr->baseChannelId());
  setCellHorizontalLayout(ptr->cellLayout());
  setNPrimitives(ptr->nprimitives());

  setLateralComb(ptr->lateralComb());
  setBxTimeValue(ptr->bxTimeValue());
  setTanPhi(ptr->tanPhi());
  setHorizPos(ptr->horizPos());
  setChiSquare(ptr->chiSquare());

  for (int i = 0; i < ptr->nprimitives(); i++) {
    prim_.push_back(ptr->primitive(i));

    setXCoorCell(ptr->xCoorCell(i), i);
    setDriftDistance(ptr->xDriftDistance(i), i);
    setXWirePos(ptr->xWirePos(i), i);
    setZWirePos(ptr->zWirePos(i), i);
    setTWireTDC(ptr->tWireTDC(i), i);
  }
}

//------------------------------------------------------------------
//--- Public
//------------------------------------------------------------------
void MuonPath::setPrimitive(DTPrimitivePtr &ptr, int layer) {
  if (ptr == nullptr)
    std::cout << "NULL 'Primitive'." << std::endl;
  prim_[layer] = std::move(ptr);
}

void MuonPath::setCellHorizontalLayout(int layout[4]) {
  for (int i = 0; i < NUM_LAYERS; i++)
    cellLayout_[i] = layout[i];
}

void MuonPath::setCellHorizontalLayout(const int *layout) {
  for (int i = 0; i < NUM_LAYERS; i++)
    cellLayout_[i] = layout[i];
}

bool MuonPath::isEqualTo(MuonPath *ptr) {
  for (int i = 0; i < ptr->nprimitives(); i++) {
    if (this->primitive(i)->isValidTime() && ptr->primitive(i)->isValidTime()) {
      if (ptr->primitive(i)->superLayerId() != this->primitive(i)->superLayerId() ||

          ptr->primitive(i)->channelId() != this->primitive(i)->channelId() ||

          ptr->primitive(i)->tdcTimeStamp() != this->primitive(i)->tdcTimeStamp() ||

          ptr->primitive(i)->orbit() != this->primitive(i)->orbit() ||
          (ptr->lateralComb())[i] != (this->lateralComb())[i])
        return false;
    } else {
      if (!this->primitive(i)->isValidTime() && !ptr->primitive(i)->isValidTime())
        continue;

      else
        return false;
    }
  }

  return true;
}

bool MuonPath::isAnalyzable() {
  short countValidHits = 0;
  for (int i = 0; i < this->nprimitives(); i++) {
    //    if (!this->primitive(i))
    //      continue;
    if (this->primitive(i)->isValidTime())
      countValidHits++;
  }

  if (countValidHits >= 3)
    return true;
  return false;
}

bool MuonPath::completeMP() {
  return (prim_[0]->isValidTime() && prim_[1]->isValidTime() && prim_[2]->isValidTime() && prim_[3]->isValidTime());
}

void MuonPath::setBxTimeValue(int time) {
  bxTimeValue_ = time;

  float auxBxId = float(time) / LHC_CLK_FREQ;
  bxNumId_ = int(auxBxId);
  if ((auxBxId - int(auxBxId)) >= 0.5)
    bxNumId_ = int(bxNumId_ + 1);
}

void MuonPath::setLateralCombFromPrimitives() {
  for (int i = 0; i < nprimitives_; i++) {
    if (!this->primitive(i)->isValidTime())
      continue;
    lateralComb_[i] = this->primitive(i)->laterality();
  }
}

void MuonPath::setLateralComb(LATERAL_CASES latComb[4]) {
  for (int i = 0; i < NUM_LAYERS; i++)
    lateralComb_[i] = latComb[i];
}

void MuonPath::setLateralComb(const LATERAL_CASES *latComb) {
  for (int i = 0; i < NUM_LAYERS; i++)
    lateralComb_[i] = latComb[i];
}
