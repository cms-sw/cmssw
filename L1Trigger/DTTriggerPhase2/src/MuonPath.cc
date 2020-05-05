#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"

#include <cstring>  // Para función "memcpy"
#include "math.h"
#include <iostream>

MuonPath::MuonPath() {
  //    std::cout<<"Creando un 'MuonPath'"<<std::endl;

  quality_ = NOPATH;
  baseChannelId_ = -1;

  for (int i = 0; i <= 3; i++) {
    prim_[i] = new DTPrimitive();
  }

  nprimitives_ = 4;
  bxTimeValue_ = -1;
  bxNumId_ = -1;
  tanPhi_ = 0;
  horizPos_ = 0;
  chiSquare_ = 0;
  for (int i = 0; i <= 3; i++) {
    lateralComb_[i] = LEFT;
    setXCoorCell(0, i);
    setDriftDistance(0, i);
  }
}

MuonPath::MuonPath(DTPrimitive *ptrPrimitive[4]) {
  //    std::cout<<"Creando un 'MuonPath'"<<std::endl;

  quality_ = NOPATH;
  baseChannelId_ = -1;

  for (int i = 0; i <= 3; i++) {
    if ((prim_[i] = ptrPrimitive[i]) == NULL)
      std::cout << "Unable to create 'MuonPath'. Null 'Primitive'." << std::endl;
  }

  nprimitives_ = 4;
  //Dummy values
  nprimitivesUp_ = 0;
  nprimitivesDown_ = 0;
  bxTimeValue_ = -1;
  bxNumId_ = -1;
  tanPhi_ = 0;
  horizPos_ = 0;
  chiSquare_ = 0;
  phi_ = 0;
  phiB_ = 0;
  rawId_ = 0;
  for (int i = 0; i <= 3; i++) {
    lateralComb_[i] = LEFT;
    setXCoorCell(0, i);
    setDriftDistance(0, i);
    setXWirePos(0, i);
    setZWirePos(0, i);
    setTWireTDC(0, i);
  }
}

MuonPath::MuonPath(DTPrimitive *ptrPrimitive[8], int nprimUp, int nprimDown) {
  //    std::cout<<"Creando un 'MuonPath'"<<std::endl;
  nprimitives_ = 8;  //Instead of nprimUp + nprimDown;
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
  for (int l = 0; l <= 3; l++) {
    lateralComb_[l] = LEFT;
  }

  for (short i = 0; i < nprimitives_; i++) {
    if ((prim_[i] = ptrPrimitive[i]) == NULL) {
      std::cout << "Unable to create 'MuonPath'. Null 'Primitive'." << std::endl;
    }
    setXCoorCell(0, i);
    setDriftDistance(0, i);
    setXWirePos(0, i);
    setZWirePos(0, i);
    setTWireTDC(0, i);
  }
}

MuonPath::MuonPath(MuonPath *ptr) {
  //  std::cout<<"Clonando un 'MuonPath'"<<std::endl;
  setRawId(ptr->rawId());
  setPhi(ptr->phi());
  setPhiB(ptr->phiB());
  setQuality(ptr->quality());
  setBaseChannelId(ptr->baseChannelId());
  setCellHorizontalLayout(ptr->cellLayout());
  setNPrimitives(ptr->nprimitives());

  for (int i = 0; i < ptr->nprimitives(); i++)
    setPrimitive(new DTPrimitive(ptr->primitive(i)), i);

  setLateralComb(ptr->lateralComb());
  setBxTimeValue(ptr->bxTimeValue());
  setTanPhi(ptr->tanPhi());
  setHorizPos(ptr->horizPos());
  setChiSquare(ptr->chiSquare());

  for (int i = 0; i < ptr->nprimitives(); i++) {
    setXCoorCell(ptr->xCoorCell(i), i);
    setDriftDistance(ptr->xDriftDistance(i), i);
    setXWirePos(ptr->xWirePos(i), i);
    setZWirePos(ptr->zWirePos(i), i);
    setTWireTDC(ptr->tWireTDC(i), i);
  }
}

MuonPath::~MuonPath() {
  //std::cout<<"Destruyendo un 'MuonPath'"<<std::endl;

  for (int i = 0; i < nprimitives_; i++)
    if (prim_[i] != NULL)
      delete prim_[i];
}

//------------------------------------------------------------------
//--- Public
//------------------------------------------------------------------
/**
 * Añade una 'DTPrimitive'
 */
void MuonPath::setPrimitive(DTPrimitive *ptr, int layer) {
  if (ptr == NULL)
    std::cout << "NULL 'Primitive'." << std::endl;
  prim_[layer] = ptr;
}

void MuonPath::setCellHorizontalLayout(int layout[4]) {
  //  std::cout << "setCellHorizontalLayout" << std::endl;
  for (int i = 0; i <= 3; i++)
    cellLayout_[i] = layout[i];
}

void MuonPath::setCellHorizontalLayout(const int *layout) {
  //  std::cout << "setCellHorizontalLayout2" << std::endl;
  for (int i = 0; i <= 3; i++)
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

bool MuonPath::isAnalyzable(void) {
  short countValidHits = 0;
  for (int i = 0; i < this->nprimitives(); i++) {
    if (this->primitive(i)->isValidTime())
      countValidHits++;
  }

  if (countValidHits >= 3)
    return true;
  return false;
}

bool MuonPath::completeMP(void) {
  return (prim_[0]->isValidTime() && prim_[1]->isValidTime() && prim_[2]->isValidTime() && prim_[3]->isValidTime());
}

void MuonPath::setBxTimeValue(int time) {
  bxTimeValue_ = time;

  float auxBxId = float(time) / LHC_CLK_FREQ;
  bxNumId_ = int(auxBxId);
  if ((auxBxId - int(auxBxId)) >= 0.5)
    bxNumId_ = int(bxNumId_ + 1);
}

void MuonPath::setLateralCombFromPrimitives(void) {
  for (int i = 0; i < nprimitives_; i++) {
    if (!this->primitive(i)->isValidTime())
      continue;
    lateralComb_[i] = this->primitive(i)->laterality();
  }
}

void MuonPath::setLateralComb(LATERAL_CASES latComb[4]) {
  for (int i = 0; i <= 3; i++)
    lateralComb_[i] = latComb[i];
}

void MuonPath::setLateralComb(const LATERAL_CASES *latComb) {
  for (int i = 0; i <= 3; i++)
    lateralComb_[i] = latComb[i];
}
