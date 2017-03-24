#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelROC.h"

CTPPSPixelROC::CTPPSPixelROC(uint32_t du, int idDU, int idLk)
  : theDetUnit(du), theIdDU(idDU), theIdLk(idLk) {
  theIndices = new CTPPSPixelIndices();
}

