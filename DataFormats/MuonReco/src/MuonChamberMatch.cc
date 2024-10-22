#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include <cmath>
using namespace reco;

int MuonChamberMatch::station() const {
  if (detector() == MuonSubdetId::DT) {  // DT
    DTChamberId segId(id.rawId());
    return segId.station();
  }
  if (detector() == MuonSubdetId::CSC) {  // CSC
    CSCDetId segId(id.rawId());
    return segId.station();
  }
  if (detector() == MuonSubdetId::RPC) {  //RPC
    RPCDetId segId(id.rawId());
    return segId.station();
  }
  if (detector() == MuonSubdetId::GEM) {  //GEM
    GEMDetId segId(id.rawId());
    return segId.station();
  }
  if (detector() == MuonSubdetId::ME0) {  //ME0
    ME0DetId segId(id.rawId());
    return segId.station();
  }
  return -1;  // is this appropriate? fix this
}

std::pair<float, float> MuonChamberMatch::getDistancePair(float edgeX, float edgeY, float xErr, float yErr) const {
  if (edgeX > 9E5 && edgeY > 9E5 && xErr > 9E5 && yErr > 9E5)  // there is no track
    return std::make_pair(999999, 999999);

  float distance = 999999;
  float error = 999999;

  if (edgeX < 0 && edgeY < 0) {
    if (edgeX < edgeY) {
      distance = edgeY;
      error = yErr;
    } else {
      distance = edgeX;
      error = xErr;
    }
  }
  if (edgeX < 0 && edgeY > 0) {
    distance = edgeY;
    error = yErr;
  }
  if (edgeX > 0 && edgeY < 0) {
    distance = edgeX;
    error = xErr;
  }
  if (edgeX > 0 && edgeY > 0) {
    distance = sqrt(edgeX * edgeX + edgeY * edgeY);
    error = distance ? sqrt(edgeX * edgeX * xErr * xErr + edgeY * edgeY * yErr * yErr) / fabs(distance) : 0;
  }

  return std::make_pair(distance, error);
}
