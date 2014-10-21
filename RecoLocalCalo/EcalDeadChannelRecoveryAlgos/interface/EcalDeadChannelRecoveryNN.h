#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalDeadChannelRecoveryNN_H
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalDeadChannelRecoveryNN_H

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include <TTree.h>
#include <TMultiLayerPerceptron.h>

#include <string>
#include <functional>

template <typename DetIdT> class EcalDeadChannelRecoveryNN {
 public:
  EcalDeadChannelRecoveryNN();
  ~EcalDeadChannelRecoveryNN();

  //  Arrangement within the M3x3Input matrix
  //
  //                  M3x3
  //   -----------------------------------
  //
  //
  //   LU  UU  RU             04  01  07
  //   LL  CC  RR      or     03  00  06
  //   LD  DD  RD             05  02  08

  //  Enumeration to switch from custom names within the 3x3 matrix.
  enum CellID {
    CC = 0,
    UU = 1,
    DD = 2,
    LL = 3,
    LU = 4,
    LD = 5,
    RR = 6,
    RU = 7,
    RD = 8
  };

  // Mapping custom names in the 3x3 to (x,y) or (ieta, iphi)
  // ex: x=+1, y=-1 (ix() == ixP && iy() == iyN -> RD)
  // ex: x=-1, y=+1 (ieta() == ietaN && iphi() == iphiP -> LU)

  const int CellX[9] = { 0, 0, 0 /* CC, UU, DD */, -1, -1, -1 /* LL, LU, LD */,
                         1, 1, 1 /* RR, RU, RD */ };

  const int CellY[9] = { 0, -1, 1 /* CC, UU, DD */, 0, -1, 1 /* LL, LU, LD */,
                         0, -1, 1 /* RR, RU, RD */ };

  void setCaloTopology(const CaloTopology *topo);
  double recover(const DetIdT id, const EcalRecHitCollection &hit_collection,
                 double Sum8Cut, bool *AcceptFlag);

 private:
  struct MultiLayerPerceptronContext {
    Double_t tmp[9];
    TTree *tree;
    TMultiLayerPerceptron *mlp;
  };

  const CaloSubdetectorTopology* topology_;
  MultiLayerPerceptronContext ctx_[9];

  void load();
  void load_file(MultiLayerPerceptronContext &ctx, std::string fn);

 public:
  double estimateEnergy(double *M3x3Input, double epsilon = 0.0000001);

  double makeNxNMatrice_RelMC(DetIdT itID,
                              const EcalRecHitCollection &hit_collection,
                              double *MNxN_RelMC, bool *AccFlag);
  double makeNxNMatrice_RelDC(DetIdT itID,
                              const EcalRecHitCollection &hit_collection,
                              double *MNxN_RelDC, bool *AccFlag);

   double reorderMxNMatrix(EBDetId it, const std::vector<DetId>& window, 
    const EcalRecHitCollection& hit_collection, double *MNxN, bool* AcceptFlag);
};

#endif
