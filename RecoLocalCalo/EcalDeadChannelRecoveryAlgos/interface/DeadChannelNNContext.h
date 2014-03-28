#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_DeadChannelNNContext_H
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_DeadChannelNNContext_H

#include <TTree.h>
#include <TMultiLayerPerceptron.h>

#include <functional>

class DeadChannelNNContext {
 //  Arrangement within the M3x3Input matrix
 //
 //                  M3x3
 //   -----------------------------------
 //   
 //   
 //   LU  UU  RU             04  01  07
 //   LL  CC  RR      or     03  00  06
 //   LD  DD  RD             05  02  08

 public:
  DeadChannelNNContext();
  ~DeadChannelNNContext();

  //  Enumeration to switch from custom names within the 3x3 matrix.
  enum CellID { CC=0, UU=1, DD=2, LL=3, LU=4, LD=5, RR=6, RU=7, RD=8 };

  // Double value_ee(CellID missing, Double_t inputs[8]);
  // Double value_eb(CellID missing, Double_t inputs[8]);

  double estimateEnergyEB(double *M3x3Input, double epsilon=0.0000001);
  double estimateEnergyEE(double *M3x3Input, double epsilon=0.0000001);

 private:
  struct MultiLayerPerceptronContext {
    Double_t tmp[9];
    TTree *tree;
    TMultiLayerPerceptron *mlp;
  };

  MultiLayerPerceptronContext ctx_eb[9];
  MultiLayerPerceptronContext ctx_ee[9];

  void load();
  void load_file(MultiLayerPerceptronContext& ctx, std::string fn);
  double estimateEnergy(MultiLayerPerceptronContext *ctx, double *M3x3Input, double epsilon);
};

#endif
