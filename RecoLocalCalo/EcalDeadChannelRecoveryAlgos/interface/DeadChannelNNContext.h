#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_DeadChannelNNContext_H
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_DeadChannelNNContext_H

#include <TTree.h>
#include <TMultiLayerPerceptron.h>

#include <functional>

class DeadChannelNNContext {
 public:
  DeadChannelNNContext();
  ~DeadChannelNNContext();

  enum NetworkID {
    ccEB = 0,
    ccEE,
    rrEB,
    rrEE,
    llEB,
    llEE,

    uuEB,
    uuEE,
    ddEB,
    ddEE,
    ruEB,
    ruEE,

    rdEB,
    rdEE,
    luEB,
    luEE,
    ldEB,
    ldEE,

    lastID
  };

  double value(NetworkID method, int index, double in0, double in1, double in2,
               double in3, double in4, double in5, double in6, double in7);

 private:
  void load();
  void load_file(NetworkID id, std::string fn);

  struct MultiLayerPerceptronContext {
    Double_t tmp[9];
    TTree *tree;
    TMultiLayerPerceptron *mlp;
  };

  MultiLayerPerceptronContext ctx[lastID];
};

#endif
