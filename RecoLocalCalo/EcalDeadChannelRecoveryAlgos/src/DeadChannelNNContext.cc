#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/DeadChannelNNContext.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <iostream>
#include <TMath.h>

DeadChannelNNContext::DeadChannelNNContext() {
  for (int id = 0; id < 9; ++id) {
    ctx_eb[id].mlp = NULL;
    ctx_ee[id].mlp = NULL;
  }

  this->load();
}

DeadChannelNNContext::~DeadChannelNNContext() {
  for (int id = 0; id < 9; ++id) {
    if (ctx_eb[id].mlp) {
      // @TODO segfaults for an uknown reason
      // delete ctx[id].mlp;
      // delete ctx[id].tree;
    }

    if (ctx_ee[id].mlp) {
      // @TODO segfaults for an uknown reason
      // delete ctx[id].mlp;
      // delete ctx[id].tree;
    }
  }
}

void DeadChannelNNContext::load_file(MultiLayerPerceptronContext& ctx, std::string fn) {
  std::string path = edm::FileInPath(fn).fullPath();

  TTree *t = new TTree("t", "dummy MLP tree");
  t->SetDirectory(0);

  t->Branch("z1", &(ctx.tmp[0]), "z1/D");
  t->Branch("z2", &(ctx.tmp[1]), "z2/D");
  t->Branch("z3", &(ctx.tmp[2]), "z3/D");
  t->Branch("z4", &(ctx.tmp[3]), "z4/D");
  t->Branch("z5", &(ctx.tmp[4]), "z5/D");
  t->Branch("z6", &(ctx.tmp[5]), "z6/D");
  t->Branch("z7", &(ctx.tmp[6]), "z7/D");
  t->Branch("z8", &(ctx.tmp[7]), "z8/D");
  t->Branch("zf", &(ctx.tmp[8]), "zf/D");

  ctx.tree = t;
  ctx.mlp =
      new TMultiLayerPerceptron("@z1,@z2,@z3,@z4,@z5,@z6,@z7,@z8:10:5:zf", t);
  ctx.mlp->LoadWeights(path.c_str());
}

void DeadChannelNNContext::load() {
  std::string p = "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/NNWeights/";

  this->load_file(ctx_eb[CellID::CC], p + "EB_ccNNWeights.txt");
  this->load_file(ctx_eb[CellID::RR], p + "EB_rrNNWeights.txt");
  this->load_file(ctx_eb[CellID::LL], p + "EB_llNNWeights.txt");
  this->load_file(ctx_eb[CellID::UU], p + "EB_uuNNWeights.txt");
  this->load_file(ctx_eb[CellID::DD], p + "EB_ddNNWeights.txt");
  this->load_file(ctx_eb[CellID::RU], p + "EB_ruNNWeights.txt");
  this->load_file(ctx_eb[CellID::RD], p + "EB_rdNNWeights.txt");
  this->load_file(ctx_eb[CellID::LU], p + "EB_luNNWeights.txt");
  this->load_file(ctx_eb[CellID::LD], p + "EB_ldNNWeights.txt");

  this->load_file(ctx_ee[CellID::CC], p + "EE_ccNNWeights.txt");
  this->load_file(ctx_ee[CellID::RR], p + "EE_rrNNWeights.txt");
  this->load_file(ctx_ee[CellID::LL], p + "EE_llNNWeights.txt");
  this->load_file(ctx_ee[CellID::UU], p + "EE_uuNNWeights.txt");
  this->load_file(ctx_ee[CellID::DD], p + "EE_ddNNWeights.txt");
  this->load_file(ctx_ee[CellID::RU], p + "EE_ruNNWeights.txt");
  this->load_file(ctx_ee[CellID::RD], p + "EE_rdNNWeights.txt");
  this->load_file(ctx_ee[CellID::LU], p + "EE_luNNWeights.txt");
  this->load_file(ctx_ee[CellID::LD], p + "EE_ldNNWeights.txt");
}

double DeadChannelNNContext::estimateEnergy(MultiLayerPerceptronContext *cts, double *M3x3Input, double epsilon) {
	int missing[9];
	int missing_index = 0;

    for (int i = 0; i < 9; i++) {
        if (TMath::Abs(M3x3Input[i]) < epsilon) {
			missing[missing_index++] = i;
		} else {
		    //  Generally the "dead" cells are allowed to have negative energies (since they will be estimated by the ANN anyway).
		    //  But all the remaining "live" ones must have positive values otherwise the logarithm fails.

			if (M3x3Input[i] < 0.0) { return -2000000.0; }
		}
    }

    //  Currently EXACTLY ONE AND ONLY ONE dead cell is corrected. Return -1000000.0 if zero DC's detected and -101.0 if more than one DC's exist.
    int idxDC = -1 ;
    if (missing_index == 0) { return -1000000.0; }    //  Zero DC's were detected
    if (missing_index  > 1) { return -1000001.0; }    //  More than one DC's detected.
    if (missing_index == 1) { idxDC = missing[0]; } 

	// Arrange inputs into an array of 8, excluding the dead cell;
	int input_order[9] = { CC, RR, LL, UU, DD, RU, RD, LU, LD };
	int input_index = 0;
	Double_t input[8];

	for (int id : input_order) {
		if (id == idxDC)
			continue;

		input[input_index++] = TMath::Log(M3x3Input[id]);
	}

    //  Select the case to apply the appropriate NN and return the result.
	M3x3Input[idxDC] = TMath::Exp(cts[idxDC].mlp->Evaluate(0, input));
	return M3x3Input[idxDC];
}

double DeadChannelNNContext::estimateEnergyEB(double *M3x3Input, double epsilon) {
  return estimateEnergy(this->ctx_eb, M3x3Input, epsilon);
}

double DeadChannelNNContext::estimateEnergyEE(double *M3x3Input, double epsilon) {
  return estimateEnergy(this->ctx_ee, M3x3Input, epsilon);
}
