#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/DeadChannelNNContext.h"

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/xyNNEB.h"

DeadChannelNNContext::DeadChannelNNContext() {
	this->load();
}

DeadChannelNNContext::~DeadChannelNNContext() {}

void DeadChannelNNContext::load() {
	ccNNEB* ccNNObjEB = new ccNNEB();
	rrNNEB* rrNNObjEB = new rrNNEB();
	llNNEB* llNNObjEB = new llNNEB();
	uuNNEB* uuNNObjEB = new uuNNEB();
	ddNNEB* ddNNObjEB = new ddNNEB();
	ruNNEB* ruNNObjEB = new ruNNEB();
	rdNNEB* rdNNObjEB = new rdNNEB();
	luNNEB* luNNObjEB = new luNNEB();
	ldNNEB* ldNNObjEB = new ldNNEB();

	ccNNEE* ccNNObjEE = new ccNNEE();
	rrNNEE* rrNNObjEE = new rrNNEE();
	llNNEE* llNNObjEE = new llNNEE();
	uuNNEE* uuNNObjEE = new uuNNEE();
	ddNNEE* ddNNObjEE = new ddNNEE();
	ruNNEE* ruNNObjEE = new ruNNEE();
	rdNNEE* rdNNObjEE = new rdNNEE();
	luNNEE* luNNObjEE = new luNNEE();
	ldNNEE* ldNNObjEE = new ldNNEE();

	implementation[NetworkID::ccEB] = [=](double x[8]) -> double { return ccNNObjEB->Value(0, x); };
	implementation[NetworkID::rrEB] = [=](double x[8]) -> double { return rrNNObjEB->Value(0, x); };
	implementation[NetworkID::llEB] = [=](double x[8]) -> double { return llNNObjEB->Value(0, x); };
	implementation[NetworkID::uuEB] = [=](double x[8]) -> double { return uuNNObjEB->Value(0, x); };
	implementation[NetworkID::ddEB] = [=](double x[8]) -> double { return ddNNObjEB->Value(0, x); };
	implementation[NetworkID::ruEB] = [=](double x[8]) -> double { return ruNNObjEB->Value(0, x); };
	implementation[NetworkID::rdEB] = [=](double x[8]) -> double { return rdNNObjEB->Value(0, x); };
	implementation[NetworkID::luEB] = [=](double x[8]) -> double { return luNNObjEB->Value(0, x); };
	implementation[NetworkID::ldEB] = [=](double x[8]) -> double { return ldNNObjEB->Value(0, x); };

	implementation[NetworkID::ccEE] = [=](double x[8]) -> double { return ccNNObjEE->Value(0, x); };
	implementation[NetworkID::rrEE] = [=](double x[8]) -> double { return rrNNObjEE->Value(0, x); };
	implementation[NetworkID::llEE] = [=](double x[8]) -> double { return llNNObjEE->Value(0, x); };
	implementation[NetworkID::uuEE] = [=](double x[8]) -> double { return uuNNObjEE->Value(0, x); };
	implementation[NetworkID::ddEE] = [=](double x[8]) -> double { return ddNNObjEE->Value(0, x); };
	implementation[NetworkID::ruEE] = [=](double x[8]) -> double { return ruNNObjEE->Value(0, x); };
	implementation[NetworkID::rdEE] = [=](double x[8]) -> double { return rdNNObjEE->Value(0, x); };
	implementation[NetworkID::luEE] = [=](double x[8]) -> double { return luNNObjEE->Value(0, x); };
	implementation[NetworkID::ldEE] = [=](double x[8]) -> double { return ldNNObjEE->Value(0, x); };

}

double DeadChannelNNContext::value(NetworkID method, int index, double in0, double in1, double in2, 
  double in3, double in4, double in5, double in6, double in7) {

  double vCC[8] = {in0, in1, in2, in3, in4, in5, in6, in7};
  return implementation[method](vCC);
}


