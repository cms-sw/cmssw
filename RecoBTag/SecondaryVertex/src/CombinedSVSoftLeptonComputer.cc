#include "RecoBTag/SecondaryVertex/interface/CombinedSVSoftLeptonComputer.h"

using namespace reco;
using namespace std;


CombinedSVSoftLeptonComputer::CombinedSVSoftLeptonComputer(const edm::ParameterSet &params) :
	CombinedSVComputer(params),
	SoftLeptonFlip(params.getParameter<bool>("SoftLeptonFlip"))
{
}
