//
// author Aris Kyriakis (NCSR "Demokritos")
//
#include "DataFormats/EgammaReco/interface/PreshowerClusterShape.h"

using namespace reco;

PreshowerClusterShape::~PreshowerClusterShape() { }


PreshowerClusterShape::PreshowerClusterShape(const std::vector<float>& stripEnergies,
				   const int plane)
{
  stripEnergies_ = stripEnergies;
  plane_ = plane;
}

PreshowerClusterShape::PreshowerClusterShape(const PreshowerClusterShape &b) 
{
  stripEnergies_ = b.stripEnergies_;
  plane_ = b.plane_;
  sc_ref_=b.sc_ref_;
}
