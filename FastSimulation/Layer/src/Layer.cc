#include "FastSimulation/Layer/interface/Layer.h"
#include "iostream"
#include "TH1F.h"

std::ostream& fastsim::operator << (std::ostream& os , const Layer & layer)
{
    os << (layer.isForward() ? "ForwardLayer" : "BarrelLayer")
       << " index=" << layer.index_
       << (layer.isForward() ? " z=" : " radius=") << layer.position_;
    return os;
}

// note: define destructor and constructor in .cc file,
//       otherwise one cannot forward declare TH1F in the header file
//       w/o compilation issues
fastsim::Layer::~Layer()
{}

fastsim::Layer::Layer(double position)
    : position_(position)
    , position2_(position_*position_)
    , index_(-1)
    , detLayer_(0)
    , nuclearInteractionThicknessFactor_(1.)
{}

