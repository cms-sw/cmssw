#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"

using namespace reco;

PFConversion::PFConversion(const reco::ConversionRef& cp, const std::vector<reco::PFRecTrackRef>& tr)
    : originalConversion_(cp), pfTracks_(tr) {}

PFConversion::PFConversion(const reco::ConversionRef cp) : originalConversion_(cp) {}

PFConversion::~PFConversion() {}
