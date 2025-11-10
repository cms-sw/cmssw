#include "DataFormats/L1TParticleFlow/interface/PFJet.h"

void l1t::PFJet::calibratePt(float newpt) { setP4(PolarLorentzVector(newpt, eta(), phi(), mass())); }

const std::unordered_map<std::string, l1t::PFJet::HWEncoding> l1t::PFJet::encodingMapping = {
    {"CT", l1t::PFJet::HWEncoding::CT}, {"GT", l1t::PFJet::HWEncoding::GT}, {"GTWide", l1t::PFJet::HWEncoding::GTWide}};

l1t::PFJet::HWEncoding l1t::PFJet::encodingFromString(const std::string& name) {
  auto it = l1t::PFJet::encodingMapping.find(name);
  if (it == l1t::PFJet::encodingMapping.end()) {
    throw cms::Exception("PFJet") << "Unknown HWEncoding: " << name;
  }
  return it->second;
}