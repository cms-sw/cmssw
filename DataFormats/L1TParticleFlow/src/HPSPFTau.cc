#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"
#include "FWCore/Utilities/interface/Exception.h"

// default constructor
l1t::HPSPFTau::HPSPFTau()
    : tauType_(kUndefined),
      sumChargedIso_(0.),
      sumNeutralIso_(0.),
      sumCombinedIso_(0.),
      sumChargedIsoPileup_(0.),
      rhoCorr_(0.),
      passTightIso_(false),
      passMediumIso_(false),
      passLooseIso_(false),
      passVLooseIso_(false),
      passTightRelIso_(false),
      passMediumRelIso_(false),
      passLooseRelIso_(false),
      passVLooseRelIso_(false) {}

// destructor
l1t::HPSPFTau::~HPSPFTau() {}

// print to stream
ostream& operator<<(ostream& os, const l1t::HPSPFTau& l1PFTau) {
  os << "pT = " << l1PFTau.pt() << ", eta = " << l1PFTau.eta() << ", phi = " << l1PFTau.phi()
     << " (type = " << l1PFTau.tauType() << ")" << std::endl;
  os << "lead. ChargedPFCand:" << std::endl;
  if (l1PFTau.leadChargedPFCand().isNonnull()) {
    printPFCand(os, *l1PFTau.leadChargedPFCand(), l1PFTau.primaryVertex());
  } else {
    os << " N/A" << std::endl;
  }
  os << "seed:";
  if (l1PFTau.isChargedPFCandSeeded()) {
    os << " chargedPFCand";
  } else if (l1PFTau.isPFJetSeeded()) {
    os << " PFJet";
  } else {
    cms::Exception ex("InconsistentTau");
    ex.addContext("Calling HPSPFTau::operator <<");
    ex.addAdditionalInfo("This tau is not seed by either a chargedPFCand or a PFJet!");
    throw ex;
  }
  os << std::endl;
  os << "signalPFCands:" << std::endl;
  for (const auto& l1PFCand : l1PFTau.signalAllL1PFCandidates()) {
    printPFCand(os, *l1PFCand, l1PFTau.primaryVertex());
  }
  os << "stripPFCands:" << std::endl;
  for (const auto& l1PFCand : l1PFTau.stripAllL1PFCandidates()) {
    printPFCand(os, *l1PFCand, l1PFTau.primaryVertex());
  }
  os << "strip pT = " << l1PFTau.strip_p4().pt() << std::endl;
  os << "isolationPFCands:" << std::endl;
  for (const auto& l1PFCand : l1PFTau.isoAllL1PFCandidates()) {
    printPFCand(os, *l1PFCand, l1PFTau.primaryVertex());
  }
  os << "isolation pT-sum: charged = " << l1PFTau.sumChargedIso() << ", neutral = " << l1PFTau.sumNeutralIso()
     << " (charged from pileup = " << l1PFTau.sumChargedIsoPileup() << ")" << std::endl;
  return os;
}

void l1t::printPFCand(ostream& os, const l1t::PFCandidate& l1PFCand, const l1t::TkPrimaryVertexRef& primaryVertex) {
  float primaryVertex_z = (primaryVertex.isNonnull()) ? primaryVertex->zvertex() : 0.;
  l1t::printPFCand(os, l1PFCand, primaryVertex_z);
}

void l1t::printPFCand(ostream& os, const l1t::PFCandidate& l1PFCand, float primaryVertex_z) {
  std::string type_string;
  if (l1PFCand.id() == l1t::PFCandidate::ChargedHadron)
    type_string = "PFChargedHadron";
  else if (l1PFCand.id() == l1t::PFCandidate::Electron)
    type_string = "PFElectron";
  else if (l1PFCand.id() == l1t::PFCandidate::NeutralHadron)
    type_string = "PFNeutralHadron";
  else if (l1PFCand.id() == l1t::PFCandidate::Photon)
    type_string = "PFPhoton";
  else if (l1PFCand.id() == l1t::PFCandidate::Muon)
    type_string = "PFMuon";
  else
    type_string = "N/A";
  os << " " << type_string << " with pT = " << l1PFCand.pt() << ", eta = " << l1PFCand.eta()
     << ", phi = " << l1PFCand.phi() << ","
     << " mass = " << l1PFCand.mass() << ", charge = " << l1PFCand.charge();
  if (l1PFCand.charge() != 0 && primaryVertex_z != 0.) {
    os << " (dz = " << std::fabs(l1PFCand.pfTrack()->vertex().z() - primaryVertex_z) << ")";
  }
  os << std::endl;
}
