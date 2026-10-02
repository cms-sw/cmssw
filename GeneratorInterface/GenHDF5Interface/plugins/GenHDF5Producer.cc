// -*- C++ -*-
//
// Package:    GeneratorInterface/GenHDF5Interface
// Class:      GenHDF5Producer
//
/**\class GenHDF5Producer

 Reads a GenHDF5 file (https://gitlab.cern.ch/tvami/genhdf5) and turns event N
 into the generator products of row N, for SIM. Driven by EmptySource.
 All columns are read in the constructor, so produce() is lock free.
*/

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "HepMC/SimpleVector.h"
#include "HepMC/Units.h"

#include "TDatabasePDG.h"
#include "TParticlePDG.h"

#include "hdf5.h"

namespace {

  class H5File {
  public:
    explicit H5File(std::string const& name) : id_(H5Fopen(name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)) {
      if (id_ < 0)
        throw cms::Exception("GenHDF5Producer") << "cannot open " << name;
    }
    ~H5File() { H5Fclose(id_); }
    H5File(H5File const&) = delete;
    H5File& operator=(H5File const&) = delete;

    bool has(std::string const& path) const { return H5Lexists(id_, path.c_str(), H5P_DEFAULT) > 0; }

    // whole column, converted to T by HDF5
    template <typename T>
    std::vector<T> read(std::string const& path, hid_t memType) const {
      hid_t d = H5Dopen2(id_, path.c_str(), H5P_DEFAULT);
      if (d < 0)
        throw cms::Exception("GenHDF5Producer") << "no dataset " << path;
      hid_t s = H5Dget_space(d);
      hsize_t dims[1] = {0};
      H5Sget_simple_extent_dims(s, dims, nullptr);
      std::vector<T> out(dims[0]);
      herr_t err = dims[0] ? H5Dread(d, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) : 0;
      H5Sclose(s);
      H5Dclose(d);
      if (err < 0)
        throw cms::Exception("GenHDF5Producer") << "cannot read " << path;
      return out;
    }

  private:
    hid_t id_;
  };

  std::vector<size_t> offsets(std::vector<int32_t> const& n) {
    std::vector<size_t> off(n.size() + 1, 0);
    for (size_t i = 0; i < n.size(); ++i)
      off[i + 1] = off[i] + n[i];
    return off;
  }

}  // namespace

class GenHDF5Producer : public edm::global::EDProducer<> {
public:
  explicit GenHDF5Producer(edm::ParameterSet const&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  double massOf(int pdgId) const {
    auto it = massTable_.find(pdgId);
    return it == massTable_.end() ? 0. : it->second;
  }
  int chargeOf(int pdgId) const {
    auto it = chargeTable_.find(pdgId);
    return it == chargeTable_.end() ? 0 : it->second;
  }
  void cachePdg(int pdgId);

  std::vector<int32_t> simN_, simPdg_, simVtxIdx_, vtxN_;
  std::vector<int8_t> simStatus_;   // 1, or 2 for a stored decay
  std::vector<int32_t> simEndVtx_;  // -1 if none
  std::vector<double> simPt_, simEta_, simPhi_;
  std::vector<double> vtxX_, vtxY_, vtxZ_, vtxT_;
  std::vector<double> weight_, qScale_, alphaQCD_, alphaQED_;

  // optional tiers
  std::vector<int32_t> trN_, trPdg_, trSimIdx_;
  std::vector<int16_t> trStatus_, trMother_;
  std::vector<uint16_t> trFlags_;
  std::vector<double> trPt_, trEta_, trPhi_, trMass_;
  std::array<std::vector<int32_t>, 2> jetN_;
  std::array<std::vector<double>, 2> jetPt_, jetEta_, jetPhi_, jetMass_;
  std::vector<double> metPt_, metPhi_;
  std::vector<double> pdfX1_, pdfX2_, pdfXpdf1_, pdfXpdf2_, pdfScale_;
  std::vector<int32_t> pdfId1_, pdfId2_;

  std::vector<size_t> simOff_, vtxOff_, trOff_;
  std::array<std::vector<size_t>, 2> jetOff_;
  size_t nEvents_ = 0;
  // TDatabasePDG is not thread safe, so cache it up front
  std::unordered_map<int, double> massTable_;
  std::unordered_map<int, int> chargeTable_;
};

void GenHDF5Producer::cachePdg(int pdgId) {
  if (massTable_.count(pdgId))
    return;
  auto* pd = TDatabasePDG::Instance()->GetParticle(pdgId);
  massTable_[pdgId] = pd ? pd->Mass() : 0.;
  chargeTable_[pdgId] = pd ? static_cast<int>(std::lround(pd->Charge() / 3.0)) : 0;
}

void GenHDF5Producer::fillDescriptions(edm::ConfigurationDescriptions& d) {
  edm::ParameterSetDescription p;
  p.add<std::string>("fileName", "gen.h5");
  d.add("genHDF5Producer", p);
}

GenHDF5Producer::GenHDF5Producer(edm::ParameterSet const& ps) {
  produces<edm::HepMCProduct>();
  produces<GenEventInfoProduct>();
  produces<reco::GenParticleCollection>();
  produces<reco::GenJetCollection>("ak4GenJetsNoNu");
  produces<reco::GenJetCollection>("ak8GenJetsNoNu");
  produces<reco::GenMETCollection>();

  H5File file(ps.getParameter<std::string>("fileName"));

  simN_ = file.read<int32_t>("/sim/n", H5T_NATIVE_INT32);
  nEvents_ = simN_.size();
  simPt_ = file.read<double>("/sim/pt", H5T_NATIVE_DOUBLE);
  simEta_ = file.read<double>("/sim/eta", H5T_NATIVE_DOUBLE);
  simPhi_ = file.read<double>("/sim/phi", H5T_NATIVE_DOUBLE);
  simPdg_ = file.read<int32_t>("/sim/pdgId", H5T_NATIVE_INT32);
  if (file.has("/sim/vtxIdx")) {
    simVtxIdx_ = file.read<int32_t>("/sim/vtxIdx", H5T_NATIVE_INT32);
    if (file.has("/sim/status")) {
      simStatus_ = file.read<int8_t>("/sim/status", H5T_NATIVE_INT8);
      simEndVtx_ = file.read<int32_t>("/sim/endVtxIdx", H5T_NATIVE_INT32);
    }
    vtxN_ = file.read<int32_t>("/sim/vertex/n", H5T_NATIVE_INT32);
    if (file.has("/sim/vertex/dx")) {
      // int32 micron offsets from /event/pv_*
      auto dx = file.read<int32_t>("/sim/vertex/dx", H5T_NATIVE_INT32);
      auto dy = file.read<int32_t>("/sim/vertex/dy", H5T_NATIVE_INT32);
      auto dz = file.read<int32_t>("/sim/vertex/dz", H5T_NATIVE_INT32);
      auto dt = file.has("/sim/vertex/dt") ? file.read<int32_t>("/sim/vertex/dt", H5T_NATIVE_INT32)
                                           : std::vector<int32_t>(dx.size(), 0);
      auto pvx = file.read<double>("/event/pv_x", H5T_NATIVE_DOUBLE);
      auto pvy = file.read<double>("/event/pv_y", H5T_NATIVE_DOUBLE);
      auto pvz = file.read<double>("/event/pv_z", H5T_NATIVE_DOUBLE);
      auto pvt = file.read<double>("/event/pv_t", H5T_NATIVE_DOUBLE);
      vtxX_.resize(dx.size());
      vtxY_.resize(dx.size());
      vtxZ_.resize(dx.size());
      vtxT_.resize(dx.size());
      size_t k = 0;
      for (size_t ev = 0; ev < vtxN_.size(); ++ev) {
        for (int j = 0; j < vtxN_[ev] && k < dx.size(); ++j, ++k) {
          vtxX_[k] = pvx[ev] + dx[k] * 1.e-3;
          vtxY_[k] = pvy[ev] + dy[k] * 1.e-3;
          vtxZ_[k] = pvz[ev] + dz[k] * 1.e-3;
          vtxT_[k] = pvt[ev] + dt[k] * 1.e-3;
        }
      }
    } else {
      vtxX_ = file.read<double>("/sim/vertex/x", H5T_NATIVE_DOUBLE);
      vtxY_ = file.read<double>("/sim/vertex/y", H5T_NATIVE_DOUBLE);
      vtxZ_ = file.read<double>("/sim/vertex/z", H5T_NATIVE_DOUBLE);
      vtxT_ = file.read<double>("/sim/vertex/t", H5T_NATIVE_DOUBLE);
    }
  }
  if (file.has("/truth/n")) {
    trN_ = file.read<int32_t>("/truth/n", H5T_NATIVE_INT32);
    trPt_ = file.read<double>("/truth/pt", H5T_NATIVE_DOUBLE);
    trEta_ = file.read<double>("/truth/eta", H5T_NATIVE_DOUBLE);
    trPhi_ = file.read<double>("/truth/phi", H5T_NATIVE_DOUBLE);
    trMass_ = file.read<double>("/truth/mass", H5T_NATIVE_DOUBLE);
    trPdg_ = file.read<int32_t>("/truth/pdgId", H5T_NATIVE_INT32);
    trStatus_ = file.read<int16_t>("/truth/status", H5T_NATIVE_INT16);
    trFlags_ = file.read<uint16_t>("/truth/statusFlags", H5T_NATIVE_UINT16);
    trMother_ = file.read<int16_t>("/truth/motherIdx", H5T_NATIVE_INT16);
    if (file.has("/truth/simIdx"))
      trSimIdx_ = file.read<int32_t>("/truth/simIdx", H5T_NATIVE_INT32);
  }
  for (int ij = 0; ij < 2; ++ij) {
    const std::string g = ij == 0 ? "/jets/ak4" : "/jets/ak8";
    if (!file.has(g + "/n"))
      continue;
    jetN_[ij] = file.read<int32_t>(g + "/n", H5T_NATIVE_INT32);
    jetPt_[ij] = file.read<double>(g + "/pt", H5T_NATIVE_DOUBLE);
    jetEta_[ij] = file.read<double>(g + "/eta", H5T_NATIVE_DOUBLE);
    jetPhi_[ij] = file.read<double>(g + "/phi", H5T_NATIVE_DOUBLE);
    jetMass_[ij] = file.read<double>(g + "/mass", H5T_NATIVE_DOUBLE);
  }
  if (file.has("/event/pdf_x1")) {
    pdfX1_ = file.read<double>("/event/pdf_x1", H5T_NATIVE_DOUBLE);
    pdfX2_ = file.read<double>("/event/pdf_x2", H5T_NATIVE_DOUBLE);
    pdfXpdf1_ = file.read<double>("/event/pdf_xpdf1", H5T_NATIVE_DOUBLE);
    pdfXpdf2_ = file.read<double>("/event/pdf_xpdf2", H5T_NATIVE_DOUBLE);
    pdfScale_ = file.read<double>("/event/pdf_scalePDF", H5T_NATIVE_DOUBLE);
    pdfId1_ = file.read<int32_t>("/event/pdf_id1", H5T_NATIVE_INT32);
    pdfId2_ = file.read<int32_t>("/event/pdf_id2", H5T_NATIVE_INT32);
  }
  if (file.has("/event/GenMET_pt")) {
    metPt_ = file.read<double>("/event/GenMET_pt", H5T_NATIVE_DOUBLE);
    metPhi_ = file.read<double>("/event/GenMET_phi", H5T_NATIVE_DOUBLE);
  }
  weight_ = file.read<double>("/event/weight", H5T_NATIVE_DOUBLE);
  qScale_ = file.read<double>("/event/qScale", H5T_NATIVE_DOUBLE);
  alphaQCD_ = file.read<double>("/event/alphaQCD", H5T_NATIVE_DOUBLE);
  alphaQED_ = file.read<double>("/event/alphaQED", H5T_NATIVE_DOUBLE);

  simOff_ = offsets(simN_);
  vtxOff_ = offsets(vtxN_);
  trOff_ = offsets(trN_);
  for (int ij = 0; ij < 2; ++ij)
    jetOff_[ij] = offsets(jetN_[ij]);

  for (int pdg : simPdg_)
    cachePdg(pdg);
  for (int pdg : trPdg_)
    cachePdg(pdg);

  edm::LogInfo("GenHDF5Producer") << nEvents_ << " events, " << (nEvents_ ? double(simOff_.back()) / nEvents_ : 0.)
                                  << " SIM particles/event";
}

void GenHDF5Producer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
  // EmptySource numbers events from 1
  const size_t idx = static_cast<size_t>(e.id().event()) - 1;
  if (idx >= nEvents_)
    throw cms::Exception("GenHDF5Producer")
        << "event " << e.id().event() << " is past the end of the file (" << nEvents_ << " events)";

  auto* evt = new HepMC::GenEvent(HepMC::Units::GEV, HepMC::Units::MM);
  evt->set_event_number(static_cast<int>(e.id().event()));
  evt->weights().push_back(weight_[idx]);

  const size_t p0 = simOff_[idx], p1 = simOff_[idx + 1];
  const bool haveVtx = !vtxN_.empty();
  const size_t v0 = haveVtx ? vtxOff_[idx] : 0;
  const size_t nv = haveVtx ? static_cast<size_t>(vtxN_[idx]) : 0;

  std::vector<HepMC::GenVertex*> vertices(nv, nullptr);
  for (size_t i = 0; i < nv; ++i)
    vertices[i] = new HepMC::GenVertex(HepMC::FourVector(vtxX_[v0 + i], vtxY_[v0 + i], vtxZ_[v0 + i], vtxT_[v0 + i]));
  // HepMC2 iterates vertices in reverse insertion order; insert backwards so
  // Generator.cc meets the PV first and a parent before its decay
  for (size_t i = nv; i-- > 0;)
    evt->add_vertex(vertices[i]);
  HepMC::GenVertex* primary = nullptr;
  if (nv > 0) {
    primary = vertices[0];
  } else {
    primary = new HepMC::GenVertex(HepMC::FourVector(0., 0., 0., 0.));
    evt->add_vertex(primary);
  }
  evt->set_signal_process_vertex(primary);

  for (size_t i = p0; i < p1; ++i) {
    const double pt = simPt_[i], eta = simEta_[i], phi = simPhi_[i];
    const int pdg = simPdg_[i];
    const double m = massOf(pdg);
    const double px = pt * std::cos(phi), py = pt * std::sin(phi), pz = pt * std::sinh(eta);
    const double en = std::sqrt(px * px + py * py + pz * pz + m * m);
    const int st = simStatus_.empty() ? 1 : static_cast<int>(simStatus_[i]);
    auto* part = new HepMC::GenParticle(HepMC::FourVector(px, py, pz, en), pdg, st);
    part->set_generated_mass(m);
    const int vi = simVtxIdx_.empty() ? -1 : simVtxIdx_[i];
    auto* v = (vi >= 0 && static_cast<size_t>(vi) < nv) ? vertices[vi] : primary;
    v->add_particle_out(part);
    // stored decay: attach to its end vertex so Geant4 gets the chain
    if (st == 2 && !simEndVtx_.empty()) {
      const int ei = simEndVtx_[i];
      if (ei >= 0 && static_cast<size_t>(ei) < nv)
        vertices[ei]->add_particle_in(part);
    }
  }

  auto product = std::make_unique<edm::HepMCProduct>();
  product->addHepMCData(evt);  // takes ownership
  e.put(std::move(product));

  // genParticles: truth tier plus the SIM-tier particles it does not hold
  auto gps = std::make_unique<reco::GenParticleCollection>();
  if (idx < trN_.size()) {
    const size_t t0 = trOff_[idx], t1 = trOff_[idx + 1];
    gps->reserve(t1 - t0);
    for (size_t i = t0; i < t1; ++i) {
      const int pdg = trPdg_[i];
      double m = trMass_[i];
      if (m <= 0.)
        m = massOf(pdg);  // mass omitted when the PDG table has it
      const reco::Particle::PolarLorentzVector p4(trPt_[i], trEta_[i], trPhi_[i], m);
      reco::GenParticle g(chargeOf(pdg), p4, reco::Particle::Point(0, 0, 0), pdg, trStatus_[i], true);
      g.statusFlags().flags_ = std::bitset<15>(static_cast<unsigned long>(trFlags_[i]));
      gps->push_back(g);
    }
    std::vector<bool> inTruth(p1 - p0, false);
    for (size_t i = t0; i < t1 && !trSimIdx_.empty(); ++i) {
      const int si = trSimIdx_[i];
      if (si >= 0 && static_cast<size_t>(si) < inTruth.size())
        inTruth[si] = true;
    }
    for (size_t i = p0; i < p1; ++i) {
      if (inTruth[i - p0])
        continue;
      const int pdg = simPdg_[i];
      reco::GenParticle g(chargeOf(pdg),
                          reco::Particle::PolarLorentzVector(simPt_[i], simEta_[i], simPhi_[i], massOf(pdg)),
                          reco::Particle::Point(0, 0, 0),
                          pdg,
                          1,
                          true);
      g.statusFlags().setIsPrompt(true);
      g.statusFlags().setIsLastCopy(true);
      gps->push_back(g);
    }
    auto ref = e.getRefBeforePut<reco::GenParticleCollection>();
    for (size_t i = t0; i < t1; ++i) {
      const int m = trMother_[i];
      if (m >= 0 && static_cast<size_t>(m) < (t1 - t0))
        (*gps)[i - t0].addMother(reco::GenParticleRef(ref, m));
    }
  }
  e.put(std::move(gps));

  for (int ij = 0; ij < 2; ++ij) {
    auto jets = std::make_unique<reco::GenJetCollection>();
    if (idx < jetN_[ij].size()) {
      const size_t j0 = jetOff_[ij][idx], j1 = jetOff_[ij][idx + 1];
      jets->reserve(j1 - j0);
      for (size_t i = j0; i < j1; ++i) {
        const reco::Particle::PolarLorentzVector p4(jetPt_[ij][i], jetEta_[ij][i], jetPhi_[ij][i], jetMass_[ij][i]);
        reco::GenJet j;
        j.setP4(reco::Particle::LorentzVector(p4));
        jets->push_back(j);
      }
    }
    e.put(std::move(jets), ij == 0 ? "ak4GenJetsNoNu" : "ak8GenJetsNoNu");
  }

  auto mets = std::make_unique<reco::GenMETCollection>();
  if (idx < metPt_.size()) {
    const double px = metPt_[idx] * std::cos(metPhi_[idx]);
    const double py = metPt_[idx] * std::sin(metPhi_[idx]);
    SpecificGenMETData spec;
    mets->push_back(reco::GenMET(
        spec, metPt_[idx], reco::Particle::LorentzVector(px, py, 0., metPt_[idx]), reco::Particle::Point(0, 0, 0)));
  }
  e.put(std::move(mets));

  auto info = std::make_unique<GenEventInfoProduct>();
  info->setWeights({weight_[idx]});
  info->setScales(qScale_[idx], alphaQCD_[idx], alphaQED_[idx]);
  if (idx < pdfX1_.size()) {
    gen::PdfInfo pdf;
    pdf.id = std::make_pair(pdfId1_[idx], pdfId2_[idx]);
    pdf.x = std::make_pair(pdfX1_[idx], pdfX2_[idx]);
    pdf.xPDF = std::make_pair(pdfXpdf1_[idx], pdfXpdf2_[idx]);
    pdf.scalePDF = pdfScale_[idx];
    info->setPDF(&pdf);
  }
  e.put(std::move(info));
}

DEFINE_FWK_MODULE(GenHDF5Producer);
