// -*- C++ -*-
//
// Package:    GeneratorInterface/GenHDF5Interface
// Class:      GenHDF5Producer
//
/**\class GenHDF5Producer

 Reads a GenHDF5 file (https://gitlab.cern.ch/tvami/genhdf5) and turns event N
 into the generator products of row N-1, for SIM. Driven by EmptySource, so a
 job that starts at firstEvent=K+1 reads rows K onwards.
 Rows are read in blocks of blockRows on first use and the last few blocks are
 kept, so memory does not grow with the file.
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
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
#include "FWStorage/StorageFactory/interface/Storage.h"
#include "FWStorage/StorageFactory/interface/StorageFactory.h"

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

    // one component at a time: H5Lexists on a path with a missing parent reports an error
    bool has(std::string const& path) const {
      for (size_t pos = path.find('/', 1);; pos = path.find('/', pos + 1)) {
        if (H5Lexists(id_, path.substr(0, pos).c_str(), H5P_DEFAULT) <= 0)
          return false;
        if (pos == std::string::npos)
          return true;
      }
    }

    // string file attribute, or "" if absent
    std::string attrString(std::string const& name) const {
      if (H5Aexists(id_, name.c_str()) <= 0)
        return "";
      hid_t a = H5Aopen(id_, name.c_str(), H5P_DEFAULT);
      hid_t t = H5Aget_type(a);
      std::string v;
      if (H5Tget_class(t) == H5T_STRING && !H5Tis_variable_str(t)) {
        v.resize(H5Tget_size(t));
        if (H5Aread(a, t, v.data()) < 0)
          v.clear();
        v.erase(std::find(v.begin(), v.end(), '\0'), v.end());
      }
      H5Tclose(t);
      H5Aclose(a);
      return v;
    }

    // integer file attribute, or def if absent
    long long attrInt(std::string const& name, long long def) const {
      if (H5Aexists(id_, name.c_str()) <= 0)
        return def;
      hid_t a = H5Aopen(id_, name.c_str(), H5P_DEFAULT);
      long long v = def;
      if (H5Aread(a, H5T_NATIVE_LLONG, &v) < 0)
        v = def;
      H5Aclose(a);
      return v;
    }

    // rows [start, start+count) of a column, converted to T by HDF5; count < 0 reads to the end
    template <typename T>
    std::vector<T> read(std::string const& path, hid_t memType, hsize_t start = 0, long long count = -1) const {
      hid_t d = H5Dopen2(id_, path.c_str(), H5P_DEFAULT);
      if (d < 0)
        throw cms::Exception("GenHDF5Producer") << "no dataset " << path;
      hid_t s = H5Dget_space(d);
      hsize_t dims[1] = {0};
      H5Sget_simple_extent_dims(s, dims, nullptr);
      hsize_t n[1] = {count < 0 ? dims[0] - std::min(start, dims[0]) : static_cast<hsize_t>(count)};
      if (start + n[0] > dims[0]) {
        H5Sclose(s);
        H5Dclose(d);
        throw cms::Exception("GenHDF5Producer")
            << path << " has " << dims[0] << " rows, asked for " << start << "+" << n[0];
      }
      std::vector<T> out(n[0]);
      herr_t err = 0;
      if (n[0]) {
        hsize_t off[1] = {start};
        H5Sselect_hyperslab(s, H5S_SELECT_SET, off, nullptr, n, nullptr);
        hid_t m = H5Screate_simple(1, n, nullptr);
        err = H5Dread(d, memType, m, s, H5P_DEFAULT, out.data());
        H5Sclose(m);
      }
      H5Sclose(s);
      H5Dclose(d);
      if (err < 0)
        throw cms::Exception("GenHDF5Producer") << "cannot read " << path;
      return out;
    }

  private:
    hid_t id_;
  };

  // a URL (root://, file:, ...) is copied to the working directory, since HDF5 reads only local files
  std::string stageIn(std::string const& name, std::string& staged) {
    if (name.find(':') == std::string::npos)
      return name;
    const auto t0 = std::chrono::steady_clock::now();
    auto in = edm::storage::StorageFactory::get()->open(name, edm::storage::IOFlags::OpenRead);
    if (!in)
      throw cms::Exception("GenHDF5Producer") << "cannot open " << name;
    staged = "genhdf5_staged_" + std::to_string(::getpid()) + ".h5";
    std::ofstream out(staged, std::ios::binary);
    std::vector<char> buf(16 << 20);
    // read exactly the file size: a read at EOF makes XrdAdaptor warn
    const long long size = in->size();
    long long total = 0;
    for (edm::storage::IOSize n;
         total < size && (n = in->read(buf.data(), std::min<long long>(buf.size(), size - total))) > 0;
         total += n)
      out.write(buf.data(), n);
    in->close();
    if (!out || total != size)
      throw cms::Exception("GenHDF5Producer")
          << "cannot stage " << name << " (" << total << " of " << size << " bytes)";
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    edm::LogInfo("GenHDF5Producer") << "staged " << name << ": " << total << " bytes in " << dt << " s";
    return staged;
  }

  std::vector<size_t> offsets(std::vector<int32_t> const& n) {
    std::vector<size_t> off(n.size() + 1, 0);
    for (size_t i = 0; i < n.size(); ++i)
      off[i + 1] = off[i] + n[i];
    return off;
  }

  // per-row offsets of a block, relative to its first row
  std::vector<size_t> localOffsets(std::vector<size_t> const& global, size_t r0, size_t r1) {
    std::vector<size_t> off;
    if (global.empty())
      return off;
    off.reserve(r1 - r0 + 1);
    for (size_t r = r0; r <= r1; ++r)
      off.push_back(global[r] - global[r0]);
    return off;
  }

  // the columns of rows [row0, row0 + nRows)
  struct Block {
    size_t row0 = 0, nRows = 0;
    std::vector<size_t> simOff, vtxOff, trOff;
    std::array<std::vector<size_t>, 2> jetOff;

    std::vector<int32_t> simPdg, simVtxIdx, simEndVtx;
    std::vector<int8_t> simStatus;
    std::vector<double> simPt, simEta, simPhi, simMass;
    std::vector<int> simCharge;
    std::vector<double> vtxX, vtxY, vtxZ, vtxT;
    std::vector<double> weight, qScale, alphaQCD, alphaQED;

    std::vector<int32_t> trPdg, trSimIdx;
    std::vector<int16_t> trStatus, trMother;  // trMother: single mother, older files
    std::vector<uint16_t> trFlags, trNMo, simFlags;
    std::vector<int16_t> simTrMo;  // nearest truth ancestor of a SIM-tier particle
    std::vector<int16_t> trAxIdx;  // truth particles with pt = 0 (beams, incoming partons) ...
    std::vector<double> trAxPz;    // ... and their pz
    std::vector<size_t> trAxOff;
    std::vector<int16_t> trMo;    // all mothers, CSR by trNMo
    std::vector<size_t> trMoOff;  // per row, into trMo
    std::vector<double> trPt, trEta, trPhi, trMass;
    std::vector<int> trCharge;
    std::array<std::vector<double>, 2> jetPt, jetEta, jetPhi, jetMass;
    std::vector<double> metPt, metPhi;
    std::vector<double> pdfX1, pdfX2, pdfXpdf1, pdfXpdf2, pdfScale;
    std::vector<int32_t> pdfId1, pdfId2;
  };

}  // namespace

class GenHDF5Producer : public edm::global::EDProducer<> {
public:
  explicit GenHDF5Producer(edm::ParameterSet const&);
  ~GenHDF5Producer() override {
    file_.reset();
    if (!staged_.empty())
      std::remove(staged_.c_str());
  }
  static void fillDescriptions(edm::ConfigurationDescriptions&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  std::shared_ptr<const Block> block(size_t row) const;
  std::shared_ptr<const Block> load(size_t b) const;
  std::pair<double, int> pdg(int pdgId) const;

  std::string staged_;  // local copy of a remote file, removed at the end
  std::unique_ptr<H5File> file_;
  size_t nEvents_ = 0;
  size_t blockRows_ = 1024;
  size_t maxBlocks_ = 4;

  // layout of the file
  bool haveVtx_ = false, quantised_ = false, haveDt_ = false, haveStatus_ = false;
  bool haveTruth_ = false, haveTrSimIdx_ = false, haveTrMothers_ = false, haveSimFlags_ = false, haveSimTrMo_ = false;
  bool havePdf_ = false, haveMet_ = false;
  bool trInvariantMass_ = false;  // truth mass is sqrt(E^2 - p^2); older files store 0 for "PDG mass"
  std::array<bool, 2> haveJets_{{false, false}};

  // global per-row offsets into the particle-level columns
  std::vector<size_t> simOff_, vtxOff_, trOff_, trMoOff_, trAxOff_;
  std::array<std::vector<size_t>, 2> jetOff_;

  // guards file_, the block cache and TDatabasePDG (not thread safe)
  mutable std::mutex mutex_;
  mutable std::vector<std::pair<size_t, std::shared_ptr<const Block>>> cache_;  // most recent first
  mutable std::unordered_map<int, std::pair<double, int>> pdgTable_;
  // per-file masses for species the PDG table gets wrong (exotics), keyed by |pdgId|
  std::unordered_map<int, double> fileMasses_;
};

void GenHDF5Producer::fillDescriptions(edm::ConfigurationDescriptions& d) {
  edm::ParameterSetDescription p;
  p.add<std::string>("fileName", "gen.h5");
  p.addUntracked<unsigned int>("blockRows", 0)
      ->setComment("rows read at a time; 0 = the file's chunk_events, else 1024");
  p.addUntracked<unsigned int>("maxCachedBlocks", 4);
  d.add("genHDF5Producer", p);
}

GenHDF5Producer::GenHDF5Producer(edm::ParameterSet const& ps)
    : file_(std::make_unique<H5File>(stageIn(ps.getParameter<std::string>("fileName"), staged_))),
      maxBlocks_(std::max(1u, ps.getUntrackedParameter<unsigned int>("maxCachedBlocks"))) {
  produces<edm::HepMCProduct>();
  produces<GenEventInfoProduct>();
  produces<reco::GenParticleCollection>();
  produces<std::vector<int>>();  // HepMC barcode per genParticle, as GenParticleProducer writes it
  produces<reco::GenJetCollection>("ak4GenJetsNoNu");
  produces<reco::GenJetCollection>("ak8GenJetsNoNu");
  produces<reco::GenMETCollection>();

  H5File const& f = *file_;
  simOff_ = offsets(f.read<int32_t>("/sim/n", H5T_NATIVE_INT32));
  nEvents_ = simOff_.size() - 1;
  haveVtx_ = f.has("/sim/vtxIdx");
  if (haveVtx_) {
    haveStatus_ = f.has("/sim/status");
    haveSimFlags_ = f.has("/sim/statusFlags");
    haveSimTrMo_ = f.has("/sim/truthMother");
    vtxOff_ = offsets(f.read<int32_t>("/sim/vertex/n", H5T_NATIVE_INT32));
    quantised_ = f.has("/sim/vertex/dx");
    haveDt_ = f.has("/sim/vertex/dt");
  }
  haveTruth_ = f.has("/truth/n");
  if (haveTruth_) {
    trOff_ = offsets(f.read<int32_t>("/truth/n", H5T_NATIVE_INT32));
    haveTrSimIdx_ = f.has("/truth/simIdx");
    trInvariantMass_ = f.attrString("truth_mass") == "invariant";
    haveTrMothers_ = f.has("/truth/nMothersEvent");
    if (haveTrMothers_)
      trMoOff_ = offsets(f.read<int32_t>("/truth/nMothersEvent", H5T_NATIVE_INT32));
    if (f.has("/truth/nAxial"))
      trAxOff_ = offsets(f.read<int32_t>("/truth/nAxial", H5T_NATIVE_INT32));
  }
  for (int ij = 0; ij < 2; ++ij) {
    const std::string g = ij == 0 ? "/jets/ak4" : "/jets/ak8";
    haveJets_[ij] = f.has(g + "/n");
    if (haveJets_[ij])
      jetOff_[ij] = offsets(f.read<int32_t>(g + "/n", H5T_NATIVE_INT32));
  }
  if (f.has("/sim/massTable/pdgId")) {
    auto ids = f.read<int32_t>("/sim/massTable/pdgId", H5T_NATIVE_INT32);
    auto ms = f.read<double>("/sim/massTable/mass", H5T_NATIVE_DOUBLE);
    for (size_t i = 0; i < ids.size() && i < ms.size(); ++i)
      fileMasses_[std::abs(ids[i])] = ms[i];
  }
  havePdf_ = f.has("/event/pdf_x1");
  haveMet_ = f.has("/event/GenMET_pt");

  long long rows = ps.getUntrackedParameter<unsigned int>("blockRows");
  if (rows == 0)
    rows = f.attrInt("chunk_events", 1024);
  blockRows_ = rows > 0 ? rows : 1024;

  edm::LogInfo("GenHDF5Producer") << nEvents_ << " events, " << (nEvents_ ? double(simOff_.back()) / nEvents_ : 0.)
                                  << " SIM particles/event, "
                                  << "blocks of " << blockRows_ << " rows";
}

std::pair<double, int> GenHDF5Producer::pdg(int pdgId) const {
  auto it = pdgTable_.find(pdgId);
  if (it != pdgTable_.end())
    return it->second;
  auto* pd = TDatabasePDG::Instance()->GetParticle(pdgId);
  std::pair<double, int> mc(pd ? pd->Mass() : 0., pd ? static_cast<int>(std::lround(pd->Charge() / 3.0)) : 0);
  auto fm = fileMasses_.find(std::abs(pdgId));
  if (fm != fileMasses_.end())
    mc.first = fm->second;
  pdgTable_.emplace(pdgId, mc);
  return mc;
}

std::shared_ptr<const Block> GenHDF5Producer::block(size_t row) const {
  const size_t b = row / blockRows_;
  std::lock_guard<std::mutex> guard(mutex_);
  for (size_t i = 0; i < cache_.size(); ++i) {
    if (cache_[i].first == b) {
      auto hit = cache_[i];
      cache_.erase(cache_.begin() + i);
      cache_.insert(cache_.begin(), hit);
      return hit.second;
    }
  }
  auto blk = load(b);
  cache_.insert(cache_.begin(), {b, blk});
  if (cache_.size() > maxBlocks_)
    cache_.pop_back();  // a stream still using it keeps its own reference
  return blk;
}

// called with mutex_ held
std::shared_ptr<const Block> GenHDF5Producer::load(size_t b) const {
  H5File const& f = *file_;
  auto blk = std::make_shared<Block>();
  const size_t r0 = b * blockRows_, r1 = std::min(nEvents_, r0 + blockRows_);
  blk->row0 = r0;
  blk->nRows = r1 - r0;
  const long long nr = r1 - r0;

  blk->simOff = localOffsets(simOff_, r0, r1);
  const size_t p0 = simOff_[r0];
  const long long np = simOff_[r1] - p0;
  blk->simPt = f.read<double>("/sim/pt", H5T_NATIVE_DOUBLE, p0, np);
  blk->simEta = f.read<double>("/sim/eta", H5T_NATIVE_DOUBLE, p0, np);
  blk->simPhi = f.read<double>("/sim/phi", H5T_NATIVE_DOUBLE, p0, np);
  blk->simPdg = f.read<int32_t>("/sim/pdgId", H5T_NATIVE_INT32, p0, np);
  if (haveSimFlags_)
    blk->simFlags = f.read<uint16_t>("/sim/statusFlags", H5T_NATIVE_UINT16, p0, np);
  if (haveSimTrMo_)
    blk->simTrMo = f.read<int16_t>("/sim/truthMother", H5T_NATIVE_INT16, p0, np);
  blk->simMass.reserve(np);
  blk->simCharge.reserve(np);
  for (int id : blk->simPdg) {
    auto mc = pdg(id);
    blk->simMass.push_back(mc.first);
    blk->simCharge.push_back(mc.second);
  }
  if (haveVtx_) {
    blk->simVtxIdx = f.read<int32_t>("/sim/vtxIdx", H5T_NATIVE_INT32, p0, np);
    if (haveStatus_) {
      blk->simStatus = f.read<int8_t>("/sim/status", H5T_NATIVE_INT8, p0, np);
      blk->simEndVtx = f.read<int32_t>("/sim/endVtxIdx", H5T_NATIVE_INT32, p0, np);
    }
    blk->vtxOff = localOffsets(vtxOff_, r0, r1);
    const size_t v0 = vtxOff_[r0];
    const long long nv = vtxOff_[r1] - v0;
    if (quantised_) {
      // int32 micron offsets from /event/pv_*
      auto dx = f.read<int32_t>("/sim/vertex/dx", H5T_NATIVE_INT32, v0, nv);
      auto dy = f.read<int32_t>("/sim/vertex/dy", H5T_NATIVE_INT32, v0, nv);
      auto dz = f.read<int32_t>("/sim/vertex/dz", H5T_NATIVE_INT32, v0, nv);
      auto dt =
          haveDt_ ? f.read<int32_t>("/sim/vertex/dt", H5T_NATIVE_INT32, v0, nv) : std::vector<int32_t>(dx.size(), 0);
      auto pvx = f.read<double>("/event/pv_x", H5T_NATIVE_DOUBLE, r0, nr);
      auto pvy = f.read<double>("/event/pv_y", H5T_NATIVE_DOUBLE, r0, nr);
      auto pvz = f.read<double>("/event/pv_z", H5T_NATIVE_DOUBLE, r0, nr);
      auto pvt = f.read<double>("/event/pv_t", H5T_NATIVE_DOUBLE, r0, nr);
      blk->vtxX.resize(nv);
      blk->vtxY.resize(nv);
      blk->vtxZ.resize(nv);
      blk->vtxT.resize(nv);
      for (long long r = 0; r < nr; ++r) {
        for (size_t k = blk->vtxOff[r]; k < blk->vtxOff[r + 1]; ++k) {
          blk->vtxX[k] = pvx[r] + dx[k] * 1.e-3;
          blk->vtxY[k] = pvy[r] + dy[k] * 1.e-3;
          blk->vtxZ[k] = pvz[r] + dz[k] * 1.e-3;
          blk->vtxT[k] = pvt[r] + dt[k] * 1.e-3;
        }
      }
    } else {
      blk->vtxX = f.read<double>("/sim/vertex/x", H5T_NATIVE_DOUBLE, v0, nv);
      blk->vtxY = f.read<double>("/sim/vertex/y", H5T_NATIVE_DOUBLE, v0, nv);
      blk->vtxZ = f.read<double>("/sim/vertex/z", H5T_NATIVE_DOUBLE, v0, nv);
      blk->vtxT = f.read<double>("/sim/vertex/t", H5T_NATIVE_DOUBLE, v0, nv);
    }
  }
  if (haveTruth_) {
    blk->trOff = localOffsets(trOff_, r0, r1);
    const size_t t0 = trOff_[r0];
    const long long nt = trOff_[r1] - t0;
    blk->trPt = f.read<double>("/truth/pt", H5T_NATIVE_DOUBLE, t0, nt);
    blk->trEta = f.read<double>("/truth/eta", H5T_NATIVE_DOUBLE, t0, nt);
    blk->trPhi = f.read<double>("/truth/phi", H5T_NATIVE_DOUBLE, t0, nt);
    blk->trMass = f.read<double>("/truth/mass", H5T_NATIVE_DOUBLE, t0, nt);
    blk->trPdg = f.read<int32_t>("/truth/pdgId", H5T_NATIVE_INT32, t0, nt);
    blk->trStatus = f.read<int16_t>("/truth/status", H5T_NATIVE_INT16, t0, nt);
    blk->trFlags = f.read<uint16_t>("/truth/statusFlags", H5T_NATIVE_UINT16, t0, nt);
    if (haveTrMothers_) {
      blk->trNMo = f.read<uint16_t>("/truth/nMothers", H5T_NATIVE_UINT16, t0, nt);
      blk->trMoOff = localOffsets(trMoOff_, r0, r1);
      const size_t m0 = trMoOff_[r0];
      blk->trMo = f.read<int16_t>("/truth/mothers", H5T_NATIVE_INT16, m0, trMoOff_[r1] - m0);
    } else {
      blk->trMother = f.read<int16_t>("/truth/motherIdx", H5T_NATIVE_INT16, t0, nt);
    }
    if (haveTrSimIdx_)
      blk->trSimIdx = f.read<int32_t>("/truth/simIdx", H5T_NATIVE_INT32, t0, nt);
    if (!trAxOff_.empty()) {
      blk->trAxOff = localOffsets(trAxOff_, r0, r1);
      const size_t a0 = trAxOff_[r0];
      blk->trAxIdx = f.read<int16_t>("/truth/axialIdx", H5T_NATIVE_INT16, a0, trAxOff_[r1] - a0);
      blk->trAxPz = f.read<double>("/truth/axialPz", H5T_NATIVE_DOUBLE, a0, trAxOff_[r1] - a0);
    }
    blk->trCharge.reserve(nt);
    for (long long i = 0; i < nt; ++i) {
      auto mc = pdg(blk->trPdg[i]);
      if (!trInvariantMass_ && blk->trMass[i] <= 0.)
        blk->trMass[i] = mc.first;  // mass omitted when the PDG table has it
      blk->trCharge.push_back(mc.second);
    }
  }
  for (int ij = 0; ij < 2; ++ij) {
    if (!haveJets_[ij])
      continue;
    const std::string g = ij == 0 ? "/jets/ak4" : "/jets/ak8";
    blk->jetOff[ij] = localOffsets(jetOff_[ij], r0, r1);
    const size_t j0 = jetOff_[ij][r0];
    const long long nj = jetOff_[ij][r1] - j0;
    blk->jetPt[ij] = f.read<double>(g + "/pt", H5T_NATIVE_DOUBLE, j0, nj);
    blk->jetEta[ij] = f.read<double>(g + "/eta", H5T_NATIVE_DOUBLE, j0, nj);
    blk->jetPhi[ij] = f.read<double>(g + "/phi", H5T_NATIVE_DOUBLE, j0, nj);
    blk->jetMass[ij] = f.read<double>(g + "/mass", H5T_NATIVE_DOUBLE, j0, nj);
  }
  if (havePdf_) {
    blk->pdfX1 = f.read<double>("/event/pdf_x1", H5T_NATIVE_DOUBLE, r0, nr);
    blk->pdfX2 = f.read<double>("/event/pdf_x2", H5T_NATIVE_DOUBLE, r0, nr);
    blk->pdfXpdf1 = f.read<double>("/event/pdf_xpdf1", H5T_NATIVE_DOUBLE, r0, nr);
    blk->pdfXpdf2 = f.read<double>("/event/pdf_xpdf2", H5T_NATIVE_DOUBLE, r0, nr);
    blk->pdfScale = f.read<double>("/event/pdf_scalePDF", H5T_NATIVE_DOUBLE, r0, nr);
    blk->pdfId1 = f.read<int32_t>("/event/pdf_id1", H5T_NATIVE_INT32, r0, nr);
    blk->pdfId2 = f.read<int32_t>("/event/pdf_id2", H5T_NATIVE_INT32, r0, nr);
  }
  if (haveMet_) {
    blk->metPt = f.read<double>("/event/GenMET_pt", H5T_NATIVE_DOUBLE, r0, nr);
    blk->metPhi = f.read<double>("/event/GenMET_phi", H5T_NATIVE_DOUBLE, r0, nr);
  }
  blk->weight = f.read<double>("/event/weight", H5T_NATIVE_DOUBLE, r0, nr);
  blk->qScale = f.read<double>("/event/qScale", H5T_NATIVE_DOUBLE, r0, nr);
  blk->alphaQCD = f.read<double>("/event/alphaQCD", H5T_NATIVE_DOUBLE, r0, nr);
  blk->alphaQED = f.read<double>("/event/alphaQED", H5T_NATIVE_DOUBLE, r0, nr);
  return blk;
}

void GenHDF5Producer::produce(edm::StreamID, edm::Event& e, edm::EventSetup const&) const {
  // EmptySource numbers events from 1
  const size_t row = static_cast<size_t>(e.id().event()) - 1;
  if (row >= nEvents_)
    throw cms::Exception("GenHDF5Producer")
        << "event " << e.id().event() << " is past the end of the file (" << nEvents_ << " events)";
  auto blkp = block(row);
  Block const& B = *blkp;
  const size_t idx = row - B.row0;

  auto* evt = new HepMC::GenEvent(HepMC::Units::GEV, HepMC::Units::MM);
  evt->set_event_number(static_cast<int>(e.id().event()));
  evt->weights().push_back(B.weight[idx]);

  const size_t p0 = B.simOff[idx], p1 = B.simOff[idx + 1];
  const size_t v0 = haveVtx_ ? B.vtxOff[idx] : 0;
  const size_t nv = haveVtx_ ? B.vtxOff[idx + 1] - v0 : 0;

  std::vector<HepMC::GenVertex*> vertices(nv, nullptr);
  for (size_t i = 0; i < nv; ++i)
    vertices[i] =
        new HepMC::GenVertex(HepMC::FourVector(B.vtxX[v0 + i], B.vtxY[v0 + i], B.vtxZ[v0 + i], B.vtxT[v0 + i]));
  // HepMC2 iterates vertices in insertion order (barcode -1 first), so
  // Generator.cc meets the PV first and a parent before its decay
  for (size_t i = 0; i < nv; ++i)
    evt->add_vertex(vertices[i]);
  HepMC::GenVertex* primary = nullptr;
  if (nv > 0) {
    primary = vertices[0];
  } else {
    primary = new HepMC::GenVertex(HepMC::FourVector(0., 0., 0., 0.));
    evt->add_vertex(primary);
  }
  evt->set_signal_process_vertex(primary);

  std::vector<int> simBarcode(p1 - p0, 0);
  for (size_t i = p0; i < p1; ++i) {
    const double pt = B.simPt[i], eta = B.simEta[i], phi = B.simPhi[i];
    const int pdgId = B.simPdg[i];
    const double m = B.simMass[i];
    const double px = pt * std::cos(phi), py = pt * std::sin(phi), pz = pt * std::sinh(eta);
    const double en = std::sqrt(px * px + py * py + pz * pz + m * m);
    const int st = B.simStatus.empty() ? 1 : static_cast<int>(B.simStatus[i]);
    auto* part = new HepMC::GenParticle(HepMC::FourVector(px, py, pz, en), pdgId, st);
    part->set_generated_mass(m);
    const int vi = B.simVtxIdx.empty() ? -1 : B.simVtxIdx[i];
    auto* v = (vi >= 0 && static_cast<size_t>(vi) < nv) ? vertices[vi] : primary;
    v->add_particle_out(part);
    simBarcode[i - p0] = part->barcode();
    // stored decay: attach to its end vertex so Geant4 gets the chain
    if (st == 2 && !B.simEndVtx.empty()) {
      const int ei = B.simEndVtx[i];
      if (ei >= 0 && static_cast<size_t>(ei) < nv)
        vertices[ei]->add_particle_in(part);
    }
  }

  auto product = std::make_unique<edm::HepMCProduct>();
  product->addHepMCData(evt);  // takes ownership
  e.put(std::move(product));

  // production vertices in cm, as GenParticleProducer sets them; never exactly zero, since HepMC3 then asks
  // the ancestors for a position (endless on a pruned graph with loops, GenParticles2HepMCConverter)
  const reco::Particle::Point pv = nv > 0 ? reco::Particle::Point(B.vtxX[v0] * 0.1, B.vtxY[v0] * 0.1, B.vtxZ[v0] * 0.1)
                                          : reco::Particle::Point(0, 0, 0);
  const reco::Particle::Point origin = pv.mag2() > 0 ? pv : reco::Particle::Point(0, 0, 1.e-9);
  auto simPos = [&](size_t i) {
    const int vi = B.simVtxIdx.empty() ? -1 : B.simVtxIdx[i];
    if (vi < 0 || static_cast<size_t>(vi) >= nv)
      return origin;
    const reco::Particle::Point x(B.vtxX[v0 + vi] * 0.1, B.vtxY[v0 + vi] * 0.1, B.vtxZ[v0 + vi] * 0.1);
    return x.mag2() > 0 ? x : origin;
  };

  // genParticles: truth tier plus the SIM-tier particles it does not hold
  auto gps = std::make_unique<reco::GenParticleCollection>();
  auto barcodes = std::make_unique<std::vector<int>>();
  if (haveTruth_) {
    const size_t t0 = B.trOff[idx], t1 = B.trOff[idx + 1];
    gps->reserve(t1 - t0);
    std::unordered_map<size_t, double> axialPz;
    if (!B.trAxOff.empty())
      for (size_t a = B.trAxOff[idx]; a < B.trAxOff[idx + 1]; ++a)
        axialPz[B.trAxIdx[a]] = B.trAxPz[a];
    for (size_t i = t0; i < t1; ++i) {
      reco::Particle::LorentzVector p4(
          reco::Particle::PolarLorentzVector(B.trPt[i], B.trEta[i], B.trPhi[i], B.trMass[i]));
      auto ax = axialPz.find(i - t0);
      if (ax != axialPz.end()) {
        const double m = std::max(B.trMass[i], 0.);
        p4 = reco::Particle::LorentzVector(0., 0., ax->second, std::sqrt(ax->second * ax->second + m * m));
      }
      const int tsi = B.trSimIdx.empty() ? -1 : B.trSimIdx[i];
      const reco::Particle::Point x = tsi >= 0 && static_cast<size_t>(tsi) < p1 - p0 ? simPos(p0 + tsi) : origin;
      reco::GenParticle g(B.trCharge[i], p4, x, B.trPdg[i], B.trStatus[i], true);
      g.statusFlags().flags_ = std::bitset<15>(static_cast<unsigned long>(B.trFlags[i]));
      gps->push_back(g);
      const int si = B.trSimIdx.empty() ? -1 : B.trSimIdx[i];
      barcodes->push_back(si >= 0 && static_cast<size_t>(si) < simBarcode.size() ? simBarcode[si] : 0);
    }
    std::vector<bool> inTruth(p1 - p0, false);
    for (size_t i = t0; i < t1 && !B.trSimIdx.empty(); ++i) {
      const int si = B.trSimIdx[i];
      if (si >= 0 && static_cast<size_t>(si) < inTruth.size())
        inTruth[si] = true;
    }
    for (size_t i = p0; i < p1; ++i) {
      if (inTruth[i - p0])
        continue;
      reco::GenParticle g(B.simCharge[i],
                          reco::Particle::PolarLorentzVector(B.simPt[i], B.simEta[i], B.simPhi[i], B.simMass[i]),
                          simPos(i),
                          B.simPdg[i],
                          B.simStatus.empty() ? 1 : B.simStatus[i],
                          true);
      if (B.simFlags.empty()) {
        g.statusFlags().setIsPrompt(true);
        g.statusFlags().setIsLastCopy(true);
      } else {
        g.statusFlags().flags_ = std::bitset<15>(static_cast<unsigned long>(B.simFlags[i]));
      }
      gps->push_back(g);
      barcodes->push_back(simBarcode[i - p0]);
    }
    // mothers as stored (nearest kept ancestors), daughters as their inverse, like GenParticlePruner
    auto ref = e.getRefBeforePut<reco::GenParticleCollection>();
    const size_t nt = t1 - t0;
    // SIM-only particles: mother only, so re-pruning never reaches them through daughters
    if (!B.simTrMo.empty()) {
      size_t k = nt;
      for (size_t i = p0; i < p1; ++i) {
        if (inTruth[i - p0])
          continue;
        const int m = B.simTrMo[i];
        if (m >= 0 && static_cast<size_t>(m) < nt)
          (*gps)[k].addMother(reco::GenParticleRef(ref, m));
        ++k;
      }
    }
    if (!B.trNMo.empty()) {
      size_t k = B.trMoOff[idx];
      for (size_t i = 0; i < nt; ++i) {
        for (unsigned j = 0; j < B.trNMo[t0 + i]; ++j, ++k) {
          const int m = B.trMo[k];
          if (m >= 0 && static_cast<size_t>(m) < nt) {
            (*gps)[i].addMother(reco::GenParticleRef(ref, m));
            if (i > 0)
              (*gps)[m].addDaughter(reco::GenParticleRef(ref, i));
          }
        }
      }
    } else {
      for (size_t i = 0; i < nt; ++i) {
        const int m = B.trMother[t0 + i];
        if (m >= 0 && static_cast<size_t>(m) < nt)
          (*gps)[i].addMother(reco::GenParticleRef(ref, m));
      }
    }
  }
  e.put(std::move(gps));
  e.put(std::move(barcodes));

  for (int ij = 0; ij < 2; ++ij) {
    auto jets = std::make_unique<reco::GenJetCollection>();
    if (haveJets_[ij]) {
      const size_t j0 = B.jetOff[ij][idx], j1 = B.jetOff[ij][idx + 1];
      jets->reserve(j1 - j0);
      for (size_t i = j0; i < j1; ++i) {
        const reco::Particle::PolarLorentzVector p4(B.jetPt[ij][i], B.jetEta[ij][i], B.jetPhi[ij][i], B.jetMass[ij][i]);
        reco::GenJet j;
        j.setP4(reco::Particle::LorentzVector(p4));
        jets->push_back(j);
      }
    }
    e.put(std::move(jets), ij == 0 ? "ak4GenJetsNoNu" : "ak8GenJetsNoNu");
  }

  auto mets = std::make_unique<reco::GenMETCollection>();
  if (haveMet_) {
    const double px = B.metPt[idx] * std::cos(B.metPhi[idx]);
    const double py = B.metPt[idx] * std::sin(B.metPhi[idx]);
    SpecificGenMETData spec;
    mets->push_back(reco::GenMET(
        spec, B.metPt[idx], reco::Particle::LorentzVector(px, py, 0., B.metPt[idx]), reco::Particle::Point(0, 0, 0)));
  }
  e.put(std::move(mets));

  auto info = std::make_unique<GenEventInfoProduct>();
  info->setWeights({B.weight[idx]});
  info->setScales(B.qScale[idx], B.alphaQCD[idx], B.alphaQED[idx]);
  if (havePdf_) {
    gen::PdfInfo pdf;
    pdf.id = std::make_pair(B.pdfId1[idx], B.pdfId2[idx]);
    pdf.x = std::make_pair(B.pdfX1[idx], B.pdfX2[idx]);
    pdf.xPDF = std::make_pair(B.pdfXpdf1[idx], B.pdfXpdf2[idx]);
    pdf.scalePDF = B.pdfScale[idx];
    info->setPDF(&pdf);
  }
  e.put(std::move(info));
}

DEFINE_FWK_MODULE(GenHDF5Producer);
