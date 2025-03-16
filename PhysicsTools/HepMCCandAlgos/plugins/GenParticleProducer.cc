/* \class GenParticleProducer
 *
 * \author Luca Lista, INFN
 *
 * Convert HepMC GenEvent format into a collection of type
 * CandidateCollection containing objects of type GenParticle
 *
 *
 */
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthHelper.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <vector>
#include <string>
#include <unordered_map>

namespace edm {
  class ParameterSet;
}
namespace HepMC {
  class GenParticle;
  class GenEvent;
}  // namespace HepMC

namespace HepMC3 {
  class GenParticle;
  class GenVertex;
  class GenEvent;
}  // namespace HepMC3

static constexpr int PDGCacheMax = 32768;

namespace {
  struct IDto3Charge {
    IDto3Charge(HepPDT::ParticleDataTable const&, bool abortOnUnknownPDGCode);

    int chargeTimesThree(int) const;

  private:
    std::vector<int> chargeP_, chargeM_;
    std::unordered_map<int, int> chargeMap_;
    bool abortOnUnknownPDGCode_;
  };

  IDto3Charge::IDto3Charge(HepPDT::ParticleDataTable const& iTable, bool iAbortOnUnknownPDGCode)
      : chargeP_(PDGCacheMax, 0), chargeM_(PDGCacheMax, 0), abortOnUnknownPDGCode_(iAbortOnUnknownPDGCode) {
    for (auto const& p : iTable) {
      const HepPDT::ParticleID& id = p.first;
      int pdgId = id.pid(), apdgId = std::abs(pdgId);
      int q3 = id.threeCharge();
      if (apdgId < PDGCacheMax && pdgId > 0) {
        chargeP_[apdgId] = q3;
        chargeM_[apdgId] = -q3;
      } else if (apdgId < PDGCacheMax) {
        chargeP_[apdgId] = -q3;
        chargeM_[apdgId] = q3;
      } else {
        chargeMap_.emplace(pdgId, q3);
        chargeMap_.emplace(-pdgId, -q3);
      }
    }
  }

  int IDto3Charge::chargeTimesThree(int id) const {
    if (std::abs(id) < PDGCacheMax)
      return id > 0 ? chargeP_[id] : chargeM_[-id];
    auto f = chargeMap_.find(id);
    if (f == chargeMap_.end()) {
      if (abortOnUnknownPDGCode_)
        throw edm::Exception(edm::errors::LogicError) << "invalid PDG id: " << id;
      else
        return HepPDT::ParticleID(id).threeCharge();
    }
    return f->second;
  }

}  // namespace

class GenParticleProducer : public edm::global::EDProducer<edm::RunCache<IDto3Charge>> {
public:
  /// constructor
  GenParticleProducer(const edm::ParameterSet&);
  /// destructor
  ~GenParticleProducer() override;

  /// process one event
  void produce(edm::StreamID, edm::Event& e, const edm::EventSetup&) const override;
  std::shared_ptr<IDto3Charge> globalBeginRun(const edm::Run&, const edm::EventSetup&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {}

  bool convertParticle(reco::GenParticle& cand, const HepMC::GenParticle* part, const IDto3Charge& id2Charge) const;
  bool fillIndices(const HepMC::GenEvent* mc,
                   std::vector<const HepMC::GenParticle*>& particles,
                   std::vector<int>& barCodeVector,
                   int offset,
                   std::unordered_map<int, size_t>& barcodes) const;
  bool fillDaughters(reco::GenParticleCollection& cand,
                     const HepMC::GenParticle* part,
                     reco::GenParticleRefProd const& ref,
                     size_t index,
                     std::unordered_map<int, size_t>& barcodes) const;

  bool convertParticle3(reco::GenParticle& cand, const HepMC3::GenParticle* part, const IDto3Charge& id2Charge) const;
  bool fillIndices3(const HepMC3::GenEvent* mc,
                    std::vector<const HepMC3::GenParticle*>& particles,
                    std::vector<int>& barCodeVector,
                    int offset,
                    std::unordered_map<int, size_t>& barcodes) const;
  bool fillDaughters3(reco::GenParticleCollection& cand,
                      const HepMC3::GenParticle* part,
                      reco::GenParticleRefProd const& ref,
                      size_t index,
                      std::unordered_map<int, size_t>& barcodes) const;

private:
  /// source collection name
  edm::EDGetTokenT<edm::HepMCProduct> srcToken_;
  edm::EDGetTokenT<edm::HepMC3Product> srcToken3_;
  std::vector<edm::EDGetTokenT<edm::HepMCProduct>> vectorSrcTokens_;
  edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct>> mixToken_;
  edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> particleTableToken_;

  /// unknown code treatment flag
  bool abortOnUnknownPDGCode_;
  /// save bar-codes
  bool saveBarCodes_;

  /// input & output modes
  bool doSubEvent_;
  bool useCF_;

  MCTruthHelper<reco::GenParticle> mcTruthHelperGenParts_;
  MCTruthHelper<HepMC::GenParticle> mcTruthHelper_;
  MCTruthHelper<HepMC3::GenParticle> mcTruthHelper3_;
};

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

//#include "SimDataFormats/HiGenData/interface/SubEventMap.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenVertex.h"

#include <fstream>
#include <algorithm>
using namespace edm;
using namespace reco;
using namespace std;
using namespace HepMC;

static const double mmToCm = 0.1;
static const double mmToNs = 1.0 / 299792458e-6;

GenParticleProducer::GenParticleProducer(const ParameterSet& cfg)
    : abortOnUnknownPDGCode_(false),
      saveBarCodes_(cfg.getUntrackedParameter<bool>("saveBarCodes", false)),
      doSubEvent_(cfg.getUntrackedParameter<bool>("doSubEvent", false)),
      useCF_(cfg.getUntrackedParameter<bool>("useCrossingFrame", false)) {
  particleTableToken_ = esConsumes<HepPDT::ParticleDataTable, edm::DefaultRecord, edm::Transition::BeginRun>();
  produces<GenParticleCollection>();
  produces<math::XYZPointF>("xyz0");
  produces<float>("t0");
  if (saveBarCodes_) {
    std::string alias(cfg.getParameter<std::string>("@module_label"));
    produces<vector<int>>().setBranchAlias(alias + "BarCodes");
  }

  if (useCF_)
    mixToken_ =
        mayConsume<CrossingFrame<HepMCProduct>>(InputTag(cfg.getParameter<std::string>("mix"), "generatorSmeared"));
  else {
    srcToken_ = mayConsume<HepMCProduct>(cfg.getParameter<InputTag>("src"));
    srcToken3_ = mayConsume<HepMC3Product>(cfg.getParameter<InputTag>("src"));
  }
}

GenParticleProducer::~GenParticleProducer() {}

std::shared_ptr<IDto3Charge> GenParticleProducer::globalBeginRun(const Run&, const EventSetup& es) const {
  ESHandle<HepPDT::ParticleDataTable> pdt = es.getHandle(particleTableToken_);
  return std::make_shared<IDto3Charge>(*pdt, abortOnUnknownPDGCode_);
}

void GenParticleProducer::produce(StreamID, Event& evt, const EventSetup& es) const {
  std::unordered_map<int, size_t> barcodes;

  size_t totalSize = 0;
  const GenEvent* mc = nullptr;
  HepMC3::GenEvent* mc3 = nullptr;
  MixCollection<HepMCProduct>* cfhepmcprod = nullptr;
  size_t npiles = 1;

  int ivhepmc = 0;
  if (useCF_) {
    Handle<CrossingFrame<HepMCProduct>> cf;
    evt.getByToken(mixToken_, cf);
    cfhepmcprod = new MixCollection<HepMCProduct>(cf.product());
    npiles = cfhepmcprod->size();
    LogDebug("GenParticleProducer") << "npiles : " << npiles << endl;
    for (unsigned int icf = 0; icf < npiles; ++icf) {
      LogDebug("GenParticleProducer") << "subSize : " << cfhepmcprod->getObject(icf).GetEvent()->particles_size()
                                      << endl;
      totalSize += cfhepmcprod->getObject(icf).GetEvent()->particles_size();
    }
    LogDebug("GenParticleProducer") << "totalSize : " << totalSize << endl;
  } else {
    Handle<HepMCProduct> mcp;
    bool found = evt.getByToken(srcToken_, mcp);
    if (found) {
      mc = mcp->GetEvent();
      if (mc == nullptr)
        throw edm::Exception(edm::errors::InvalidReference) << "HepMC has null pointer to GenEvent" << endl;
      ivhepmc = 2;
      totalSize = mc->particles_size();
    } else {
      Handle<HepMC3Product> mcp3;
      bool found = evt.getByToken(srcToken3_, mcp3);
      if (!found)
        throw cms::Exception("ProductAbsent") << "No HepMCProduct, tried to get HepMC3Product, but it is also absent.";
      ivhepmc = 3;
      mc3 = new HepMC3::GenEvent();
      mc3->read_data(*mcp3->GetEvent());
      totalSize = (mc3->particles()).size();
    }
  }

  // initialise containers
  const size_t size = totalSize;
  vector<const HepMC::GenParticle*> particles(size);
  vector<const HepMC3::GenParticle*> particles3(size);
  auto candsPtr = std::make_unique<GenParticleCollection>(size);
  auto barCodeVector = std::make_unique<vector<int>>(size);
  std::unique_ptr<math::XYZPointF> xyz0Ptr(new math::XYZPointF(0., 0., 0.));
  std::unique_ptr<float> t0Ptr(new float(0.f));
  reco::GenParticleRefProd ref = evt.getRefBeforePut<GenParticleCollection>();
  GenParticleCollection& cands = *candsPtr;
  size_t offset = 0;
  size_t suboffset = 0;

  IDto3Charge const& id2Charge = *runCache(evt.getRun().index());
  /// fill indices
  if (doSubEvent_ || useCF_) {
    for (size_t ipile = 0; ipile < npiles; ++ipile) {
      LogDebug("GenParticleProducer") << "mixed object ipile : " << ipile << endl;
      barcodes.clear();
      if (useCF_)
        mc = cfhepmcprod->getObject(ipile).GetEvent();

      //Look whether heavy ion/signal event
      bool isHI = false;
      const HepMC::HeavyIon* hi = mc->heavy_ion();
      if (hi && hi->Ncoll_hard() > 1)
        isHI = true;
      size_t num_particles = mc->particles_size();
      LogDebug("GenParticleProducer") << "num_particles : " << num_particles << endl;
      if (ipile == 0) {
        auto origin = (*mc->vertices_begin())->position();
        xyz0Ptr->SetXYZ(origin.x() * mmToCm, origin.y() * mmToCm, origin.z() * mmToCm);
        *t0Ptr = origin.t() * mmToNs;
      }
      fillIndices(mc, particles, *barCodeVector, offset, barcodes);
      // fill output collection and save association
      for (size_t ipar = offset; ipar < offset + num_particles; ++ipar) {
        const HepMC::GenParticle* part = particles[ipar];
        reco::GenParticle& cand = cands[ipar];
        // convert HepMC::GenParticle to new reco::GenParticle
        convertParticle(cand, part, id2Charge);
        cand.resetDaughters(ref.id());
      }

      for (size_t d = offset; d < offset + num_particles; ++d) {
        const HepMC::GenParticle* part = particles[d];
        const GenVertex* productionVertex = part->production_vertex();
        int sub_id = 0;
        if (productionVertex != nullptr) {
          sub_id = productionVertex->id();
          if (!isHI)
            sub_id = 0;
          // search barcode map and attach daughters
          fillDaughters(cands, part, ref, d, barcodes);
        } else {
          const GenVertex* endVertex = part->end_vertex();
          if (endVertex != nullptr)
            sub_id = endVertex->id();
          else
            throw cms::Exception("SubEventID")
                << "SubEvent not determined. Particle has no production and no end vertex!" << endl;
        }
        if (sub_id < 0)
          sub_id = 0;
        int new_id = sub_id + suboffset;
        GenParticleRef dref(ref, d);
        cands[d].setCollisionId(new_id);  // For new GenParticle
        LogDebug("VertexId") << "SubEvent offset 3 : " << suboffset;
      }
      int nsub = -2;
      if (isHI) {
        nsub = hi->Ncoll_hard() + 1;
        suboffset += nsub;
      } else {
        suboffset += 1;
      }
      offset += num_particles;
    }
  } else {  // Normal way from HepMCProduct or HepMC3Product

    if (ivhepmc == 2) {
      if (totalSize) {
        auto origin = (*mc->vertices_begin())->position();
        xyz0Ptr->SetXYZ(origin.x() * mmToCm, origin.y() * mmToCm, origin.z() * mmToCm);
        *t0Ptr = origin.t() * mmToNs;
        fillIndices(mc, particles, *barCodeVector, 0, barcodes);

        // fill output collection and save association
        for (size_t i = 0; i < particles.size(); ++i) {
          const HepMC::GenParticle* part = particles[i];
          reco::GenParticle& cand = cands[i];
          // convert HepMC::GenParticle to new reco::GenParticle
          convertParticle(cand, part, id2Charge);
          cand.resetDaughters(ref.id());
        }

        // fill references to daughters
        for (size_t d = 0; d < cands.size(); ++d) {
          const HepMC::GenParticle* part = particles[d];
          const GenVertex* productionVertex = part->production_vertex();
          // search barcode map and attach daughters
          if (productionVertex != nullptr) {
            fillDaughters(cands, part, ref, d, barcodes);
            cands[d].setCollisionId(0);
          }
        }
      }
    }

    if (ivhepmc == 3) {
      if (totalSize) {
        for (HepMC3::GenVertexPtr v : mc3->vertices()) {
          xyz0Ptr->SetXYZ((v->position()).x() * mmToCm, (v->position()).y() * mmToCm, (v->position()).z() * mmToCm);
          *t0Ptr = (v->position()).t() * mmToNs;
          break;
        }
        fillIndices3(mc3, particles3, *barCodeVector, 0, barcodes);

        // fill output collection and save association
        for (size_t i = 0; i < particles3.size(); ++i) {
          const HepMC3::GenParticle* part = particles3[i];
          reco::GenParticle& cand = cands[i];
          // convert HepMC::GenParticle to new reco::GenParticle
          convertParticle3(cand, part, id2Charge);
          cand.resetDaughters(ref.id());
        }

        // fill references to daughters
        for (size_t d = 0; d < cands.size(); ++d) {
          const HepMC3::GenParticle* part = particles3[d];
          // search barcode map and attach daughters
          if (part->production_vertex() != nullptr) {
            fillDaughters3(cands, part, ref, d, barcodes);
            cands[d].setCollisionId(0);
          }
        }
      }
    }
  }

  if (ivhepmc == 3)
    delete mc3;
  evt.put(std::move(candsPtr));
  if (saveBarCodes_)
    evt.put(std::move(barCodeVector));
  if (cfhepmcprod)
    delete cfhepmcprod;
  evt.put(std::move(xyz0Ptr), "xyz0");
  evt.put(std::move(t0Ptr), "t0");
}

bool GenParticleProducer::convertParticle(reco::GenParticle& cand,
                                          const HepMC::GenParticle* part,
                                          IDto3Charge const& id2Charge) const {
  Candidate::LorentzVector p4(part->momentum());
  int pdgId = part->pdg_id();
  cand.setThreeCharge(id2Charge.chargeTimesThree(pdgId));
  cand.setPdgId(pdgId);
  cand.setStatus(part->status());
  cand.setP4(p4);
  cand.setCollisionId(0);
  const GenVertex* v = part->production_vertex();
  if (v != nullptr) {
    ThreeVector vtx = v->point3d();
    Candidate::Point vertex(vtx.x() * mmToCm, vtx.y() * mmToCm, vtx.z() * mmToCm);
    cand.setVertex(vertex);
  } else {
    cand.setVertex(Candidate::Point(0, 0, 0));
  }
  mcTruthHelper_.fillGenStatusFlags(*part, cand.statusFlags());
  return true;
}

bool GenParticleProducer::convertParticle3(reco::GenParticle& cand,
                                           const HepMC3::GenParticle* part,
                                           IDto3Charge const& id2Charge) const {
  Candidate::LorentzVector p4(part->momentum());
  int pdgId = part->pdg_id();
  cand.setThreeCharge(id2Charge.chargeTimesThree(pdgId));
  cand.setPdgId(pdgId);
  cand.setStatus(part->status());
  cand.setP4(p4);
  cand.setCollisionId(0);
  if (part->production_vertex() != nullptr) {
    ThreeVector vtx((part->production_vertex()->position()).x(),
                    (part->production_vertex()->position()).y(),
                    (part->production_vertex()->position()).z());
    Candidate::Point vertex(vtx.x() * mmToCm, vtx.y() * mmToCm, vtx.z() * mmToCm);
    cand.setVertex(vertex);
  } else {
    cand.setVertex(Candidate::Point(0, 0, 0));
  }
  mcTruthHelper3_.fillGenStatusFlags(*part, cand.statusFlags());
  return true;
}

bool GenParticleProducer::fillDaughters(reco::GenParticleCollection& cands,
                                        const HepMC::GenParticle* part,
                                        reco::GenParticleRefProd const& ref,
                                        size_t index,
                                        std::unordered_map<int, size_t>& barcodes) const {
  const GenVertex* productionVertex = part->production_vertex();
  size_t numberOfMothers = productionVertex->particles_in_size();
  if (numberOfMothers > 0) {
    GenVertex::particles_in_const_iterator motherIt = productionVertex->particles_in_const_begin();
    for (; motherIt != productionVertex->particles_in_const_end(); motherIt++) {
      const HepMC::GenParticle* mother = *motherIt;
      size_t m = barcodes.find(mother->barcode())->second;
      cands[m].addDaughter(GenParticleRef(ref, index));
      cands[index].addMother(GenParticleRef(ref, m));
    }
  }
  return true;
}

bool GenParticleProducer::fillDaughters3(reco::GenParticleCollection& cands,
                                         const HepMC3::GenParticle* part,
                                         reco::GenParticleRefProd const& ref,
                                         size_t index,
                                         std::unordered_map<int, size_t>& barcodes) const {
  size_t numberOfMothers = (part->production_vertex())->particles_in_size();
  if (numberOfMothers > 0) {
    for (HepMC3::ConstGenParticlePtr mother : (part->production_vertex())->particles_in()) {
      size_t m = barcodes.find(mother->id())->second;
      cands[m].addDaughter(GenParticleRef(ref, index));
      cands[index].addMother(GenParticleRef(ref, m));
    }
  }
  return true;
}

bool GenParticleProducer::fillIndices(const HepMC::GenEvent* mc,
                                      vector<const HepMC::GenParticle*>& particles,
                                      vector<int>& barCodeVector,
                                      int offset,
                                      std::unordered_map<int, size_t>& barcodes) const {
  size_t idx = offset;
  HepMC::GenEvent::particle_const_iterator begin = mc->particles_begin(), end = mc->particles_end();
  for (HepMC::GenEvent::particle_const_iterator p = begin; p != end; ++p) {
    const HepMC::GenParticle* particle = *p;
    size_t barCode_this_event = particle->barcode();
    size_t barCode = barCode_this_event + offset;
    if (barcodes.find(barCode) != barcodes.end())
      throw cms::Exception("WrongReference") << "barcodes are duplicated! " << endl;
    particles[idx] = particle;
    barCodeVector[idx] = barCode;
    barcodes.insert(make_pair(barCode_this_event, idx++));
  }
  return true;
}

bool GenParticleProducer::fillIndices3(const HepMC3::GenEvent* mc,
                                       vector<const HepMC3::GenParticle*>& particles,
                                       vector<int>& barCodeVector,
                                       int offset,
                                       std::unordered_map<int, size_t>& barcodes) const {
  size_t idx = offset;
  for (const auto& p : mc->particles()) {
    const HepMC3::GenParticle* particle = p.get();
    size_t barCode_this_event = particle->id();
    size_t barCode = barCode_this_event + offset;
    if (barcodes.find(barCode) != barcodes.end())
      throw cms::Exception("WrongReference") << "barcodes are duplicated! " << endl;
    particles[idx] = particle;
    barCodeVector[idx] = barCode;
    barcodes.insert(make_pair(barCode_this_event, idx++));
  }
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenParticleProducer);
