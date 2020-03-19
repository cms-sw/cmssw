#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <iostream>
#include <map>

using namespace std;

class GenParticles2HepMCConverter : public edm::stream::EDProducer<> {
public:
  explicit GenParticles2HepMCConverter(const edm::ParameterSet& pset);
  ~GenParticles2HepMCConverter() override{};

  void beginRun(edm::Run const& iRun, edm::EventSetup const&) override;
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

private:
  edm::EDGetTokenT<reco::CandidateView> genParticlesToken_;
  edm::EDGetTokenT<GenEventInfoProduct> genEventInfoToken_;
  edm::EDGetTokenT<GenRunInfoProduct> genRunInfoToken_;
  edm::ESHandle<ParticleDataTable> pTable_;

  std::vector<int> signalParticlePdgIds_;
  const double cmEnergy_;
  HepMC::GenCrossSection xsec_;

private:
  inline HepMC::FourVector FourVector(const reco::Candidate::Point& point) {
    return HepMC::FourVector(10 * point.x(), 10 * point.y(), 10 * point.z(), 0);
  };

  inline HepMC::FourVector FourVector(const reco::Candidate::LorentzVector& lvec) {
    // Avoid negative mass, set minimum m^2 = 0
    return HepMC::FourVector(lvec.px(), lvec.py(), lvec.pz(), std::hypot(lvec.P(), std::max(0., lvec.mass())));
  };
};

GenParticles2HepMCConverter::GenParticles2HepMCConverter(const edm::ParameterSet& pset)
    // dummy value to set incident proton pz for particle gun samples
    : cmEnergy_(pset.getUntrackedParameter<double>("cmEnergy", 13000)) {
  genParticlesToken_ = consumes<reco::CandidateView>(pset.getParameter<edm::InputTag>("genParticles"));
  genEventInfoToken_ = consumes<GenEventInfoProduct>(pset.getParameter<edm::InputTag>("genEventInfo"));
  genRunInfoToken_ = consumes<GenRunInfoProduct, edm::InRun>(pset.getParameter<edm::InputTag>("genEventInfo"));
  signalParticlePdgIds_ = pset.getParameter<std::vector<int>>("signalParticlePdgIds");

  produces<edm::HepMCProduct>("unsmeared");
}

void GenParticles2HepMCConverter::beginRun(edm::Run const& iRun, edm::EventSetup const&) {
  edm::Handle<GenRunInfoProduct> genRunInfoHandle;
  iRun.getByToken(genRunInfoToken_, genRunInfoHandle);

  xsec_.set_cross_section(genRunInfoHandle->internalXSec().value(), genRunInfoHandle->internalXSec().error());
}

void GenParticles2HepMCConverter::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  edm::Handle<reco::CandidateView> genParticlesHandle;
  event.getByToken(genParticlesToken_, genParticlesHandle);

  edm::Handle<GenEventInfoProduct> genEventInfoHandle;
  event.getByToken(genEventInfoToken_, genEventInfoHandle);

  eventSetup.getData(pTable_);

  HepMC::GenEvent* hepmc_event = new HepMC::GenEvent();
  hepmc_event->set_event_number(event.id().event());
  hepmc_event->set_signal_process_id(genEventInfoHandle->signalProcessID());
  hepmc_event->set_event_scale(genEventInfoHandle->qScale());
  hepmc_event->set_alphaQED(genEventInfoHandle->alphaQED());
  hepmc_event->set_alphaQCD(genEventInfoHandle->alphaQCD());

  hepmc_event->weights() = genEventInfoHandle->weights();

  hepmc_event->set_cross_section(xsec_);

  // Set PDF
  const gen::PdfInfo* pdf = genEventInfoHandle->pdf();
  if (pdf != nullptr) {
    const int pdf_id1 = pdf->id.first, pdf_id2 = pdf->id.second;
    const double pdf_x1 = pdf->x.first, pdf_x2 = pdf->x.second;
    const double pdf_scalePDF = pdf->scalePDF;
    const double pdf_xPDF1 = pdf->xPDF.first, pdf_xPDF2 = pdf->xPDF.second;
    HepMC::PdfInfo hepmc_pdfInfo(pdf_id1, pdf_id2, pdf_x1, pdf_x2, pdf_scalePDF, pdf_xPDF1, pdf_xPDF2);
    hepmc_event->set_pdf_info(hepmc_pdfInfo);
  }

  // Prepare list of HepMC::GenParticles
  std::map<const reco::Candidate*, HepMC::GenParticle*> genCandToHepMCMap;
  HepMC::GenParticle *hepmc_parton1 = nullptr, *hepmc_parton2 = nullptr;
  std::vector<HepMC::GenParticle*> hepmc_particles;
  const reco::Candidate *parton1 = nullptr, *parton2 = nullptr;
  for (unsigned int i = 0, n = genParticlesHandle->size(); i < n; ++i) {
    const reco::Candidate* p = &genParticlesHandle->at(i);
    HepMC::GenParticle* hepmc_particle = new HepMC::GenParticle(FourVector(p->p4()), p->pdgId(), p->status());
    hepmc_particle->suggest_barcode(i + 1);

    // Assign particle's generated mass from the standard particle data table
    double particleMass;
    if (pTable_->particle(p->pdgId()))
      particleMass = pTable_->particle(p->pdgId())->mass();
    else
      particleMass = p->mass();

    hepmc_particle->set_generated_mass(particleMass);

    hepmc_particles.push_back(hepmc_particle);
    genCandToHepMCMap[p] = hepmc_particle;

    // Find incident proton pair
    if (p->mother() == nullptr and std::abs(p->eta()) > 5 and std::abs(p->pz()) > 1000) {
      if (!parton1 and p->pz() > 0) {
        parton1 = p;
        hepmc_parton1 = hepmc_particle;
      } else if (!parton2 and p->pz() < 0) {
        parton2 = p;
        hepmc_parton2 = hepmc_particle;
      }
    }
  }

  HepMC::GenVertex* vertex1 = nullptr;
  HepMC::GenVertex* vertex2 = nullptr;
  if (parton1 == nullptr || parton2 == nullptr) {
    // Particle gun samples do not have incident partons. Put dummy incident particle and prod vertex
    // Note: leave parton1 and parton2 as nullptr since it is not used anymore after creating hepmc_parton1 and 2
    const reco::Candidate::LorentzVector nullP4(0, 0, 0, 0);
    const reco::Candidate::LorentzVector beamP4(0, 0, cmEnergy_ / 2, cmEnergy_ / 2);
    vertex1 = new HepMC::GenVertex(FourVector(nullP4));
    vertex2 = new HepMC::GenVertex(FourVector(nullP4));
    hepmc_parton1 = new HepMC::GenParticle(FourVector(+beamP4), 2212, 4);
    hepmc_parton2 = new HepMC::GenParticle(FourVector(-beamP4), 2212, 4);
  } else {
    // Put incident beam particles : proton -> parton vertex
    vertex1 = new HepMC::GenVertex(FourVector(parton1->vertex()));
    vertex2 = new HepMC::GenVertex(FourVector(parton2->vertex()));
  }
  hepmc_event->add_vertex(vertex1);
  hepmc_event->add_vertex(vertex2);
  vertex1->add_particle_in(hepmc_parton1);
  vertex2->add_particle_in(hepmc_parton2);
  //hepmc_event->set_beam_particles(hepmc_parton1, hepmc_parton2);

  // Prepare vertex list
  typedef std::map<const reco::Candidate*, HepMC::GenVertex*> ParticleToVertexMap;
  ParticleToVertexMap particleToVertexMap;
  particleToVertexMap[parton1] = vertex1;
  particleToVertexMap[parton2] = vertex2;
  for (unsigned int i = 0, n = genParticlesHandle->size(); i < n; ++i) {
    const reco::Candidate* p = &genParticlesHandle->at(i);
    if (p == parton1 or p == parton2)
      continue;

    // Connect mother-daughters for the other cases
    for (unsigned int j = 0, nMothers = p->numberOfMothers(); j < nMothers; ++j) {
      // Mother-daughter hierarchy defines vertex
      const reco::Candidate* elder = p->mother(j)->daughter(0);
      HepMC::GenVertex* vertex;
      if (particleToVertexMap.find(elder) == particleToVertexMap.end()) {
        vertex = new HepMC::GenVertex(FourVector(elder->vertex()));
        hepmc_event->add_vertex(vertex);
        particleToVertexMap[elder] = vertex;
      } else {
        vertex = particleToVertexMap[elder];
      }

      // Vertex is found. Now connect each other
      const reco::Candidate* mother = p->mother(j);
      vertex->add_particle_in(genCandToHepMCMap[mother]);
      vertex->add_particle_out(hepmc_particles[i]);
    }
  }

  // Finalize HepMC event record
  bool hasSignalVertex = false;
  if (!signalParticlePdgIds_.empty()) {
    // Loop over all vertices to assign the signal vertex, decaying to a signal particle
    for (auto v = hepmc_event->vertices_begin(); v != hepmc_event->vertices_end(); ++v) {
      for (auto p = (*v)->particles_begin(HepMC::children); p != (*v)->particles_end(HepMC::children); ++p) {
        const int pdgId = (*p)->pdg_id();
        if (std::find(signalParticlePdgIds_.begin(), signalParticlePdgIds_.end(), pdgId) !=
            signalParticlePdgIds_.end()) {
          hepmc_event->set_signal_process_vertex(*v);
          hasSignalVertex = true;
          break;
        }
      }
      if (hasSignalVertex)
        break;
    }
  }
  // Set the default signal vertex if still not set
  if (!hasSignalVertex)
    hepmc_event->set_signal_process_vertex(*(vertex1->vertices_begin()));

  std::unique_ptr<edm::HepMCProduct> hepmc_product(new edm::HepMCProduct());
  hepmc_product->addHepMCData(hepmc_event);
  event.put(std::move(hepmc_product), "unsmeared");
}

DEFINE_FWK_MODULE(GenParticles2HepMCConverter);
