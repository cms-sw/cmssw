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
//#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <iostream>
#include <map>

using namespace std;

class GenParticles2HepMCConverter : public edm::stream::EDProducer<>
{
public:
  explicit GenParticles2HepMCConverter(const edm::ParameterSet& pset);
  ~GenParticles2HepMCConverter() {};

  //void beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) override;
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

private:
//  edm::InputTag lheEventToken_;
  edm::EDGetTokenT<reco::CandidateView> genParticlesToken_;
//  edm::InputTag genRunInfoToken_;
  edm::EDGetTokenT<GenEventInfoProduct> genEventInfoToken_;
  edm::ESHandle<ParticleDataTable> pTable_;

private:
  inline HepMC::FourVector FourVector(const reco::Candidate::Point& point)
  {
    return HepMC::FourVector(10*point.x(), 10*point.y(), 10*point.z(), 0);
  };

  inline HepMC::FourVector FourVector(const reco::Candidate::LorentzVector& lvec)
  {
    // Avoid negative mass, set minimum m^2 = 0
    return HepMC::FourVector(lvec.px(), lvec.py(), lvec.pz(), std::hypot(lvec.P(), std::max(0., lvec.mass())));
  };


};

GenParticles2HepMCConverter::GenParticles2HepMCConverter(const edm::ParameterSet& pset)
{
//  lheEventToken_ = pset.getParameter<edm::InputTag>("lheEvent");
  genParticlesToken_ = consumes<reco::CandidateView>(pset.getParameter<edm::InputTag>("genParticles"));
  //genRunInfoToken_ = pset.getParameter<edm::InputTag>("genRunInfo");
  genEventInfoToken_ = consumes<GenEventInfoProduct>(pset.getParameter<edm::InputTag>("genEventInfo"));

  produces<edm::HepMCProduct>("unsmeared");
}

//void GenParticles2HepMCConverter::beginRun(edm::Run& run, const edm::EventSetup& eventSetup)
//{
  //edm::Handle<GenRunInfoProduct> genRunInfoHandle;
  //event.getByToken(genRunInfoToken_, genRunInfoHandle);
  // const double xsecIn = genRunInfoHandle->internalXSec().value();
  // const double xsecInErr = genRunInfoHandle->internalXSec().error();
  // const double xsecLO = genRunInfoHandle->externalXSecLO().value();
  // const double xsecLOErr = genRunInfoHandle->externalXSecLO().error();
  // const double xsecNLO = genRunInfoHandle->externalXSecNLO().value();
  // const double xsecNLOErr = genRunInfoHandle->externalXSecNLO().error();
//}

void GenParticles2HepMCConverter::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{
//  edm::Handle<LHEEventProduct> lheEventHandle;
//  event.getByToken(lheEventToken_, lheEventHandle);

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

  // Set PDF
  const gen::PdfInfo* pdf = genEventInfoHandle->pdf();
  const int pdf_id1 = pdf->id.first, pdf_id2 = pdf->id.second;
  const double pdf_x1 = pdf->x.first, pdf_x2 = pdf->x.second;
  const double pdf_scalePDF = pdf->scalePDF;
  const double pdf_xPDF1 = pdf->xPDF.first, pdf_xPDF2 = pdf->xPDF.second;
  HepMC::PdfInfo hepmc_pdfInfo(pdf_id1, pdf_id2, pdf_x1, pdf_x2, pdf_scalePDF, pdf_xPDF1, pdf_xPDF2);
  hepmc_event->set_pdf_info(hepmc_pdfInfo);

  // Load LHE
//  const lhef::HEPEUP& lheEvent = lheEventHandle->hepeup();
//  std::vector<int> lhe_meIndex; // Particle indices with preserved mass, status=2
//  for ( int i=0, n=lheEvent.ISTUP.size(); i<n; ++i )
//  {
//    if ( lheEvent.ISTUP[i] == 2 ) lhe_meIndex.push_back(i);
//  }

  // Prepare list of HepMC::GenParticles
  std::map<const reco::Candidate*, HepMC::GenParticle*> genCandToHepMCMap;
  std::vector<HepMC::GenParticle*> hepmc_particles;
  for ( unsigned int i=0, n=genParticlesHandle->size(); i<n; ++i )
  {
    const reco::Candidate* p = &genParticlesHandle->at(i);
    HepMC::GenParticle* hepmc_particle = new HepMC::GenParticle(FourVector(p->p4()), p->pdgId(), p->status());
    hepmc_particle->suggest_barcode(i+1);

    // Assign particle's generated mass from the standard particle data table
    double particleMass;
    if ( pTable_->particle(p->pdgId()) ) particleMass = pTable_->particle(p->pdgId())->mass();
    else particleMass = p->mass();
//    // Re-assign generated mass from LHE, find particle among the LHE
//    for ( unsigned int j=0, m=lhe_meIndex.size(); j<m; ++j )
//    {
//      const unsigned int lheIndex = lhe_meIndex[j];
//      if ( p->pdgId() != lheEvent.IDUP[lheIndex] ) continue;
//
//      const lhef::HEPEUP::FiveVector& vp = lheEvent.PUP[lheIndex];
//      if ( std::abs(vp[0] - p->px()) > 1e-7 or std::abs(vp[1] - p->py()) > 1e-7 ) continue;
//      if ( std::abs(vp[2] - p->pz()) > 1e-7 or std::abs(vp[3] - p->energy()) > 1e-7 ) continue;
//
//      particleMass = vp[4];
//      break;
//    }
    hepmc_particle->set_generated_mass(particleMass);

    hepmc_particles.push_back(hepmc_particle);
    genCandToHepMCMap[p] = hepmc_particle;
  }

  // Put incident beam particles : proton -> parton vertex
  const reco::Candidate* parton1 = genParticlesHandle->at(0).daughter(0);
  const reco::Candidate* parton2 = genParticlesHandle->at(1).daughter(0);
  HepMC::GenVertex* vertex1 = new HepMC::GenVertex(FourVector(parton1->vertex()));
  HepMC::GenVertex* vertex2 = new HepMC::GenVertex(FourVector(parton2->vertex()));
  hepmc_event->add_vertex(vertex1);
  hepmc_event->add_vertex(vertex2);
  //hepmc_particles[0]->set_status(4);
  //hepmc_particles[1]->set_status(4);
  vertex1->add_particle_in(hepmc_particles[0]);
  vertex2->add_particle_in(hepmc_particles[1]);
  hepmc_event->set_beam_particles(hepmc_particles[0], hepmc_particles[1]);

  // Prepare vertex list
  typedef std::map<const reco::Candidate*, HepMC::GenVertex*> ParticleToVertexMap;
  ParticleToVertexMap particleToVertexMap;
  particleToVertexMap[parton1] = vertex1;
  particleToVertexMap[parton2] = vertex2;
  for ( unsigned int i=2, n=genParticlesHandle->size(); i<n; ++i )
  {
    const reco::Candidate* p = &genParticlesHandle->at(i);

    // Connect mother-daughters for the other cases
    for ( unsigned int j=0, nMothers=p->numberOfMothers(); j<nMothers; ++j )
    {
      // Mother-daughter hierarchy defines vertex
      const reco::Candidate* elder = p->mother(j)->daughter(0);
      HepMC::GenVertex* vertex;
      if ( particleToVertexMap.find(elder) == particleToVertexMap.end() )
      {
        vertex = new HepMC::GenVertex(FourVector(elder->vertex()));
        hepmc_event->add_vertex(vertex);
        particleToVertexMap[elder] = vertex;
      }
      else
      {
        vertex = particleToVertexMap[elder];
      }

      // Vertex is found. Now connect each other
      const reco::Candidate* mother = p->mother(j);
      vertex->add_particle_in(genCandToHepMCMap[mother]);
      vertex->add_particle_out(hepmc_particles[i]);
    }
  }

  // Finalize HepMC event record
  hepmc_event->set_signal_process_vertex(*(vertex1->vertices_begin()));

  std::auto_ptr<edm::HepMCProduct> hepmc_product(new edm::HepMCProduct());
  hepmc_product->addHepMCData(hepmc_event);
  event.put(hepmc_product, "unsmeared");

}

DEFINE_FWK_MODULE(GenParticles2HepMCConverter);
