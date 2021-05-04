/****************************************************************************
 * Authors:
 *   Jan Kašpar
 ****************************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"

#include <CLHEP/Random/RandGauss.h>

//----------------------------------------------------------------------------------------------------

class BeamDivergenceVtxGenerator : public edm::stream::EDProducer<> {
public:
  explicit BeamDivergenceVtxGenerator(const edm::ParameterSet &);
  ~BeamDivergenceVtxGenerator() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::EDGetTokenT<edm::HepMCProduct> sourceToken_;
  edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> beamParametersToken_;
  std::vector<edm::EDGetTokenT<reco::GenParticleCollection>> genParticleTokens_;

  bool simulateVertex_;
  bool simulateBeamDivergence_;

  struct SmearingParameters {
    double vtx_x, vtx_y, vtx_z, vtx_t;          // cm
    double bd_x_45, bd_y_45, bd_x_56, bd_y_56;  // rad
  };

  template <typename T>
  static HepMC::FourVector smearMomentum(const T &mom, const SmearingParameters &sp);

  void applySmearingHepMC(const SmearingParameters &sp, HepMC::GenEvent *genEvt);

  void addSmearedGenParticle(const reco::GenParticle &gp, const SmearingParameters &sp, HepMC::GenEvent *genEvt);
};

//----------------------------------------------------------------------------------------------------

BeamDivergenceVtxGenerator::BeamDivergenceVtxGenerator(const edm::ParameterSet &iConfig)
    : beamParametersToken_(esConsumes<CTPPSBeamParameters, CTPPSBeamParametersRcd>()),
      simulateVertex_(iConfig.getParameter<bool>("simulateVertex")),
      simulateBeamDivergence_(iConfig.getParameter<bool>("simulateBeamDivergence")) {
  const edm::InputTag tagSrcHepMC = iConfig.getParameter<edm::InputTag>("src");
  if (!tagSrcHepMC.label().empty())
    sourceToken_ = consumes<edm::HepMCProduct>(tagSrcHepMC);

  for (const auto &it : iConfig.getParameter<std::vector<edm::InputTag>>("srcGenParticle"))
    genParticleTokens_.push_back(consumes<reco::GenParticleCollection>(it));

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable())
    throw cms::Exception("Configuration")
        << "The BeamDivergenceVtxGenerator requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file. \n"
           "You must add the service\n"
           "in the configuration file or remove the modules that require it.";

  produces<edm::HepMCProduct>();
}

//----------------------------------------------------------------------------------------------------

void BeamDivergenceVtxGenerator::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("generator", "unsmeared"))
      ->setComment("input collection in HepMC format");

  desc.add<std::vector<edm::InputTag>>("srcGenParticle", std::vector<edm::InputTag>())
      ->setComment("input collections in GenParticle format");

  desc.add<bool>("simulateBeamDivergence", true)->setComment("account for the beam angular divergence?");
  desc.add<bool>("simulateVertex", true)->setComment("account for the vertex transverse smearing?");

  descriptions.add("beamDivergenceVtxGenerator", desc);
}

//----------------------------------------------------------------------------------------------------

void BeamDivergenceVtxGenerator::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get random engine
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine *rnd = &(rng->getEngine(iEvent.streamID()));

  // get conditions
  edm::ESHandle<CTPPSBeamParameters> hBeamParameters = iSetup.getHandle(beamParametersToken_);

  // get HepMC input (if given)
  HepMC::GenEvent *genEvt;
  if (sourceToken_.isUninitialized()) {
    genEvt = new HepMC::GenEvent();
  } else {
    edm::Handle<edm::HepMCProduct> hepUnsmearedMCEvt;
    iEvent.getByToken(sourceToken_, hepUnsmearedMCEvt);

    genEvt = new HepMC::GenEvent(*hepUnsmearedMCEvt->GetEvent());
  }

  // prepare output
  std::unique_ptr<edm::HepMCProduct> output(new edm::HepMCProduct(genEvt));

  // generate smearing parameters
  SmearingParameters sp;

  if (simulateVertex_) {
    // NB: the separtion between effective offsets in LHC sectors 45 and 56 cannot be applied, thus the values for 45 are used
    sp.vtx_x = hBeamParameters->getVtxOffsetX45() + CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getVtxStddevX();
    sp.vtx_y = hBeamParameters->getVtxOffsetY45() + CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getVtxStddevY();
    sp.vtx_z = hBeamParameters->getVtxOffsetZ45() + CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getVtxStddevZ();
    sp.vtx_t = hBeamParameters->getVtxOffsetT45() + CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getVtxStddevT();
  }

  if (simulateBeamDivergence_) {
    sp.bd_x_45 = CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getBeamDivergenceX45();
    sp.bd_x_56 = CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getBeamDivergenceX56();
    sp.bd_y_45 = CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getBeamDivergenceY45();
    sp.bd_y_56 = CLHEP::RandGauss::shoot(rnd) * hBeamParameters->getBeamDivergenceY56();
  }

  // apply smearing
  applySmearingHepMC(sp, genEvt);

  for (const auto &token : genParticleTokens_) {
    edm::Handle<reco::GenParticleCollection> hGPCollection;
    iEvent.getByToken(token, hGPCollection);

    for (const auto &gp : *hGPCollection)
      addSmearedGenParticle(gp, sp, genEvt);
  }

  // save output
  iEvent.put(std::move(output));
}

//----------------------------------------------------------------------------------------------------

template <typename T>
HepMC::FourVector BeamDivergenceVtxGenerator::smearMomentum(const T &mom, const SmearingParameters &sp) {
  // TODO: this is an oversimplified implemetation
  // the TOTEM smearing module should be taken as reference

  double th_x = mom.x() / mom.z();
  double th_y = mom.y() / mom.z();

  if (mom.z() > 0.) {
    th_x += sp.bd_x_45;
    th_y += sp.bd_y_45;
  } else {
    th_x += sp.bd_x_56;
    th_y += sp.bd_y_56;
  }

  // calculate consistent p_z component
  const double sign = (mom.z() > 0.) ? 1. : -1.;
  const double p = sqrt(mom.x() * mom.x() + mom.y() * mom.y() + mom.z() * mom.z());
  const double p_z = sign * p / sqrt(1. + th_x * th_x + th_y * th_y);

  return HepMC::FourVector(p_z * th_x, p_z * th_y, p_z, mom.e());
}

//----------------------------------------------------------------------------------------------------

void BeamDivergenceVtxGenerator::applySmearingHepMC(const SmearingParameters &sp, HepMC::GenEvent *genEvt) {
  if (simulateVertex_) {
    for (HepMC::GenEvent::vertex_iterator vit = genEvt->vertices_begin(); vit != genEvt->vertices_end(); ++vit) {
      const auto &pos = (*vit)->position();
      (*vit)->set_position(HepMC::FourVector(pos.x() + sp.vtx_x * 1E1,  // conversion: cm to mm
                                             pos.y() + sp.vtx_y * 1E1,
                                             pos.z() + sp.vtx_z * 1E1,
                                             pos.t() + sp.vtx_t * 1E1));
    }
  }

  if (simulateBeamDivergence_) {
    for (HepMC::GenEvent::particle_iterator part = genEvt->particles_begin(); part != genEvt->particles_end(); ++part)
      (*part)->set_momentum(smearMomentum((*part)->momentum(), sp));
  }
}

//----------------------------------------------------------------------------------------------------

void BeamDivergenceVtxGenerator::addSmearedGenParticle(const reco::GenParticle &gp,
                                                       const SmearingParameters &sp,
                                                       HepMC::GenEvent *genEvt) {
  // add vertex of the particle
  HepMC::GenVertex *vtx = new HepMC::GenVertex(HepMC::FourVector(
      (gp.vx() + sp.vtx_x) * 1E1,  // conversion: cm to mm
      (gp.vy() + sp.vtx_y) * 1E1,
      (gp.vz() + sp.vtx_z) * 1E1,
      (/*gp.vt()*/ +sp.vtx_t) * 1E1));  // TODO: GenParticle doesn't seem to have time component of the vertex
  genEvt->add_vertex(vtx);

  // add the particle itself
  HepMC::GenParticle *particle = new HepMC::GenParticle(smearMomentum(gp.p4(), sp), gp.pdgId(), gp.status());
  vtx->add_particle_out(particle);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(BeamDivergenceVtxGenerator);
