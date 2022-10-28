/** \class ShiftedParticleProducer
 *
 * Vary energy of electrons/muons/tau-jets by +/- 1 standard deviation,
 * in order to estimate resulting uncertainty on MET
 *
 * NOTE: energy scale uncertainties need to be specified in python config
 *
 * \author Matthieu marionneau ETH
 * 
 *
 *
 */

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include <TF2.h>

#include <string>
#include <vector>

class ShiftedParticleProducer : public edm::stream::EDProducer<> {
  typedef edm::View<reco::Candidate> CandidateView;

public:
  explicit ShiftedParticleProducer(const edm::ParameterSet& cfg);
  ~ShiftedParticleProducer() override;

private:
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

  double getUncShift(const reco::Candidate& originalParticle);

  std::string moduleLabel_;

  edm::EDGetTokenT<CandidateView> srcToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> weightsToken_;

  struct binningEntryType {
    binningEntryType(std::string uncertainty, std::string moduleLabel)
        : binSelection_(nullptr), binUncertainty_(uncertainty), energyDep_(false) {
      binUncFormula_ =
          std::make_unique<TF2>(std::string(moduleLabel).append("_uncFormula").c_str(), binUncertainty_.c_str());
    }
    binningEntryType(const edm::ParameterSet& cfg, std::string moduleLabel)
        : binSelection_(new StringCutObjectSelector<reco::Candidate>(cfg.getParameter<std::string>("binSelection"))),
          binUncertainty_(cfg.getParameter<std::string>("binUncertainty")),
          energyDep_(false) {
      binUncFormula_ =
          std::make_unique<TF2>(std::string(moduleLabel).append("_uncFormula").c_str(), binUncertainty_.c_str());
      if (cfg.exists("energyDependency")) {
        energyDep_ = cfg.getParameter<bool>("energyDependency");
      }
    }
    ~binningEntryType() {}
    std::unique_ptr<StringCutObjectSelector<reco::Candidate>> binSelection_;
    //double binUncertainty_;
    std::string binUncertainty_;
    std::unique_ptr<TF2> binUncFormula_;
    bool energyDep_;
  };
  std::vector<binningEntryType*> binning_;

  double shiftBy_;  // set to +1.0/-1.0 for up/down variation of energy scale
};

ShiftedParticleProducer::ShiftedParticleProducer(const edm::ParameterSet& cfg) {
  moduleLabel_ = cfg.getParameter<std::string>("@module_label");
  srcToken_ = consumes<CandidateView>(cfg.getParameter<edm::InputTag>("src"));
  shiftBy_ = cfg.getParameter<double>("shiftBy");
  edm::InputTag srcWeights = cfg.getParameter<edm::InputTag>("srcWeights");
  if (!srcWeights.label().empty())
    weightsToken_ = consumes<edm::ValueMap<float>>(srcWeights);

  if (cfg.exists("binning")) {
    typedef std::vector<edm::ParameterSet> vParameterSet;
    vParameterSet cfgBinning = cfg.getParameter<vParameterSet>("binning");
    for (vParameterSet::const_iterator cfgBinningEntry = cfgBinning.begin(); cfgBinningEntry != cfgBinning.end();
         ++cfgBinningEntry) {
      binning_.push_back(new binningEntryType(*cfgBinningEntry, moduleLabel_));
    }
  } else {
    std::string uncertainty = cfg.getParameter<std::string>("uncertainty");
    binning_.push_back(new binningEntryType(uncertainty, moduleLabel_));
  }

  produces<reco::CandidateCollection>();
}

ShiftedParticleProducer::~ShiftedParticleProducer() {
  for (std::vector<binningEntryType*>::const_iterator it = binning_.begin(); it != binning_.end(); ++it) {
    delete (*it);
  }
}

void ShiftedParticleProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<CandidateView> originalParticles;
  evt.getByToken(srcToken_, originalParticles);

  edm::Handle<edm::ValueMap<float>> weights;
  if (!weightsToken_.isUninitialized())
    evt.getByToken(weightsToken_, weights);

  auto shiftedParticles = std::make_unique<reco::CandidateCollection>();

  for (unsigned i = 0; i < originalParticles->size(); ++i) {
    float weight = 1.0;
    if (!weightsToken_.isUninitialized()) {
      edm::Ptr<reco::Candidate> particlePtr = originalParticles->ptrAt(i);
      while (!weights->contains(particlePtr.id()) && (particlePtr->numberOfSourceCandidatePtrs() > 0))
        particlePtr = particlePtr->sourceCandidatePtr(0);
      weight = (*weights)[particlePtr];
    }
    const reco::Candidate& originalParticle = originalParticles->at(i);
    reco::LeafCandidate weightedParticle(originalParticle);
    weightedParticle.setP4(originalParticle.p4() * weight);
    double uncertainty = getUncShift(weightedParticle);
    double shift = shiftBy_ * uncertainty;

    reco::Candidate::LorentzVector shiftedParticleP4 = originalParticle.p4();
    //leave 0*nan = 0
    if ((weight > 0) && (!(edm::isNotFinite(shift) && shiftedParticleP4.mag2() == 0)))
      shiftedParticleP4 *= (1. + shift);

    std::unique_ptr<reco::Candidate> shiftedParticle = std::make_unique<reco::LeafCandidate>(originalParticle);
    shiftedParticle->setP4(shiftedParticleP4);

    shiftedParticles->push_back(shiftedParticle.release());
  }

  evt.put(std::move(shiftedParticles));
}

double ShiftedParticleProducer::getUncShift(const reco::Candidate& originalParticle) {
  double valx = 0;
  double valy = 0;
  for (std::vector<binningEntryType*>::iterator binningEntry = binning_.begin(); binningEntry != binning_.end();
       ++binningEntry) {
    if ((!(*binningEntry)->binSelection_) || (*(*binningEntry)->binSelection_)(originalParticle)) {
      if ((*binningEntry)->energyDep_)
        valx = originalParticle.energy();
      else
        valx = originalParticle.pt();

      valy = originalParticle.eta();
      return (*binningEntry)->binUncFormula_->Eval(valx, valy);
    }
  }
  return 0;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ShiftedParticleProducer);
