#include "RecoJets/JetProducers/interface/ECFAdder.h"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

ECFAdder::ECFAdder(const edm::ParameterSet& iConfig)
    : src_(iConfig.getParameter<edm::InputTag>("src")),
      src_token_(consumes<edm::View<reco::Jet>>(src_)),
      Njets_(iConfig.getParameter<std::vector<unsigned>>("Njets")),
      cuts_(iConfig.getParameter<std::vector<std::string>>("cuts")),
      ecftype_(iConfig.getParameter<std::string>("ecftype")),
      alpha_(iConfig.getParameter<double>("alpha")),
      beta_(iConfig.getParameter<double>("beta")) {
  if (cuts_.size() != Njets_.size()) {
    throw cms::Exception("ConfigurationError") << "cuts and Njets must be the same size in ECFAdder" << std::endl;
  }

  edm::InputTag srcWeights = iConfig.getParameter<edm::InputTag>("srcWeights");
  if (!srcWeights.label().empty())
    input_weights_token_ = consumes<edm::ValueMap<float>>(srcWeights);

  for (std::vector<unsigned>::const_iterator n = Njets_.begin(); n != Njets_.end(); ++n) {
    std::ostringstream ecfN_str;
    std::shared_ptr<fastjet::FunctionOfPseudoJet<double>> pfunc;

    if (ecftype_ == "ECF" || ecftype_.empty()) {
      ecfN_str << "ecf" << *n;
      pfunc.reset(new fastjet::contrib::EnergyCorrelator(*n, beta_, fastjet::contrib::EnergyCorrelator::pt_R));
    } else if (ecftype_ == "C") {
      ecfN_str << "ecfC" << *n;
      pfunc.reset(new fastjet::contrib::EnergyCorrelatorCseries(*n, beta_, fastjet::contrib::EnergyCorrelator::pt_R));
    } else if (ecftype_ == "D") {
      ecfN_str << "ecfD" << *n;
      pfunc.reset(
          new fastjet::contrib::EnergyCorrelatorGeneralizedD2(alpha_, beta_, fastjet::contrib::EnergyCorrelator::pt_R));
    } else if (ecftype_ == "N") {
      ecfN_str << "ecfN" << *n;
      pfunc.reset(new fastjet::contrib::EnergyCorrelatorNseries(*n, beta_, fastjet::contrib::EnergyCorrelator::pt_R));
    } else if (ecftype_ == "M") {
      ecfN_str << "ecfM" << *n;
      pfunc.reset(new fastjet::contrib::EnergyCorrelatorMseries(*n, beta_, fastjet::contrib::EnergyCorrelator::pt_R));
    } else if (ecftype_ == "U") {
      ecfN_str << "ecfU" << *n;
      pfunc.reset(new fastjet::contrib::EnergyCorrelatorUseries(*n, beta_, fastjet::contrib::EnergyCorrelator::pt_R));
    }
    variables_.push_back(ecfN_str.str());
    produces<edm::ValueMap<float>>(ecfN_str.str());
    routine_.push_back(pfunc);

    selectors_.push_back(StringCutObjectSelector<reco::Jet>(cuts_[n - Njets_.begin()]));
  }
}

void ECFAdder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // read input collection
  edm::Handle<edm::View<reco::Jet>> jets;
  iEvent.getByToken(src_token_, jets);

  // Get Weights Collection
  if (!input_weights_token_.isUninitialized())
    weightsHandle_ = &iEvent.get(input_weights_token_);

  unsigned i = 0;
  for (std::vector<unsigned>::const_iterator n = Njets_.begin(); n != Njets_.end(); ++n) {
    // prepare room for output
    std::vector<float> ecfN;
    ecfN.reserve(jets->size());

    for (typename edm::View<reco::Jet>::const_iterator jetIt = jets->begin(); jetIt != jets->end(); ++jetIt) {
      edm::Ptr<reco::Jet> jetPtr = jets->ptrAt(jetIt - jets->begin());

      float t = -1.0;
      if (selectors_[n - Njets_.begin()](*jetIt))
        t = getECF(i, jetPtr);

      ecfN.push_back(t);
    }

    auto outT = std::make_unique<edm::ValueMap<float>>();
    edm::ValueMap<float>::Filler fillerT(*outT);
    fillerT.insert(jets, ecfN.begin(), ecfN.end());
    fillerT.fill();

    iEvent.put(std::move(outT), variables_[i]);
    ++i;
  }
}

float ECFAdder::getECF(unsigned index, const edm::Ptr<reco::Jet>& object) const {
  std::vector<fastjet::PseudoJet> fjParticles;
  for (unsigned k = 0; k < object->numberOfDaughters(); ++k) {
    const reco::CandidatePtr& dp = object->daughterPtr(k);
    if (dp.isNonnull() && dp.isAvailable()) {
      // Here, the daughters are the "end" node, so this is a PFJet
      if (dp->numberOfDaughters() == 0) {
        if (object->isWeighted()) {
          if (input_weights_token_.isUninitialized())
            throw cms::Exception("MissingConstituentWeight")
                << "ECFAdder: No weights (e.g. PUPPI) given for weighted jet collection" << std::endl;
          float w = (*weightsHandle_)[dp];
          fjParticles.push_back(fastjet::PseudoJet(dp->px() * w, dp->py() * w, dp->pz() * w, dp->energy() * w));
        } else
          fjParticles.push_back(fastjet::PseudoJet(dp->px(), dp->py(), dp->pz(), dp->energy()));
      } else {  // Otherwise, this is a BasicJet, so you need to descend further.
        auto subjet = dynamic_cast<reco::Jet const*>(dp.get());
        for (unsigned l = 0; l < subjet->numberOfDaughters(); ++l) {
          if (subjet != nullptr) {
            const reco::CandidatePtr& ddp = subjet->daughterPtr(l);
            if (subjet->isWeighted()) {
              if (input_weights_token_.isUninitialized())
                throw cms::Exception("MissingConstituentWeight")
                    << "ECFAdder: No weights (e.g. PUPPI) given for weighted jet collection" << std::endl;
              float w = (*weightsHandle_)[ddp];
              fjParticles.push_back(fastjet::PseudoJet(ddp->px() * w, ddp->py() * w, ddp->pz() * w, ddp->energy() * w));
            } else
              fjParticles.push_back(fastjet::PseudoJet(ddp->px(), ddp->py(), ddp->pz(), ddp->energy()));
          } else {
            edm::LogWarning("MissingJetConstituent") << "BasicJet constituent required for ECF computation is missing!";
          }
        }
      }  // end if basic jet
    }    // end if daughter pointer is nonnull and available
    else
      edm::LogWarning("MissingJetConstituent") << "Jet constituent required for ECF computation is missing!";
  }
  if (fjParticles.size() > Njets_[index]) {
    return routine_[index]->result(join(fjParticles));
  } else {
    return -1.0;
  }
}

// ParameterSet description for module
void ECFAdder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("Energy Correlation Functions adder");

  iDesc.add<edm::InputTag>("src", edm::InputTag("no default"))->setComment("input collection");
  iDesc.add<std::vector<unsigned>>("Njets", {1, 2, 3})->setComment("Number of jets to emulate");
  iDesc.add<std::vector<std::string>>("cuts", {"", "", ""})
      ->setComment("Jet selections for each N value. Size must match Njets.");
  iDesc.add<double>("alpha", 1.0)->setComment("alpha factor, only valid for N2");
  iDesc.add<double>("beta", 1.0)->setComment("angularity factor");
  iDesc.add<std::string>("ecftype", "")->setComment("ECF type: ECF or empty; C; D; N; M; U;");
  iDesc.add<edm::InputTag>("srcWeights", edm::InputTag("puppi"));
  descriptions.add("ECFAdder", iDesc);
}

DEFINE_FWK_MODULE(ECFAdder);
