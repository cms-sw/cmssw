// -*- C++ -*-
//
// Package:    PileupJetIdProducer
// Class:      PileupJetIdProducer
//
/**\class PileupJetIdProducer PileupJetIdProducer.cc CMGTools/PileupJetIdProducer/src/PileupJetIdProducer.cc

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Pasquale Musella,40 2-A12,+41227671706,
//         Created:  Wed Apr 18 15:48:47 CEST 2012
//
//

#include "RecoJets/JetProducers/plugins/PileupJetIdProducer.h"

#include <memory>

GBRForestsAndConstants::GBRForestsAndConstants(edm::ParameterSet const& iConfig)
    : runMvas_(iConfig.getParameter<bool>("runMvas")),
      produceJetIds_(iConfig.getParameter<bool>("produceJetIds")),
      inputIsCorrected_(iConfig.getParameter<bool>("inputIsCorrected")),
      applyJec_(iConfig.getParameter<bool>("applyJec")),
      jec_(iConfig.getParameter<std::string>("jec")),
      residualsFromTxt_(iConfig.getParameter<bool>("residualsFromTxt")),
      applyConstituentWeight_(false) {
  if (residualsFromTxt_) {
    residualsTxt_ = iConfig.getParameter<edm::FileInPath>("residualsTxt");
  }

  std::vector<edm::ParameterSet> algos = iConfig.getParameter<std::vector<edm::ParameterSet>>("algos");
  for (auto const& algoPset : algos) {
    vAlgoGBRForestsAndConstants_.emplace_back(algoPset, runMvas_);
  }

  if (!runMvas_) {
    assert(algos.size() == 1);
  }

  edm::InputTag srcConstituentWeights = iConfig.getParameter<edm::InputTag>("srcConstituentWeights");
  if (!srcConstituentWeights.label().empty()) {
    applyConstituentWeight_ = true;
  }
}

// ------------------------------------------------------------------------------------------
PileupJetIdProducer::PileupJetIdProducer(const edm::ParameterSet& iConfig, GBRForestsAndConstants const* globalCache) {
  if (globalCache->produceJetIds()) {
    produces<edm::ValueMap<StoredPileupJetIdentifier>>("");
  }
  for (auto const& algoGBRForestsAndConstants : globalCache->vAlgoGBRForestsAndConstants()) {
    std::string const& label = algoGBRForestsAndConstants.label();
    algos_.emplace_back(label, std::make_unique<PileupJetIdAlgo>(&algoGBRForestsAndConstants));
    if (globalCache->runMvas()) {
      produces<edm::ValueMap<float>>(label + "Discriminant");
      produces<edm::ValueMap<int>>(label + "Id");
    }
  }

  input_jet_token_ = consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"));
  input_vertex_token_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexes"));
  input_vm_pujetid_token_ =
      consumes<edm::ValueMap<StoredPileupJetIdentifier>>(iConfig.getParameter<edm::InputTag>("jetids"));
  input_rho_token_ = consumes<double>(iConfig.getParameter<edm::InputTag>("rho"));
  parameters_token_ = esConsumes(edm::ESInputTag("", globalCache->jec()));

  edm::InputTag srcConstituentWeights = iConfig.getParameter<edm::InputTag>("srcConstituentWeights");
  if (!srcConstituentWeights.label().empty()) {
    input_constituent_weights_token_ = consumes<edm::ValueMap<float>>(srcConstituentWeights);
  }
}

// ------------------------------------------------------------------------------------------
PileupJetIdProducer::~PileupJetIdProducer() {}

// ------------------------------------------------------------------------------------------
void PileupJetIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  GBRForestsAndConstants const* gc = globalCache();

  using namespace edm;
  using namespace std;
  using namespace reco;

  // Input jets
  Handle<View<Jet>> jetHandle;
  iEvent.getByToken(input_jet_token_, jetHandle);
  const View<Jet>& jets = *jetHandle;

  // Constituent weight (e.g PUPPI) Value Map
  edm::ValueMap<float> constituentWeights;
  if (!input_constituent_weights_token_.isUninitialized()) {
    constituentWeights = iEvent.get(input_constituent_weights_token_);
  }

  // input variables
  Handle<ValueMap<StoredPileupJetIdentifier>> vmap;
  if (!gc->produceJetIds()) {
    iEvent.getByToken(input_vm_pujetid_token_, vmap);
  }
  // rho
  edm::Handle<double> rhoH;
  double rho = 0.;

  // products
  vector<StoredPileupJetIdentifier> ids;
  map<string, vector<float>> mvas;
  map<string, vector<int>> idflags;

  const VertexCollection* vertexes = nullptr;
  VertexCollection::const_iterator vtx;
  if (gc->produceJetIds()) {
    // vertexes
    Handle<VertexCollection> vertexHandle;
    iEvent.getByToken(input_vertex_token_, vertexHandle);

    vertexes = vertexHandle.product();

    // require basic quality cuts on the vertexes
    vtx = vertexes->begin();
    while (vtx != vertexes->end() && (vtx->isFake() || vtx->ndof() < 4)) {
      ++vtx;
    }
    if (vtx == vertexes->end()) {
      vtx = vertexes->begin();
    }
  }

  // Loop over input jets
  bool ispat = true;
  for (unsigned int i = 0; i < jets.size(); ++i) {
    // Pick the first algo to compute the input variables
    auto algoi = algos_.begin();
    PileupJetIdAlgo* ialgo = algoi->second.get();

    const Jet& jet = jets.at(i);
    const pat::Jet* patjet = nullptr;
    if (ispat) {
      patjet = dynamic_cast<const pat::Jet*>(&jet);
      ispat = patjet != nullptr;
    }

    // Get jet energy correction
    float jec = 0.;
    if (gc->applyJec()) {
      // If haven't done it get rho from the event
      if (rho == 0.) {
        iEvent.getByToken(input_rho_token_, rhoH);
        rho = *rhoH;
      }
      // jet corrector
      if (not jecCor_) {
        initJetEnergyCorrector(iSetup, iEvent.isRealData());
      }
      if (ispat) {
        jecCor_->setJetPt(patjet->correctedJet(0).pt());
      } else {
        jecCor_->setJetPt(jet.pt());
      }
      jecCor_->setJetEta(jet.eta());
      jecCor_->setJetA(jet.jetArea());
      jecCor_->setRho(rho);
      jec = jecCor_->getCorrection();
    }
    // If it was requested AND the input is an uncorrected jet apply the JEC
    bool applyJec = gc->applyJec() && (ispat || !gc->inputIsCorrected());
    std::unique_ptr<reco::Jet> corrJet;

    if (applyJec) {
      float scale = jec;
      if (ispat) {
        corrJet = std::make_unique<pat::Jet>(patjet->correctedJet(0));
      } else {
        corrJet.reset(dynamic_cast<reco::Jet*>(jet.clone()));
      }
      corrJet->scaleEnergy(scale);
    }
    const reco::Jet* theJet = (applyJec ? corrJet.get() : &jet);

    PileupJetIdentifier puIdentifier;
    if (gc->produceJetIds()) {
      // Compute the input variables
      ////////////////////////////// added PUPPI weight Value Map
      puIdentifier = ialgo->computeIdVariables(
          theJet, jec, &(*vtx), *vertexes, rho, constituentWeights, gc->applyConstituentWeight());
      ids.push_back(puIdentifier);
    } else {
      // Or read it from the value map
      puIdentifier = (*vmap)[jets.refAt(i)];
      puIdentifier.jetPt(theJet->pt());  // make sure JEC is applied when computing the MVA
      puIdentifier.jetEta(theJet->eta());
      puIdentifier.jetPhi(theJet->phi());
      ialgo->set(puIdentifier);
      puIdentifier = ialgo->computeMva();
    }

    if (gc->runMvas()) {
      // Compute the MVA and WP
      mvas[algoi->first].push_back(puIdentifier.mva());
      idflags[algoi->first].push_back(puIdentifier.idFlag());
      for (++algoi; algoi != algos_.end(); ++algoi) {
        ialgo = algoi->second.get();
        ialgo->set(puIdentifier);
        PileupJetIdentifier id = ialgo->computeMva();
        mvas[algoi->first].push_back(id.mva());
        idflags[algoi->first].push_back(id.idFlag());
      }
    }
  }

  // Produce the output value maps
  if (gc->runMvas()) {
    for (const auto& ialgo : algos_) {
      // MVA
      vector<float>& mva = mvas[ialgo.first];
      auto mvaout = std::make_unique<ValueMap<float>>();
      ValueMap<float>::Filler mvafiller(*mvaout);
      mvafiller.insert(jetHandle, mva.begin(), mva.end());
      mvafiller.fill();
      iEvent.put(std::move(mvaout), ialgo.first + "Discriminant");

      // WP
      vector<int>& idflag = idflags[ialgo.first];
      auto idflagout = std::make_unique<ValueMap<int>>();
      ValueMap<int>::Filler idflagfiller(*idflagout);
      idflagfiller.insert(jetHandle, idflag.begin(), idflag.end());
      idflagfiller.fill();
      iEvent.put(std::move(idflagout), ialgo.first + "Id");
    }
  }
  // input variables
  if (gc->produceJetIds()) {
    assert(jetHandle->size() == ids.size());
    auto idsout = std::make_unique<ValueMap<StoredPileupJetIdentifier>>();
    ValueMap<StoredPileupJetIdentifier>::Filler idsfiller(*idsout);
    idsfiller.insert(jetHandle, ids.begin(), ids.end());
    idsfiller.fill();
    iEvent.put(std::move(idsout));
  }
}

// ------------------------------------------------------------------------------------------
void PileupJetIdProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// ------------------------------------------------------------------------------------------
void PileupJetIdProducer::initJetEnergyCorrector(const edm::EventSetup& iSetup, bool isData) {
  GBRForestsAndConstants const* gc = globalCache();

  //jet energy correction levels to apply on raw jet
  std::vector<std::string> jecLevels;
  jecLevels.push_back("L1FastJet");
  jecLevels.push_back("L2Relative");
  jecLevels.push_back("L3Absolute");
  if (isData && !gc->residualsFromTxt())
    jecLevels.push_back("L2L3Residual");

  //check the corrector parameters needed according to the correction levels
  auto const& parameters = iSetup.getData(parameters_token_);
  for (std::vector<std::string>::const_iterator ll = jecLevels.begin(); ll != jecLevels.end(); ++ll) {
    const JetCorrectorParameters& ip = parameters[*ll];
    jetCorPars_.push_back(ip);
  }
  if (isData && gc->residualsFromTxt()) {
    jetCorPars_.push_back(JetCorrectorParameters(gc->residualsTxt().fullPath()));
  }

  //instantiate the jet corrector
  jecCor_ = std::make_unique<FactorizedJetCorrector>(jetCorPars_);
}
//define this as a plug-in
DEFINE_FWK_MODULE(PileupJetIdProducer);
