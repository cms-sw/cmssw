#include "DQMOffline/RecoB/plugins/MiniAODTaggerAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"

MiniAODTaggerAnalyzer::MiniAODTaggerAnalyzer(const edm::ParameterSet& pSet)
    : jetToken_(consumes<std::vector<pat::Jet> >(pSet.getParameter<edm::InputTag>("JetTag"))),
      discrParameters_(pSet.getParameter<edm::ParameterSet>("parameters")),

      folder_(pSet.getParameter<std::string>("folder")),
      discrNumerator_(pSet.getParameter<vstring>("numerator")),
      discrDenominator_(pSet.getParameter<vstring>("denominator")),

      mclevel_(pSet.getParameter<int>("MClevel")),
      doCTagPlots_(pSet.getParameter<bool>("CTagPlots")),
      dodifferentialPlots_(pSet.getParameter<bool>("differentialPlots")),
      discrCut_(pSet.getParameter<double>("discrCut")),

      etaActive_(pSet.getParameter<bool>("etaActive")),
      etaMin_(pSet.getParameter<double>("etaMin")),
      etaMax_(pSet.getParameter<double>("etaMax")),
      ptActive_(pSet.getParameter<bool>("ptActive")),
      ptMin_(pSet.getParameter<double>("ptMin")),
      ptMax_(pSet.getParameter<double>("ptMax"))

{}

MiniAODTaggerAnalyzer::~MiniAODTaggerAnalyzer() {}

void MiniAODTaggerAnalyzer::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& es) {
  jetTagPlotter_ = std::make_unique<JetTagPlotter>(folder_,
                                                   EtaPtBin(etaActive_, etaMin_, etaMax_, ptActive_, ptMin_, ptMax_),
                                                   discrParameters_,
                                                   mclevel_,
                                                   false,
                                                   ibook,
                                                   doCTagPlots_,
                                                   dodifferentialPlots_,
                                                   discrCut_);
}

void MiniAODTaggerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<pat::Jet> > jetCollection;
  iEvent.getByToken(jetToken_, jetCollection);

  // Loop over the pat::Jets
  for (std::vector<pat::Jet>::const_iterator jet = jetCollection->begin(); jet != jetCollection->end(); ++jet) {
    // fill numerator
    float numerator = 0;
    for (const auto& discrLabel : discrNumerator_) {
      numerator += jet->bDiscriminator(discrLabel);
    }

    // fill denominator
    float denominator;
    if (discrDenominator_.empty()) {
      denominator = 1;  // no division performed
    } else {
      denominator = 0;

      for (const auto& discrLabel : discrDenominator_) {
        denominator += jet->bDiscriminator(discrLabel);
      }
    }

    const float jec = 1.;  // JEC not implemented!

    // only add to histograms when discriminator values are valid
    if (numerator >= 0 && denominator > 0) {
      reco::Jet recoJet = *jet;
      if (jetTagPlotter_->etaPtBin().inBin(recoJet, jec)) {
        jetTagPlotter_->analyzeTag(recoJet, jec, numerator / denominator, jet->partonFlavour());
      }
    }
  }

  // fill JetMultiplicity
  if (mclevel_ > 0) {
    jetTagPlotter_->analyzeTag(1.);
  } else {
    jetTagPlotter_->analyzeTag();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(MiniAODTaggerAnalyzer);
