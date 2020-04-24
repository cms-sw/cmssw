// -*- C++ -*-
//
// Package:    METProducers
// Class:      PFMETProducer
//
//

//____________________________________________________________________________||
#include "RecoMET/METProducers/interface/PFMETProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"

//____________________________________________________________________________||
namespace cms {

  //____________________________________________________________________________||
  PFMETProducer::PFMETProducer(const edm::ParameterSet& iConfig)
      : src_(iConfig.getParameter<edm::InputTag>("src")),
        inputToken_(consumes<edm::View<reco::Candidate>>(src_)),
        calculateSignificance_(iConfig.getParameter<bool>("calculateSignificance")),
        globalThreshold_(iConfig.getParameter<double>("globalThreshold")),
        applyWeight_(iConfig.getParameter<bool>("applyWeight")),
        weights_(nullptr) {
    if (applyWeight_) {
      edm::InputTag srcWeights = iConfig.getParameter<edm::InputTag>("srcWeights");
      if (srcWeights.label().empty())
        throw cms::Exception("InvalidInput") << "applyWeight set to True, but no weights given in PFMETProducer\n";
      if (srcWeights.label() == src_.label())
        edm::LogWarning("PFMETProducer")
            << "Particle and weights collection have the same label. You may be applying the same weights twice.\n";
      weightsToken_ = consumes<edm::ValueMap<float>>(srcWeights);
    }
    if (calculateSignificance_) {
      metSigAlgo_ = new metsig::METSignificance(iConfig);

      jetToken_ = mayConsume<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("srcJets"));
      std::vector<edm::InputTag> srcLeptonsTags = iConfig.getParameter<std::vector<edm::InputTag>>("srcLeptons");
      for (std::vector<edm::InputTag>::const_iterator it = srcLeptonsTags.begin(); it != srcLeptonsTags.end(); it++) {
        lepTokens_.push_back(mayConsume<edm::View<reco::Candidate>>(*it));
      }

      jetSFType_ = iConfig.getParameter<std::string>("srcJetSF");
      jetResPtType_ = iConfig.getParameter<std::string>("srcJetResPt");
      jetResPhiType_ = iConfig.getParameter<std::string>("srcJetResPhi");
      rhoToken_ = consumes<double>(iConfig.getParameter<edm::InputTag>("srcRho"));
    }

    std::string alias = iConfig.exists("alias") ? iConfig.getParameter<std::string>("alias") : "";

    produces<reco::PFMETCollection>().setBranchAlias(alias);
  }

  //____________________________________________________________________________||
  void PFMETProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
    edm::Handle<edm::View<reco::Candidate>> input;
    event.getByToken(inputToken_, input);

    if (applyWeight_)
      weights_ = &event.get(weightsToken_);

    METAlgo algo;
    CommonMETData commonMETdata;
    commonMETdata = algo.run(*input.product(), globalThreshold_, weights_);

    const math::XYZTLorentzVector p4(commonMETdata.mex, commonMETdata.mey, 0.0, commonMETdata.met);
    const math::XYZPoint vtx(0.0, 0.0, 0.0);

    PFSpecificAlgo pf;
    SpecificPFMETData specific;
    specific = pf.run(*input.product(), weights_);

    reco::PFMET pfmet(specific, commonMETdata.sumet, p4, vtx);
    pfmet.setIsWeighted(applyWeight_);

    if (calculateSignificance_) {
      reco::METCovMatrix sigcov = getMETCovMatrix(event, setup, input);
      pfmet.setSignificanceMatrix(sigcov);
    }

    auto pfmetcoll = std::make_unique<reco::PFMETCollection>();

    pfmetcoll->push_back(pfmet);
    event.put(std::move(pfmetcoll));
  }

  reco::METCovMatrix PFMETProducer::getMETCovMatrix(const edm::Event& event,
                                                    const edm::EventSetup& setup,
                                                    const edm::Handle<edm::View<reco::Candidate>>& candInput) const {
    // leptons
    std::vector<edm::Handle<reco::CandidateView>> leptons;
    for (std::vector<edm::EDGetTokenT<edm::View<reco::Candidate>>>::const_iterator srcLeptons_i = lepTokens_.begin();
         srcLeptons_i != lepTokens_.end();
         ++srcLeptons_i) {
      edm::Handle<reco::CandidateView> leptons_i;
      event.getByToken(*srcLeptons_i, leptons_i);
      leptons.push_back(leptons_i);
      /*
	  for ( reco::CandidateView::const_iterator lepton = leptons_i->begin();
		lepton != leptons_i->end(); ++lepton ) {
	    leptons.push_back(*lepton);
	  }
     */
    }

    // jets
    edm::Handle<edm::View<reco::Jet>> inputJets;
    event.getByToken(jetToken_, inputJets);

    JME::JetResolution resPtObj = JME::JetResolution::get(setup, jetResPtType_);
    JME::JetResolution resPhiObj = JME::JetResolution::get(setup, jetResPhiType_);
    JME::JetResolutionScaleFactor resSFObj = JME::JetResolutionScaleFactor::get(setup, jetSFType_);

    edm::Handle<double> rho;
    event.getByToken(rhoToken_, rho);

    //Compute the covariance matrix and fill it
    double sumPtUnclustered = 0;
    reco::METCovMatrix cov = metSigAlgo_->getCovariance(*inputJets,
                                                        leptons,
                                                        candInput,
                                                        *rho,
                                                        resPtObj,
                                                        resPhiObj,
                                                        resSFObj,
                                                        event.isRealData(),
                                                        sumPtUnclustered,
                                                        weights_);

    return cov;
  }

  void PFMETProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("particleFlow"));
    desc.add<double>("globalThreshold", 0.);
    desc.add<std::string>("alias", "@module_label");
    desc.add<bool>("calculateSignificance", false);
    desc.addOptional<edm::InputTag>("srcJets");
    desc.addOptional<std::vector<edm::InputTag>>("srcLeptons");
    desc.addOptional<std::string>("srcJetSF");
    desc.addOptional<std::string>("srcJetResPt");
    desc.addOptional<std::string>("srcJetResPhi");
    desc.addOptional<edm::InputTag>("srcRho");
    edm::ParameterSetDescription params;
    params.setAllowAnything();  // FIXME: This still needs to be defined in METSignficance
    desc.addOptional<edm::ParameterSetDescription>("parameters", params);
    edm::ParameterSetDescription desc1 = desc;
    edm::ParameterSetDescription desc2 = desc;
    desc1.add<bool>("applyWeight", false);
    desc1.add<edm::InputTag>("srcWeights", edm::InputTag(""));
    descriptions.add("pfMet", desc1);
    desc2.add<bool>("applyWeight", true);
    desc2.add<edm::InputTag>("srcWeights", edm::InputTag("puppiNoLep"));
    descriptions.add("pfMetPuppi", desc2);
  }

  //____________________________________________________________________________||
  DEFINE_FWK_MODULE(PFMETProducer);
}  // namespace cms

//____________________________________________________________________________||
