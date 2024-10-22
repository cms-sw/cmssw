#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class GEDGsfElectronFinalizer : public edm::stream::EDProducer<> {
public:
  explicit GEDGsfElectronFinalizer(const edm::ParameterSet&);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<reco::GsfElectronCollection> previousGsfElectrons_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > tokenElectronIsoVals_;
  std::unique_ptr<ModifyObjectValueBase> gedRegression_;

  const edm::EDPutTokenT<reco::GsfElectronCollection> putToken_;
};

using edm::InputTag;
using edm::ValueMap;

using reco::GsfElectronCollection;

GEDGsfElectronFinalizer::GEDGsfElectronFinalizer(const edm::ParameterSet& cfg)
    : previousGsfElectrons_(consumes<GsfElectronCollection>(cfg.getParameter<InputTag>("previousGsfElectronsTag"))),
      pfCandidates_(consumes<reco::PFCandidateCollection>(cfg.getParameter<InputTag>("pfCandidatesTag"))),
      putToken_{produces<reco::GsfElectronCollection>()} {
  edm::ParameterSet pfIsoVals(cfg.getParameter<edm::ParameterSet>("pfIsolationValues"));

  tokenElectronIsoVals_ = {consumes<ValueMap<float> >(pfIsoVals.getParameter<InputTag>("pfSumChargedHadronPt")),
                           consumes<ValueMap<float> >(pfIsoVals.getParameter<InputTag>("pfSumPhotonEt")),
                           consumes<ValueMap<float> >(pfIsoVals.getParameter<InputTag>("pfSumNeutralHadronEt")),
                           consumes<ValueMap<float> >(pfIsoVals.getParameter<InputTag>("pfSumPUPt")),
                           consumes<ValueMap<float> >(pfIsoVals.getParameter<InputTag>("pfSumEcalClusterEt")),
                           consumes<ValueMap<float> >(pfIsoVals.getParameter<InputTag>("pfSumHcalClusterEt"))};

  if (cfg.existsAs<edm::ParameterSet>("regressionConfig")) {
    auto const& iconf = cfg.getParameterSet("regressionConfig");
    auto const& mname = iconf.getParameter<std::string>("modifierName");
    auto cc = consumesCollector();
    gedRegression_ = ModifyObjectValueFactory::get()->create(mname, iconf, cc);
  }
}

void GEDGsfElectronFinalizer::produce(edm::Event& event, const edm::EventSetup& setup) {
  // Output collection
  reco::GsfElectronCollection outputElectrons;

  if (gedRegression_) {
    gedRegression_->setEvent(event);
    gedRegression_->setEventContent(setup);
  }

  // read input collections
  // electrons
  auto gedElectronHandle = event.getHandle(previousGsfElectrons_);

  // PFCandidates
  auto pfCandidateHandle = event.getHandle(pfCandidates_);
  // value maps
  std::vector<edm::ValueMap<float> const*> isolationValueMaps(tokenElectronIsoVals_.size());

  for (unsigned i = 0; i < tokenElectronIsoVals_.size(); ++i) {
    isolationValueMaps[i] = &event.get(tokenElectronIsoVals_[i]);
  }

  // prepare a map of PFCandidates having a valid GsfTrackRef to save time
  std::map<reco::GsfTrackRef, const reco::PFCandidate*> gsfPFMap;
  for (auto const& pfCand : *pfCandidateHandle) {
    // First check that the GsfTrack is non null
    if (pfCand.gsfTrackRef().isNonnull()) {
      if (abs(pfCand.pdgId()) == 11)  // consider only the electrons
        gsfPFMap[pfCand.gsfTrackRef()] = &pfCand;
    }
  }

  // Now loop on the electrons
  unsigned nele = gedElectronHandle->size();
  for (unsigned iele = 0; iele < nele; ++iele) {
    reco::GsfElectronRef myElectronRef(gedElectronHandle, iele);

    reco::GsfElectron newElectron(*myElectronRef);
    reco::GsfElectron::PflowIsolationVariables isoVariables;
    isoVariables.sumChargedHadronPt = (*(isolationValueMaps)[0])[myElectronRef];
    isoVariables.sumPhotonEt = (*(isolationValueMaps)[1])[myElectronRef];
    isoVariables.sumNeutralHadronEt = (*(isolationValueMaps)[2])[myElectronRef];
    isoVariables.sumPUPt = (*(isolationValueMaps)[3])[myElectronRef];
    isoVariables.sumEcalClusterEt = (*(isolationValueMaps)[4])[myElectronRef];
    isoVariables.sumHcalClusterEt = (*(isolationValueMaps)[5])[myElectronRef];

    newElectron.setPfIsolationVariables(isoVariables);

    // now set a status if not already done (in GEDGsfElectronProducer.cc)
    //     std::cout << " previous status " << newElectron.mvaOutput().status << std::endl;
    if (newElectron.mvaOutput().status <= 0) {
      reco::GsfElectron::MvaOutput myMvaOutput(newElectron.mvaOutput());
      if (gsfPFMap.find(newElectron.gsfTrack()) != gsfPFMap.end()) {
        // it means that there is a PFCandidate with the same GsfTrack
        myMvaOutput.status = 3;  //as defined in PFCandidateEGammaExtra.h
        //this is currently fully redundant with mvaOutput.stats so candidate for removal
        newElectron.setPassPflowPreselection(true);
      } else {
        myMvaOutput.status = 4;  //
        //this is currently fully redundant with mvaOutput.stats so candidate for removal
        newElectron.setPassPflowPreselection(false);
      }
      newElectron.setMvaOutput(myMvaOutput);
    }

    if (gedRegression_) {
      gedRegression_->modifyObject(newElectron);
    }
    outputElectrons.push_back(newElectron);
  }

  event.emplace(putToken_, std::move(outputElectrons));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEDGsfElectronFinalizer);
