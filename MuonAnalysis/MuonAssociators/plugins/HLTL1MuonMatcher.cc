//
//

/**
  \class    pat::L1MuonMatcher L1MuonMatcher.h "MuonAnalysis/MuonAssociators/interface/L1MuonMatcher.h"
  \brief    Matcher of reconstructed objects to L1 Muons

  \author   Giovanni Petrucciani
  \version  $Id: HLTL1MuonMatcher.cc,v 1.3 2010/07/12 20:56:11 gpetrucc Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/View.h"

#include "MuonAnalysis/MuonAssociators/interface/L1MuonMatcherAlgo.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace pat {

  class HLTL1MuonMatcher : public edm::EDProducer {
  public:
    explicit HLTL1MuonMatcher(const edm::ParameterSet &iConfig);
    ~HLTL1MuonMatcher() override {}

    void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

    void beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;

    /// select L1s with patName_ and filterLabel_ (public, so it can be used by L1MuonMatcherAlgo)
    bool operator()(const pat::TriggerObjectStandAlone &l1) const {
      if (resolveAmbiguities_ && (std::find(lockedItems_.begin(), lockedItems_.end(), &l1) != lockedItems_.end()))
        return false;
      return selector_(l1);
    }

  private:
    typedef pat::TriggerObjectStandAlone PATPrimitive;
    typedef pat::TriggerObjectStandAloneCollection PATPrimitiveCollection;
    typedef pat::TriggerObjectStandAloneMatch PATTriggerAssociation;

    L1MuonMatcherAlgo matcher_;

    /// Tokens for input collections
    edm::EDGetTokenT<edm::View<reco::Candidate> > recoToken_;
    edm::EDGetTokenT<PATPrimitiveCollection> l1Token_;

    /// Select HLT objects.
    StringCutObjectSelector<PATPrimitive> selector_;
    bool resolveAmbiguities_;

    /// Labels to set as filter names in the output
    std::string labelProp_;

    /// Write out additional info as ValueMaps
    bool writeExtraInfo_;

    /// Store extra information in a ValueMap
    template <typename Hand, typename T>
    void storeExtraInfo(edm::Event &iEvent,
                        const Hand &handle,
                        const std::vector<T> &values,
                        const std::string &label) const;

    // for ambiguity resolution
    std::vector<const pat::TriggerObjectStandAlone *> lockedItems_;
  };

}  // namespace pat

pat::HLTL1MuonMatcher::HLTL1MuonMatcher(const edm::ParameterSet &iConfig)
    : matcher_(iConfig),
      recoToken_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("src"))),
      l1Token_(consumes<PATPrimitiveCollection>(iConfig.getParameter<edm::InputTag>("matched"))),
      selector_{iConfig.getParameter<std::string>("matchedCuts")},
      resolveAmbiguities_(iConfig.getParameter<bool>("resolveAmbiguities")),
      labelProp_(iConfig.getParameter<std::string>("setPropLabel")),
      writeExtraInfo_(iConfig.existsAs<bool>("writeExtraInfo") ? iConfig.getParameter<bool>("writeExtraInfo") : false) {
  produces<PATPrimitiveCollection>("propagatedReco");  // reco to muon station 2
  produces<PATTriggerAssociation>("propagatedReco");   // asso reco to propagated reco
  produces<PATTriggerAssociation>();                   // asso reco to l1
  if (writeExtraInfo_) {
    produces<edm::ValueMap<float> >("deltaR");
    produces<edm::ValueMap<float> >("deltaPhi");
  }
}

void pat::HLTL1MuonMatcher::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace std;

  Handle<View<reco::Candidate> > reco;
  Handle<PATPrimitiveCollection> l1s;

  iEvent.getByToken(recoToken_, reco);
  iEvent.getByToken(l1Token_, l1s);

  unique_ptr<PATPrimitiveCollection> propOut(new PATPrimitiveCollection());
  vector<int> propMatches(reco->size(), -1);
  vector<int> fullMatches(reco->size(), -1);
  vector<float> deltaRs(reco->size(), 999), deltaPhis(reco->size(), 999);
  lockedItems_.clear();
  for (int i = 0, n = reco->size(); i < n; ++i) {
    TrajectoryStateOnSurface propagated;
    const reco::Candidate &mu = (*reco)[i];
    int match = matcher_.matchGeneric(mu, *l1s, *this, deltaRs[i], deltaPhis[i], propagated);
    if (propagated.isValid()) {
      GlobalPoint pos = propagated.globalPosition();
      propMatches[i] = propOut->size();
      propOut->push_back(PATPrimitive(math::PtEtaPhiMLorentzVector(mu.pt(), pos.eta(), pos.phi(), mu.mass())));
      propOut->back().addFilterLabel(labelProp_);
      propOut->back().setCharge(mu.charge());
    }
    fullMatches[i] = match;
    if (match != -1) {
      lockedItems_.push_back(&(*l1s)[match]);
    }
  }
  lockedItems_.clear();

  OrphanHandle<PATPrimitiveCollection> propDone = iEvent.put(std::move(propOut), "propagatedReco");

  unique_ptr<PATTriggerAssociation> propAss(new PATTriggerAssociation(propDone));
  PATTriggerAssociation::Filler propFiller(*propAss);
  propFiller.insert(reco, propMatches.begin(), propMatches.end());
  propFiller.fill();
  iEvent.put(std::move(propAss), "propagatedReco");

  unique_ptr<PATTriggerAssociation> fullAss(new PATTriggerAssociation(l1s));
  PATTriggerAssociation::Filler fullFiller(*fullAss);
  fullFiller.insert(reco, fullMatches.begin(), fullMatches.end());
  fullFiller.fill();
  iEvent.put(std::move(fullAss));

  if (writeExtraInfo_) {
    storeExtraInfo(iEvent, reco, deltaRs, "deltaR");
    storeExtraInfo(iEvent, reco, deltaPhis, "deltaPhi");
  }
}

template <typename Hand, typename T>
void pat::HLTL1MuonMatcher::storeExtraInfo(edm::Event &iEvent,
                                           const Hand &handle,
                                           const std::vector<T> &values,
                                           const std::string &label) const {
  using namespace edm;
  using namespace std;
  unique_ptr<ValueMap<T> > valMap(new ValueMap<T>());
  typename edm::ValueMap<T>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap), label);
}

void pat::HLTL1MuonMatcher::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) { matcher_.init(iSetup); }

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(HLTL1MuonMatcher);
