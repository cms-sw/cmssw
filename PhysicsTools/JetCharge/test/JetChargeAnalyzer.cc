#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"

#include "DataFormats/JetMatching/interface/JetFlavour.h"
#include "DataFormats/JetMatching/interface/JetFlavourMatching.h"

//#include "RecoBTag/MCTools/interface/JetFlavourIdentifier.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH1.h>
#include <string>

class JetChargeAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  struct JetRefCompare {
    inline bool operator()(const edm::RefToBase<reco::Jet>& j1, const edm::RefToBase<reco::Jet>& j2) const {
      return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key());
    }
  };

  typedef std::map<edm::RefToBase<reco::Jet>, unsigned int, JetRefCompare> FlavourMap;
  typedef reco::JetFloatAssociation::Container JetChargeCollection;

  explicit JetChargeAnalyzer(const edm::ParameterSet&);
  ~JetChargeAnalyzer() {}

  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  // physics stuff
  edm::EDGetTokenT<JetChargeCollection> srcToken_;
  edm::EDGetTokenT<reco::JetFlavourMatchingCollection> jetMCSrcToken_;
  double minET_;
  // plot stuff
  std::string dir_;
  TH1D* charge_[12];
};

const int pdgIds[12] = {0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 21};
const char* const pdgs[12] = {"?", "u", "-u", "d", "-d", "s", "-s", "c", "-c", "b", "-b", "g"};

JetChargeAnalyzer::JetChargeAnalyzer(const edm::ParameterSet& iConfig)
    : srcToken_(consumes<JetChargeCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      jetMCSrcToken_(consumes<reco::JetFlavourMatchingCollection>(iConfig.getParameter<edm::InputTag>("jetFlavour"))),
      minET_(iConfig.getParameter<double>("minET")),
      dir_(iConfig.getParameter<std::string>("dir")) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  TFileDirectory cwd = fs->mkdir(dir_.c_str());
  char buff[255], biff[255];
  for (int i = 0; i < 12; i++) {
    sprintf(biff, "jch_id_%s%d", (pdgIds[i] >= 0 ? "p" : "m"), abs(pdgIds[i]));
    sprintf(buff, "Jet charge for '%s' jets (pdgId %d) [ET > %f]", pdgs[i], pdgIds[i], minET_);
    charge_[i] = cwd.make<TH1D>(biff, buff, 22, -1.1, 1.1);
  }
}

void JetChargeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace reco;

  Handle<JetChargeCollection> hJC;
  iEvent.getByToken(srcToken_, hJC);
  //
  // get jet flavour via genparticles
  //
  edm::Handle<JetFlavourMatchingCollection> jetMC;
  FlavourMap flavours;

  iEvent.getByToken(jetMCSrcToken_, jetMC);
  for (JetFlavourMatchingCollection::const_iterator iter = jetMC->begin(); iter != jetMC->end(); iter++) {
    unsigned int fl = abs(iter->second.getFlavour());
    flavours.insert(FlavourMap::value_type(iter->first, fl));
  }
  //          for (JetChargeCollection::const_iterator it = hJC->begin(), ed = hJC->end(); it != ed; ++it) {
  for (unsigned int i = 0; i < hJC->size(); ++i) {
    const Jet& jet = *(hJC->key(i));
    if (jet.et() < minET_)
      continue;
    //        int id = jf_.identifyBasedOnPartons(jet).mainFlavour();

    edm::RefToBase<reco::Jet> jetr = hJC->key(i);
    int id = flavours[jetr];
    int k;
    for (k = 0; k < 12; k++) {
      if (id == pdgIds[k])
        break;
    }
    if (k == 12) {
      std::cerr << "Error: jet with flavour " << id << ". !??" << std::endl;
      continue;
    }
    charge_[k]->Fill(hJC->value(i));
  }
}

DEFINE_FWK_MODULE(JetChargeAnalyzer);
