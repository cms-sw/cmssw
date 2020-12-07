/**\class Type1PFMET
\brief Computes the Type-1 corrections for pfMET. A specific version of the Type1MET class from the JetMETCorrections/Type1MET package.

\todo Unify with the Type1MET class from the JetMETCorrections/Type1MET package

\author Michal Bluj
\date   February 2009
*/

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <memory>
#include <cstring>

class Type1PFMET : public edm::EDProducer {
public:
  explicit Type1PFMET(const edm::ParameterSet&);
  explicit Type1PFMET();
  ~Type1PFMET() override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<reco::METCollection> tokenUncorMet;
  edm::EDGetTokenT<reco::PFJetCollection> tokenUncorJets;
  edm::EDGetTokenT<reco::JetCorrector> correctorToken;
  double jetPTthreshold;
  double jetEMfracLimit;
  double jetMufracLimit;
  void run(const reco::METCollection& uncorMET,
           const reco::JetCorrector& corrector,
           const reco::PFJetCollection& uncorJet,
           double jetPTthreshold,
           double jetEMfracLimit,
           double jetMufracLimit,
           reco::METCollection* corMET);
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Type1PFMET);

void Type1PFMET::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfType1MET
  // Type-1 met corrections (AK4PFJets)
  // remember about including ES producer definition e.g. JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputUncorJetsTag", edm::InputTag("ak4PFJets"));
  desc.add<double>("jetEMfracLimit", 0.95);  // to remove electron which give rise to jets
  desc.add<double>("jetMufracLimit", 0.95);  // to remove muon which give rise to jets
  desc.add<std::string>("metType", "PFMET");
  desc.add<double>("jetPTthreshold", 20.0);
  // pfMET should be not corrected for HF 0.7
  desc.add<edm::InputTag>("inputUncorMetLabel", edm::InputTag("pfMET"));
  desc.add<edm::InputTag>("corrector", edm::InputTag("ak4PFL2L3Corrector"));
  descriptions.add("pfType1MET", desc);
}

using namespace reco;

// PRODUCER CONSTRUCTORS ------------------------------------------
Type1PFMET::Type1PFMET(const edm::ParameterSet& iConfig) {
  tokenUncorMet = consumes<METCollection>(iConfig.getParameter<edm::InputTag>("inputUncorMetLabel"));
  tokenUncorJets = consumes<PFJetCollection>(iConfig.getParameter<edm::InputTag>("inputUncorJetsTag"));
  correctorToken = consumes<JetCorrector>(iConfig.getParameter<edm::InputTag>("corrector"));
  jetPTthreshold = iConfig.getParameter<double>("jetPTthreshold");
  jetEMfracLimit = iConfig.getParameter<double>("jetEMfracLimit");
  jetMufracLimit = iConfig.getParameter<double>("jetMufracLimit");
  produces<METCollection>();
}
Type1PFMET::Type1PFMET() {}

// PRODUCER DESTRUCTORS -------------------------------------------
Type1PFMET::~Type1PFMET() {}

// PRODUCER METHODS -----------------------------------------------
void Type1PFMET::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  Handle<PFJetCollection> inputUncorJets;
  iEvent.getByToken(tokenUncorJets, inputUncorJets);
  Handle<JetCorrector> corrector;
  iEvent.getByToken(correctorToken, corrector);
  Handle<METCollection> inputUncorMet;                         //Define Inputs
  iEvent.getByToken(tokenUncorMet, inputUncorMet);             //Get Inputs
  std::unique_ptr<METCollection> output(new METCollection());  //Create empty output
  run(*(inputUncorMet.product()),
      *(corrector.product()),
      *(inputUncorJets.product()),
      jetPTthreshold,
      jetEMfracLimit,
      jetMufracLimit,
      &*output);                  //Invoke the algorithm
  iEvent.put(std::move(output));  //Put output into Event
}

void Type1PFMET::run(const METCollection& uncorMET,
                     const reco::JetCorrector& corrector,
                     const PFJetCollection& uncorJet,
                     double jetPTthreshold,
                     double jetEMfracLimit,
                     double jetMufracLimit,
                     METCollection* corMET) {
  if (!corMET) {
    std::cerr << "Type1METAlgo_run-> undefined output MET collection. Stop. " << std::endl;
    return;
  }

  double DeltaPx = 0.0;
  double DeltaPy = 0.0;
  double DeltaSumET = 0.0;
  // ---------------- Calculate jet corrections, but only for those uncorrected jets
  // ---------------- which are above the given threshold.  This requires that the
  // ---------------- uncorrected jets be matched with the corrected jets.
  for (PFJetCollection::const_iterator jet = uncorJet.begin(); jet != uncorJet.end(); ++jet) {
    if (jet->pt() > jetPTthreshold) {
      double emEFrac = jet->chargedEmEnergyFraction() + jet->neutralEmEnergyFraction();
      double muEFrac = jet->chargedMuEnergyFraction();
      if (emEFrac < jetEMfracLimit && muEFrac < jetMufracLimit) {
        double corr = corrector.correction(*jet) - 1.;  // correction itself
        DeltaPx += jet->px() * corr;
        DeltaPy += jet->py() * corr;
        DeltaSumET += jet->et() * corr;
      }
    }
  }
  //----------------- Calculate and set deltas for new MET correction
  CorrMETData delta;
  delta.mex = -DeltaPx;  //correction to MET (from Jets) is negative,
  delta.mey = -DeltaPy;  //since MET points in direction opposite of jets
  delta.sumet = DeltaSumET;
  //----------------- Fill holder with corrected MET (= uncorrected + delta) values
  const MET* u = &(uncorMET.front());
  double corrMetPx = u->px() + delta.mex;
  double corrMetPy = u->py() + delta.mey;
  MET::LorentzVector correctedMET4vector(corrMetPx, corrMetPy, 0., sqrt(corrMetPx * corrMetPx + corrMetPy * corrMetPy));
  //----------------- get previous corrections and push into new corrections
  std::vector<CorrMETData> corrections = u->mEtCorr();
  corrections.push_back(delta);
  //----------------- Push onto MET Collection
  MET result = MET(u->sumEt() + delta.sumet, corrections, correctedMET4vector, u->vertex(), u->isWeighted());
  corMET->push_back(result);

  return;
}
