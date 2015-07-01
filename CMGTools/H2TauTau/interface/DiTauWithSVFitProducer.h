
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "TauAnalysis/SVfitStandalone/interface/SVfitStandaloneAlgorithm.h"

#include <sstream>


template< typename T, typename U>
class DiTauWithSVFitProducer : public edm::EDProducer {

typedef pat::CompositeCandidate DiTauObject;

public:
  typedef std::vector<DiTauObject> DiTauCollection;
  typedef std::auto_ptr<DiTauCollection> OutPtr;
  explicit DiTauWithSVFitProducer(const edm::ParameterSet& iConfig);
  virtual ~DiTauWithSVFitProducer() {}

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

  /// source diobject inputtag
  edm::InputTag diTauSrc_;

  unsigned warningNumbers_;
  bool verbose_;
  int SVFitVersion_;
  std::string fitAlgo_;
};


template< typename T, typename U >
DiTauWithSVFitProducer<T, U>::DiTauWithSVFitProducer(const edm::ParameterSet& iConfig) :
  diTauSrc_(iConfig.getParameter<edm::InputTag>("diTauSrc")),
  warningNumbers_(0),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
  SVFitVersion_(iConfig.getParameter<int>("SVFitVersion")),
  fitAlgo_(iConfig.getParameter<std::string>("fitAlgo")) {

  // will produce a collection containing a copy of each di-object in input,
  // with the SVFit mass set.
  produces<std::vector<DiTauObject>>();
}


template<typename T, typename U>
void DiTauWithSVFitProducer<T, U>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<DiTauCollection> diTauH;
  iEvent.getByLabel(diTauSrc_, diTauH);

  if(verbose_ && !diTauH->empty()) {
    std::cout << "DiTauWithSVFitProducer" << std::endl;
    std::cout << "+++" << std::endl;
  }

  std::string warningMessage;

  svFitStandalone::kDecayType leg1type = svFitStandalone::kUndefinedDecayType;
  svFitStandalone::kDecayType leg2type = svFitStandalone::kUndefinedDecayType;

  if (typeid(T) == typeid(pat::Tau)) {
      leg1type=svFitStandalone::kTauToHadDecay;
      warningMessage += " - first leg is hadronic tau";
  } else if (typeid(T) == typeid(pat::Electron)) {
      leg1type=svFitStandalone::kTauToElecDecay;
      warningMessage += " - first leg is electron from tau";
  } else if (typeid(T) == typeid(pat::Muon)) {
      leg1type=svFitStandalone::kTauToMuDecay;
      warningMessage += " - first leg is muon from tau";
  } else {
      warningMessage += " - first leg - COULD NOT IDENTIFY TYPE";
  }

  if (typeid(U) == typeid(pat::Tau)) {
      leg2type=svFitStandalone::kTauToHadDecay;
      warningMessage += " - second leg is hadronic tau";
  } else if (typeid(U) == typeid(pat::Electron)) {
      leg2type=svFitStandalone::kTauToElecDecay;
      warningMessage += " - second leg is electron from tau";
  } else if (typeid(U) == typeid(pat::Muon)) {
      leg2type=svFitStandalone::kTauToMuDecay;
      warningMessage += " - second leg is muon from tau";
  }

  const unsigned maxWarnings = 5;
  if(warningNumbers_<maxWarnings) {
    std::cout << warningMessage << std::endl;
    warningNumbers_ += 1;
  }

  OutPtr pOut(new DiTauCollection());

  if(verbose_ && !diTauH->empty()) {
    std::cout << "Looping on " << diTauH->size() << " input di-objects:" << std::endl;
  }

  for (auto& diTauOriginal : *diTauH) {
    DiTauObject diTau(diTauOriginal);
    const reco::MET& met(dynamic_cast<const reco::MET&>(*diTau.daughter(2)));

    const auto& smsig = met.getSignificanceMatrix();

    TMatrixD tmsig(2, 2);
    // tmsig.SetMatrixArray(smsig.Array());
    // Set elements by hand to avoid array gymnastics/assumptions
    tmsig(0,0) = smsig(0,0);
    tmsig(0,1) = smsig(0,1);
    tmsig(1,0) = smsig(1,0);
    tmsig(1,1) = smsig(1,1);

    if(verbose_) {
      std::cout << "  ---------------- " << std::endl;
      std::cout << "\trec boson: " << diTau << std::endl;
      std::cout << "\t\tleg1: " << *diTau.daughter(0) << std::endl;
      std::cout << "\t\tleg2: " << *diTau.daughter(1) << std::endl;
      std::cout << "\t\tMET = " << met.et() << ", phi_MET = " << met.phi() << std::endl;
    }

    double massSVFit = 0.;
    float det=tmsig.Determinant();
    if(det>1e-8) {
      if (SVFitVersion_ >= 1) {
        //Note that this works only for di-objects where the tau is the leg1 and mu is leg2
        std::vector<svFitStandalone::MeasuredTauLepton> measuredTauLeptons;
        int leg1DecayMode = -1;
        int leg2DecayMode = -1;
        auto leg2Mass = diTau.daughter(1)->mass();
        auto leg1Mass = diTau.daughter(0)->mass();

        if (leg1type == svFitStandalone::kTauToHadDecay) {
          leg1DecayMode = static_cast<pat::Tau*>(diTau.daughter(0))->decayMode();
        }
        else if (leg1type == svFitStandalone::kTauToElecDecay)
        {
          // Reconstructed GSF electrons have non-fixed mass in CMS
          leg1Mass = 0.000511;
        }
        else if (leg1type == svFitStandalone::kTauToMuDecay)
        {
          // Muons may sometimes have the charged pion mass
          leg1Mass = 0.10566;
        }

        if (leg2type == svFitStandalone::kTauToHadDecay) {
          leg2DecayMode = static_cast<pat::Tau*>(diTau.daughter(1))->decayMode();
        }
        else if (leg2type == svFitStandalone::kTauToElecDecay)
        {
          leg2Mass = 0.000511;
        }
        else if (leg2type == svFitStandalone::kTauToMuDecay)
        {
          leg2Mass = 0.10566;
        }

        measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(leg2type, diTau.daughter(1)->pt(), diTau.daughter(1)->eta(), diTau.daughter(1)->phi(), leg2Mass, leg2DecayMode));
        measuredTauLeptons.push_back(svFitStandalone::MeasuredTauLepton(leg1type, diTau.daughter(0)->pt(), diTau.daughter(0)->eta(), diTau.daughter(0)->phi(), leg1Mass, leg1DecayMode));
        SVfitStandaloneAlgorithm algo(measuredTauLeptons, met.px(), met.py(), tmsig, 0);
        algo.addLogM(false);

        if (fitAlgo_ == "VEGAS")
          algo.integrateVEGAS();
        else if (fitAlgo_ == "MC")
          algo.integrateMarkovChain();
        else
          algo.integrate();

        massSVFit = algo.mass();

        // Add more fit results as user floats
        diTau.addUserFloat("massUncert", algo.massUncert());

        if (fitAlgo_ == "MC"){
          diTau.addUserFloat("pt"        , algo.pt()        );
          diTau.addUserFloat("ptUncert"  , algo.ptUncert()  );
          diTau.addUserFloat("fittedEta" , algo.eta()       );
          diTau.addUserFloat("fittedPhi" , algo.phi()       );
        }
        else {
          diTau.addUserFloat("pt"        , -99.);
          diTau.addUserFloat("ptUncert"  , -99.);
          diTau.addUserFloat("fittedEta" , -99.);
          diTau.addUserFloat("fittedPhi" , -99.);
        }

      }
    }
    // This is now handled via the user floats so we can keep the visible mass
    diTau.addUserFloat("mass", massSVFit);

    pOut->push_back(diTau);

    if(verbose_) {
      std::cout << "\tm_vis = " << diTau.mass() << ", m_svfit = " << massSVFit << std::endl;
    }
  }

  iEvent.put(pOut);

  if(verbose_ && !diTauH->empty()) {
    std::cout << "DiTauWithSVFitProducer done" << std::endl;
    std::cout << "***" << std::endl;
  }
}
