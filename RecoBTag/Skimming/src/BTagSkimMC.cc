#include "RecoBTag/Skimming/interface/BTagSkimMC.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace edm;
using namespace std;
using namespace reco;

BTagSkimMC::BTagSkimMC(const ParameterSet& p, const BTagSkimMCCount::Counters* count) : nEvents_(0), nAccepted_(0) {
  verbose = p.getUntrackedParameter<bool>("verbose", false);
  pthatMin = p.getParameter<double>("pthat_min");
  pthatMax = p.getParameter<double>("pthat_max");
  process_ = p.getParameter<string>("mcProcess");
  if (verbose)
    cout << " Requested:  " << process_ << endl;
}

bool BTagSkimMC::filter(Event& evt, const EventSetup& es) {
  nEvents_++;

  Handle<int> genProcessID;
  evt.getByLabel("genEventProcID", genProcessID);
  double processID = *genProcessID;

  Handle<double> genEventScale;
  evt.getByLabel("genEventScale", genEventScale);
  double pthat = *genEventScale;

  if (verbose)
    cout << "processID: " << processID << " - pthat: " << pthat;

  if ((processID != 4) && (process_ == "QCD")) {  // the Pythia events (for ALPGEN see below)

    Handle<double> genFilterEff;
    evt.getByLabel("genEventRunInfo", "FilterEfficiency", genFilterEff);
    double filter_eff = *genFilterEff;
    if (verbose)
      cout << " Is QCD ";
    // qcd (including min bias HS)
    if ((filter_eff == 1. || filter_eff == 0.964) && (processID == 11 || processID == 12 || processID == 13 ||
                                                      processID == 28 || processID == 68 || processID == 53)) {
      if (pthat > pthatMin && pthat < pthatMax) {
        if (verbose)
          cout << " ACCEPTED " << endl;
        nAccepted_++;
        return true;
      }
    }

  }  // ALPGEN
  else if (processID == 4) {  // this is the number for external ALPGEN events

    Handle<GenParticleCollection> genParticles;
    evt.getByLabel("genParticles", genParticles);

    for (size_t i = 0; i < genParticles->size(); ++i) {
      const Candidate& p = (*genParticles)[i];
      int id = p.pdgId();
      int st = p.status();

      // tt+jets
      if (st == 3 && (id == 6 || id == -6)) {
        if (verbose)
          cout << "We have a ttbar event" << endl;
        nAccepted_++;
        return true;
      }
    }
  }
  if (verbose)
    cout << " REJECTED " << endl;

  return false;
}

void BTagSkimMC::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("mcProcess", "ttbar");
  desc.add<double>("pthat_min", 50.0);
  desc.addUntracked<bool>("verbose", false);
  desc.add<double>("pthat_max", 80.0);
  descriptions.addWithDefaultLabel(desc);
}

void BTagSkimMC::endStream() {
  globalCache()->nEvents_ += nEvents_;
  globalCache()->nAccepted_ += nAccepted_;
}

void BTagSkimMC::globalEndJob(const BTagSkimMCCount::Counters* count) {
  edm::LogVerbatim("BTagSkimMC") << "=============================================================================\n"
                                 << " Events read: " << count->nEvents_
                                 << "\n Events accepted by BTagSkimMC: " << count->nAccepted_
                                 << "\n Efficiency: " << (double)(count->nAccepted_) / (double)(count->nEvents_)
                                 << "\n==========================================================================="
                                 << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(BTagSkimMC);
