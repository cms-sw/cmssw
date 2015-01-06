/*
 *  See header file for a description of this class.
 *
 *  \author S. Bolognesi, Eric - CERN
 */

#include "DQM/Physics/src/QcdHighPtDQM.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <string>
#include <cmath>

using namespace std;
using namespace edm;
using namespace reco;
using namespace math;

// Get Jets and MET (no MET plots yet pending converging w/JetMET group)

QcdHighPtDQM::QcdHighPtDQM(const ParameterSet &iConfig)
    : jetToken_(consumes<CaloJetCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("jetTag"))),
      metToken1_(consumes<CaloMETCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("metTag1"))),
      metToken2_(consumes<CaloMETCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("metTag2"))),
      metToken3_(consumes<CaloMETCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("metTag3"))),
      metToken4_(consumes<CaloMETCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("metTag4"))) {
}

QcdHighPtDQM::~QcdHighPtDQM() {}

void QcdHighPtDQM::bookHistograms(DQMStore::IBooker &iBooker,
                                     edm::Run const &,
                                     edm::EventSetup const &) {
  iBooker.setCurrentFolder("Physics/QcdHighPt");

  MEcontainer_["dijet_mass"] = iBooker.book1D(
      "dijet_mass", "dijet resonance invariant mass, barrel region", 100, 0,
      1000);
  MEcontainer_["njets"] =
      iBooker.book1D("njets", "jet multiplicity", 10, 0, 10);
  MEcontainer_["etaphi"] = iBooker.book2D("etaphi", "eta/phi distribution", 83,
                                          -42, 42, 72, -M_PI, M_PI);
  MEcontainer_["njets30"] =
      iBooker.book1D("njets30", "jet multiplicity, pt > 30 GeV", 10, 0, 10);

  // book histograms for inclusive jet quantities
  MEcontainer_["inclusive_jet_pt"] = iBooker.book1D(
      "inclusive_jet_pt", "inclusive jet Pt spectrum", 100, 0, 1000);
  MEcontainer_["inclusive_jet_pt_barrel"] = iBooker.book1D(
      "inclusive_jet_pt_barrel", "inclusive jet Pt, eta < 1.3", 100, 0, 1000);
  MEcontainer_["inclusive_jet_pt_forward"] =
      iBooker.book1D("inclusive_jet_pt_forward",
                     "inclusive jet Pt, 3.0 < eta < 5.0", 100, 0, 1000);
  MEcontainer_["inclusive_jet_pt_endcap"] =
      iBooker.book1D("inclusive_jet_pt_endcap",
                     "inclusive jet Pt, 1.3 < eta < 3.0", 100, 0, 1000);

  // book histograms for leading jet quantities
  MEcontainer_["leading_jet_pt"] =
      iBooker.book1D("leading_jet_pt", "leading jet Pt", 100, 0, 1000);
  MEcontainer_["leading_jet_pt_barrel"] = iBooker.book1D(
      "leading_jet_pt_barrel", "leading jet Pt, eta < 1.3", 100, 0, 1000);
  MEcontainer_["leading_jet_pt_forward"] =
      iBooker.book1D("leading_jet_pt_forward",
                     "leading jet Pt, 3.0 < eta < 5.0", 100, 0, 1000);
  MEcontainer_["leading_jet_pt_endcap"] = iBooker.book1D(
      "leading_jet_pt_endcap", "leading jet Pt, 1.3 < eta < 3.0", 100, 0, 1000);

  // book histograms for met over sum et and met over leading jet pt for various
  // flavors of MET
  MEcontainer_["movers_met"] = iBooker.book1D(
      "movers_met", "MET over Sum ET for basic MET collection", 50, 0, 1);
  MEcontainer_["moverl_met"] = iBooker.book1D(
      "moverl_met", "MET over leading jet Pt for basic MET collection", 50, 0,
      2);

  MEcontainer_["movers_metho"] = iBooker.book1D(
      "movers_metho", "MET over Sum ET for MET HO collection", 50, 0, 1);
  MEcontainer_["moverl_metho"] =
      iBooker.book1D("moverl_metho",
                     "MET over leading jet Pt for MET HO collection", 50, 0, 2);

  MEcontainer_["movers_metnohf"] = iBooker.book1D(
      "movers_metnohf", "MET over Sum ET for MET no HF collection", 50, 0, 1);
  MEcontainer_["moverl_metnohf"] = iBooker.book1D(
      "moverl_metnohf", "MET over leading jet Pt for MET no HF collection", 50,
      0, 2);

  MEcontainer_["movers_metnohfho"] =
      iBooker.book1D("movers_metnohfho",
                     "MET over Sum ET for MET no HF HO collection", 50, 0, 1);
  MEcontainer_["moverl_metnohfho"] = iBooker.book1D(
      "moverl_metnohfho", "MET over leading jet Pt for MET no HF HO collection",
      50, 0, 2);

  // book histograms for EMF fraction for all jets and first 3 jets
  MEcontainer_["inclusive_jet_EMF"] =
      iBooker.book1D("inclusive_jet_EMF", "inclusive jet EMF", 50, -1, 1);
  MEcontainer_["leading_jet_EMF"] =
      iBooker.book1D("leading_jet_EMF", "leading jet EMF", 50, -1, 1);
  MEcontainer_["second_jet_EMF"] =
      iBooker.book1D("second_jet_EMF", "second jet EMF", 50, -1, 1);
  MEcontainer_["third_jet_EMF"] =
      iBooker.book1D("third_jet_EMF", "third jet EMF", 50, -1, 1);
}

// method to calculate MET over Sum ET from a particular MET collection
float QcdHighPtDQM::movers(const CaloMETCollection &metcollection) {
  float metovers = 0;
  CaloMETCollection::const_iterator met_iter;
  for (met_iter = metcollection.begin(); met_iter != metcollection.end();
       ++met_iter) {
    float mex = met_iter->momentum().x();
    float mey = met_iter->momentum().y();
    float met = sqrt(mex * mex + mey * mey);
    float sumet = met_iter->sumEt();
    metovers = (met / sumet);
  }
  return metovers;
}

// method to calculate MET over Leading jet PT for a particular MET collection
float QcdHighPtDQM::moverl(const CaloMETCollection &metcollection,
                           float &ljpt) {
  float metoverl = 0;
  CaloMETCollection::const_iterator met_iter;
  for (met_iter = metcollection.begin(); met_iter != metcollection.end();
       ++met_iter) {
    float mex = met_iter->momentum().x();
    float mey = met_iter->momentum().y();
    float met = sqrt(mex * mex + mey * mey);
    metoverl = (met / ljpt);
  }
  return metoverl;
}

void QcdHighPtDQM::analyze(const Event &iEvent, const EventSetup &iSetup) {
  // Get Jets
  edm::Handle<CaloJetCollection> jetHandle;
  iEvent.getByToken(jetToken_, jetHandle);
  const CaloJetCollection &jets = *jetHandle;
  CaloJetCollection::const_iterator jet_iter;

  // Get MET collections
  edm::Handle<CaloMETCollection> metHandle;
  iEvent.getByToken(metToken1_, metHandle);
  const CaloMETCollection &met = *metHandle;

  edm::Handle<CaloMETCollection> metHOHandle;
  iEvent.getByToken(metToken2_, metHOHandle);
  const CaloMETCollection &metHO = *metHOHandle;

  edm::Handle<CaloMETCollection> metNoHFHandle;
  iEvent.getByToken(metToken3_, metNoHFHandle);
  const CaloMETCollection &metNoHF = *metNoHFHandle;

  edm::Handle<CaloMETCollection> metNoHFHOHandle;
  iEvent.getByToken(metToken4_, metNoHFHOHandle);
  const CaloMETCollection &metNoHFHO = *metNoHFHOHandle;

  // initialize leading jet value and jet multiplicity counter
  int njets = 0;
  int njets30 = 0;
  float leading_jetpt = 0;
  float leading_jeteta = 0;

  // initialize variables for picking out leading 2 barrel jets
  reco::CaloJet leadingbarreljet;
  reco::CaloJet secondbarreljet;
  int nbarreljets = 0;

  // get bins in eta.
  // Bins correspond to calotower regions.

  const float etabins[83] = {
      -5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664,
      -3.489, -3.314, -3.139, -2.964, -2.853, -2.650, -2.500, -2.322, -2.172,
      -2.043, -1.930, -1.830, -1.740, -1.653, -1.566, -1.479, -1.392, -1.305,
      -1.218, -1.131, -1.044, -.957,  -.879,  -.783,  -.696,  -.609,  -.522,
      -.435,  -.348,  -.261,  -.174,  -.087,  0,      .087,   .174,   .261,
      .348,   .435,   .522,   .609,   .696,   .783,   .879,   .957,   1.044,
      1.131,  1.218,  1.305,  1.392,  1.479,  1.566,  1.653,  1.740,  1.830,
      1.930,  2.043,  2.172,  2.322,  2.500,  2.650,  2.853,  2.964,  3.139,
      3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,  4.889,
      5.191};

  for (jet_iter = jets.begin(); jet_iter != jets.end(); ++jet_iter) {
    njets++;

    // get Jet stats
    float jet_pt = jet_iter->pt();
    float jet_eta = jet_iter->eta();
    float jet_phi = jet_iter->phi();

    // fill jet Pt and jet EMF
    MEcontainer_["inclusive_jet_pt"]->Fill(jet_pt);
    MEcontainer_["inclusive_jet_EMF"]->Fill(jet_iter->emEnergyFraction());

    // pick out up to the first 2 leading barrel jets
    // for use in calculating dijet mass in barrel region
    // also fill jet Pt histogram for barrel

    if (jet_eta <= 1.3) {
      MEcontainer_["inclusive_jet_pt_barrel"]->Fill(jet_pt);
      if (nbarreljets == 0) {
        leadingbarreljet = jets[(njets - 1)];
        nbarreljets++;
      } else if (nbarreljets == 1) {
        secondbarreljet = jets[(njets - 1)];
        nbarreljets++;
      }

    }

    // fill jet Pt for endcap and forward regions
    else if (jet_eta <= 3.0 && jet_eta > 1.3) {
      MEcontainer_["inclusive_jet_pt_endcap"]->Fill(jet_pt);
    } else if (jet_eta <= 5.0 && jet_eta > 3.0) {
      MEcontainer_["inclusive_jet_pt_forward"]->Fill(jet_pt);
    }

    // count jet multiplicity for jets with Pt > 30
    if ((jet_pt) > 30) njets30++;

    // check leading jet quantities
    if (jet_pt > leading_jetpt) {
      leading_jetpt = jet_pt;
      leading_jeteta = jet_eta;
    }

    // fill eta-phi plot
    for (int eit = 0; eit < 81; eit++) {
      for (int pit = 0; pit < 72; pit++) {
        float low_eta = etabins[eit];
        float high_eta = etabins[eit + 1];
        float low_phi = (-M_PI) + pit * (M_PI / 36);
        float high_phi = low_phi + (M_PI / 36);
        if (jet_eta > low_eta && jet_eta < high_eta && jet_phi > low_phi &&
            jet_phi < high_phi) {
          MEcontainer_["etaphi"]->Fill((eit - 41), jet_phi);
        }
      }
    }
  }

  // after iterating over all jets, fill leading jet quantity histograms
  // and jet multiplicity histograms

  MEcontainer_["leading_jet_pt"]->Fill(leading_jetpt);

  if (leading_jeteta <= 1.3) {
    MEcontainer_["leading_jet_pt_barrel"]->Fill(leading_jetpt);
  } else if (leading_jeteta <= 3.0 && leading_jeteta > 1.3) {
    MEcontainer_["leading_jet_pt_endcap"]->Fill(leading_jetpt);
  } else if (leading_jeteta <= 5.0 && leading_jeteta > 3.0) {
    MEcontainer_["leading_jet_pt_forward"]->Fill(leading_jetpt);
  }

  MEcontainer_["njets"]->Fill(njets);

  MEcontainer_["njets30"]->Fill(njets30);

  // fill MET over Sum ET and Leading jet PT for all MET flavors
  MEcontainer_["movers_met"]->Fill(movers(met));
  MEcontainer_["moverl_met"]->Fill(movers(met), leading_jetpt);
  MEcontainer_["movers_metho"]->Fill(movers(metHO));
  MEcontainer_["moverl_metho"]->Fill(movers(metHO), leading_jetpt);
  MEcontainer_["movers_metnohf"]->Fill(movers(metNoHF));
  MEcontainer_["moverl_metnohf"]->Fill(movers(metNoHF), leading_jetpt);
  MEcontainer_["movers_metnohfho"]->Fill(movers(metNoHFHO));
  MEcontainer_["moverl_metnohfho"]->Fill(movers(metNoHFHO), leading_jetpt);

  // fetch first 3 jet EMF

  if (jets.size() >= 1) {
    MEcontainer_["leading_jet_EMF"]->Fill(jets[0].emEnergyFraction());
    if (jets.size() >= 2) {
      MEcontainer_["second_jet_EMF"]->Fill(jets[1].emEnergyFraction());
      if (jets.size() >= 3) {
        MEcontainer_["third_jet_EMF"]->Fill(jets[2].emEnergyFraction());
      }
    }
  }

  // if 2 nontrivial barrel jets, reconstruct dijet mass

  if (nbarreljets == 2) {
    if (leadingbarreljet.energy() > 0 && secondbarreljet.energy() > 0) {
      math::XYZTLorentzVector DiJet =
          leadingbarreljet.p4() + secondbarreljet.p4();
      float dijet_mass = DiJet.mass();
      MEcontainer_["dijet_mass"]->Fill(dijet_mass);
    }
  }
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
