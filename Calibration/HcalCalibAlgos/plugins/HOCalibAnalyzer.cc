//April 2015 : Removal of itrg1, itrg2, but addition of isect2, same is true in HOCalibVariables.h
// -*- C++ -*-
//
// Package:    HOCalibAnalyzer
// Class:      HOCalibAnalyzer
//
/**\class HOCalibAnalyzer HOCalibAnalyzer.cc Calibration/HOCalibAnalyzer/src/HOCalibAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>

April 2015
// Addition of these variables, ilumi (analyser), pileup (analyser), nprim


*/
//
// Original Author:  Gobinda Majumder
//         Created:  Sat Jul  7 09:51:31 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Math/interface/angle_units.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
#include "TProfile.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

//
// class decleration
//

class HOCalibAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HOCalibAnalyzer(const edm::ParameterSet&);
  ~HOCalibAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  static constexpr int netamx = 30;
  static constexpr int nphimx = 72;
  static constexpr int ringmx = 5;
  static constexpr int ncut = 14;

  const char* varcrit[3] = {"All", "steps", "n-1"};  // or opposite

  const double elosfact = (14.9 + 0.96 * std::fabs(std::log(8 * 2.8)) + 0.033 * 8 * (1.0 - std::pow(8, -0.33)));

  int getHOieta(int ij) { return (ij < netamx / 2) ? -netamx / 2 + ij : -netamx / 2 + ij + 1; }
  int invert_HOieta(int ieta) { return (ieta < 0) ? netamx / 2 + ieta : netamx / 2 + ieta - 1; }

  int mypow_2[31];

  const bool m_cosmic;
  const bool m_zeroField;
  const int m_bins;
  const double m_low;
  const double m_ahigh;
  const bool m_histFill;  //Stored signals of individual HO towers with default selection criteria
  const bool m_treeFill;  //Store rootuple without almost any selection criteria except a quality on muon
  const bool m_verbose;

  int ipass;

  TTree* T1;

  TH1F* ho_indenergy[netamx][nphimx];

  TH1F* muonnm;
  TH1F* muonmm;
  TH1F* muonth;
  TH1F* muonph;
  TH1F* muonch;

  TH1F* sel_muonnm;
  TH1F* sel_muonmm;
  TH1F* sel_muonth;
  TH1F* sel_muonph;
  TH1F* sel_muonch;

  TH2F* sig_eta_evt[3 * netamx][ncut];  //For individual eta
  TH2F* sigvsevt[3 * netamx][ncut];
  TH1F* variab[3 * netamx][ncut];

  TH2F* mu_projection[ncut + 1];

  unsigned ievt, hoflag;
  int irun, ilumi, nprim, isect, isect2, ndof, nmuon;

  float pileup, trkdr, trkdz, trkvx, trkvy, trkvz, trkmm, trkth, trkph, chisq, therr, pherr, hodx, hody, hoang, htime,
      hosig[9], hocorsig[18], hocro, hbhesig[9], caloen[3];
  float momatho, tkpt03, ecal03, hcal03;
  float tmphoang;

  int nevents[10];

  float ncount[ringmx][ncut + 10];

  edm::InputTag hoCalibVariableCollectionTag;
  const edm::EDGetTokenT<HOCalibVariableCollection> tok_ho_;
  const edm::EDGetTokenT<HORecHitCollection> tok_allho_;
  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

HOCalibAnalyzer::HOCalibAnalyzer(const edm::ParameterSet& iConfig)
    : m_cosmic(iConfig.getUntrackedParameter<bool>("cosmic", true)),
      m_zeroField(iConfig.getUntrackedParameter<bool>("zeroField", false)),
      m_bins(iConfig.getUntrackedParameter<int>("HOSignalBins", 120)),
      m_low(iConfig.getUntrackedParameter<double>("lowerRange", -1.0)),
      m_ahigh(iConfig.getUntrackedParameter<double>("upperRange", 29.0)),
      m_histFill(iConfig.getUntrackedParameter<bool>("histFill", true)),
      m_treeFill(iConfig.getUntrackedParameter<bool>("treeFill", false)),
      m_verbose(iConfig.getUntrackedParameter<bool>("verbose", false)),
      tok_ho_(consumes<HOCalibVariableCollection>(iConfig.getParameter<edm::InputTag>("hoCalibVariableCollectionTag"))),
      tok_allho_(consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInputTag"))) {
  // It is very likely you want the following in your configuration
  // hoCalibVariableCollectionTag = cms.InputTag('hoCalibProducer', 'HOCalibVariableCollection')

  usesResource(TFileService::kSharedResource);

  //now do what ever initialization is needed
  ipass = 0;
  for (int ij = 0; ij < 10; ij++) {
    nevents[ij] = 0;
  }

  edm::Service<TFileService> fs;

  T1 = fs->make<TTree>("T1", "HOSignal");

  T1->Branch("irun", &irun, "irun/I");
  T1->Branch("ievt", &ievt, "ievt/i");

  T1->Branch("isect", &isect, "isect/I");
  T1->Branch("isect2", &isect2, "isect2/I");
  T1->Branch("ndof", &ndof, "ndof/I");
  T1->Branch("nmuon", &nmuon, "nmuon/I");

  T1->Branch("ilumi", &ilumi, "ilumi/I");
  if (!m_cosmic) {
    T1->Branch("pileup", &pileup, "pileup/F");
    T1->Branch("nprim", &nprim, "nprim/I");
    T1->Branch("tkpt03", &tkpt03, " tkpt03/F");
    T1->Branch("ecal03", &ecal03, " ecal03/F");
    T1->Branch("hcal03", &hcal03, " hcal03/F");
  }

  T1->Branch("trkdr", &trkdr, "trkdr/F");
  T1->Branch("trkdz", &trkdz, "trkdz/F");

  T1->Branch("trkvx", &trkvx, "trkvx/F");
  T1->Branch("trkvy", &trkvy, "trkvy/F");
  T1->Branch("trkvz", &trkvz, "trkvz/F");
  T1->Branch("trkmm", &trkmm, "trkmm/F");
  T1->Branch("trkth", &trkth, "trkth/F");
  T1->Branch("trkph", &trkph, "trkph/F");

  T1->Branch("chisq", &chisq, "chisq/F");
  T1->Branch("therr", &therr, "therr/F");
  T1->Branch("pherr", &pherr, "pherr/F");
  T1->Branch("hodx", &hodx, "hodx/F");
  T1->Branch("hody", &hody, "hody/F");
  T1->Branch("hoang", &hoang, "hoang/F");

  T1->Branch("momatho", &momatho, "momatho/F");
  T1->Branch("hoflag", &hoflag, "hoflag/i");
  T1->Branch("htime", &htime, "htime/F");
  T1->Branch("hosig", hosig, "hosig[9]/F");
  T1->Branch("hocro", &hocro, "hocro/F");
  T1->Branch("hocorsig", hocorsig, "hocorsig[18]/F");
  T1->Branch("caloen", caloen, "caloen[3]/F");

  char name[200];
  char title[200];

  if (m_histFill) {
    for (int ij = 0; ij < netamx; ij++) {
      int ieta = getHOieta(ij);
      for (int jk = 0; jk < nphimx; jk++) {
        sprintf(name, "ho_indenergy_%i_%i", ij, jk);
        sprintf(title, "ho IndEnergy (GeV) i#eta=%i i#phi=%i", ieta, jk + 1);
        ho_indenergy[ij][jk] = fs->make<TH1F>(name, title, 1200, m_low, m_ahigh);
      }
    }
  }

  muonnm = fs->make<TH1F>("muonnm", "No of muon", 10, -0.5, 9.5);
  muonmm = fs->make<TH1F>("muonmm", "P_{mu}", 200, -100., 100.);
  muonth = fs->make<TH1F>("muonth", "{Theta}_{mu}", 180, 0., 180.);
  muonph = fs->make<TH1F>("muonph", "{Phi}_{mu}", 180, -180., 180.);
  muonch = fs->make<TH1F>("muonch", "{chi^2}/ndf", 100, 0., 1000.);

  sel_muonnm = fs->make<TH1F>("sel_muonnm", "No of muon(sel)", 10, -0.5, 9.5);
  sel_muonmm = fs->make<TH1F>("sel_muonmm", "P_{mu}(sel)", 200, -100., 100.);
  sel_muonth = fs->make<TH1F>("sel_muonth", "{Theta}_{mu}(sel)", 180, 0., 180.);
  sel_muonph = fs->make<TH1F>("sel_muonph", "{Phi}_{mu}(sel)", 180, -180., 180.);
  sel_muonch = fs->make<TH1F>("sel_muonch", "{chi^2}/ndf(sel)", 100, 0., 1000.);

  //if change order, change in iselect_wotime also and other efficiency numbers
  const char* varnam[ncut] = {"ndof",
                              "chisq",
                              "th",
                              "ph",
                              "therr",
                              "pherr",
                              "dircos",
                              "trkmm",
                              "nmuon",
                              "calo",
                              "trkiso",
                              "#phi-dir",
                              "#eta-dir",
                              "time"};
  int nbinxx[ncut] = {25, 60, 60, 60, 60, 60, 60, 120, 6, 60, 60, 120, 120, 60};
  double alowxx[ncut] = {5.5, 0., 0., -angle_units::piRadians, 0.0, 0.0, 0.0, 0., 0.5, 0.0, 0.0, -20., -32., -45.0};
  double ahghxx[ncut] = {
      30.5, 40., angle_units::piRadians, angle_units::piRadians, 0.8, 0.02, 0.5, 300., 6.5, 10.0, 24.0, 20.0, 32.0, 45.0};

  for (int kl = 0; kl < ncut; kl++) {
    for (int jk = 0; jk < 3; jk++) {
      for (int ij = 0; ij < netamx; ij++) {
        sprintf(name, "sigeta_%i_%i_%i", kl, jk, ij);
        sprintf(title, "sigeta %s %s i#eta=%i", varnam[kl], varcrit[jk], getHOieta(ij));
        sig_eta_evt[netamx * jk + ij][kl] =
            fs->make<TH2F>(name, title, nbinxx[kl], alowxx[kl], ahghxx[kl], m_bins, m_low, m_ahigh);
      }
    }
  }

  for (int kl = 0; kl < ncut; kl++) {
    for (int ij = 0; ij < ringmx * 3; ij++) {
      int iring = ij % ringmx - 2;
      int iset = ij / ringmx;
      sprintf(name, "sigring_%i_%i", kl, ij);
      sprintf(title, "Signal %s %s Ring%i", varnam[kl], varcrit[iset], iring);
      sigvsevt[ij][kl] = fs->make<TH2F>(name, title, nbinxx[kl], alowxx[kl], ahghxx[kl], m_bins, m_low, m_ahigh);
    }
  }

  for (int kl = 0; kl < ncut; kl++) {
    for (int ij = 0; ij < ringmx * 3; ij++) {
      int iring = ij % ringmx - 2;
      int iset = ij / ringmx;
      sprintf(name, "varring_%i_%i", kl, ij);
      sprintf(title, "%s %s Ring%i", varnam[kl], varcrit[iset], iring);
      variab[ij][kl] = fs->make<TH1F>(name, title, nbinxx[kl], alowxx[kl], ahghxx[kl]);
    }
  }

  for (int ij = 0; ij <= ncut; ij++) {
    sprintf(name, "mu_projection_%i", ij);
    if (ij == 0) {
      sprintf(title, "All projected muon");
    } else {
      sprintf(title, "Projected muon with selection %s", varnam[ij - 1]);
    }
    mu_projection[ij] =
        fs->make<TH2F>(name, title, netamx + 1, -netamx / 2 - 0.5, netamx / 2 + 0.5, nphimx, 0.5, nphimx + 0.5);
  }

  for (int ij = 0; ij < 31; ij++) {
    mypow_2[ij] = std::pow(2, ij);
  }
  for (int ij = 0; ij < ringmx; ij++) {
    for (int jk = 0; jk < ncut + 10; jk++) {
      ncount[ij][jk] = 0.0;
    }
  }
}

HOCalibAnalyzer::~HOCalibAnalyzer() {
  edm::LogVerbatim("HOCalibAnalyzer") << " Total events = " << std::setw(7) << nevents[0] << " " << std::setw(7)
                                      << nevents[1] << " " << std::setw(7) << nevents[2] << " " << std::setw(7)
                                      << nevents[3] << " " << std::setw(7) << nevents[4] << " " << std::setw(7)
                                      << nevents[5] << " Selected events # is " << ipass;
}

//
// member functions
//
void HOCalibAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hoCalibVariableCollectionTag",
                          edm::InputTag("hoCalibProducer", "HOCalibVariableCollection"));
  desc.add<edm::InputTag>("hoInputTag", edm::InputTag("horeco"));
  desc.addUntracked<bool>("cosmic", true);
  desc.addUntracked<bool>("zeroField", false);
  desc.addUntracked<int>("HOSignalBins", 120);
  desc.addUntracked<double>("lowerRange", -1.0);
  desc.addUntracked<double>("upperRange", 29.0);
  desc.addUntracked<bool>("histFill", true);
  desc.addUntracked<bool>("treeFill", false);
  desc.addUntracked<double>("sigma", 0.05);
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("hoCalibAnalyzer", desc);
}

// ------------ method called to for each event  ------------
void HOCalibAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  nevents[0]++;

  ievt = iEvent.id().event();
  ilumi = iEvent.luminosityBlock();

  const edm::Handle<HOCalibVariableCollection>& HOCalib = iEvent.getHandle(tok_ho_);

  if (nevents[0] % 20000 == 1) {
    edm::LogVerbatim("HOCalibAnalyzer") << "nmuon event # " << std::setw(7) << nevents[0] << " " << std::setw(7)
                                        << nevents[1] << " " << std::setw(7) << nevents[2] << " " << std::setw(7)
                                        << nevents[3] << " " << std::setw(7) << nevents[4] << " " << std::setw(7)
                                        << nevents[5];
    edm::LogVerbatim("HOCalibAnalyzer") << " Run # " << iEvent.id().run() << " Evt # " << iEvent.id().event() << " "
                                        << int(HOCalib.isValid()) << " " << ipass;
  }

  if (HOCalib.isValid()) {
    nevents[1]++;
    nmuon = (*HOCalib).size();

    for (HOCalibVariableCollection::const_iterator hoC = (*HOCalib).begin(); hoC != (*HOCalib).end(); hoC++) {
      trkdr = (*hoC).trkdr;
      trkdz = (*hoC).trkdz;

      trkvx = (*hoC).trkvx;
      trkvy = (*hoC).trkvy;
      trkvz = (*hoC).trkvz;

      trkmm = (*hoC).trkmm;
      trkth = (*hoC).trkth;
      trkph = (*hoC).trkph;

      ndof = static_cast<int>((*hoC).ndof);
      chisq = (*hoC).chisq;
      momatho = (*hoC).momatho;

      therr = (*hoC).therr;
      pherr = (*hoC).pherr;
      trkph = (*hoC).trkph;

      if (!m_cosmic) {
        nprim = (*hoC).nprim;
        pileup = (*hoC).pileup;
        tkpt03 = (*hoC).tkpt03;
        ecal03 = (*hoC).ecal03;
        hcal03 = (*hoC).hcal03;
      }

      isect = (*hoC).isect;
      isect2 = (*hoC).isect2;
      hodx = (*hoC).hodx;
      hody = (*hoC).hody;
      hoang = (*hoC).hoang;

      tmphoang = std::sin(trkth) - hoang;

      htime = (*hoC).htime;
      hoflag = (*hoC).hoflag;
      for (int ij = 0; ij < 9; ij++) {
        hosig[ij] = (*hoC).hosig[ij];
        if (m_verbose)
          edm::LogVerbatim("HOCalibAnalyzer") << "hosig " << ij << " " << hosig[ij];
      }
      for (int ij = 0; ij < 18; ij++) {
        hocorsig[ij] = (*hoC).hocorsig[ij];
        if (m_verbose)
          edm::LogVerbatim("HOCalibAnalyzer") << "hocorsig " << ij << " " << hocorsig[ij];
      }
      hocro = (*hoC).hocro;
      for (int ij = 0; ij < 3; ij++) {
        caloen[ij] = (*hoC).caloen[ij];
      }

      int ipsall = 0;
      int ips0 = 0;
      int ips1 = 0;
      int ips2 = 0;
      int ips3 = 0;
      int ips4 = 0;
      int ips5 = 0;
      int ips6 = 0;
      int ips7 = 0;
      int ips8 = 0;
      int ips9 = 0;
      int ips10 = 0;
      int ips11 = 0;
      int ips12 = 0;
      int ips13 = 0;

      nevents[2]++;
      bool isZSps = (hosig[4] < -99.0) ? false : true;

      if ((!m_cosmic) && std::fabs(trkmm) < momatho)
        continue;

      nevents[3]++;
      if (std::fabs(trkth - angle_units::piRadians / 2) < 0.000001)
        continue;  //22OCT07
      nevents[4]++;

      int ieta = int((std::abs(isect) % 10000) / 100.) - 50;  //an offset to acodate -ve eta values
      if (std::abs(ieta) >= 16)
        continue;
      nevents[5]++;
      int iphi = std::abs(isect) % 100;

      int iring = 0;

      int iring2 = iring + 2;

      double abshoang = (m_cosmic) ? std::fabs(hoang) : hoang;

      double elos = 1.0 / std::max(0.1, std::abs(static_cast<double>(hoang)));

      if (!m_zeroField)
        elos *= ((14.9 + 0.96 * std::fabs(log(momatho * 2.8)) + 0.033 * momatho * (1.0 - std::pow(momatho, -0.33))) /
                 elosfact);

      if (m_cosmic) {
        if (std::abs(ndof) >= 20 && std::abs(ndof) < 55) {
          ips0 = mypow_2[0];
          ipsall += ips0;
        }
        if (chisq > 0 && chisq < 12) {
          ips1 = mypow_2[1];
          ipsall += ips1;
        }  //18Jan2008

        if (trkth > 0.3 && trkth < angle_units::piRadians - 0.3) {
          ips2 = mypow_2[2];
          ipsall += ips2;
        }  //No nead for pp evt
        if (trkph > -angle_units::piRadians + 0.1 && trkph < -0.1) {
          ips3 = mypow_2[3];
          ipsall += ips3;
        }  //No nead for pp evt

        if (therr < 0.02) {
          ips4 = mypow_2[4];
          ipsall += ips4;
        }
        if (pherr < 0.0002) {
          ips5 = mypow_2[5];
          ipsall += ips5;
        }
        if (abshoang > 0.60 && abshoang < 1.0) {
          ips6 = mypow_2[6];
          ipsall += ips6;
        }

        if (m_zeroField || (std::fabs(momatho) > 5.0 && std::fabs(momatho) < 2000.0)) {
          ips7 = mypow_2[7];
          ipsall += ips7;
        }

        if (nmuon >= 1 && nmuon <= 3) {
          ips8 = mypow_2[8];
          ipsall += ips8;
        }

        // initially for: if (hodx>0 && hody>0) { }
        ips9 = mypow_2[9];
        ipsall += ips9;

        ips10 = mypow_2[10];
        ipsall += ips10;

        if (iring2 == 2) {
          if (std::fabs(hodx) < 100 && std::fabs(hodx) > 2 && std::fabs(hocorsig[8]) < 40 &&
              std::fabs(hocorsig[8]) > 2) {
            ips11 = mypow_2[11];
            ipsall += ips11;
          }

          if (std::fabs(hody) < 100 && std::fabs(hody) > 2 && std::fabs(hocorsig[9]) < 40 &&
              std::fabs(hocorsig[9]) > 2) {
            ips12 = mypow_2[12];
            ipsall += ips12;
          }

        } else {
          if (std::fabs(hodx) < 100 && std::fabs(hodx) > 2) {
            ips11 = mypow_2[11];
            ipsall += ips11;
          }

          if (std::fabs(hody) < 100 && std::fabs(hody) > 2) {
            ips12 = mypow_2[12];
            ipsall += ips12;
          }
        }

        if (m_zeroField) {
          if (iring2 == 0) {
            if (htime > -60 && htime < 60) {
              ips13 = mypow_2[13];
              ipsall += ips13;
            }
          }
          if (iring2 == 1) {
            if (htime > -60 && htime < 60) {
              ips13 = mypow_2[13];
              ipsall += ips13;
            }
          }
          if (iring2 == 2) {
            if (htime > -60 && htime < 60) {
              ips13 = mypow_2[13];
              ipsall += ips13;
            }
          }
          if (iring2 == 3) {
            if (htime > -60 && htime < 60) {
              ips13 = mypow_2[13];
              ipsall += ips13;
            }
          }
          if (iring2 == 4) {
            if (htime > -60 && htime < 60) {
              ips13 = mypow_2[13];
              ipsall += ips13;
            }
          }
        } else {
          if (htime > -100 && htime < 100) {
            ips13 = mypow_2[13];
            ipsall += ips13;
          }
        }
      } else {
        if (std::abs(ndof) >= 10 && std::abs(ndof) < 25) {
          ips0 = mypow_2[0];
          ipsall += ips0;
        }
        if (chisq > 0 && chisq < 10) {
          ips1 = mypow_2[1];
          ipsall += ips1;
        }  //18Jan2008

        if (std::fabs(trkth - angle_units::piRadians / 2) < 21.5) {
          ips2 = mypow_2[2];
          ipsall += ips2;
        }  //No nead for pp evt
        if (std::fabs(trkph + angle_units::piRadians / 2) < 21.5) {
          ips3 = mypow_2[3];
          ipsall += ips3;
        }  //No nead for pp evt

        if (therr < 0.00002) {
          ips4 = mypow_2[4];
          ipsall += ips4;
        }
        if (pherr < 0.000002) {
          ips5 = mypow_2[5];
          ipsall += ips5;
        }
        // earlier: if (abshoang >0.40 && abshoang <1.0) {ips6 = mypow_2[6];  ipsall +=ips6;}
        if (tmphoang < 0.065) {
          ips6 = mypow_2[6];
          ipsall += ips6;
        }

        if (std::fabs(momatho) < 250.0 && std::fabs(momatho) > 15.0) {
          if (iring2 == 2) {
            ips7 = mypow_2[7];
            ipsall += ips7;
          }
          if ((iring2 == 1 || iring2 == 3) && std::fabs(momatho) > 17.0) {
            ips7 = mypow_2[7];
            ipsall += ips7;
          }
          if ((iring2 == 0 || iring2 == 4) && std::fabs(momatho) > 20.0) {
            ips7 = mypow_2[7];
            ipsall += ips7;
          }
        }

        if (nmuon >= 1 && nmuon <= 3) {
          ips8 = mypow_2[8];
          ipsall += ips8;
        }

        if (ndof > 0 && caloen[0] < 15.0) {
          ips9 = mypow_2[9];
          ipsall += ips9;
        }  //5.0
        if (tkpt03 < 5.0) {
          ips10 = mypow_2[10];
          ipsall += ips10;
        }  //4.0

        if (iring2 == 2) {
          if (std::fabs(hodx) < 100 && std::fabs(hodx) > 2 && std::fabs(hocorsig[8]) < 40 &&
              std::fabs(hocorsig[8]) > 2) {
            ips11 = mypow_2[11];
            ipsall += ips11;
          }

          if (std::fabs(hody) < 100 && std::fabs(hody) > 2 && std::fabs(hocorsig[9]) < 40 &&
              std::fabs(hocorsig[9]) > 2) {
            ips12 = mypow_2[12];
            ipsall += ips12;
          }

        } else {
          if (std::fabs(hodx) < 100 && std::fabs(hodx) > 2) {
            ips11 = mypow_2[11];
            ipsall += ips11;
          }

          if (std::fabs(hody) < 100 && std::fabs(hody) > 2) {
            ips12 = mypow_2[12];
            ipsall += ips12;
          }
        }

        if (iring2 == 0) {
          if (htime > -20 && htime < 20) {
            ips13 = mypow_2[13];
            ipsall += ips13;
          }
        }
        if (iring2 == 1) {
          if (htime > -20 && htime < 20) {
            ips13 = mypow_2[13];
            ipsall += ips13;
          }
        }
        if (iring2 == 2) {
          if (htime > -30 && htime < 20) {
            ips13 = mypow_2[13];
            ipsall += ips13;
          }
        }
        if (iring2 == 3) {
          if (htime > -20 && htime < 20) {
            ips13 = mypow_2[13];
            ipsall += ips13;
          }
        }
        if (iring2 == 4) {
          if (htime > -20 && htime < 20) {
            ips13 = mypow_2[13];
            ipsall += ips13;
          }
        }
      }
      int tmpxet = invert_HOieta(ieta);
      double nomHOSig = hosig[4] / elos;

      if (ipsall - ips0 == mypow_2[ncut] - mypow_2[0] - 1) {
        if (isZSps) {
          sigvsevt[iring2][0]->Fill(std::abs(ndof), nomHOSig);
          sig_eta_evt[tmpxet][0]->Fill(std::abs(ndof), nomHOSig);
        }
        variab[iring2][0]->Fill(std::abs(ndof));
      }
      if (ipsall - ips1 == mypow_2[ncut] - mypow_2[1] - 1) {
        if (isZSps) {
          sigvsevt[iring2][1]->Fill(chisq, nomHOSig);
          sig_eta_evt[tmpxet][1]->Fill(chisq, nomHOSig);
        }
        variab[iring2][1]->Fill(chisq);
      }
      if (ipsall - ips2 == mypow_2[ncut] - mypow_2[2] - 1) {
        if (isZSps) {
          sigvsevt[iring2][2]->Fill(trkth, nomHOSig);
          sig_eta_evt[tmpxet][2]->Fill(trkth, nomHOSig);
        }
        variab[iring2][2]->Fill(trkth);
      }
      if (ipsall - ips3 == mypow_2[ncut] - mypow_2[3] - 1) {
        if (isZSps) {
          sigvsevt[iring2][3]->Fill(trkph, nomHOSig);
          sig_eta_evt[tmpxet][3]->Fill(trkph, nomHOSig);
        }
        variab[iring2][3]->Fill(trkph);
      }
      if (ipsall - ips4 == mypow_2[ncut] - mypow_2[4] - 1) {
        if (isZSps) {
          sigvsevt[iring2][4]->Fill(1000 * therr, nomHOSig);
          sig_eta_evt[tmpxet][4]->Fill(1000 * therr, nomHOSig);
        }
        variab[iring2][4]->Fill(1000 * therr);
      }
      if (ipsall - ips5 == mypow_2[ncut] - mypow_2[5] - 1) {
        if (isZSps) {
          sigvsevt[iring2][5]->Fill(1000 * pherr, nomHOSig);
          sig_eta_evt[tmpxet][5]->Fill(1000 * pherr, nomHOSig);
        }
        variab[iring2][5]->Fill(1000 * pherr);
      }
      if (ipsall - ips6 == mypow_2[ncut] - mypow_2[6] - 1) {
        if (isZSps) {
          sigvsevt[iring2][6]->Fill(tmphoang, (nomHOSig)*abshoang);
          sig_eta_evt[tmpxet][6]->Fill(tmphoang, (nomHOSig)*abshoang);
        }
        variab[iring2][6]->Fill(tmphoang);
      }
      if (ipsall - ips7 == mypow_2[ncut] - mypow_2[7] - 1) {
        if (isZSps) {
          sigvsevt[iring2][7]->Fill(std::fabs(trkmm), nomHOSig);
          sig_eta_evt[tmpxet][7]->Fill(std::fabs(trkmm), nomHOSig);
        }
        variab[iring2][7]->Fill(std::fabs(trkmm));
      }
      if (ipsall - ips8 == mypow_2[ncut] - mypow_2[8] - 1) {
        if (isZSps) {
          sigvsevt[iring2][8]->Fill(nmuon, nomHOSig);
          sig_eta_evt[tmpxet][8]->Fill(nmuon, nomHOSig);
        }
        variab[iring2][8]->Fill(nmuon);
      }
      if (!m_cosmic) {
        if (ipsall - ips9 == mypow_2[ncut] - mypow_2[9] - 1) {
          if (isZSps) {
            sigvsevt[iring2][9]->Fill(caloen[0], nomHOSig);
            sig_eta_evt[tmpxet][9]->Fill(caloen[0], nomHOSig);
          }
          variab[iring2][9]->Fill(caloen[0]);
        }
      }
      if (ipsall - ips10 == mypow_2[ncut] - mypow_2[10] - 1) {
        if (isZSps) {
          sigvsevt[iring2][10]->Fill(tkpt03, nomHOSig);
          sig_eta_evt[tmpxet][10]->Fill(tkpt03, nomHOSig);
        }
        variab[iring2][10]->Fill(tkpt03);
      }
      if (ipsall - ips11 == mypow_2[ncut] - mypow_2[11] - 1) {
        if (isZSps) {
          sigvsevt[iring2][11]->Fill(hodx, nomHOSig);
          sig_eta_evt[tmpxet][11]->Fill(hodx, nomHOSig);
        }
        variab[iring2][11]->Fill(hodx);
      }
      if (ipsall - ips12 == mypow_2[ncut] - mypow_2[12] - 1) {
        if (isZSps) {
          sigvsevt[iring2][12]->Fill(hody, nomHOSig);
          sig_eta_evt[tmpxet][12]->Fill(hody, nomHOSig);
        }
        variab[iring2][12]->Fill(hody);
      }

      if (ipsall - ips13 == mypow_2[ncut] - mypow_2[13] - 1) {
        if (isZSps) {
          sigvsevt[iring2][13]->Fill(htime, nomHOSig);
          sig_eta_evt[tmpxet][13]->Fill(htime, nomHOSig);
        }
        variab[iring2][13]->Fill(htime);
      }

      if (isZSps) {
        sigvsevt[iring2 + ringmx][0]->Fill(std::abs(ndof), nomHOSig);
        sig_eta_evt[netamx + tmpxet][0]->Fill(std::abs(ndof), nomHOSig);
      }
      variab[iring2 + 5][0]->Fill(std::abs(ndof));

      ncount[iring2][0]++;
      if (isZSps) {
        ncount[iring2][1]++;
      }
      if (ips0 > 0) {
        if (isZSps) {
          ncount[iring2][10]++;
          sigvsevt[iring2 + ringmx][1]->Fill(chisq, nomHOSig);
          sig_eta_evt[netamx + tmpxet][1]->Fill(chisq, nomHOSig);
        }
        variab[iring2 + ringmx][1]->Fill(chisq);
        mu_projection[1]->Fill(ieta, iphi);
        if (ips1 > 0) {
          if (isZSps) {
            ncount[iring2][11]++;
            sigvsevt[iring2 + ringmx][2]->Fill(trkth, nomHOSig);
            sig_eta_evt[netamx + tmpxet][2]->Fill(trkth, nomHOSig);
          }
          variab[iring2 + ringmx][2]->Fill(trkth);
          mu_projection[2]->Fill(ieta, iphi);
          if (ips2 > 0) {
            if (isZSps) {
              ncount[iring2][12]++;
              sigvsevt[iring2 + ringmx][3]->Fill(trkph, nomHOSig);
              sig_eta_evt[netamx + tmpxet][3]->Fill(trkph, nomHOSig);
            }
            variab[iring2 + ringmx][3]->Fill(trkph);
            mu_projection[3]->Fill(ieta, iphi);
            if (ips3 > 0) {
              if (isZSps) {
                ncount[iring2][13]++;
                sigvsevt[iring2 + ringmx][4]->Fill(1000 * therr, nomHOSig);
                sig_eta_evt[netamx + tmpxet][4]->Fill(1000 * therr, nomHOSig);
              }
              variab[iring2 + ringmx][4]->Fill(1000 * therr);
              mu_projection[4]->Fill(ieta, iphi);
              if (ips4 > 0) {
                if (isZSps) {
                  ncount[iring2][14]++;
                  sigvsevt[iring2 + ringmx][5]->Fill(1000 * pherr, nomHOSig);
                  sig_eta_evt[netamx + tmpxet][5]->Fill(1000 * pherr, nomHOSig);
                }
                variab[iring2 + ringmx][5]->Fill(1000 * pherr);
                mu_projection[5]->Fill(ieta, iphi);
                if (ips5 > 0) {
                  if (isZSps) {
                    ncount[iring2][15]++;
                    sigvsevt[iring2 + ringmx][6]->Fill(tmphoang, (nomHOSig)*abshoang);
                    sig_eta_evt[netamx + tmpxet][6]->Fill(tmphoang, (nomHOSig)*abshoang);
                  }
                  variab[iring2 + ringmx][6]->Fill(tmphoang);
                  mu_projection[6]->Fill(ieta, iphi);
                  if (ips6 > 0) {
                    if (isZSps) {
                      ncount[iring2][16]++;
                      sigvsevt[iring2 + ringmx][7]->Fill(std::fabs(trkmm), nomHOSig);
                      sig_eta_evt[netamx + tmpxet][7]->Fill(std::fabs(trkmm), nomHOSig);
                    }
                    variab[iring2 + ringmx][7]->Fill(std::fabs(trkmm));
                    mu_projection[7]->Fill(ieta, iphi);
                    if (ips7 > 0) {
                      ncount[iring2][4]++;  //Efficiency of Muon detection
                      if (isZSps) {
                        ncount[iring2][17]++;
                        sigvsevt[iring2 + ringmx][8]->Fill(nmuon, nomHOSig);
                        sig_eta_evt[netamx + tmpxet][8]->Fill(nmuon, nomHOSig);
                      }
                      variab[iring2 + ringmx][8]->Fill(nmuon);
                      mu_projection[8]->Fill(ieta, iphi);
                      if (ips8 > 0) {
                        if (!m_cosmic) {
                          if (isZSps) {
                            ncount[iring2][18]++;
                            sigvsevt[iring2 + ringmx][9]->Fill(caloen[0], nomHOSig);
                            sig_eta_evt[netamx + tmpxet][9]->Fill(caloen[0], nomHOSig);
                          }
                          variab[iring2 + ringmx][9]->Fill(caloen[0]);
                          mu_projection[9]->Fill(ieta, iphi);
                        }
                        if (ips9 > 0) {
                          if (isZSps) {
                            ncount[iring2][19]++;
                            sigvsevt[iring2 + ringmx][10]->Fill(tkpt03, nomHOSig);
                            sig_eta_evt[netamx + tmpxet][10]->Fill(tkpt03, nomHOSig);
                          }
                          variab[iring2 + ringmx][10]->Fill(tkpt03);
                          mu_projection[10]->Fill(ieta, iphi);
                          if (ips10 > 0) {
                            ncount[iring2][3]++;  //Efficiency of Muon detection
                            if (isZSps) {
                              ncount[iring2][20]++;
                              sigvsevt[iring2 + ringmx][11]->Fill(hodx, nomHOSig);
                              sig_eta_evt[netamx + tmpxet][11]->Fill(hodx, nomHOSig);
                            }
                            variab[iring2 + ringmx][11]->Fill(hodx);
                            mu_projection[11]->Fill(ieta, iphi);

                            if (ips11 > 0) {
                              if (isZSps) {
                                ncount[iring2][21]++;
                                sigvsevt[iring2 + ringmx][12]->Fill(hody, nomHOSig);
                                sig_eta_evt[netamx + tmpxet][12]->Fill(hody, nomHOSig);
                              }
                              variab[iring2 + ringmx][12]->Fill(hody);
                              mu_projection[12]->Fill(ieta, iphi);

                              if (ips12 > 0) {
                                ncount[iring2][2]++;  //Efficiency of Muon detection
                                if (isZSps) {
                                  ncount[iring2][22]++;
                                  sigvsevt[iring2 + ringmx][13]->Fill(htime, nomHOSig);
                                  sig_eta_evt[tmpxet + ringmx][13]->Fill(htime, nomHOSig);
                                }
                                variab[iring2 + ringmx][13]->Fill(htime);
                                mu_projection[13]->Fill(ieta, iphi);

                                if (ips13 > 0) {
                                  if (isZSps) {
                                    ncount[iring2][23]++;
                                    mu_projection[14]->Fill(ieta, iphi);
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      if (isZSps) {
        sigvsevt[iring2 + 2 * ringmx][0]->Fill(std::abs(ndof), nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][1]->Fill(chisq, nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][2]->Fill(trkth, nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][3]->Fill(trkph, nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][4]->Fill(1000 * therr, nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][5]->Fill(1000 * pherr, nomHOSig);
        if (abshoang > 0.01) {
          sigvsevt[iring2 + 2 * ringmx][6]->Fill(tmphoang, (nomHOSig)*abshoang);
        }
        sigvsevt[iring2 + 2 * ringmx][7]->Fill(std::fabs(trkmm), nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][8]->Fill(nmuon, nomHOSig);
        if (!m_cosmic)
          sigvsevt[iring2 + 2 * ringmx][9]->Fill(caloen[0], nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][10]->Fill(tkpt03, nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][11]->Fill(hodx, nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][12]->Fill(hody, nomHOSig);
        sigvsevt[iring2 + 2 * ringmx][13]->Fill(htime, nomHOSig);

        sig_eta_evt[2 * netamx + tmpxet][0]->Fill(std::abs(ndof), nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][1]->Fill(chisq, nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][2]->Fill(trkth, nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][3]->Fill(trkph, nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][4]->Fill(1000 * therr, nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][5]->Fill(1000 * pherr, nomHOSig);
        if (abshoang > 0.01) {
          sig_eta_evt[2 * netamx + tmpxet][6]->Fill(tmphoang, (nomHOSig)*abshoang);
        }
        sig_eta_evt[2 * netamx + tmpxet][7]->Fill(std::fabs(trkmm), nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][8]->Fill(nmuon, nomHOSig);
        if (!m_cosmic)
          sig_eta_evt[2 * netamx + tmpxet][9]->Fill(caloen[0], nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][10]->Fill(tkpt03, nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][11]->Fill(hodx, nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][12]->Fill(hody, nomHOSig);
        sig_eta_evt[2 * netamx + tmpxet][13]->Fill(htime, nomHOSig);
      }

      variab[iring2 + 2 * ringmx][0]->Fill(std::abs(ndof));
      variab[iring2 + 2 * ringmx][1]->Fill(chisq);
      variab[iring2 + 2 * ringmx][2]->Fill(trkth);
      variab[iring2 + 2 * ringmx][3]->Fill(trkph);
      variab[iring2 + 2 * ringmx][4]->Fill(1000 * therr);
      variab[iring2 + 2 * ringmx][5]->Fill(1000 * pherr);
      variab[iring2 + 2 * ringmx][6]->Fill(tmphoang);
      variab[iring2 + 2 * ringmx][7]->Fill(std::fabs(trkmm));
      variab[iring2 + 2 * ringmx][8]->Fill(nmuon);
      if (!m_cosmic)
        variab[iring2 + 2 * ringmx][9]->Fill(caloen[0]);
      variab[iring2 + 2 * ringmx][10]->Fill(tkpt03);
      variab[iring2 + 2 * ringmx][11]->Fill(hodx);
      variab[iring2 + 2 * ringmx][12]->Fill(hody);
      variab[iring2 + 2 * ringmx][13]->Fill(htime);

      muonnm->Fill(nmuon);
      muonmm->Fill(trkmm);
      muonth->Fill(trkth * 180 / angle_units::piRadians);
      muonph->Fill(trkph * 180 / angle_units::piRadians);
      muonch->Fill(chisq);

      int iselect = (ipsall == mypow_2[ncut] - 1) ? 1 : 0;

      if (iselect == 1) {
        ipass++;
        sel_muonnm->Fill(nmuon);
        sel_muonmm->Fill(trkmm);
        sel_muonth->Fill(trkth * 180 / angle_units::piRadians);
        sel_muonph->Fill(trkph * 180 / angle_units::piRadians);
        sel_muonch->Fill(chisq);
        if (m_histFill && tmpxet >= 0 && tmpxet < netamx && iphi >= 0 && iphi < nphimx) {
          ho_indenergy[tmpxet][iphi - 1]->Fill(nomHOSig);
        }
        if (m_treeFill) {
          T1->Fill();
        }
      }
    }  //close the for loop: (HOCalibVariableCollection::const_iterator hoC=(*HOCalib).begin(); hoC!=(*HOCalib).end(); hoC++){
  }  //end of the if loop (isCosMu)
}

//define this as a plug-in
DEFINE_FWK_MODULE(HOCalibAnalyzer);
