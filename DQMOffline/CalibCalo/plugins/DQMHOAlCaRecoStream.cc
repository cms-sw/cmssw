// -*- C++ -*-
//
// Package:    DQMHOAlCaRecoStream
// Class:      DQMHOAlCaRecoStream
//
/**\class DQMHOAlCaRecoStream DQMHOAlCaRecoStream.cc
 DQMOffline/DQMHOAlCaRecoStream/src/DQMHOAlCaRecoStream.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gobinda Majumder
//         Created:  Mon Mar  2 12:33:08 CET 2009
//
//

// system include files
#include <string>

// user include files
#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class decleration
//

class DQMHOAlCaRecoStream : public DQMEDAnalyzer {
public:
  explicit DQMHOAlCaRecoStream(const edm::ParameterSet &);
  ~DQMHOAlCaRecoStream() override;

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  // ----------member data ---------------------------

  MonitorElement *hMuonMultipl;
  MonitorElement *hMuonMom;
  MonitorElement *hMuonEta;
  MonitorElement *hMuonPhi;

  MonitorElement *hDirCosine;
  MonitorElement *hHOTime;

  MonitorElement *hSigRing[5];
  MonitorElement *hPedRing[5];
  MonitorElement *hSignal3x3[9];

  int Nevents;
  int Nmuons;

  std::string theRootFileName;
  std::string folderName_;
  double m_sigmaValue;

  double m_lowRadPosInMuch;
  double m_highRadPosInMuch;

  int m_nbins;
  double m_lowEdge;
  double m_highEdge;

  bool saveToFile_;
  edm::EDGetTokenT<HOCalibVariableCollection> hoCalibVariableCollectionTag;
};

//
// constructors and destructor
//
DQMHOAlCaRecoStream::DQMHOAlCaRecoStream(const edm::ParameterSet &iConfig)
    : hoCalibVariableCollectionTag(
          consumes<HOCalibVariableCollection>(iConfig.getParameter<edm::InputTag>("hoCalibVariableCollectionTag"))) {
  // now do what ever initialization is needed

  theRootFileName = iConfig.getUntrackedParameter<std::string>("RootFileName", "tmp.root");
  folderName_ = iConfig.getUntrackedParameter<std::string>("folderName");
  m_sigmaValue = iConfig.getUntrackedParameter<double>("sigmaval", 0.2);
  m_lowRadPosInMuch = iConfig.getUntrackedParameter<double>("lowradposinmuch", 400.0);
  m_highRadPosInMuch = iConfig.getUntrackedParameter<double>("highradposinmuch", 480.0);
  m_lowEdge = iConfig.getUntrackedParameter<double>("lowedge", -2.0);
  m_highEdge = iConfig.getUntrackedParameter<double>("highedge", 6.0);
  m_nbins = iConfig.getUntrackedParameter<int>("nbins", 40);
  saveToFile_ = iConfig.getUntrackedParameter<bool>("saveToFile", false);
}

DQMHOAlCaRecoStream::~DQMHOAlCaRecoStream() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void DQMHOAlCaRecoStream::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  Nevents++;

  edm::Handle<HOCalibVariableCollection> HOCalib;
  bool isCosMu = true;

  iEvent.getByToken(hoCalibVariableCollectionTag, HOCalib);

  if (!HOCalib.isValid()) {
    LogDebug("") << "DQMHOAlCaRecoStream:: Error! can't get HOCalib product!" << std::endl;
    return;
  }

  if (isCosMu) {
    hMuonMultipl->Fill((*HOCalib).size(), 1.);
    if (!(*HOCalib).empty()) {
      for (HOCalibVariableCollection::const_iterator hoC = (*HOCalib).begin(); hoC != (*HOCalib).end(); hoC++) {
        // OK!!!!
        float okt = 2.;
        double okx = std::pow((*hoC).trkvx, okt) + std::pow((*hoC).trkvy, okt);
        ///////
        double dr = std::pow(okx, 0.5);
        if (dr < m_lowRadPosInMuch || dr > m_highRadPosInMuch)
          continue;

        if ((*hoC).isect < 0)
          continue;
        if (fabs((*hoC).trkth - acos(-1.) / 2) < 0.000001)
          continue;
        int ieta = int((std::abs((*hoC).isect) % 10000) / 100.) - 30;

        if (std::abs(ieta) >= 16)
          continue;

        Nmuons++;

        hMuonMom->Fill((*hoC).trkmm, 1.0);
        hMuonEta->Fill(-log(tan((*hoC).trkth / 2.0)), 1.0);
        hMuonPhi->Fill((*hoC).trkph, 1.0);
        hDirCosine->Fill((*hoC).hoang, 1.0);
        hHOTime->Fill((*hoC).htime, 1.0);

        double energy = (*hoC).hosig[4];
        double pedval = (*hoC).hocro;
        int iring = 0;
        if (ieta >= -15 && ieta <= -11) {
          iring = -2;
        } else if (ieta >= -10 && ieta <= -5) {
          iring = -1;
        } else if (ieta >= 5 && ieta <= 10) {
          iring = 1;
        } else if (ieta >= 11 && ieta <= 15) {
          iring = 2;
        }

        hSigRing[iring + 2]->Fill(energy, 1.0);
        hPedRing[iring + 2]->Fill(pedval, 1.0);

        for (int k = 0; k < 9; k++) {
          hSignal3x3[k]->Fill((*hoC).hosig[k]);
        }
      }  // for (HOCalibVariableCollection::const_iterator hoC=(*HOCalib).begin()
    }    // if ((*HOCalib).size() >0 ) {
  }      // if (isCosMu) {
}

// ------------ method called once each job just before starting event loop
// ------------
void DQMHOAlCaRecoStream::bookHistograms(DQMStore::IBooker &ibooker,
                                         edm::Run const &irun,
                                         edm::EventSetup const &isetup) {
  ibooker.setCurrentFolder(folderName_);

  char title[200];
  char name[200];

  hMuonMom = ibooker.book1D("hMuonMom", "Muon momentum (GeV)", 50, -100, 100);
  hMuonMom->setAxisTitle("Muon momentum (GeV)", 1);

  hMuonEta = ibooker.book1D("hMuonEta", "Pseudo-rapidity of muon", 50, -1.5, 1.5);
  hMuonEta->setAxisTitle("Pseudo-rapidity of muon", 1);

  hMuonPhi = ibooker.book1D("hMuonPhi", "Azimuthal angle of muon", 24, -acos(-1), acos(-1));
  hMuonPhi->setAxisTitle("Azimuthal angle of muon", 1);

  hMuonMultipl = ibooker.book1D("hMuonMultipl", "Muon Multiplicity", 10, 0.5, 10.5);
  hMuonMultipl->setAxisTitle("Muon Multiplicity", 1);

  hDirCosine = ibooker.book1D("hDirCosine", "Direction Cosine of muon at HO tower", 50, -1., 1.);
  hDirCosine->setAxisTitle("Direction Cosine of muon at HO tower", 1);

  hHOTime = ibooker.book1D("hHOTime", "HO time distribution", 60, -20, 100.);
  hHOTime->setAxisTitle("HO time distribution", 1);

  for (int i = 0; i < 5; i++) {
    sprintf(name, "hSigRing_%i", i - 2);
    sprintf(title, "HO signal in Ring_%i", i - 2);
    hSigRing[i] = ibooker.book1D(name, title, m_nbins, m_lowEdge, m_highEdge);
    hSigRing[i]->setAxisTitle(title, 1);

    sprintf(name, "hPedRing_%i", i - 2);
    sprintf(title, "HO Pedestal in Ring_%i", i - 2);
    hPedRing[i] = ibooker.book1D(name, title, m_nbins, m_lowEdge, m_highEdge);
    hPedRing[i]->setAxisTitle(title, 1);
  }

  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int k = 3 * (i + 1) + j + 1;

      sprintf(title, "hSignal3x3_deta%i_dphi%i", i, j);
      hSignal3x3[k] = ibooker.book1D(title, title, m_nbins, m_lowEdge, m_highEdge);
      hSignal3x3[k]->setAxisTitle(title, 1);
    }
  }

  Nevents = 0;
  Nmuons = 0;
}

DEFINE_FWK_MODULE(DQMHOAlCaRecoStream);
