#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/GenSimInfo.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
// track associator
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1I.h"
#include "TH2D.h"
#include "TTree.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <list>
#include <vector>

class StudyCaloGen : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit StudyCaloGen(const edm::ParameterSet &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override {}

  void fillTrack(
      GlobalPoint &posVec, math::XYZTLorentzVector &momVec, GlobalPoint &posECAL, int pdgId, bool okECAL, bool accpet);
  void fillIsolatedTrack(math::XYZTLorentzVector &momVec, GlobalPoint &posECAL, int pdgId);
  void bookHistograms();
  void clearTreeVectors();
  int particleCode(int);

  static constexpr int NPBins_ = 3;
  static constexpr int NEtaBins_ = 4;
  static constexpr int PBins_ = 32, EtaBins_ = 60, Particles = 12;
  int nEventProc;
  double genPartPBins_[NPBins_ + 1], genPartEtaBins_[NEtaBins_ + 1];
  double ptMin_, etaMax_, pCutIsolate_;
  bool a_Isolation_;
  std::string genSrc_;

  edm::EDGetTokenT<edm::HepMCProduct> tok_hepmc_;
  edm::EDGetTokenT<reco::GenParticleCollection> tok_genParticles_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_caloTopology_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_topo_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
  edm::ESGetToken<HepPDT::ParticleDataTable, PDTRecord> tok_pdt_;

  bool useHepMC_;
  double a_coneR_, a_charIsoR_, a_neutIsoR_, a_mipR_;
  int verbosity_;

  TH1I *h_NEventProc;
  TH2D *h_pEta[Particles];
  TTree *tree_;

  std::vector<double> *t_isoTrkPAll;
  std::vector<double> *t_isoTrkPtAll;
  std::vector<double> *t_isoTrkPhiAll;
  std::vector<double> *t_isoTrkEtaAll;
  std::vector<double> *t_isoTrkPdgIdAll;
  std::vector<double> *t_isoTrkDEtaAll;
  std::vector<double> *t_isoTrkDPhiAll;

  std::vector<double> *t_isoTrkP;
  std::vector<double> *t_isoTrkPt;
  std::vector<double> *t_isoTrkEne;
  std::vector<double> *t_isoTrkEta;
  std::vector<double> *t_isoTrkPhi;
  std::vector<double> *t_isoTrkEtaEC;
  std::vector<double> *t_isoTrkPhiEC;
  std::vector<double> *t_isoTrkPdgId;

  std::vector<double> *t_maxNearP31x31;
  std::vector<double> *t_cHadronEne31x31, *t_cHadronEne31x31_1, *t_cHadronEne31x31_2, *t_cHadronEne31x31_3;
  std::vector<double> *t_nHadronEne31x31;
  std::vector<double> *t_photonEne31x31;
  std::vector<double> *t_eleEne31x31;
  std::vector<double> *t_muEne31x31;

  std::vector<double> *t_maxNearP25x25;
  std::vector<double> *t_cHadronEne25x25, *t_cHadronEne25x25_1, *t_cHadronEne25x25_2, *t_cHadronEne25x25_3;
  std::vector<double> *t_nHadronEne25x25;
  std::vector<double> *t_photonEne25x25;
  std::vector<double> *t_eleEne25x25;
  std::vector<double> *t_muEne25x25;

  std::vector<double> *t_maxNearP21x21;
  std::vector<double> *t_cHadronEne21x21, *t_cHadronEne21x21_1, *t_cHadronEne21x21_2, *t_cHadronEne21x21_3;
  std::vector<double> *t_nHadronEne21x21;
  std::vector<double> *t_photonEne21x21;
  std::vector<double> *t_eleEne21x21;
  std::vector<double> *t_muEne21x21;

  std::vector<double> *t_maxNearP15x15;
  std::vector<double> *t_cHadronEne15x15, *t_cHadronEne15x15_1, *t_cHadronEne15x15_2, *t_cHadronEne15x15_3;
  std::vector<double> *t_nHadronEne15x15;
  std::vector<double> *t_photonEne15x15;
  std::vector<double> *t_eleEne15x15;
  std::vector<double> *t_muEne15x15;

  std::vector<double> *t_maxNearP11x11;
  std::vector<double> *t_cHadronEne11x11, *t_cHadronEne11x11_1, *t_cHadronEne11x11_2, *t_cHadronEne11x11_3;
  std::vector<double> *t_nHadronEne11x11;
  std::vector<double> *t_photonEne11x11;
  std::vector<double> *t_eleEne11x11;
  std::vector<double> *t_muEne11x11;

  std::vector<double> *t_maxNearP9x9;
  std::vector<double> *t_cHadronEne9x9, *t_cHadronEne9x9_1, *t_cHadronEne9x9_2, *t_cHadronEne9x9_3;
  std::vector<double> *t_nHadronEne9x9;
  std::vector<double> *t_photonEne9x9;
  std::vector<double> *t_eleEne9x9;
  std::vector<double> *t_muEne9x9;

  std::vector<double> *t_maxNearP7x7;
  std::vector<double> *t_cHadronEne7x7, *t_cHadronEne7x7_1, *t_cHadronEne7x7_2, *t_cHadronEne7x7_3;
  std::vector<double> *t_nHadronEne7x7;
  std::vector<double> *t_photonEne7x7;
  std::vector<double> *t_eleEne7x7;
  std::vector<double> *t_muEne7x7;

  std::vector<double> *t_maxNearP3x3;
  std::vector<double> *t_cHadronEne3x3, *t_cHadronEne3x3_1, *t_cHadronEne3x3_2, *t_cHadronEne3x3_3;
  std::vector<double> *t_nHadronEne3x3;
  std::vector<double> *t_photonEne3x3;
  std::vector<double> *t_eleEne3x3;
  std::vector<double> *t_muEne3x3;

  std::vector<double> *t_maxNearP1x1;
  std::vector<double> *t_cHadronEne1x1, *t_cHadronEne1x1_1, *t_cHadronEne1x1_2, *t_cHadronEne1x1_3;
  std::vector<double> *t_nHadronEne1x1;
  std::vector<double> *t_photonEne1x1;
  std::vector<double> *t_eleEne1x1;
  std::vector<double> *t_muEne1x1;

  std::vector<double> *t_maxNearPHC1x1;
  std::vector<double> *t_cHadronEneHC1x1, *t_cHadronEneHC1x1_1, *t_cHadronEneHC1x1_2, *t_cHadronEneHC1x1_3;
  std::vector<double> *t_nHadronEneHC1x1;
  std::vector<double> *t_photonEneHC1x1;
  std::vector<double> *t_eleEneHC1x1;
  std::vector<double> *t_muEneHC1x1;

  std::vector<double> *t_maxNearPHC3x3;
  std::vector<double> *t_cHadronEneHC3x3, *t_cHadronEneHC3x3_1, *t_cHadronEneHC3x3_2, *t_cHadronEneHC3x3_3;
  std::vector<double> *t_nHadronEneHC3x3;
  std::vector<double> *t_photonEneHC3x3;
  std::vector<double> *t_eleEneHC3x3;
  std::vector<double> *t_muEneHC3x3;

  std::vector<double> *t_maxNearPHC5x5;
  std::vector<double> *t_cHadronEneHC5x5, *t_cHadronEneHC5x5_1, *t_cHadronEneHC5x5_2, *t_cHadronEneHC5x5_3;
  std::vector<double> *t_nHadronEneHC5x5;
  std::vector<double> *t_photonEneHC5x5;
  std::vector<double> *t_eleEneHC5x5;
  std::vector<double> *t_muEneHC5x5;

  std::vector<double> *t_maxNearPHC7x7;
  std::vector<double> *t_cHadronEneHC7x7, *t_cHadronEneHC7x7_1, *t_cHadronEneHC7x7_2, *t_cHadronEneHC7x7_3;
  std::vector<double> *t_nHadronEneHC7x7;
  std::vector<double> *t_photonEneHC7x7;
  std::vector<double> *t_eleEneHC7x7;
  std::vector<double> *t_muEneHC7x7;

  std::vector<double> *t_maxNearPR;
  std::vector<double> *t_cHadronEneR, *t_cHadronEneR_1, *t_cHadronEneR_2, *t_cHadronEneR_3;
  std::vector<double> *t_nHadronEneR;
  std::vector<double> *t_photonEneR;
  std::vector<double> *t_eleEneR;
  std::vector<double> *t_muEneR;

  std::vector<double> *t_maxNearPIsoR;
  std::vector<double> *t_cHadronEneIsoR, *t_cHadronEneIsoR_1, *t_cHadronEneIsoR_2, *t_cHadronEneIsoR_3;
  std::vector<double> *t_nHadronEneIsoR;
  std::vector<double> *t_photonEneIsoR;
  std::vector<double> *t_eleEneIsoR;
  std::vector<double> *t_muEneIsoR;

  std::vector<double> *t_maxNearPHCR;
  std::vector<double> *t_cHadronEneHCR, *t_cHadronEneHCR_1, *t_cHadronEneHCR_2, *t_cHadronEneHCR_3;
  std::vector<double> *t_nHadronEneHCR;
  std::vector<double> *t_photonEneHCR;
  std::vector<double> *t_eleEneHCR;
  std::vector<double> *t_muEneHCR;

  std::vector<double> *t_maxNearPIsoHCR;
  std::vector<double> *t_cHadronEneIsoHCR, *t_cHadronEneIsoHCR_1, *t_cHadronEneIsoHCR_2, *t_cHadronEneIsoHCR_3;
  std::vector<double> *t_nHadronEneIsoHCR;
  std::vector<double> *t_photonEneIsoHCR;
  std::vector<double> *t_eleEneIsoHCR;
  std::vector<double> *t_muEneIsoHCR;

  spr::genSimInfo isoinfo1x1, isoinfo3x3, isoinfo7x7, isoinfo9x9, isoinfo11x11;
  spr::genSimInfo isoinfo15x15, isoinfo21x21, isoinfo25x25, isoinfo31x31;
  spr::genSimInfo isoinfoHC1x1, isoinfoHC3x3, isoinfoHC5x5, isoinfoHC7x7;
  spr::genSimInfo isoinfoR, isoinfoIsoR, isoinfoHCR, isoinfoIsoHCR;
};

StudyCaloGen::StudyCaloGen(const edm::ParameterSet &iConfig)
    : ptMin_(iConfig.getUntrackedParameter<double>("PTMin", 1.0)),
      etaMax_(iConfig.getUntrackedParameter<double>("MaxChargedHadronEta", 2.5)),
      pCutIsolate_(iConfig.getUntrackedParameter<double>("PMaxIsolation", 20.0)),
      a_Isolation_(iConfig.getUntrackedParameter<bool>("UseConeIsolation", false)),
      genSrc_(iConfig.getUntrackedParameter("GenSrc", std::string("generatorSmeared"))),
      useHepMC_(iConfig.getUntrackedParameter<bool>("UseHepMC", false)),
      a_coneR_(iConfig.getUntrackedParameter<double>("ConeRadius", 34.98)),
      a_mipR_(iConfig.getUntrackedParameter<double>("ConeRadiusMIP", 14.0)),
      verbosity_(iConfig.getUntrackedParameter<int>("Verbosity", 0)) {
  usesResource(TFileService::kSharedResource);

  a_charIsoR_ = a_coneR_ + 28.9;
  a_neutIsoR_ = a_charIsoR_ * 0.726;

  tok_hepmc_ = consumes<edm::HepMCProduct>(edm::InputTag(genSrc_));
  tok_genParticles_ = consumes<reco::GenParticleCollection>(edm::InputTag(genSrc_));

  if (!strcmp("Dummy", genSrc_.c_str())) {
    if (useHepMC_)
      genSrc_ = "generatorSmeared";
    else
      genSrc_ = "genParticles";
  }
  edm::LogVerbatim("IsoTrack") << "Generator Source " << genSrc_ << " Use HepMC " << useHepMC_ << " ptMin " << ptMin_
                               << " etaMax " << etaMax_ << "\n a_coneR " << a_coneR_ << " a_charIsoR " << a_charIsoR_
                               << " a_neutIsoR " << a_neutIsoR_ << " a_mipR " << a_mipR_ << " debug " << verbosity_
                               << "\nIsolation Flag " << a_Isolation_ << " with cut " << pCutIsolate_ << " GeV";

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_caloTopology_ = esConsumes<CaloTopology, CaloTopologyRecord>();
  tok_topo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  tok_pdt_ = esConsumes<HepPDT::ParticleDataTable, PDTRecord>();
}

void StudyCaloGen::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("GenSrc", "genParticles");
  desc.addUntracked<bool>("UseHepMC", false);
  desc.addUntracked<double>("ChargedHadronSeedP", 1.0);
  desc.addUntracked<double>("PTMin", 1.0);
  desc.addUntracked<double>("MaxChargedHadronEta", 2.5);
  desc.addUntracked<double>("ConeRadius", 34.98);
  desc.addUntracked<double>("ConeRadiusMIP", 14.0);
  desc.addUntracked<bool>("UseConeIsolation", true);
  desc.addUntracked<double>("PMaxIsolation", 5.0);
  desc.addUntracked<int>("Verbosity", 0);
  descriptions.add("studyCaloGen", desc);
}

void StudyCaloGen::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  clearTreeVectors();

  nEventProc++;
  const MagneticField *bField = &iSetup.getData(tok_magField_);

  // get particle data table
  const HepPDT::ParticleDataTable *pdt = &iSetup.getData(tok_pdt_);

  // get handle to HEPMCProduct
  edm::Handle<edm::HepMCProduct> hepmc;
  edm::Handle<reco::GenParticleCollection> genParticles;
  if (useHepMC_)
    iEvent.getByToken(tok_hepmc_, hepmc);
  else
    iEvent.getByToken(tok_genParticles_, genParticles);

  const CaloGeometry *geo = &iSetup.getData(tok_geom_);
  const CaloTopology *caloTopology = &iSetup.getData(tok_caloTopology_);
  const HcalTopology *theHBHETopology = &iSetup.getData(tok_topo_);

  GlobalPoint posVec, posECAL;
  math::XYZTLorentzVector momVec;
  if (verbosity_ > 0)
    edm::LogVerbatim("IsoTrack") << "event number " << iEvent.id().event();
  if (useHepMC_) {
    const HepMC::GenEvent *myGenEvent = hepmc->GetEvent();
    std::vector<spr::propagatedGenTrackID> trackIDs = spr::propagateCALO(myGenEvent, pdt, geo, bField, etaMax_, false);

    for (unsigned int indx = 0; indx < trackIDs.size(); ++indx) {
      int charge = trackIDs[indx].charge;
      HepMC::GenEvent::particle_const_iterator p = trackIDs[indx].trkItr;
      momVec = math::XYZTLorentzVector(
          (*p)->momentum().px(), (*p)->momentum().py(), (*p)->momentum().pz(), (*p)->momentum().e());
      if (verbosity_ > 1)
        edm::LogVerbatim("IsoTrack") << "trkIndx " << indx << " pdgid " << trackIDs[indx].pdgId << " charge " << charge
                                     << " momVec " << momVec;
      // only stable particles avoiding electrons and muons
      if (trackIDs[indx].ok && (std::abs(trackIDs[indx].pdgId) < 11 || std::abs(trackIDs[indx].pdgId) >= 21)) {
        // consider particles within a phased space
        if (momVec.Pt() > ptMin_ && std::abs(momVec.eta()) < etaMax_) {
          posVec = GlobalPoint(0.1 * (*p)->production_vertex()->position().x(),
                               0.1 * (*p)->production_vertex()->position().y(),
                               0.1 * (*p)->production_vertex()->position().z());
          posECAL = trackIDs[indx].pointECAL;
          fillTrack(posVec, momVec, posECAL, trackIDs[indx].pdgId, trackIDs[indx].okECAL, true);
          if (verbosity_ > 1)
            edm::LogVerbatim("IsoTrack") << "posECAL " << posECAL << " okECAL " << trackIDs[indx].okECAL << "okHCAL "
                                         << trackIDs[indx].okHCAL;
          if (trackIDs[indx].okECAL) {
            if (std::abs(charge) > 0) {
              spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 0, 0, isoinfo1x1, false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 1, 1, isoinfo3x3, false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 3, 3, isoinfo7x7, false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 4, 4, isoinfo9x9, false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 5, 5, isoinfo11x11, false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 7, 7, isoinfo15x15, false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 10, 10, isoinfo21x21, false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 12, 12, isoinfo25x25, false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 15, 15, isoinfo31x31, false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL,
                               p,
                               trackIDs,
                               geo,
                               caloTopology,
                               a_mipR_,
                               trackIDs[indx].directionECAL,
                               isoinfoR,
                               false);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL,
                               p,
                               trackIDs,
                               geo,
                               caloTopology,
                               a_neutIsoR_,
                               trackIDs[indx].directionECAL,
                               isoinfoIsoR,
                               false);
              if (trackIDs[indx].okHCAL) {
                spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 0, 0, isoinfoHC1x1, false);
                spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 1, 1, isoinfoHC3x3, false);
                spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 2, 2, isoinfoHC5x5, false);
                spr::hGenSimInfo(trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 3, 3, isoinfoHC7x7, false);
                spr::hGenSimInfo(trackIDs[indx].detIdHCAL,
                                 p,
                                 trackIDs,
                                 geo,
                                 theHBHETopology,
                                 a_coneR_,
                                 trackIDs[indx].directionHCAL,
                                 isoinfoHCR,
                                 false);
                spr::hGenSimInfo(trackIDs[indx].detIdHCAL,
                                 p,
                                 trackIDs,
                                 geo,
                                 theHBHETopology,
                                 a_charIsoR_,
                                 trackIDs[indx].directionHCAL,
                                 isoinfoIsoHCR,
                                 false);
              }

              bool saveTrack = true;
              if (a_Isolation_)
                saveTrack = (isoinfoR.maxNearP < pCutIsolate_);
              else
                saveTrack = (isoinfo7x7.maxNearP < pCutIsolate_);
              if (saveTrack)
                fillIsolatedTrack(momVec, posECAL, trackIDs[indx].pdgId);
            }
          }
        } else {  // stabale particles within |eta|=2.5
          fillTrack(posVec, momVec, posECAL, 0, false, false);
        }
      }
    }

    unsigned int indx;
    HepMC::GenEvent::particle_const_iterator p;
    for (p = myGenEvent->particles_begin(), indx = 0; p != myGenEvent->particles_end(); ++p, ++indx) {
      int pdgId = ((*p)->pdg_id());
      int ix = particleCode(pdgId);
      if (ix >= 0) {
        double pp = (*p)->momentum().rho();
        double eta = (*p)->momentum().eta();
        h_pEta[ix]->Fill(pp, eta);
      }
    }
  } else {  // loop over gen particles
    std::vector<spr::propagatedGenParticleID> trackIDs =
        spr::propagateCALO(genParticles, pdt, geo, bField, etaMax_, (verbosity_ > 0));

    for (unsigned int indx = 0; indx < trackIDs.size(); ++indx) {
      int charge = trackIDs[indx].charge;
      reco::GenParticleCollection::const_iterator p = trackIDs[indx].trkItr;

      momVec = math::XYZTLorentzVector(p->momentum().x(), p->momentum().y(), p->momentum().z(), p->energy());
      if (verbosity_ > 1)
        edm::LogVerbatim("IsoTrack") << "trkIndx " << indx << " pdgid " << trackIDs[indx].pdgId << " charge " << charge
                                     << " momVec " << momVec;
      // only stable particles avoiding electrons and muons
      if (trackIDs[indx].ok && std::abs(trackIDs[indx].pdgId) > 21) {
        // consider particles within a phased space
        if (verbosity_ > 1)
          edm::LogVerbatim("IsoTrack") << " pt " << momVec.Pt() << " eta " << momVec.eta();
        if (momVec.Pt() > ptMin_ && std::abs(momVec.eta()) < etaMax_) {
          posVec = GlobalPoint(p->vertex().x(), p->vertex().y(), p->vertex().z());
          posECAL = trackIDs[indx].pointECAL;
          if (verbosity_ > 0)
            edm::LogVerbatim("IsoTrack") << "posECAL " << posECAL << " okECAL " << trackIDs[indx].okECAL << "okHCAL "
                                         << trackIDs[indx].okHCAL;
          fillTrack(posVec, momVec, posECAL, trackIDs[indx].pdgId, trackIDs[indx].okECAL, true);
          if (trackIDs[indx].okECAL) {
            if (std::abs(charge) > 0) {
              spr::eGenSimInfo(
                  trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 0, 0, isoinfo1x1, verbosity_ > 1);
              spr::eGenSimInfo(
                  trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 1, 1, isoinfo3x3, verbosity_ > 0);
              spr::eGenSimInfo(
                  trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 3, 3, isoinfo7x7, verbosity_ > 1);
              spr::eGenSimInfo(
                  trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 4, 4, isoinfo9x9, verbosity_ > 1);
              spr::eGenSimInfo(
                  trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 5, 5, isoinfo11x11, verbosity_ > 1);
              spr::eGenSimInfo(
                  trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 7, 7, isoinfo15x15, verbosity_ > 1);
              spr::eGenSimInfo(
                  trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 10, 10, isoinfo21x21, verbosity_ > 1);
              spr::eGenSimInfo(
                  trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 12, 12, isoinfo25x25, verbosity_ > 1);
              spr::eGenSimInfo(
                  trackIDs[indx].detIdECAL, p, trackIDs, geo, caloTopology, 15, 15, isoinfo31x31, verbosity_ > 1);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL,
                               p,
                               trackIDs,
                               geo,
                               caloTopology,
                               a_mipR_,
                               trackIDs[indx].directionECAL,
                               isoinfoR,
                               verbosity_ > 1);
              spr::eGenSimInfo(trackIDs[indx].detIdECAL,
                               p,
                               trackIDs,
                               geo,
                               caloTopology,
                               a_neutIsoR_,
                               trackIDs[indx].directionECAL,
                               isoinfoIsoR,
                               verbosity_ > 1);
              if (trackIDs[indx].okHCAL) {
                spr::hGenSimInfo(
                    trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 0, 0, isoinfoHC1x1, verbosity_ > 1);
                spr::hGenSimInfo(
                    trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 1, 1, isoinfoHC3x3, verbosity_ > 1);
                spr::hGenSimInfo(
                    trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 2, 2, isoinfoHC5x5, verbosity_ > 1);
                spr::hGenSimInfo(
                    trackIDs[indx].detIdHCAL, p, trackIDs, theHBHETopology, 3, 3, isoinfoHC7x7, verbosity_ > 1);
                spr::hGenSimInfo(trackIDs[indx].detIdHCAL,
                                 p,
                                 trackIDs,
                                 geo,
                                 theHBHETopology,
                                 a_coneR_,
                                 trackIDs[indx].directionHCAL,
                                 isoinfoHCR,
                                 verbosity_ > 1);
                spr::hGenSimInfo(trackIDs[indx].detIdHCAL,
                                 p,
                                 trackIDs,
                                 geo,
                                 theHBHETopology,
                                 a_charIsoR_,
                                 trackIDs[indx].directionHCAL,
                                 isoinfoIsoHCR,
                                 verbosity_ > 1);
              }

              bool saveTrack = true;
              if (a_Isolation_)
                saveTrack = (isoinfoIsoR.maxNearP < pCutIsolate_);
              else
                saveTrack = (isoinfo7x7.maxNearP < pCutIsolate_);
              if (saveTrack)
                fillIsolatedTrack(momVec, posECAL, trackIDs[indx].pdgId);
            }
          }
        } else {  // stabale particles within |eta|=2.5
          fillTrack(posVec, momVec, posECAL, 0, false, false);
        }
      }
    }  // loop over gen particles

    unsigned int indx;
    reco::GenParticleCollection::const_iterator p;
    for (p = genParticles->begin(), indx = 0; p != genParticles->end(); ++p, ++indx) {
      int pdgId = (p->pdgId());
      int ix = particleCode(pdgId);
      if (ix >= 0) {
        double pp = (p->momentum()).R();
        double eta = (p->momentum()).Eta();
        h_pEta[ix]->Fill(pp, eta);
      }
    }
  }

  //t_nEvtProc->push_back(nEventProc);
  h_NEventProc->SetBinContent(1, nEventProc);
  tree_->Fill();
}

void StudyCaloGen::beginJob() {
  nEventProc = 0;

  double tempgen_TH[NPBins_ + 1] = {0.0, 5.0, 12.0, 300.0};
  for (int i = 0; i <= NPBins_; i++)
    genPartPBins_[i] = tempgen_TH[i];

  double tempgen_Eta[NEtaBins_ + 1] = {0.0, 0.5, 1.1, 1.7, 2.3};
  for (int i = 0; i <= NEtaBins_; i++)
    genPartEtaBins_[i] = tempgen_Eta[i];

  bookHistograms();
}

void StudyCaloGen::fillTrack(
    GlobalPoint &posVec, math::XYZTLorentzVector &momVec, GlobalPoint &posECAL, int pdgId, bool okECAL, bool accept) {
  if (accept) {
    t_isoTrkPAll->push_back(momVec.P());
    t_isoTrkPtAll->push_back(momVec.Pt());
    t_isoTrkPhiAll->push_back(momVec.phi());
    t_isoTrkEtaAll->push_back(momVec.eta());
    t_isoTrkPdgIdAll->push_back(pdgId);
    if (okECAL) {
      double phi1 = momVec.phi();
      double phi2 = (posECAL - posVec).phi();
      double dphi = reco::deltaPhi(phi1, phi2);
      double deta = momVec.eta() - (posECAL - posVec).eta();
      t_isoTrkDPhiAll->push_back(dphi);
      t_isoTrkDEtaAll->push_back(deta);
    } else {
      t_isoTrkDPhiAll->push_back(999.0);
      t_isoTrkDEtaAll->push_back(999.0);
    }
  } else {
    t_isoTrkDPhiAll->push_back(-999.0);
    t_isoTrkDEtaAll->push_back(-999.0);
  }
}

void StudyCaloGen::fillIsolatedTrack(math::XYZTLorentzVector &momVec, GlobalPoint &posECAL, int pdgId) {
  t_isoTrkP->push_back(momVec.P());
  t_isoTrkPt->push_back(momVec.Pt());
  t_isoTrkEne->push_back(momVec.E());
  t_isoTrkEta->push_back(momVec.eta());
  t_isoTrkPhi->push_back(momVec.phi());
  t_isoTrkEtaEC->push_back(posECAL.eta());
  t_isoTrkPhiEC->push_back(posECAL.phi());
  t_isoTrkPdgId->push_back(pdgId);

  t_maxNearP31x31->push_back(isoinfo31x31.maxNearP);
  t_cHadronEne31x31->push_back(isoinfo31x31.cHadronEne);
  t_cHadronEne31x31_1->push_back(isoinfo31x31.cHadronEne_[0]);
  t_cHadronEne31x31_2->push_back(isoinfo31x31.cHadronEne_[1]);
  t_cHadronEne31x31_3->push_back(isoinfo31x31.cHadronEne_[2]);
  t_nHadronEne31x31->push_back(isoinfo31x31.nHadronEne);
  t_photonEne31x31->push_back(isoinfo31x31.photonEne);
  t_eleEne31x31->push_back(isoinfo31x31.eleEne);
  t_muEne31x31->push_back(isoinfo31x31.muEne);

  t_maxNearP25x25->push_back(isoinfo25x25.maxNearP);
  t_cHadronEne25x25->push_back(isoinfo25x25.cHadronEne);
  t_cHadronEne25x25_1->push_back(isoinfo25x25.cHadronEne_[0]);
  t_cHadronEne25x25_2->push_back(isoinfo25x25.cHadronEne_[1]);
  t_cHadronEne25x25_3->push_back(isoinfo25x25.cHadronEne_[2]);
  t_nHadronEne25x25->push_back(isoinfo25x25.nHadronEne);
  t_photonEne25x25->push_back(isoinfo25x25.photonEne);
  t_eleEne25x25->push_back(isoinfo25x25.eleEne);
  t_muEne25x25->push_back(isoinfo25x25.muEne);

  t_maxNearP21x21->push_back(isoinfo21x21.maxNearP);
  t_cHadronEne21x21->push_back(isoinfo21x21.cHadronEne);
  t_cHadronEne21x21_1->push_back(isoinfo21x21.cHadronEne_[0]);
  t_cHadronEne21x21_2->push_back(isoinfo21x21.cHadronEne_[1]);
  t_cHadronEne21x21_3->push_back(isoinfo21x21.cHadronEne_[2]);
  t_nHadronEne21x21->push_back(isoinfo21x21.nHadronEne);
  t_photonEne21x21->push_back(isoinfo21x21.photonEne);
  t_eleEne21x21->push_back(isoinfo21x21.eleEne);
  t_muEne21x21->push_back(isoinfo21x21.muEne);

  t_maxNearP15x15->push_back(isoinfo15x15.maxNearP);
  t_cHadronEne15x15->push_back(isoinfo15x15.cHadronEne);
  t_cHadronEne15x15_1->push_back(isoinfo15x15.cHadronEne_[0]);
  t_cHadronEne15x15_2->push_back(isoinfo15x15.cHadronEne_[1]);
  t_cHadronEne15x15_3->push_back(isoinfo15x15.cHadronEne_[2]);
  t_nHadronEne15x15->push_back(isoinfo15x15.nHadronEne);
  t_photonEne15x15->push_back(isoinfo15x15.photonEne);
  t_eleEne15x15->push_back(isoinfo15x15.eleEne);
  t_muEne15x15->push_back(isoinfo15x15.muEne);

  t_maxNearP11x11->push_back(isoinfo11x11.maxNearP);
  t_cHadronEne11x11->push_back(isoinfo11x11.cHadronEne);
  t_cHadronEne11x11_1->push_back(isoinfo11x11.cHadronEne_[0]);
  t_cHadronEne11x11_2->push_back(isoinfo11x11.cHadronEne_[1]);
  t_cHadronEne11x11_3->push_back(isoinfo11x11.cHadronEne_[2]);
  t_nHadronEne11x11->push_back(isoinfo11x11.nHadronEne);
  t_photonEne11x11->push_back(isoinfo11x11.photonEne);
  t_eleEne11x11->push_back(isoinfo11x11.eleEne);
  t_muEne11x11->push_back(isoinfo11x11.muEne);

  t_maxNearP9x9->push_back(isoinfo9x9.maxNearP);
  t_cHadronEne9x9->push_back(isoinfo9x9.cHadronEne);
  t_cHadronEne9x9_1->push_back(isoinfo9x9.cHadronEne_[0]);
  t_cHadronEne9x9_2->push_back(isoinfo9x9.cHadronEne_[1]);
  t_cHadronEne9x9_3->push_back(isoinfo9x9.cHadronEne_[2]);
  t_nHadronEne9x9->push_back(isoinfo9x9.nHadronEne);
  t_photonEne9x9->push_back(isoinfo9x9.photonEne);
  t_eleEne9x9->push_back(isoinfo9x9.eleEne);
  t_muEne9x9->push_back(isoinfo9x9.muEne);

  t_maxNearP7x7->push_back(isoinfo7x7.maxNearP);
  t_cHadronEne7x7->push_back(isoinfo7x7.cHadronEne);
  t_cHadronEne7x7_1->push_back(isoinfo7x7.cHadronEne_[0]);
  t_cHadronEne7x7_2->push_back(isoinfo7x7.cHadronEne_[1]);
  t_cHadronEne7x7_3->push_back(isoinfo7x7.cHadronEne_[2]);
  t_nHadronEne7x7->push_back(isoinfo7x7.nHadronEne);
  t_photonEne7x7->push_back(isoinfo7x7.photonEne);
  t_eleEne7x7->push_back(isoinfo7x7.eleEne);
  t_muEne7x7->push_back(isoinfo7x7.muEne);

  t_maxNearP3x3->push_back(isoinfo3x3.maxNearP);
  t_cHadronEne3x3->push_back(isoinfo3x3.cHadronEne);
  t_cHadronEne3x3_1->push_back(isoinfo3x3.cHadronEne_[0]);
  t_cHadronEne3x3_2->push_back(isoinfo3x3.cHadronEne_[1]);
  t_cHadronEne3x3_3->push_back(isoinfo3x3.cHadronEne_[2]);
  t_nHadronEne3x3->push_back(isoinfo3x3.nHadronEne);
  t_photonEne3x3->push_back(isoinfo3x3.photonEne);
  t_eleEne3x3->push_back(isoinfo3x3.eleEne);
  t_muEne3x3->push_back(isoinfo3x3.muEne);

  t_maxNearP1x1->push_back(isoinfo1x1.maxNearP);
  t_cHadronEne1x1->push_back(isoinfo1x1.cHadronEne);
  t_cHadronEne1x1_1->push_back(isoinfo1x1.cHadronEne_[0]);
  t_cHadronEne1x1_2->push_back(isoinfo1x1.cHadronEne_[1]);
  t_cHadronEne1x1_3->push_back(isoinfo1x1.cHadronEne_[2]);
  t_nHadronEne1x1->push_back(isoinfo1x1.nHadronEne);
  t_photonEne1x1->push_back(isoinfo1x1.photonEne);
  t_eleEne1x1->push_back(isoinfo1x1.eleEne);
  t_muEne1x1->push_back(isoinfo1x1.muEne);

  t_maxNearPHC1x1->push_back(isoinfoHC1x1.maxNearP);
  t_cHadronEneHC1x1->push_back(isoinfoHC1x1.cHadronEne);
  t_cHadronEneHC1x1_1->push_back(isoinfoHC1x1.cHadronEne_[0]);
  t_cHadronEneHC1x1_2->push_back(isoinfoHC1x1.cHadronEne_[1]);
  t_cHadronEneHC1x1_3->push_back(isoinfoHC1x1.cHadronEne_[2]);
  t_nHadronEneHC1x1->push_back(isoinfoHC1x1.nHadronEne);
  t_photonEneHC1x1->push_back(isoinfoHC1x1.photonEne);
  t_eleEneHC1x1->push_back(isoinfoHC1x1.eleEne);
  t_muEneHC1x1->push_back(isoinfoHC1x1.muEne);

  t_maxNearPHC3x3->push_back(isoinfoHC3x3.maxNearP);
  t_cHadronEneHC3x3->push_back(isoinfoHC3x3.cHadronEne);
  t_cHadronEneHC3x3_1->push_back(isoinfoHC3x3.cHadronEne_[0]);
  t_cHadronEneHC3x3_2->push_back(isoinfoHC3x3.cHadronEne_[1]);
  t_cHadronEneHC3x3_3->push_back(isoinfoHC3x3.cHadronEne_[2]);
  t_nHadronEneHC3x3->push_back(isoinfoHC3x3.nHadronEne);
  t_photonEneHC3x3->push_back(isoinfoHC3x3.photonEne);
  t_eleEneHC3x3->push_back(isoinfoHC3x3.eleEne);
  t_muEneHC3x3->push_back(isoinfoHC3x3.muEne);

  t_maxNearPHC5x5->push_back(isoinfoHC5x5.maxNearP);
  t_cHadronEneHC5x5->push_back(isoinfoHC5x5.cHadronEne);
  t_cHadronEneHC5x5_1->push_back(isoinfoHC5x5.cHadronEne_[0]);
  t_cHadronEneHC5x5_2->push_back(isoinfoHC5x5.cHadronEne_[1]);
  t_cHadronEneHC5x5_3->push_back(isoinfoHC5x5.cHadronEne_[2]);
  t_nHadronEneHC5x5->push_back(isoinfoHC5x5.nHadronEne);
  t_photonEneHC5x5->push_back(isoinfoHC5x5.photonEne);
  t_eleEneHC5x5->push_back(isoinfoHC5x5.eleEne);
  t_muEneHC5x5->push_back(isoinfoHC5x5.muEne);

  t_maxNearPHC7x7->push_back(isoinfoHC7x7.maxNearP);
  t_cHadronEneHC7x7->push_back(isoinfoHC7x7.cHadronEne);
  t_cHadronEneHC7x7_1->push_back(isoinfoHC7x7.cHadronEne_[0]);
  t_cHadronEneHC7x7_2->push_back(isoinfoHC7x7.cHadronEne_[1]);
  t_cHadronEneHC7x7_3->push_back(isoinfoHC7x7.cHadronEne_[2]);
  t_nHadronEneHC7x7->push_back(isoinfoHC7x7.nHadronEne);
  t_photonEneHC7x7->push_back(isoinfoHC7x7.photonEne);
  t_eleEneHC7x7->push_back(isoinfoHC7x7.eleEne);
  t_muEneHC7x7->push_back(isoinfoHC7x7.muEne);

  t_maxNearPR->push_back(isoinfoR.maxNearP);
  t_cHadronEneR->push_back(isoinfoR.cHadronEne);
  t_cHadronEneR_1->push_back(isoinfoR.cHadronEne_[0]);
  t_cHadronEneR_2->push_back(isoinfoR.cHadronEne_[1]);
  t_cHadronEneR_3->push_back(isoinfoR.cHadronEne_[2]);
  t_nHadronEneR->push_back(isoinfoR.nHadronEne);
  t_photonEneR->push_back(isoinfoR.photonEne);
  t_eleEneR->push_back(isoinfoR.eleEne);
  t_muEneR->push_back(isoinfoR.muEne);

  t_maxNearPIsoR->push_back(isoinfoIsoR.maxNearP);
  t_cHadronEneIsoR->push_back(isoinfoIsoR.cHadronEne);
  t_cHadronEneIsoR_1->push_back(isoinfoIsoR.cHadronEne_[0]);
  t_cHadronEneIsoR_2->push_back(isoinfoIsoR.cHadronEne_[1]);
  t_cHadronEneIsoR_3->push_back(isoinfoIsoR.cHadronEne_[2]);
  t_nHadronEneIsoR->push_back(isoinfoIsoR.nHadronEne);
  t_photonEneIsoR->push_back(isoinfoIsoR.photonEne);
  t_eleEneIsoR->push_back(isoinfoIsoR.eleEne);
  t_muEneIsoR->push_back(isoinfoIsoR.muEne);

  t_maxNearPHCR->push_back(isoinfoHCR.maxNearP);
  t_cHadronEneHCR->push_back(isoinfoHCR.cHadronEne);
  t_cHadronEneHCR_1->push_back(isoinfoHCR.cHadronEne_[0]);
  t_cHadronEneHCR_2->push_back(isoinfoHCR.cHadronEne_[1]);
  t_cHadronEneHCR_3->push_back(isoinfoHCR.cHadronEne_[2]);
  t_nHadronEneHCR->push_back(isoinfoHCR.nHadronEne);
  t_photonEneHCR->push_back(isoinfoHCR.photonEne);
  t_eleEneHCR->push_back(isoinfoHCR.eleEne);
  t_muEneHCR->push_back(isoinfoHCR.muEne);

  t_maxNearPIsoHCR->push_back(isoinfoIsoHCR.maxNearP);
  t_cHadronEneIsoHCR->push_back(isoinfoIsoHCR.cHadronEne);
  t_cHadronEneIsoHCR_1->push_back(isoinfoIsoHCR.cHadronEne_[0]);
  t_cHadronEneIsoHCR_2->push_back(isoinfoIsoHCR.cHadronEne_[1]);
  t_cHadronEneIsoHCR_3->push_back(isoinfoIsoHCR.cHadronEne_[2]);
  t_nHadronEneIsoHCR->push_back(isoinfoIsoHCR.nHadronEne);
  t_photonEneIsoHCR->push_back(isoinfoIsoHCR.photonEne);
  t_eleEneIsoHCR->push_back(isoinfoIsoHCR.eleEne);
  t_muEneIsoHCR->push_back(isoinfoIsoHCR.muEne);
}

void StudyCaloGen::bookHistograms() {
  edm::Service<TFileService> fs;
  //char hname[100], htit[100];

  h_NEventProc = fs->make<TH1I>("h_NEventProc", "h_NEventProc", 2, -0.5, 0.5);

  double pBin[PBins_ + 1] = {0.0,   2.0,   4.0,   6.0,   8.0,   10.0,  20.0,  30.0,  40.0,  50.0,  60.0,
                             70.0,  80.0,  90.0,  100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0,
                             500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0};
  double etaBin[EtaBins_ + 1] = {-3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8,
                                 -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5,
                                 -0.4, -0.3, -0.2, -0.1, 0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,
                                 0.9,  1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.0,  2.1,
                                 2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0};
  std::string particle[Particles] = {
      "electron", "positron", "#gamma", "#pi^+", "#pi^-", "K^+", "K^-", "p", "n", "pbar", "nbar", "K^0_L"};
  TFileDirectory dir1 = fs->mkdir("pEta");
  char name[20], title[50];
  for (int i = 0; i < Particles; ++i) {
    sprintf(name, "pEta%d", i);
    sprintf(title, "#eta vs momentum for %s", particle[i].c_str());
    h_pEta[i] = dir1.make<TH2D>(name, title, PBins_, pBin, EtaBins_, etaBin);
  }

  // build the tree
  tree_ = fs->make<TTree>("StudyCaloGen", "StudyCaloGen");

  t_isoTrkPAll = new std::vector<double>();
  t_isoTrkPtAll = new std::vector<double>();
  t_isoTrkPhiAll = new std::vector<double>();
  t_isoTrkEtaAll = new std::vector<double>();
  t_isoTrkDPhiAll = new std::vector<double>();
  t_isoTrkDEtaAll = new std::vector<double>();
  t_isoTrkPdgIdAll = new std::vector<double>();

  t_isoTrkP = new std::vector<double>();
  t_isoTrkPt = new std::vector<double>();
  t_isoTrkEne = new std::vector<double>();
  t_isoTrkEta = new std::vector<double>();
  t_isoTrkPhi = new std::vector<double>();
  t_isoTrkEtaEC = new std::vector<double>();
  t_isoTrkPhiEC = new std::vector<double>();
  t_isoTrkPdgId = new std::vector<double>();

  t_maxNearP31x31 = new std::vector<double>();
  t_cHadronEne31x31 = new std::vector<double>();
  t_cHadronEne31x31_1 = new std::vector<double>();
  t_cHadronEne31x31_2 = new std::vector<double>();
  t_cHadronEne31x31_3 = new std::vector<double>();
  t_nHadronEne31x31 = new std::vector<double>();
  t_photonEne31x31 = new std::vector<double>();
  t_eleEne31x31 = new std::vector<double>();
  t_muEne31x31 = new std::vector<double>();

  t_maxNearP25x25 = new std::vector<double>();
  t_cHadronEne25x25 = new std::vector<double>();
  t_cHadronEne25x25_1 = new std::vector<double>();
  t_cHadronEne25x25_2 = new std::vector<double>();
  t_cHadronEne25x25_3 = new std::vector<double>();
  t_nHadronEne25x25 = new std::vector<double>();
  t_photonEne25x25 = new std::vector<double>();
  t_eleEne25x25 = new std::vector<double>();
  t_muEne25x25 = new std::vector<double>();

  t_maxNearP21x21 = new std::vector<double>();
  t_cHadronEne21x21 = new std::vector<double>();
  t_cHadronEne21x21_1 = new std::vector<double>();
  t_cHadronEne21x21_2 = new std::vector<double>();
  t_cHadronEne21x21_3 = new std::vector<double>();
  t_nHadronEne21x21 = new std::vector<double>();
  t_photonEne21x21 = new std::vector<double>();
  t_eleEne21x21 = new std::vector<double>();
  t_muEne21x21 = new std::vector<double>();

  t_maxNearP15x15 = new std::vector<double>();
  t_cHadronEne15x15 = new std::vector<double>();
  t_cHadronEne15x15_1 = new std::vector<double>();
  t_cHadronEne15x15_2 = new std::vector<double>();
  t_cHadronEne15x15_3 = new std::vector<double>();
  t_nHadronEne15x15 = new std::vector<double>();
  t_photonEne15x15 = new std::vector<double>();
  t_eleEne15x15 = new std::vector<double>();
  t_muEne15x15 = new std::vector<double>();

  t_maxNearP11x11 = new std::vector<double>();
  t_cHadronEne11x11 = new std::vector<double>();
  t_cHadronEne11x11_1 = new std::vector<double>();
  t_cHadronEne11x11_2 = new std::vector<double>();
  t_cHadronEne11x11_3 = new std::vector<double>();
  t_nHadronEne11x11 = new std::vector<double>();
  t_photonEne11x11 = new std::vector<double>();
  t_eleEne11x11 = new std::vector<double>();
  t_muEne11x11 = new std::vector<double>();

  t_maxNearP9x9 = new std::vector<double>();
  t_cHadronEne9x9 = new std::vector<double>();
  t_cHadronEne9x9_1 = new std::vector<double>();
  t_cHadronEne9x9_2 = new std::vector<double>();
  t_cHadronEne9x9_3 = new std::vector<double>();
  t_nHadronEne9x9 = new std::vector<double>();
  t_photonEne9x9 = new std::vector<double>();
  t_eleEne9x9 = new std::vector<double>();
  t_muEne9x9 = new std::vector<double>();

  t_maxNearP7x7 = new std::vector<double>();
  t_cHadronEne7x7 = new std::vector<double>();
  t_cHadronEne7x7_1 = new std::vector<double>();
  t_cHadronEne7x7_2 = new std::vector<double>();
  t_cHadronEne7x7_3 = new std::vector<double>();
  t_nHadronEne7x7 = new std::vector<double>();
  t_photonEne7x7 = new std::vector<double>();
  t_eleEne7x7 = new std::vector<double>();
  t_muEne7x7 = new std::vector<double>();

  t_maxNearP3x3 = new std::vector<double>();
  t_cHadronEne3x3 = new std::vector<double>();
  t_cHadronEne3x3_1 = new std::vector<double>();
  t_cHadronEne3x3_2 = new std::vector<double>();
  t_cHadronEne3x3_3 = new std::vector<double>();
  t_nHadronEne3x3 = new std::vector<double>();
  t_photonEne3x3 = new std::vector<double>();
  t_eleEne3x3 = new std::vector<double>();
  t_muEne3x3 = new std::vector<double>();

  t_maxNearP1x1 = new std::vector<double>();
  t_cHadronEne1x1 = new std::vector<double>();
  t_cHadronEne1x1_1 = new std::vector<double>();
  t_cHadronEne1x1_2 = new std::vector<double>();
  t_cHadronEne1x1_3 = new std::vector<double>();
  t_nHadronEne1x1 = new std::vector<double>();
  t_photonEne1x1 = new std::vector<double>();
  t_eleEne1x1 = new std::vector<double>();
  t_muEne1x1 = new std::vector<double>();

  t_maxNearPHC1x1 = new std::vector<double>();
  t_cHadronEneHC1x1 = new std::vector<double>();
  t_cHadronEneHC1x1_1 = new std::vector<double>();
  t_cHadronEneHC1x1_2 = new std::vector<double>();
  t_cHadronEneHC1x1_3 = new std::vector<double>();
  t_nHadronEneHC1x1 = new std::vector<double>();
  t_photonEneHC1x1 = new std::vector<double>();
  t_eleEneHC1x1 = new std::vector<double>();
  t_muEneHC1x1 = new std::vector<double>();

  t_maxNearPHC3x3 = new std::vector<double>();
  t_cHadronEneHC3x3 = new std::vector<double>();
  t_cHadronEneHC3x3_1 = new std::vector<double>();
  t_cHadronEneHC3x3_2 = new std::vector<double>();
  t_cHadronEneHC3x3_3 = new std::vector<double>();
  t_nHadronEneHC3x3 = new std::vector<double>();
  t_photonEneHC3x3 = new std::vector<double>();
  t_eleEneHC3x3 = new std::vector<double>();
  t_muEneHC3x3 = new std::vector<double>();

  t_maxNearPHC5x5 = new std::vector<double>();
  t_cHadronEneHC5x5 = new std::vector<double>();
  t_cHadronEneHC5x5_1 = new std::vector<double>();
  t_cHadronEneHC5x5_2 = new std::vector<double>();
  t_cHadronEneHC5x5_3 = new std::vector<double>();
  t_nHadronEneHC5x5 = new std::vector<double>();
  t_photonEneHC5x5 = new std::vector<double>();
  t_eleEneHC5x5 = new std::vector<double>();
  t_muEneHC5x5 = new std::vector<double>();

  t_maxNearPHC7x7 = new std::vector<double>();
  t_cHadronEneHC7x7 = new std::vector<double>();
  t_cHadronEneHC7x7_1 = new std::vector<double>();
  t_cHadronEneHC7x7_2 = new std::vector<double>();
  t_cHadronEneHC7x7_3 = new std::vector<double>();
  t_nHadronEneHC7x7 = new std::vector<double>();
  t_photonEneHC7x7 = new std::vector<double>();
  t_eleEneHC7x7 = new std::vector<double>();
  t_muEneHC7x7 = new std::vector<double>();

  t_maxNearPR = new std::vector<double>();
  t_cHadronEneR = new std::vector<double>();
  t_cHadronEneR_1 = new std::vector<double>();
  t_cHadronEneR_2 = new std::vector<double>();
  t_cHadronEneR_3 = new std::vector<double>();
  t_nHadronEneR = new std::vector<double>();
  t_photonEneR = new std::vector<double>();
  t_eleEneR = new std::vector<double>();
  t_muEneR = new std::vector<double>();

  t_maxNearPIsoR = new std::vector<double>();
  t_cHadronEneIsoR = new std::vector<double>();
  t_cHadronEneIsoR_1 = new std::vector<double>();
  t_cHadronEneIsoR_2 = new std::vector<double>();
  t_cHadronEneIsoR_3 = new std::vector<double>();
  t_nHadronEneIsoR = new std::vector<double>();
  t_photonEneIsoR = new std::vector<double>();
  t_eleEneIsoR = new std::vector<double>();
  t_muEneIsoR = new std::vector<double>();

  t_maxNearPHCR = new std::vector<double>();
  t_cHadronEneHCR = new std::vector<double>();
  t_cHadronEneHCR_1 = new std::vector<double>();
  t_cHadronEneHCR_2 = new std::vector<double>();
  t_cHadronEneHCR_3 = new std::vector<double>();
  t_nHadronEneHCR = new std::vector<double>();
  t_photonEneHCR = new std::vector<double>();
  t_eleEneHCR = new std::vector<double>();
  t_muEneHCR = new std::vector<double>();

  t_maxNearPIsoHCR = new std::vector<double>();
  t_cHadronEneIsoHCR = new std::vector<double>();
  t_cHadronEneIsoHCR_1 = new std::vector<double>();
  t_cHadronEneIsoHCR_2 = new std::vector<double>();
  t_cHadronEneIsoHCR_3 = new std::vector<double>();
  t_nHadronEneIsoHCR = new std::vector<double>();
  t_photonEneIsoHCR = new std::vector<double>();
  t_eleEneIsoHCR = new std::vector<double>();
  t_muEneIsoHCR = new std::vector<double>();

  tree_->Branch("t_isoTrkPAll", "std::vector<double>", &t_isoTrkPAll);
  tree_->Branch("t_isoTrkPtAll", "std::vector<double>", &t_isoTrkPtAll);
  tree_->Branch("t_isoTrkPhiAll", "std::vector<double>", &t_isoTrkPhiAll);
  tree_->Branch("t_isoTrkEtaAll", "std::vector<double>", &t_isoTrkEtaAll);
  tree_->Branch("t_isoTrkDPhiAll", "std::vector<double>", &t_isoTrkDPhiAll);
  tree_->Branch("t_isoTrkDEtaAll", "std::vector<double>", &t_isoTrkDEtaAll);
  tree_->Branch("t_isoTrkPdgIdAll", "std::vector<double>", &t_isoTrkPdgIdAll);

  tree_->Branch("t_isoTrkP", "std::vector<double>", &t_isoTrkP);
  tree_->Branch("t_isoTrkPt", "std::vector<double>", &t_isoTrkPt);
  tree_->Branch("t_isoTrkEne", "std::vector<double>", &t_isoTrkEne);
  tree_->Branch("t_isoTrkEta", "std::vector<double>", &t_isoTrkEta);
  tree_->Branch("t_isoTrkPhi", "std::vector<double>", &t_isoTrkPhi);
  tree_->Branch("t_isoTrkEtaEC", "std::vector<double>", &t_isoTrkEtaEC);
  tree_->Branch("t_isoTrkPhiEC", "std::vector<double>", &t_isoTrkPhiEC);
  tree_->Branch("t_isoTrkPdgId", "std::vector<double>", &t_isoTrkPdgId);

  tree_->Branch("t_maxNearP31x31", "std::vector<double>", &t_maxNearP31x31);
  tree_->Branch("t_cHadronEne31x31", "std::vector<double>", &t_cHadronEne31x31);
  tree_->Branch("t_cHadronEne31x31_1", "std::vector<double>", &t_cHadronEne31x31_1);
  tree_->Branch("t_cHadronEne31x31_2", "std::vector<double>", &t_cHadronEne31x31_2);
  tree_->Branch("t_cHadronEne31x31_3", "std::vector<double>", &t_cHadronEne31x31_3);
  tree_->Branch("t_nHadronEne31x31", "std::vector<double>", &t_nHadronEne31x31);
  tree_->Branch("t_photonEne31x31", "std::vector<double>", &t_photonEne31x31);
  tree_->Branch("t_eleEne31x31", "std::vector<double>", &t_eleEne31x31);
  tree_->Branch("t_muEne31x31", "std::vector<double>", &t_muEne31x31);

  tree_->Branch("t_maxNearP25x25", "std::vector<double>", &t_maxNearP25x25);
  tree_->Branch("t_cHadronEne25x25", "std::vector<double>", &t_cHadronEne25x25);
  tree_->Branch("t_cHadronEne25x25_1", "std::vector<double>", &t_cHadronEne25x25_1);
  tree_->Branch("t_cHadronEne25x25_2", "std::vector<double>", &t_cHadronEne25x25_2);
  tree_->Branch("t_cHadronEne25x25_3", "std::vector<double>", &t_cHadronEne25x25_3);
  tree_->Branch("t_nHadronEne25x25", "std::vector<double>", &t_nHadronEne25x25);
  tree_->Branch("t_photonEne25x25", "std::vector<double>", &t_photonEne25x25);
  tree_->Branch("t_eleEne25x25", "std::vector<double>", &t_eleEne25x25);
  tree_->Branch("t_muEne25x25", "std::vector<double>", &t_muEne25x25);

  tree_->Branch("t_maxNearP21x21", "std::vector<double>", &t_maxNearP21x21);
  tree_->Branch("t_cHadronEne21x21", "std::vector<double>", &t_cHadronEne21x21);
  tree_->Branch("t_cHadronEne21x21_1", "std::vector<double>", &t_cHadronEne21x21_1);
  tree_->Branch("t_cHadronEne21x21_2", "std::vector<double>", &t_cHadronEne21x21_2);
  tree_->Branch("t_cHadronEne21x21_3", "std::vector<double>", &t_cHadronEne21x21_3);
  tree_->Branch("t_nHadronEne21x21", "std::vector<double>", &t_nHadronEne21x21);
  tree_->Branch("t_photonEne21x21", "std::vector<double>", &t_photonEne21x21);
  tree_->Branch("t_eleEne21x21", "std::vector<double>", &t_eleEne21x21);
  tree_->Branch("t_muEne21x21", "std::vector<double>", &t_muEne21x21);

  tree_->Branch("t_maxNearP15x15", "std::vector<double>", &t_maxNearP15x15);
  tree_->Branch("t_cHadronEne15x15", "std::vector<double>", &t_cHadronEne15x15);
  tree_->Branch("t_cHadronEne15x15_1", "std::vector<double>", &t_cHadronEne15x15_1);
  tree_->Branch("t_cHadronEne15x15_2", "std::vector<double>", &t_cHadronEne15x15_2);
  tree_->Branch("t_cHadronEne15x15_3", "std::vector<double>", &t_cHadronEne15x15_3);
  tree_->Branch("t_nHadronEne15x15", "std::vector<double>", &t_nHadronEne15x15);
  tree_->Branch("t_photonEne15x15", "std::vector<double>", &t_photonEne15x15);
  tree_->Branch("t_eleEne15x15", "std::vector<double>", &t_eleEne15x15);
  tree_->Branch("t_muEne15x15", "std::vector<double>", &t_muEne15x15);

  tree_->Branch("t_maxNearP11x11", "std::vector<double>", &t_maxNearP11x11);
  tree_->Branch("t_cHadronEne11x11", "std::vector<double>", &t_cHadronEne11x11);
  tree_->Branch("t_cHadronEne11x11_1", "std::vector<double>", &t_cHadronEne11x11_1);
  tree_->Branch("t_cHadronEne11x11_2", "std::vector<double>", &t_cHadronEne11x11_2);
  tree_->Branch("t_cHadronEne11x11_3", "std::vector<double>", &t_cHadronEne11x11_3);
  tree_->Branch("t_nHadronEne11x11", "std::vector<double>", &t_nHadronEne11x11);
  tree_->Branch("t_photonEne11x11", "std::vector<double>", &t_photonEne11x11);
  tree_->Branch("t_eleEne11x11", "std::vector<double>", &t_eleEne11x11);
  tree_->Branch("t_muEne11x11", "std::vector<double>", &t_muEne11x11);

  tree_->Branch("t_maxNearP9x9", "std::vector<double>", &t_maxNearP9x9);
  tree_->Branch("t_cHadronEne9x9", "std::vector<double>", &t_cHadronEne9x9);
  tree_->Branch("t_cHadronEne9x9_1", "std::vector<double>", &t_cHadronEne9x9_1);
  tree_->Branch("t_cHadronEne9x9_2", "std::vector<double>", &t_cHadronEne9x9_2);
  tree_->Branch("t_cHadronEne9x9_3", "std::vector<double>", &t_cHadronEne9x9_3);
  tree_->Branch("t_nHadronEne9x9", "std::vector<double>", &t_nHadronEne9x9);
  tree_->Branch("t_photonEne9x9", "std::vector<double>", &t_photonEne9x9);
  tree_->Branch("t_eleEne9x9", "std::vector<double>", &t_eleEne9x9);
  tree_->Branch("t_muEne9x9", "std::vector<double>", &t_muEne9x9);

  tree_->Branch("t_maxNearP7x7", "std::vector<double>", &t_maxNearP7x7);
  tree_->Branch("t_cHadronEne7x7", "std::vector<double>", &t_cHadronEne7x7);
  tree_->Branch("t_cHadronEne7x7_1", "std::vector<double>", &t_cHadronEne7x7_1);
  tree_->Branch("t_cHadronEne7x7_2", "std::vector<double>", &t_cHadronEne7x7_2);
  tree_->Branch("t_cHadronEne7x7_3", "std::vector<double>", &t_cHadronEne7x7_3);
  tree_->Branch("t_nHadronEne7x7", "std::vector<double>", &t_nHadronEne7x7);
  tree_->Branch("t_photonEne7x7", "std::vector<double>", &t_photonEne7x7);
  tree_->Branch("t_eleEne7x7", "std::vector<double>", &t_eleEne7x7);
  tree_->Branch("t_muEne7x7", "std::vector<double>", &t_muEne7x7);

  tree_->Branch("t_maxNearP3x3", "std::vector<double>", &t_maxNearP3x3);
  tree_->Branch("t_cHadronEne3x3", "std::vector<double>", &t_cHadronEne3x3);
  tree_->Branch("t_cHadronEne3x3_1", "std::vector<double>", &t_cHadronEne3x3_1);
  tree_->Branch("t_cHadronEne3x3_2", "std::vector<double>", &t_cHadronEne3x3_2);
  tree_->Branch("t_cHadronEne3x3_3", "std::vector<double>", &t_cHadronEne3x3_3);
  tree_->Branch("t_nHadronEne3x3", "std::vector<double>", &t_nHadronEne3x3);
  tree_->Branch("t_photonEne3x3", "std::vector<double>", &t_photonEne3x3);
  tree_->Branch("t_eleEne3x3", "std::vector<double>", &t_eleEne3x3);
  tree_->Branch("t_muEne3x3", "std::vector<double>", &t_muEne3x3);

  tree_->Branch("t_maxNearP1x1", "std::vector<double>", &t_maxNearP1x1);
  tree_->Branch("t_cHadronEne1x1", "std::vector<double>", &t_cHadronEne1x1);
  tree_->Branch("t_cHadronEne1x1_1", "std::vector<double>", &t_cHadronEne1x1_1);
  tree_->Branch("t_cHadronEne1x1_2", "std::vector<double>", &t_cHadronEne1x1_2);
  tree_->Branch("t_cHadronEne1x1_3", "std::vector<double>", &t_cHadronEne1x1_3);
  tree_->Branch("t_nHadronEne1x1", "std::vector<double>", &t_nHadronEne1x1);
  tree_->Branch("t_photonEne1x1", "std::vector<double>", &t_photonEne1x1);
  tree_->Branch("t_eleEne1x1", "std::vector<double>", &t_eleEne1x1);
  tree_->Branch("t_muEne1x1", "std::vector<double>", &t_muEne1x1);

  tree_->Branch("t_maxNearPHC1x1", "std::vector<double>", &t_maxNearPHC1x1);
  tree_->Branch("t_cHadronEneHC1x1", "std::vector<double>", &t_cHadronEneHC1x1);
  tree_->Branch("t_cHadronEneHC1x1_1", "std::vector<double>", &t_cHadronEneHC1x1_1);
  tree_->Branch("t_cHadronEneHC1x1_2", "std::vector<double>", &t_cHadronEneHC1x1_2);
  tree_->Branch("t_cHadronEneHC1x1_3", "std::vector<double>", &t_cHadronEneHC1x1_3);
  tree_->Branch("t_nHadronEneHC1x1", "std::vector<double>", &t_nHadronEneHC1x1);
  tree_->Branch("t_photonEneHC1x1", "std::vector<double>", &t_photonEneHC1x1);
  tree_->Branch("t_eleEneHC1x1", "std::vector<double>", &t_eleEneHC1x1);
  tree_->Branch("t_muEneHC1x1", "std::vector<double>", &t_muEneHC1x1);

  tree_->Branch("t_maxNearPHC3x3", "std::vector<double>", &t_maxNearPHC3x3);
  tree_->Branch("t_cHadronEneHC3x3", "std::vector<double>", &t_cHadronEneHC3x3);
  tree_->Branch("t_cHadronEneHC3x3_1", "std::vector<double>", &t_cHadronEneHC3x3_1);
  tree_->Branch("t_cHadronEneHC3x3_2", "std::vector<double>", &t_cHadronEneHC3x3_2);
  tree_->Branch("t_cHadronEneHC3x3_3", "std::vector<double>", &t_cHadronEneHC3x3_3);
  tree_->Branch("t_nHadronEneHC3x3", "std::vector<double>", &t_nHadronEneHC3x3);
  tree_->Branch("t_photonEneHC3x3", "std::vector<double>", &t_photonEneHC3x3);
  tree_->Branch("t_eleEneHC3x3", "std::vector<double>", &t_eleEneHC3x3);
  tree_->Branch("t_muEneHC3x3", "std::vector<double>", &t_muEneHC3x3);

  tree_->Branch("t_maxNearPHC5x5", "std::vector<double>", &t_maxNearPHC5x5);
  tree_->Branch("t_cHadronEneHC5x5", "std::vector<double>", &t_cHadronEneHC5x5);
  tree_->Branch("t_cHadronEneHC5x5_1", "std::vector<double>", &t_cHadronEneHC5x5_1);
  tree_->Branch("t_cHadronEneHC5x5_2", "std::vector<double>", &t_cHadronEneHC5x5_2);
  tree_->Branch("t_cHadronEneHC5x5_3", "std::vector<double>", &t_cHadronEneHC5x5_3);
  tree_->Branch("t_nHadronEneHC5x5", "std::vector<double>", &t_nHadronEneHC5x5);
  tree_->Branch("t_photonEneHC5x5", "std::vector<double>", &t_photonEneHC5x5);
  tree_->Branch("t_eleEneHC5x5", "std::vector<double>", &t_eleEneHC5x5);
  tree_->Branch("t_muEneHC5x5", "std::vector<double>", &t_muEneHC5x5);

  tree_->Branch("t_maxNearPHC7x7", "std::vector<double>", &t_maxNearPHC7x7);
  tree_->Branch("t_cHadronEneHC7x7", "std::vector<double>", &t_cHadronEneHC7x7);
  tree_->Branch("t_cHadronEneHC7x7_1", "std::vector<double>", &t_cHadronEneHC7x7_1);
  tree_->Branch("t_cHadronEneHC7x7_2", "std::vector<double>", &t_cHadronEneHC7x7_2);
  tree_->Branch("t_cHadronEneHC7x7_3", "std::vector<double>", &t_cHadronEneHC7x7_3);
  tree_->Branch("t_nHadronEneHC7x7", "std::vector<double>", &t_nHadronEneHC7x7);
  tree_->Branch("t_photonEneHC7x7", "std::vector<double>", &t_photonEneHC7x7);
  tree_->Branch("t_eleEneHC7x7", "std::vector<double>", &t_eleEneHC7x7);
  tree_->Branch("t_muEneHC7x7", "std::vector<double>", &t_muEneHC7x7);

  tree_->Branch("t_maxNearPR", "std::vector<double>", &t_maxNearPR);
  tree_->Branch("t_cHadronEneR", "std::vector<double>", &t_cHadronEneR);
  tree_->Branch("t_cHadronEneR_1", "std::vector<double>", &t_cHadronEneR_1);
  tree_->Branch("t_cHadronEneR_2", "std::vector<double>", &t_cHadronEneR_2);
  tree_->Branch("t_cHadronEneR_3", "std::vector<double>", &t_cHadronEneR_3);
  tree_->Branch("t_nHadronEneR", "std::vector<double>", &t_nHadronEneR);
  tree_->Branch("t_photonEneR", "std::vector<double>", &t_photonEneR);
  tree_->Branch("t_eleEneR", "std::vector<double>", &t_eleEneR);
  tree_->Branch("t_muEneR", "std::vector<double>", &t_muEneR);

  tree_->Branch("t_maxNearPIsoR", "std::vector<double>", &t_maxNearPIsoR);
  tree_->Branch("t_cHadronEneIsoR", "std::vector<double>", &t_cHadronEneIsoR);
  tree_->Branch("t_cHadronEneIsoR_1", "std::vector<double>", &t_cHadronEneIsoR_1);
  tree_->Branch("t_cHadronEneIsoR_2", "std::vector<double>", &t_cHadronEneIsoR_2);
  tree_->Branch("t_cHadronEneIsoR_3", "std::vector<double>", &t_cHadronEneIsoR_3);
  tree_->Branch("t_nHadronEneIsoR", "std::vector<double>", &t_nHadronEneIsoR);
  tree_->Branch("t_photonEneIsoR", "std::vector<double>", &t_photonEneIsoR);
  tree_->Branch("t_eleEneIsoR", "std::vector<double>", &t_eleEneIsoR);
  tree_->Branch("t_muEneIsoR", "std::vector<double>", &t_muEneIsoR);

  tree_->Branch("t_maxNearPHCR", "std::vector<double>", &t_maxNearPHCR);
  tree_->Branch("t_cHadronEneHCR", "std::vector<double>", &t_cHadronEneHCR);
  tree_->Branch("t_cHadronEneHCR_1", "std::vector<double>", &t_cHadronEneHCR_1);
  tree_->Branch("t_cHadronEneHCR_2", "std::vector<double>", &t_cHadronEneHCR_2);
  tree_->Branch("t_cHadronEneHCR_3", "std::vector<double>", &t_cHadronEneHCR_3);
  tree_->Branch("t_nHadronEneHCR", "std::vector<double>", &t_nHadronEneHCR);
  tree_->Branch("t_photonEneHCR", "std::vector<double>", &t_photonEneHCR);
  tree_->Branch("t_eleEneHCR", "std::vector<double>", &t_eleEneHCR);
  tree_->Branch("t_muEneHCR", "std::vector<double>", &t_muEneHCR);

  tree_->Branch("t_maxNearPIsoHCR", "std::vector<double>", &t_maxNearPIsoHCR);
  tree_->Branch("t_cHadronEneIsoHCR", "std::vector<double>", &t_cHadronEneIsoHCR);
  tree_->Branch("t_cHadronEneIsoHCR_1", "std::vector<double>", &t_cHadronEneIsoHCR_1);
  tree_->Branch("t_cHadronEneIsoHCR_2", "std::vector<double>", &t_cHadronEneIsoHCR_2);
  tree_->Branch("t_cHadronEneIsoHCR_3", "std::vector<double>", &t_cHadronEneIsoHCR_3);
  tree_->Branch("t_nHadronEneIsoHCR", "std::vector<double>", &t_nHadronEneIsoHCR);
  tree_->Branch("t_photonEneIsoHCR", "std::vector<double>", &t_photonEneIsoHCR);
  tree_->Branch("t_eleEneIsoHCR", "std::vector<double>", &t_eleEneIsoHCR);
  tree_->Branch("t_muEneIsoHCR", "std::vector<double>", &t_muEneIsoHCR);
}

void StudyCaloGen::clearTreeVectors() {
  // t_maxNearP31x31     ->clear();
  // t_nEvtProc          ->clear();

  t_isoTrkPAll->clear();
  t_isoTrkPtAll->clear();
  t_isoTrkPhiAll->clear();
  t_isoTrkEtaAll->clear();
  t_isoTrkDPhiAll->clear();
  t_isoTrkDEtaAll->clear();
  t_isoTrkPdgIdAll->clear();

  t_isoTrkP->clear();
  t_isoTrkPt->clear();
  t_isoTrkEne->clear();
  t_isoTrkEta->clear();
  t_isoTrkPhi->clear();
  t_isoTrkEtaEC->clear();
  t_isoTrkPhiEC->clear();
  t_isoTrkPdgId->clear();

  t_maxNearP31x31->clear();
  t_cHadronEne31x31->clear();
  t_cHadronEne31x31_1->clear();
  t_cHadronEne31x31_2->clear();
  t_cHadronEne31x31_3->clear();
  t_nHadronEne31x31->clear();
  t_photonEne31x31->clear();
  t_eleEne31x31->clear();
  t_muEne31x31->clear();

  t_maxNearP25x25->clear();
  t_cHadronEne25x25->clear();
  t_cHadronEne25x25_1->clear();
  t_cHadronEne25x25_2->clear();
  t_cHadronEne25x25_3->clear();
  t_nHadronEne25x25->clear();
  t_photonEne25x25->clear();
  t_eleEne25x25->clear();
  t_muEne25x25->clear();

  t_maxNearP21x21->clear();
  t_cHadronEne21x21->clear();
  t_cHadronEne21x21_1->clear();
  t_cHadronEne21x21_2->clear();
  t_cHadronEne21x21_3->clear();
  t_nHadronEne21x21->clear();
  t_photonEne21x21->clear();
  t_eleEne21x21->clear();
  t_muEne21x21->clear();

  t_maxNearP15x15->clear();
  t_cHadronEne15x15->clear();
  t_cHadronEne15x15_1->clear();
  t_cHadronEne15x15_2->clear();
  t_cHadronEne15x15_3->clear();
  t_nHadronEne15x15->clear();
  t_photonEne15x15->clear();
  t_eleEne15x15->clear();
  t_muEne15x15->clear();

  t_maxNearP11x11->clear();
  t_cHadronEne11x11->clear();
  t_cHadronEne11x11_1->clear();
  t_cHadronEne11x11_2->clear();
  t_cHadronEne11x11_3->clear();
  t_nHadronEne11x11->clear();
  t_photonEne11x11->clear();
  t_eleEne11x11->clear();
  t_muEne11x11->clear();

  t_maxNearP9x9->clear();
  t_cHadronEne9x9->clear();
  t_cHadronEne9x9_1->clear();
  t_cHadronEne9x9_2->clear();
  t_cHadronEne9x9_3->clear();
  t_nHadronEne9x9->clear();
  t_photonEne9x9->clear();
  t_eleEne9x9->clear();
  t_muEne9x9->clear();

  t_maxNearP7x7->clear();
  t_cHadronEne7x7->clear();
  t_cHadronEne7x7_1->clear();
  t_cHadronEne7x7_2->clear();
  t_cHadronEne7x7_3->clear();
  t_nHadronEne7x7->clear();
  t_photonEne7x7->clear();
  t_eleEne7x7->clear();
  t_muEne7x7->clear();

  t_maxNearP3x3->clear();
  t_cHadronEne3x3->clear();
  t_cHadronEne3x3_1->clear();
  t_cHadronEne3x3_2->clear();
  t_cHadronEne3x3_3->clear();
  t_nHadronEne3x3->clear();
  t_photonEne3x3->clear();
  t_eleEne3x3->clear();
  t_muEne3x3->clear();

  t_maxNearP1x1->clear();
  t_cHadronEne1x1->clear();
  t_cHadronEne1x1_1->clear();
  t_cHadronEne1x1_2->clear();
  t_cHadronEne1x1_3->clear();
  t_nHadronEne1x1->clear();
  t_photonEne1x1->clear();
  t_eleEne1x1->clear();
  t_muEne1x1->clear();

  t_maxNearPHC1x1->clear();
  t_cHadronEneHC1x1->clear();
  t_cHadronEneHC1x1_1->clear();
  t_cHadronEneHC1x1_2->clear();
  t_cHadronEneHC1x1_3->clear();
  t_nHadronEneHC1x1->clear();
  t_photonEneHC1x1->clear();
  t_eleEneHC1x1->clear();
  t_muEneHC1x1->clear();

  t_maxNearPHC3x3->clear();
  t_cHadronEneHC3x3->clear();
  t_cHadronEneHC3x3_1->clear();
  t_cHadronEneHC3x3_2->clear();
  t_cHadronEneHC3x3_3->clear();
  t_nHadronEneHC3x3->clear();
  t_photonEneHC3x3->clear();
  t_eleEneHC3x3->clear();
  t_muEneHC3x3->clear();

  t_maxNearPHC5x5->clear();
  t_cHadronEneHC5x5->clear();
  t_cHadronEneHC5x5_1->clear();
  t_cHadronEneHC5x5_2->clear();
  t_cHadronEneHC5x5_3->clear();
  t_nHadronEneHC5x5->clear();
  t_photonEneHC5x5->clear();
  t_eleEneHC5x5->clear();
  t_muEneHC5x5->clear();

  t_maxNearPHC7x7->clear();
  t_cHadronEneHC7x7->clear();
  t_cHadronEneHC7x7_1->clear();
  t_cHadronEneHC7x7_2->clear();
  t_cHadronEneHC7x7_3->clear();
  t_nHadronEneHC7x7->clear();
  t_photonEneHC7x7->clear();
  t_eleEneHC7x7->clear();
  t_muEneHC7x7->clear();

  t_maxNearPR->clear();
  t_cHadronEneR->clear();
  t_cHadronEneR_1->clear();
  t_cHadronEneR_2->clear();
  t_cHadronEneR_3->clear();
  t_nHadronEneR->clear();
  t_photonEneR->clear();
  t_eleEneR->clear();
  t_muEneR->clear();

  t_maxNearPIsoR->clear();
  t_cHadronEneIsoR->clear();
  t_cHadronEneIsoR_1->clear();
  t_cHadronEneIsoR_2->clear();
  t_cHadronEneIsoR_3->clear();
  t_nHadronEneIsoR->clear();
  t_photonEneIsoR->clear();
  t_eleEneIsoR->clear();
  t_muEneIsoR->clear();

  t_maxNearPHCR->clear();
  t_cHadronEneHCR->clear();
  t_cHadronEneHCR_1->clear();
  t_cHadronEneHCR_2->clear();
  t_cHadronEneHCR_3->clear();
  t_nHadronEneHCR->clear();
  t_photonEneHCR->clear();
  t_eleEneHCR->clear();
  t_muEneHCR->clear();

  t_maxNearPIsoHCR->clear();
  t_cHadronEneIsoHCR->clear();
  t_cHadronEneIsoHCR_1->clear();
  t_cHadronEneIsoHCR_2->clear();
  t_cHadronEneIsoHCR_3->clear();
  t_nHadronEneIsoHCR->clear();
  t_photonEneIsoHCR->clear();
  t_eleEneIsoHCR->clear();
  t_muEneIsoHCR->clear();
}

int StudyCaloGen::particleCode(int pdgId) {
  int partID[Particles] = {11, -11, 21, 211, -211, 321, -321, 2212, 2112, -2212, -2112, 130};
  int ix = -1;
  for (int ik = 0; ik < Particles; ++ik) {
    if (pdgId == partID[ik]) {
      ix = ik;
      break;
    }
  }
  return ix;
}

//define this as a plug-in
DEFINE_FWK_MODULE(StudyCaloGen);
