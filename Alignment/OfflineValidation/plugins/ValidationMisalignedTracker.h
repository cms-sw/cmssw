#ifndef Alignment_OfflineValidation_ValidationMisalignedTracker_h
#define Alignment_OfflineValidation_ValidationMisalignedTracker_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

//
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "TTree.h"
#include "TFile.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TCanvas.h"
#include <cmath>
#include "TStyle.h"

//
// class decleration
//

class ValidationMisalignedTracker : public edm::one::EDAnalyzer<> {
public:
  explicit ValidationMisalignedTracker(const edm::ParameterSet&);
  ~ValidationMisalignedTracker() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const bool selection_eff, selection_fake, ZmassSelection_;
  const std::string simobject, trackassociator;
  const std::vector<std::string> associators;
  const std::vector<edm::InputTag> label;
  const edm::InputTag label_tp_effic;
  const edm::InputTag label_tp_fake;
  const std::string rootfile_;
  const edm::EDGetTokenT<edm::HepMCProduct> evtToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> tpeffToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> tpfakeToken_;
  const edm::EDGetTokenT<edm::View<reco::Track>> trackToken_;
  const std::vector<edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator>> assocToken_;

  bool skip;
  int eventCount_;
  TTree* tree_eff;
  TTree* tree_fake;
  TFile* file_;

  int irun, ievt;

  float mzmu, recmzmu, ptzmu, recptzmu, etazmu, thetazmu, phizmu, recetazmu, recthetazmu, recphizmu;
  float recenezmu, enezmu, pLzmu, recpLzmu, yzmu, recyzmu, mxptmu, recmxptmu, minptmu, recminptmu;
  int countpart[2], countpartrec[2];
  int flag, flagrec, count, countrec;
  // int countsim;
  float ene[2][2], p[2][2], px[2][2], py[2][2], pz[2][2], ptmu[2][2];
  float recene[2][2], recp[2][2], recpx[2][2], recpy[2][2], recpz[2][2], recptmu[2][2];

  int trackType;
  float pt, eta, cottheta, theta, costheta, phi, d0, z0;
  int nhit;
  float recpt, receta, rectheta, reccottheta, recphi, recd0, recz0;
  int nAssoc, recnhit;
  float recchiq;
  float reseta, respt, resd0, resz0, resphi, rescottheta, eff;

  float fakemzmu, fakerecmzmu, fakeptzmu, fakerecptzmu, fakeetazmu, fakethetazmu, fakephizmu, fakerecetazmu,
      fakerecthetazmu, fakerecphizmu;
  float fakerecenezmu, fakeenezmu, fakepLzmu, fakerecpLzmu, fakeyzmu, fakerecyzmu, fakemxptmu, fakerecmxptmu,
      fakeminptmu, fakerecminptmu;
  int fakecountpart[2], fakecountpartrec[2], fakeflag, fakeflagrec, fakecount, fakecountsim, fakecountrec;
  float fakeene[2][2], fakep[2][2], fakepx[2][2], fakepy[2][2], fakepz[2][2], fakeptmu[2][2];
  float fakerecene[2][2], fakerecp[2][2], fakerecpx[2][2], fakerecpy[2][2], fakerecpz[2][2], fakerecptmu[2][2];

  int faketrackType;
  float fakept, fakeeta, fakecottheta, faketheta, fakecostheta, fakephi, faked0, fakez0;
  int fakenhit;
  float fakerecpt, fakereceta, fakerectheta, fakereccottheta, fakerecphi, fakerecd0, fakerecz0;
  int fakenAssoc, fakerecnhit;
  float fakerecchiq;
  float fakereseta, fakerespt, fakeresd0, fakeresz0, fakeresphi, fakerescottheta, fake;

  double chi2tmp;
  float fractiontmp;
  bool onlyDiag;

  GlobalVector magField;
  std::vector<float> ptused;
};

#endif
