/*
  JetMETHLTOffline DQM code
  Migrated to use DQMEDAnalyzer by: Jyothsna Rani Komaragiri, Oct 2014
*/

#include "DQMOffline/Trigger/interface/JetMETHLTOfflineSource.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "TMath.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TPRegexp.h"

#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;

JetMETHLTOfflineSource::JetMETHLTOfflineSource(const edm::ParameterSet& iConfig) : isSetup_(false) {
  LogDebug("JetMETHLTOfflineSource") << "constructor....";

  //
  dirname_ = iConfig.getUntrackedParameter("dirname", std::string("HLT/JetMET/"));
  processname_ = iConfig.getParameter<std::string>("processname");
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  triggerSummaryToken = consumes<trigger::TriggerEvent>(triggerSummaryLabel_);
  triggerResultsToken = consumes<edm::TriggerResults>(triggerResultsLabel_);
  triggerSummaryFUToken = consumes<trigger::TriggerEvent>(
      edm::InputTag(triggerSummaryLabel_.label(), triggerSummaryLabel_.instance(), std::string("FU")));
  triggerResultsFUToken = consumes<edm::TriggerResults>(
      edm::InputTag(triggerResultsLabel_.label(), triggerResultsLabel_.instance(), std::string("FU")));
  //
  verbose_ = iConfig.getUntrackedParameter<bool>("verbose", false);
  runStandalone_ = iConfig.getUntrackedParameter<bool>("runStandalone", false);
  //
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", true);
  plotEff_ = iConfig.getUntrackedParameter<bool>("plotEff", true);
  nameForEff_ = iConfig.getUntrackedParameter<bool>("nameForEff", true);
  MuonTrigPaths_ = iConfig.getUntrackedParameter<vector<std::string> >("pathnameMuon");
  MBTrigPaths_ = iConfig.getUntrackedParameter<vector<std::string> >("pathnameMB");
  //CaloJet, CaloMET
  caloJetsToken = consumes<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("CaloJetCollectionLabel"));
  caloMetToken = consumes<reco::CaloMETCollection>(iConfig.getParameter<edm::InputTag>("CaloMETCollectionLabel"));
  //PFJet, PFMET
  pfJetsToken = consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("PFJetCollectionLabel"));
  pfMetToken = consumes<reco::PFMETCollection>(iConfig.getParameter<edm::InputTag>("PFMETCollectionLabel"));
  //pfmhtTag_       = iConfig.getParameter<edm::InputTag>("PFMHTCollectionLabel");
  //Vertex info
  vertexToken = consumes<reco::VertexCollection>(std::string("offlinePrimaryVertices"));
  //
  CaloJetCorToken_ = consumes<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("CaloJetCorLabel"));
  PFJetCorToken_ = consumes<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("PFJetCorLabel"));
  //JetID
  jetID = new reco::helper::JetIDHelper(iConfig.getParameter<ParameterSet>("JetIDParams"), consumesCollector());
  _fEMF = iConfig.getUntrackedParameter<double>("fEMF", 0.01);
  _feta = iConfig.getUntrackedParameter<double>("feta", 2.60);
  _fHPD = iConfig.getUntrackedParameter<double>("fHPD", 0.98);
  _n90Hits = iConfig.getUntrackedParameter<double>("n90Hits", 1.0);
  _min_NHEF = iConfig.getUntrackedParameter<double>("minNHEF", 0.);
  _max_NHEF = iConfig.getUntrackedParameter<double>("maxNHEF", 0.99);
  _min_CHEF = iConfig.getUntrackedParameter<double>("minCHEF", 0.);
  _max_CHEF = iConfig.getUntrackedParameter<double>("maxCHEF", 1.);
  _min_NEMF = iConfig.getUntrackedParameter<double>("minNEMF", 0.);
  _max_NEMF = iConfig.getUntrackedParameter<double>("maxNEMF", 0.99);
  _min_CEMF = iConfig.getUntrackedParameter<double>("minCEMF", 0.);
  _max_CEMF = iConfig.getUntrackedParameter<double>("maxCEMF", 0.99);
  //Paths
  pathFilter_ = iConfig.getUntrackedParameter<vector<std::string> >("pathFilter");
  pathRejectKeyword_ = iConfig.getUntrackedParameter<vector<std::string> >("pathRejectKeyword");
  std::vector<edm::ParameterSet> paths = iConfig.getParameter<std::vector<edm::ParameterSet> >("pathPairs");
  for (auto& path : paths) {
    custompathnamepairs_.push_back(
        make_pair(path.getParameter<std::string>("pathname"), path.getParameter<std::string>("denompathname")));
  }
}

//------------------------------------------------------------------------//
JetMETHLTOfflineSource::~JetMETHLTOfflineSource() {
  //
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete jetID;
}

//------------------------------------------------------------------------//
void JetMETHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (verbose_) {
    cout << endl;
    cout << "============================================================" << endl;
    cout << " New event" << endl << endl;
  }

  //---------- triggerResults ----------
  iEvent.getByToken(triggerResultsToken, triggerResults_);
  if (!triggerResults_.isValid()) {
    iEvent.getByToken(triggerResultsFUToken, triggerResults_);
    if (!triggerResults_.isValid()) {
      if (verbose_)
        cout << " triggerResults not valid" << endl;
      edm::LogInfo("JetMETHLTOfflineSource") << "TriggerResults not found, "
                                                "skipping event";
      return;
    }
  }
  if (verbose_)
    cout << " done triggerResults" << endl;

  //---------- triggerResults ----------
  triggerNames_ = iEvent.triggerNames(*triggerResults_);

  //---------- triggerSummary ----------
  iEvent.getByToken(triggerSummaryToken, triggerObj_);
  if (!triggerObj_.isValid()) {
    iEvent.getByToken(triggerSummaryFUToken, triggerObj_);
    if (!triggerObj_.isValid()) {
      edm::LogInfo("JetMETHLTOfflineSource") << "TriggerEvent not found, "
                                                "skipping event";
      return;
    }
  }
  if (verbose_)
    cout << " done triggerSummary" << endl;

  if (verbose_) {
    cout << endl;
    cout << "============================================================" << endl;
    cout << " Reading in offline objects" << endl << endl;
  }

  //------------ Offline Objects -------
  iEvent.getByToken(caloJetsToken, calojetColl_);
  if (!calojetColl_.isValid())
    return;
  calojet = *calojetColl_;
  //std::stable_sort( calojet.begin(), calojet.end(), PtSorter() );

  if (verbose_)
    cout << " done calo" << endl;

  iEvent.getByToken(pfJetsToken, pfjetColl_);
  if (!pfjetColl_.isValid())
    return;
  pfjet = *pfjetColl_;
  //std::stable_sort( pfjet.begin(), pfjet.end(), PtSorter() );

  if (verbose_)
    cout << " done pf" << endl;

  iEvent.getByToken(caloMetToken, calometColl_);
  if (!calometColl_.isValid())
    return;

  iEvent.getByToken(pfMetToken, pfmetColl_);
  if (!pfmetColl_.isValid())
    return;

  if (verbose_) {
    cout << endl;
    cout << "============================================================" << endl;
    cout << " Read in offline objects" << endl << endl;
  }

  //---------- Event counting (DEBUG) ----------
  if (verbose_ && iEvent.id().event() % 10000 == 0)
    cout << "Run = " << iEvent.id().run() << ", LS = " << iEvent.luminosityBlock()
         << ", Event = " << iEvent.id().event() << endl;

  //Define on-the-fly correction Jet
  for (int i = 0; i < 2; i++) {
    CaloJetPx[i] = 0.;
    CaloJetPy[i] = 0.;
    CaloJetPt[i] = 0.;
    CaloJetEta[i] = 0.;
    CaloJetPhi[i] = 0.;
    CaloJetEMF[i] = 0.;
    CaloJetfHPD[i] = 0.;
    CaloJetn90[i] = 0.;
    PFJetPx[i] = 0.;
    PFJetPy[i] = 0.;
    PFJetPt[i] = 0.;
    PFJetEta[i] = 0.;
    PFJetPhi[i] = 0.;
    PFJetNHEF[i] = 0.;
    PFJetCHEF[i] = 0.;
    PFJetNEMF[i] = 0.;
    PFJetCEMF[i] = 0.;
  }

  //---------- CaloJet Correction (on-the-fly) ----------
  edm::Handle<reco::JetCorrector> calocorrector;
  iEvent.getByToken(CaloJetCorToken_, calocorrector);
  auto calojet_ = calojet.begin();
  for (; calojet_ != calojet.end(); ++calojet_) {
    double scale = calocorrector->correction(*calojet_);
    jetID->calculate(iEvent, iSetup, *calojet_);

    if (scale * calojet_->pt() > CaloJetPt[0]) {
      CaloJetPt[1] = CaloJetPt[0];
      CaloJetPx[1] = CaloJetPx[0];
      CaloJetPy[1] = CaloJetPy[0];
      CaloJetEta[1] = CaloJetEta[0];
      CaloJetPhi[1] = CaloJetPhi[0];
      CaloJetEMF[1] = CaloJetEMF[0];
      CaloJetfHPD[1] = CaloJetfHPD[0];
      CaloJetn90[1] = CaloJetn90[0];
      //
      CaloJetPt[0] = scale * calojet_->pt();
      CaloJetPx[0] = scale * calojet_->px();
      CaloJetPy[0] = scale * calojet_->py();
      CaloJetEta[0] = calojet_->eta();
      CaloJetPhi[0] = calojet_->phi();
      CaloJetEMF[0] = calojet_->emEnergyFraction();
      CaloJetfHPD[0] = jetID->fHPD();
      CaloJetn90[0] = jetID->n90Hits();
    } else if (scale * calojet_->pt() < CaloJetPt[0] && scale * calojet_->pt() > CaloJetPt[1]) {
      CaloJetPt[1] = scale * calojet_->pt();
      CaloJetPx[1] = scale * calojet_->px();
      CaloJetPy[1] = scale * calojet_->py();
      CaloJetEta[1] = calojet_->eta();
      CaloJetPhi[1] = calojet_->phi();
      CaloJetEMF[1] = calojet_->emEnergyFraction();
      CaloJetfHPD[1] = jetID->fHPD();
      CaloJetn90[1] = jetID->n90Hits();
    } else {
    }
  }

  //---------- PFJet Correction (on-the-fly) ----------
  pfMHTx_All = 0.;
  pfMHTy_All = 0.;
  edm::Handle<reco::JetCorrector> pfcorrector;
  iEvent.getByToken(PFJetCorToken_, pfcorrector);
  auto pfjet_ = pfjet.begin();
  for (; pfjet_ != pfjet.end(); ++pfjet_) {
    double scale = pfcorrector->correction(*pfjet_);
    pfMHTx_All = pfMHTx_All + scale * pfjet_->px();
    pfMHTy_All = pfMHTy_All + scale * pfjet_->py();
    if (scale * pfjet_->pt() > PFJetPt[0]) {
      PFJetPt[1] = PFJetPt[0];
      PFJetPx[1] = PFJetPx[0];
      PFJetPy[1] = PFJetPy[0];
      PFJetEta[1] = PFJetEta[0];
      PFJetPhi[1] = PFJetPhi[0];
      PFJetNHEF[1] = PFJetNHEF[0];
      PFJetCHEF[1] = PFJetCHEF[0];
      PFJetNEMF[1] = PFJetNEMF[0];
      PFJetCEMF[1] = PFJetCEMF[0];
      //
      PFJetPt[0] = scale * pfjet_->pt();
      PFJetPx[0] = scale * pfjet_->px();
      PFJetPy[0] = scale * pfjet_->py();
      PFJetEta[0] = pfjet_->eta();
      PFJetPhi[0] = pfjet_->phi();
      PFJetNHEF[0] = pfjet_->neutralHadronEnergyFraction();
      PFJetCHEF[0] = pfjet_->chargedHadronEnergyFraction();
      PFJetNEMF[0] = pfjet_->neutralEmEnergyFraction();
      PFJetCEMF[0] = pfjet_->chargedEmEnergyFraction();
    } else if (scale * pfjet_->pt() < PFJetPt[0] && scale * pfjet_->pt() > PFJetPt[1]) {
      PFJetPt[1] = scale * pfjet_->pt();
      PFJetPx[1] = scale * pfjet_->px();
      PFJetPy[1] = scale * pfjet_->py();
      PFJetEta[1] = pfjet_->eta();
      PFJetPhi[1] = pfjet_->phi();
      PFJetNHEF[1] = pfjet_->neutralHadronEnergyFraction();
      PFJetCHEF[1] = pfjet_->chargedHadronEnergyFraction();
      PFJetNEMF[1] = pfjet_->neutralEmEnergyFraction();
      PFJetCEMF[1] = pfjet_->chargedEmEnergyFraction();
    } else {
    }
  }

  if (verbose_) {
    for (int i = 0; i < 2; i++) {
      cout << "CaloJet-0: " << CaloJetPt[i] << ", Eta = " << CaloJetEta[i] << ", Phi = " << CaloJetPhi[i] << endl;
      cout << "fHPD = " << CaloJetfHPD[0] << ", n90 = " << CaloJetn90[0] << endl;
    }
    for (int i = 0; i < 2; i++) {
      cout << "PFJet-0: " << PFJetPt[i] << ", Eta = " << PFJetEta[i] << ", Phi = " << PFJetPhi[i] << endl;
    }
  }

  //---------- RUN ----------
  fillMEforMonTriggerSummary(iEvent, iSetup);
  if (plotAll_)
    fillMEforMonAllTrigger(iEvent, iSetup);
  if (plotEff_)
    fillMEforEffAllTrigger(iEvent, iSetup);
  if (runStandalone_)
    fillMEforTriggerNTfired();
}

//------------------------------------------------------------------------//
// Trigger summary for all paths
void JetMETHLTOfflineSource::fillMEforMonTriggerSummary(const Event& iEvent, const edm::EventSetup& iSetup) {
  if (verbose_)
    cout << ">> Inside fillMEforMonTriggerSummary " << endl;
  bool muTrig = false;

  for (auto const& MuonTrigPath : MuonTrigPaths_) {
    const unsigned int nPath(hltConfig_.size());
    for (unsigned int j = 0; j != nPath; ++j) {
      std::string pathname = hltConfig_.triggerName(j);
      if (pathname.find(MuonTrigPath) != std::string::npos) {
        if (isHLTPathAccepted(pathname)) {
          muTrig = true;
          if (verbose_)
            cout << "fillMEforMonTriggerSummary: Muon Match" << endl;
        }
      }
      if (muTrig)
        break;
    }
    if (muTrig)
      break;
  }

  bool mbTrig = false;
  for (auto const& MBTrigPath : MBTrigPaths_) {
    const unsigned int nPath(hltConfig_.size());
    for (unsigned int j = 0; j != nPath; ++j) {
      std::string pathname = hltConfig_.triggerName(j);
      if (pathname.find(MBTrigPath) != std::string::npos) {
        if (isHLTPathAccepted(pathname)) {
          mbTrig = true;
          if (verbose_)
            cout << "fillMEforMonTriggerSummary: MinBias Match" << endl;
        }
      }
      if (mbTrig)
        break;
    }
    if (mbTrig)
      break;
  }

  auto v = hltPathsAll_.begin();
  for (; v != hltPathsAll_.end(); ++v) {
    bool trigFirst = false;
    double binV = TriggerPosition(v->getPath());
    if (isHLTPathAccepted(v->getPath()))
      trigFirst = true;
    if (!trigFirst)
      continue;
    if (trigFirst) {
      rate_All->Fill(binV);
      correlation_All->Fill(binV, binV);
      if (muTrig && runStandalone_) {
        rate_AllWrtMu->Fill(binV);
        correlation_AllWrtMu->Fill(binV, binV);
      }
      if (mbTrig && runStandalone_) {
        rate_AllWrtMB->Fill(binV);
        correlation_AllWrtMB->Fill(binV, binV);
      }
    }
    for (auto w = v + 1; w != hltPathsAll_.end(); ++w) {
      bool trigSec = false;
      double binW = TriggerPosition(w->getPath());
      if (isHLTPathAccepted(w->getPath()))
        trigSec = true;
      if (trigSec && trigFirst) {
        correlation_All->Fill(binV, binW);
        if (muTrig && runStandalone_)
          correlation_AllWrtMu->Fill(binV, binW);
        if (mbTrig && runStandalone_)
          correlation_AllWrtMB->Fill(binV, binW);
      }
      if (!trigSec && trigFirst) {
        correlation_All->Fill(binW, binV);
        if (muTrig && runStandalone_)
          correlation_AllWrtMu->Fill(binW, binV);
        if (mbTrig && runStandalone_)
          correlation_AllWrtMB->Fill(binW, binV);
      }
    }
  }

  //Vertex
  edm::Handle<VertexCollection> Vtx;
  iEvent.getByToken(vertexToken, Vtx);
  int vtxcnt = 0;
  for (auto const& itv : *Vtx) {
    //if(vtxcnt>=20) break;
    PVZ->Fill(itv.z());
    //chi2vtx[vtxcnt] = itv->chi2();
    //ndofvtx[vtxcnt] = itv->ndof();
    //ntrkvtx[vtxcnt] = itv->tracksSize();
    vtxcnt++;
  }
  NVertices->Fill(vtxcnt);
}

//------------------------------------------------------------------------//
void JetMETHLTOfflineSource::fillMEforTriggerNTfired() {
  if (verbose_)
    cout << ">> Inside fillMEforTriggerNTfired" << endl;
  if (!triggerResults_.isValid())
    return;
  if (verbose_)
    cout << "   ... and triggerResults is valid" << endl;

  //
  for (auto& v : hltPathsAll_) {
    unsigned index = triggerNames_.triggerIndex(v.getPath());
    if (index < triggerNames_.size()) {
      v.getMEhisto_TriggerSummary()->Fill(0.);
      edm::InputTag l1Tag(v.getl1Path(), "", processname_);
      const int l1Index = triggerObj_->filterIndex(l1Tag);
      bool l1found = false;
      if (l1Index < triggerObj_->sizeFilters())
        l1found = true;
      if (!l1found)
        v.getMEhisto_TriggerSummary()->Fill(1.);
      if (!l1found && !(triggerResults_->accept(index)))
        v.getMEhisto_TriggerSummary()->Fill(2.);
      if (!l1found && (triggerResults_->accept(index)))
        v.getMEhisto_TriggerSummary()->Fill(3.);
      if (l1found)
        v.getMEhisto_TriggerSummary()->Fill(4.);
      if (l1found && (triggerResults_->accept(index)))
        v.getMEhisto_TriggerSummary()->Fill(5.);
      if (l1found && !(triggerResults_->accept(index)))
        v.getMEhisto_TriggerSummary()->Fill(6.);
      if (!(triggerResults_->accept(index)) && l1found) {
        //cout<<v->getTriggerType()<<endl;
        if ((v.getTriggerType() == "SingleJet_Trigger") && (calojetColl_.isValid()) && !calojet.empty()) {
          auto jet = calojet.begin();
          v.getMEhisto_JetPt()->Fill(jet->pt());
          v.getMEhisto_EtavsPt()->Fill(jet->eta(), jet->pt());
          v.getMEhisto_PhivsPt()->Fill(jet->phi(), jet->pt());
        }
        // single jet trigger is not fired

        if ((v.getTriggerType() == "DiJet_Trigger") && calojetColl_.isValid() && !calojet.empty()) {
          v.getMEhisto_JetSize()->Fill(calojet.size());
          if (calojet.size() >= 2) {
            auto jet = calojet.begin();
            auto jet2 = calojet.begin();
            jet2++;
            double jet3pt = 0.;
            if (calojet.size() > 2) {
              auto jet3 = jet2++;
              jet3pt = jet3->pt();
            }
            v.getMEhisto_Pt12()->Fill((jet->pt() + jet2->pt()) / 2.);
            v.getMEhisto_Eta12()->Fill((jet->eta() + jet2->eta()) / 2.);
            v.getMEhisto_Phi12()->Fill(deltaPhi(jet->phi(), jet2->phi()));
            v.getMEhisto_Pt3()->Fill(jet3pt);
            v.getMEhisto_Pt12Pt3()->Fill((jet->pt() + jet2->pt()) / 2., jet3pt);
            v.getMEhisto_Pt12Phi12()->Fill((jet->pt() + jet2->pt()) / 2., deltaPhi(jet->phi(), jet2->phi()));
          }
        }  // di jet trigger is not fired

        if (((v.getTriggerType() == "MET_Trigger") || (v.getTriggerType() == "TET_Trigger")) &&
            calometColl_.isValid()) {
          const CaloMETCollection* calometcol = calometColl_.product();
          const CaloMET met = calometcol->front();
          v.getMEhisto_JetPt()->Fill(met.pt());
        }  //MET trigger is not fired
      }    // L1 is fired
    }      //
  }        // trigger not fired
}

//------------------------------------------------------------------------//
void JetMETHLTOfflineSource::fillMEforMonAllTrigger(const Event& iEvent, const edm::EventSetup& iSetup) {
  if (verbose_)
    cout << ">> Inside fillMEforMonAllTrigger " << endl;
  if (!triggerResults_.isValid())
    return;
  if (verbose_)
    cout << "   ... and triggerResults is valid" << endl;

  const trigger::TriggerObjectCollection& toc(triggerObj_->getObjects());
  for (auto& v : hltPathsAll_) {
    if (verbose_)
      cout << "   + Checking path " << v.getPath();
    if (isHLTPathAccepted(v.getPath()) == false) {
      if (verbose_)
        cout << " - failed" << endl;
      continue;
    }
    if (verbose_)
      cout << " - PASSED! " << endl;

    //New jet collection (after apply JEC)
    std::vector<double> jetPtVec;
    std::vector<double> jetPhiVec;
    std::vector<double> jetEtaVec;
    std::vector<double> jetPxVec;
    std::vector<double> jetPyVec;
    std::vector<double> hltPtVec;
    std::vector<double> hltPhiVec;
    std::vector<double> hltEtaVec;
    std::vector<double> hltPxVec;
    std::vector<double> hltPyVec;

    //This will be used to find out punch through trigger
    //bool fillL1HLT = false;

    //L1 and HLT indices
    if (verbose_) {
      cout << "     - L1Path = " << v.getl1Path() << endl;
      cout << "     - Label  = " << v.getLabel() << endl;
    }

    //edm::InputTag l1Tag(v->getl1Path(),"",processname_);
    edm::InputTag l1Tag(v.getLabel(), "", processname_);
    const int l1Index = triggerObj_->filterIndex(l1Tag);
    if (verbose_)
      cout << "     - l1Index = " << l1Index << " - l1Tag = [" << l1Tag << "]" << endl;

    edm::InputTag hltTag(v.getLabel(), "", processname_);
    const int hltIndex = triggerObj_->filterIndex(hltTag);
    if (verbose_)
      cout << "     - hltIndex = " << hltIndex << " - hltTag = [" << hltTag << "]" << endl;

    //bool l1TrigBool = false;
    bool hltTrigBool = false;
    bool diJetFire = false;
    int jetsize = 0;

    if (l1Index >= triggerObj_->sizeFilters()) {
      edm::LogInfo("JetMETHLTOfflineSource") << "no index " << l1Index << " of that name " << l1Tag;
      if (verbose_)
        cout << "[JetMETHLTOfflineSource::fillMEforMonAllTrigger] - No index l1Index=" << l1Index << " of that name \""
             << l1Tag << "\"" << endl;
    } else {
      //l1TrigBool = true;
      const trigger::Keys& kl1 = triggerObj_->filterKeys(l1Index);
      //
      if (v.getObjectType() == trigger::TriggerJet && v.getTriggerType() == "SingleJet_Trigger")
        v.getMEhisto_N_L1()->Fill(kl1.size());
      //
      auto ki = kl1.begin();
      for (; ki != kl1.end(); ++ki) {
        double l1TrigEta = -100;
        double l1TrigPhi = -100;
        //
        if (v.getObjectType() == trigger::TriggerJet) {
          l1TrigEta = toc[*ki].eta();
          l1TrigPhi = toc[*ki].phi();
          if (v.getTriggerType() == "SingleJet_Trigger") {
            v.getMEhisto_Pt_L1()->Fill(toc[*ki].pt());
            if (isBarrel(toc[*ki].eta()))
              v.getMEhisto_PtBarrel_L1()->Fill(toc[*ki].pt());
            if (isEndCap(toc[*ki].eta()))
              v.getMEhisto_PtEndcap_L1()->Fill(toc[*ki].pt());
            if (isForward(toc[*ki].eta()))
              v.getMEhisto_PtForward_L1()->Fill(toc[*ki].pt());
            v.getMEhisto_Eta_L1()->Fill(toc[*ki].eta());
            v.getMEhisto_Phi_L1()->Fill(toc[*ki].phi());
            v.getMEhisto_EtaPhi_L1()->Fill(toc[*ki].eta(), toc[*ki].phi());
          }
        }
        if (v.getObjectType() == trigger::TriggerMET || v.getObjectType() == trigger::TriggerTET) {
          v.getMEhisto_Pt_L1()->Fill(toc[*ki].pt());
          v.getMEhisto_Phi_L1()->Fill(toc[*ki].phi());
        }

        //-----------------------------------------------
        if (hltIndex >= triggerObj_->sizeFilters()) {
          edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt" << hltIndex << " of that name ";
          if (verbose_)
            cout << "[JetMETHLTOfflineSource::fillMEforMonAllTrigger] - No index hltIndex=" << hltIndex
                 << " of that name " << endl;
        } else {
          const trigger::Keys& khlt = triggerObj_->filterKeys(hltIndex);
          if (v.getObjectType() == trigger::TriggerJet && ki == kl1.begin() &&
              v.getTriggerType() == "SingleJet_Trigger")
            v.getMEhisto_N_HLT()->Fill(khlt.size());
          //
          auto kj = khlt.begin();
          //Define hltTrigBool
          for (; kj != khlt.end(); ++kj) {
            if (v.getObjectType() == trigger::TriggerJet) {
              double hltTrigEta = -100;
              double hltTrigPhi = -100;
              hltTrigEta = toc[*kj].eta();
              hltTrigPhi = toc[*kj].phi();
              if ((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi)) < 0.4 &&
                  (v.getTriggerType() == "DiJet_Trigger"))
                hltTrigBool = true;
            }
          }
          //
          kj = khlt.begin();
          for (; kj != khlt.end(); ++kj) {
            double hltTrigEta = -100.;
            double hltTrigPhi = -100.;
            //fillL1HLT = true;
            //MET Triggers
            if (verbose_)
              cout << "+ MET Triggers plots" << endl;
            if (v.getObjectType() == trigger::TriggerMET || (v.getObjectType() == trigger::TriggerTET)) {
              v.getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
              v.getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
              v.getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(), toc[*kj].pt());
              v.getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(), toc[*kj].phi());
              v.getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt() - toc[*kj].pt()) / (toc[*ki].pt()));
              v.getMEhisto_PhiResolution_L1HLT()->Fill(toc[*ki].phi() - toc[*kj].phi());
            }
            //Jet Triggers
            if (verbose_)
              cout << "+ Jet Trigger plots" << endl;
            if (v.getObjectType() == trigger::TriggerJet) {
              if (verbose_)
                cout << "  - Going for those..." << endl;
              hltTrigEta = toc[*kj].eta();
              hltTrigPhi = toc[*kj].phi();
              if ((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi)) < 0.4) {
                if (v.getTriggerType() == "SingleJet_Trigger") {
                  v.getMEhisto_PtCorrelation_L1HLT()->Fill(toc[*ki].pt(), toc[*kj].pt());
                  v.getMEhisto_EtaCorrelation_L1HLT()->Fill(toc[*ki].eta(), toc[*kj].eta());
                  v.getMEhisto_PhiCorrelation_L1HLT()->Fill(toc[*ki].phi(), toc[*kj].phi());
                  v.getMEhisto_PtResolution_L1HLT()->Fill((toc[*ki].pt() - toc[*kj].pt()) / (toc[*ki].pt()));
                  v.getMEhisto_EtaResolution_L1HLT()->Fill(toc[*ki].eta() - toc[*kj].eta());
                  v.getMEhisto_PhiResolution_L1HLT()->Fill(toc[*ki].phi() - toc[*kj].phi());
                }
              }
              if (((deltaR(hltTrigEta, hltTrigPhi, l1TrigEta, l1TrigPhi) < 0.4) ||
                   ((v.getTriggerType() == "DiJet_Trigger") && hltTrigBool)) &&
                  !diJetFire) {
                if (v.getTriggerType() == "SingleJet_Trigger") {
                  v.getMEhisto_Pt_HLT()->Fill(toc[*kj].pt());
                  if (isBarrel(toc[*kj].eta()))
                    v.getMEhisto_PtBarrel_HLT()->Fill(toc[*kj].pt());
                  if (isEndCap(toc[*kj].eta()))
                    v.getMEhisto_PtEndcap_HLT()->Fill(toc[*kj].pt());
                  if (isForward(toc[*kj].eta()))
                    v.getMEhisto_PtForward_HLT()->Fill(toc[*kj].pt());
                  v.getMEhisto_Eta_HLT()->Fill(toc[*kj].eta());
                  v.getMEhisto_Phi_HLT()->Fill(toc[*kj].phi());
                  v.getMEhisto_EtaPhi_HLT()->Fill(toc[*kj].eta(), toc[*kj].phi());
                }

                //Calojet
                if (calojetColl_.isValid() && (v.getObjectType() == trigger::TriggerJet) && (v.getPath() == "PFJet")) {
                  //CaloJetCollection::const_iterator jet = calojet.begin();
                  //for(; jet != calojet.end(); ++jet) {
                  for (int iCalo = 0; iCalo < 2; iCalo++) {
                    if (deltaR(hltTrigEta, hltTrigPhi, CaloJetEta[iCalo], CaloJetPhi[iCalo]) < 0.4) {
                      jetsize++;
                      if (v.getTriggerType() == "SingleJet_Trigger") {
                        v.getMEhisto_Pt()->Fill(CaloJetPt[iCalo]);
                        if (isBarrel(CaloJetEta[iCalo]))
                          v.getMEhisto_PtBarrel()->Fill(CaloJetPt[iCalo]);
                        if (isEndCap(CaloJetEta[iCalo]))
                          v.getMEhisto_PtEndcap()->Fill(CaloJetPt[iCalo]);
                        if (isForward(CaloJetEta[iCalo]))
                          v.getMEhisto_PtForward()->Fill(CaloJetPt[iCalo]);
                        //
                        v.getMEhisto_Eta()->Fill(CaloJetEta[iCalo]);
                        v.getMEhisto_Phi()->Fill(CaloJetPhi[iCalo]);
                        v.getMEhisto_EtaPhi()->Fill(CaloJetEta[iCalo], CaloJetPhi[iCalo]);
                        //
                        v.getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(), CaloJetPt[iCalo]);
                        v.getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*kj].eta(), CaloJetEta[iCalo]);
                        v.getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(), CaloJetPhi[iCalo]);
                        //
                        v.getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt() - CaloJetPt[iCalo]) /
                                                                    (toc[*kj].pt()));
                        v.getMEhisto_EtaResolution_HLTRecObj()->Fill(toc[*kj].eta() - CaloJetEta[iCalo]);
                        v.getMEhisto_PhiResolution_HLTRecObj()->Fill(toc[*kj].phi() - CaloJetPhi[iCalo]);
                      }

                      //-------------------------------------------------------
                      if ((v.getTriggerType() == "DiJet_Trigger")) {
                        jetPhiVec.push_back(CaloJetPhi[iCalo]);
                        jetPtVec.push_back(CaloJetPt[iCalo]);
                        jetEtaVec.push_back(CaloJetEta[iCalo]);
                        jetPxVec.push_back(CaloJetPx[iCalo]);
                        jetPyVec.push_back(CaloJetPy[iCalo]);
                        //
                        hltPhiVec.push_back(toc[*kj].phi());
                        hltPtVec.push_back(toc[*kj].pt());
                        hltEtaVec.push_back(toc[*kj].eta());
                        hltPxVec.push_back(toc[*kj].px());
                        hltPyVec.push_back(toc[*kj].py());
                      }
                    }  // matching jet
                  }    // Jet Loop
                }      // valid calojet collection, with calojet trigger

                //PFJet trigger
                if (pfjetColl_.isValid() && (v.getObjectType() == trigger::TriggerJet) && (v.getPath() != "PFJet")) {
                  //PFJetCollection::const_iterator jet = pfjet.begin();
                  //for(; jet != pfjet.end(); ++jet){
                  for (int iPF = 0; iPF < 2; iPF++) {
                    if (deltaR(hltTrigEta, hltTrigPhi, PFJetEta[iPF], PFJetPhi[iPF]) < 0.4) {
                      jetsize++;
                      if (v.getTriggerType() == "SingleJet_Trigger") {
                        v.getMEhisto_Pt()->Fill(PFJetPt[iPF]);
                        if (isBarrel(PFJetEta[iPF]))
                          v.getMEhisto_PtBarrel()->Fill(PFJetPt[iPF]);
                        if (isEndCap(PFJetEta[iPF]))
                          v.getMEhisto_PtEndcap()->Fill(PFJetPt[iPF]);
                        if (isForward(PFJetEta[iPF]))
                          v.getMEhisto_PtForward()->Fill(PFJetPt[iPF]);
                        //
                        v.getMEhisto_Eta()->Fill(PFJetEta[iPF]);
                        v.getMEhisto_Phi()->Fill(PFJetPhi[iPF]);
                        v.getMEhisto_EtaPhi()->Fill(PFJetEta[iPF], PFJetPhi[iPF]);
                        //
                        v.getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].pt(), PFJetPt[iPF]);
                        v.getMEhisto_EtaCorrelation_HLTRecObj()->Fill(toc[*kj].eta(), PFJetEta[iPF]);
                        v.getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(), PFJetPhi[iPF]);
                        //
                        v.getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].pt() - PFJetPt[iPF]) / (toc[*kj].pt()));
                        v.getMEhisto_EtaResolution_HLTRecObj()->Fill(toc[*kj].eta() - PFJetEta[iPF]);
                        v.getMEhisto_PhiResolution_HLTRecObj()->Fill(toc[*kj].phi() - PFJetPhi[iPF]);
                      }

                      //-------------------------------------------------------
                      if ((v.getTriggerType() == "DiJet_Trigger")) {
                        jetPhiVec.push_back(PFJetPhi[iPF]);
                        jetPtVec.push_back(PFJetPt[iPF]);
                        jetEtaVec.push_back(PFJetEta[iPF]);
                        jetPxVec.push_back(PFJetPx[iPF]);
                        jetPyVec.push_back(PFJetPy[iPF]);
                        //
                        hltPhiVec.push_back(toc[*kj].phi());
                        hltPtVec.push_back(toc[*kj].pt());
                        hltEtaVec.push_back(toc[*kj].eta());
                        hltPxVec.push_back(toc[*kj].px());
                        hltPyVec.push_back(toc[*kj].py());
                      }
                    }  // matching jet
                  }    //PFJet loop
                }      //valid pfjet collection, with pfjet trigger
                       //
              }        // hlt matching with l1
            }          // jet trigger

            //------------------------------------------------------
            if (calometColl_.isValid() &&
                ((v.getObjectType() == trigger::TriggerMET) || (v.getObjectType() == trigger::TriggerTET)) &&
                (v.getPath().find("HLT_PFMET") == std::string::npos)) {
              const CaloMETCollection* calometcol = calometColl_.product();
              const CaloMET met = calometcol->front();
              //
              v.getMEhisto_Pt()->Fill(met.et());
              v.getMEhisto_Phi()->Fill(met.phi());
              //
              v.getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].et(), met.et());
              v.getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(), met.phi());
              v.getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].et() - met.et()) / (toc[*kj].et()));
              v.getMEhisto_PhiResolution_HLTRecObj()->Fill(toc[*kj].phi() - met.phi());
            }

            //--------------------------------------------------------
            if (pfmetColl_.isValid() &&
                ((v.getObjectType() == trigger::TriggerMET) || (v.getObjectType() == trigger::TriggerTET)) &&
                (v.getPath().find("HLT_PFMET") != std::string::npos)) {
              const PFMETCollection* pfmetcol = pfmetColl_.product();
              const PFMET pfmet = pfmetcol->front();
              //
              v.getMEhisto_Pt()->Fill(pfmet.et());
              v.getMEhisto_Phi()->Fill(pfmet.phi());
              //
              v.getMEhisto_PtCorrelation_HLTRecObj()->Fill(toc[*kj].et(), pfmet.et());
              v.getMEhisto_PhiCorrelation_HLTRecObj()->Fill(toc[*kj].phi(), pfmet.phi());
              v.getMEhisto_PtResolution_HLTRecObj()->Fill((toc[*kj].et() - pfmet.et()) / (toc[*kj].et()));
              v.getMEhisto_PhiResolution_HLTRecObj()->Fill(toc[*kj].phi() - pfmet.phi());
            }
          }  //Loop over HLT trigger candidates
          if ((v.getTriggerType() == "DiJet_Trigger"))
            diJetFire = true;
        }  // Valid hlt trigger object
      }    // Loop over L1 objects
    }      // Valid L1 trigger object
    v.getMEhisto_N()->Fill(jetsize);

    //--------------------------------------------------------
    if ((v.getTriggerType() == "DiJet_Trigger") && jetPtVec.size() > 1) {
      double AveJetPt = (jetPtVec[0] + jetPtVec[1]) / 2;
      double AveJetEta = (jetEtaVec[0] + jetEtaVec[1]) / 2;
      double JetDelPhi = deltaPhi(jetPhiVec[0], jetPhiVec[1]);
      double AveHLTPt = (hltPtVec[0] + hltPtVec[1]) / 2;
      double AveHLTEta = (hltEtaVec[0] + hltEtaVec[1]) / 2;
      double HLTDelPhi = deltaPhi(hltPhiVec[0], hltPhiVec[1]);
      v.getMEhisto_AveragePt_RecObj()->Fill(AveJetPt);
      v.getMEhisto_AverageEta_RecObj()->Fill(AveJetEta);
      v.getMEhisto_DeltaPhi_RecObj()->Fill(JetDelPhi);
      //
      v.getMEhisto_AveragePt_HLTObj()->Fill(AveHLTPt);
      v.getMEhisto_AverageEta_HLTObj()->Fill(AveHLTEta);
      v.getMEhisto_DeltaPhi_HLTObj()->Fill(HLTDelPhi);
    }
  }
  if (verbose_)
    cout << "<< Exiting fillMEforMonAllTrigger " << endl;
}

//------------------------------------------------------------------------//
void JetMETHLTOfflineSource::fillMEforEffAllTrigger(const Event& iEvent, const edm::EventSetup& iSetup) {
  if (!triggerResults_.isValid())
    return;

  int num = -1;
  int denom = -1;
  bool denompassed = false;
  bool numpassed = false;
  const trigger::TriggerObjectCollection& toc(triggerObj_->getObjects());

  for (auto& v : hltPathsEff_) {
    num++;
    denom++;
    denompassed = false;
    numpassed = false;

    unsigned indexNum = triggerNames_.triggerIndex(v.getPath());
    unsigned indexDenom = triggerNames_.triggerIndex(v.getDenomPath());

    if (indexNum < triggerNames_.size() && triggerResults_->accept(indexNum))
      numpassed = true;
    if (indexDenom < triggerNames_.size() && triggerResults_->accept(indexDenom))
      denompassed = true;

    if (denompassed == false)
      continue;

    //if(numpassed==true){
    edm::InputTag hltTag(v.getLabel(), "", processname_);
    const int hltIndex = triggerObj_->filterIndex(hltTag);
    edm::InputTag l1Tag(v.getl1Path(), "", processname_);
    const int l1Index = triggerObj_->filterIndex(l1Tag);
    //}

    //----------------------------------------------------------------------
    //double pTcut = 0;
    double trigLowpTcut = 0;
    double trigMedpTcut = 0;
    double trigHighpTcut = 0;
    double trigLowpTcutFwd = 0;
    double trigMedpTcutFwd = 0;
    double trigHighpTcutFwd = 0;
    //
    //double pTPFcut = 0 ;
    double trigLowpTPFcut = 0;
    double trigMedpTPFcut = 0;
    double trigHighpTPFcut = 0;
    double trigLowpTPFcutFwd = 0;
    double trigMedpTPFcutFwd = 0;
    double trigHighpTPFcutFwd = 0;
    //
    //cout<<"pre-path" << v->getPath()<<endl;
    size_t jetstrfound = v.getPath().find("Jet");
    //size_t censtrfound = v->getPath().find("Central"); //shoouldn't be needed?
    string tpath = v.getPath();
    string jetTrigVal;
    float jetVal = 0.;
    //
    if (jetstrfound != string::npos) {  // && ustrfound != string::npos ){
      //cout<<v->getPath()<<endl;
      for (int trig = int(jetstrfound) + 3; trig < int(jetstrfound) + 7; trig++) {  // int(ustrfound); trig++){
        if (!isdigit(tpath[trig]))
          break;
        jetTrigVal += tpath[trig];
      }
      jetVal = atof(jetTrigVal.c_str());
      //
      if (jetVal > 0.) {
        if (jetVal < 50.) {
          //pTcut = jetVal / 2.;
          trigMedpTcut = jetVal + 5.;
          trigHighpTcut = jetVal + 10.;
          //
          trigLowpTcutFwd = jetVal + 9.;
          trigMedpTcutFwd = jetVal + 15.;
          trigHighpTcutFwd = jetVal + 21.;
        } else {
          //pTcut = jetVal - 20. ;
          trigMedpTcut = jetVal + 2.;
          trigHighpTcut = jetVal + 60.;
          //
          trigLowpTcutFwd = jetVal + 22.;
          trigMedpTcutFwd = jetVal + 25.;
          trigHighpTcutFwd = jetVal + 110.;
        }
        trigLowpTcut = jetVal;
      }
      //
      if (jetVal > 0.) {
        if (jetVal < 50.) {
          //pTPFcut = jetVal ;
          trigMedpTPFcut = jetVal + 20.;
          trigHighpTPFcut = jetVal + 40.;
          //
          trigLowpTPFcutFwd = jetVal + 60.;
          trigMedpTPFcutFwd = jetVal + 80.;
          trigHighpTPFcutFwd = jetVal + 100.;
        } else {
          //pTPFcut = jetVal  ;
          trigMedpTPFcut = jetVal + 40.;
          trigHighpTPFcut = jetVal + 140.;
          //
          trigLowpTPFcutFwd = jetVal + 110.;
          trigMedpTPFcutFwd = jetVal + 130.;
          trigHighpTPFcutFwd = jetVal + 190.;
        }
        trigLowpTPFcut = jetVal;
      }
    }
    //----------------------------------------------------------------------

    //CaloJet paths
    if (verbose_)
      std::cout << "fillMEforEffAllTrigger: CaloJet -------------------" << std::endl;
    if (calojetColl_.isValid() && (v.getObjectType() == trigger::TriggerJet)) {
      //cout<<"   - CaloJet "<<endl;
      //&& (v->getPath().find("HLT_PFJet")==std::string::npos)
      //&& (v->getPath().find("HLT_DiPFJet")==std::string::npos)){
      bool jetIDbool = false;
      double leadjpt = CaloJetPt[0];
      double leadjeta = CaloJetEta[0];
      double leadjphi = CaloJetPhi[0];
      //double ljemf    = CaloJetEMF[0];
      double ljfhpd = CaloJetfHPD[0];
      double ljn90 = CaloJetn90[0];
      if ((v.getTriggerType() == "SingleJet_Trigger") && !calojet.empty()) {  //this line stops the central jets
        if ((ljfhpd < _fHPD) && (ljn90 > _n90Hits)) {
          if (verbose_)
            cout << "Passed CaloJet ID -------------------" << endl;
          jetIDbool = true;
          //Denominator fill
          v.getMEhisto_DenominatorPt()->Fill(leadjpt);
          if (isBarrel(leadjeta))
            v.getMEhisto_DenominatorPtBarrel()->Fill(leadjpt);
          if (isEndCap(leadjeta))
            v.getMEhisto_DenominatorPtEndcap()->Fill(leadjpt);
          if (isForward(leadjeta))
            v.getMEhisto_DenominatorPtForward()->Fill(leadjpt);
          v.getMEhisto_DenominatorEta()->Fill(leadjeta);
          v.getMEhisto_DenominatorPhi()->Fill(leadjphi);
          v.getMEhisto_DenominatorEtaPhi()->Fill(leadjeta, leadjphi);
          if (isBarrel(leadjeta)) {
            v.getMEhisto_DenominatorEtaBarrel()->Fill(leadjeta);
            v.getMEhisto_DenominatorPhiBarrel()->Fill(leadjphi);
          }
          if (isEndCap(leadjeta)) {
            v.getMEhisto_DenominatorEtaEndcap()->Fill(leadjeta);
            v.getMEhisto_DenominatorPhiEndcap()->Fill(leadjphi);
          }
          if (isForward(leadjeta)) {
            v.getMEhisto_DenominatorEtaForward()->Fill(leadjeta);
            v.getMEhisto_DenominatorPhiForward()->Fill(leadjphi);
          }
          if ((leadjpt > trigLowpTcut && !isForward(leadjeta)) || (leadjpt > trigLowpTcutFwd && isForward(leadjeta))) {
            v.getMEhisto_DenominatorEta_LowpTcut()->Fill(leadjeta);
            v.getMEhisto_DenominatorPhi_LowpTcut()->Fill(leadjphi);
            v.getMEhisto_DenominatorEtaPhi_LowpTcut()->Fill(leadjeta, leadjphi);
          }
          if ((leadjpt > trigMedpTcut && !isForward(leadjeta)) || (leadjpt > trigMedpTcutFwd && isForward(leadjeta))) {
            v.getMEhisto_DenominatorEta_MedpTcut()->Fill(leadjeta);
            v.getMEhisto_DenominatorPhi_MedpTcut()->Fill(leadjphi);
            v.getMEhisto_DenominatorEtaPhi_MedpTcut()->Fill(leadjeta, leadjphi);
          }
          if ((leadjpt > trigHighpTcut && !isForward(leadjeta)) ||
              (leadjpt > trigHighpTcutFwd && isForward(leadjeta))) {
            v.getMEhisto_DenominatorEta_HighpTcut()->Fill(leadjeta);
            v.getMEhisto_DenominatorPhi_HighpTcut()->Fill(leadjphi);
            v.getMEhisto_DenominatorEtaPhi_HighpTcut()->Fill(leadjeta, leadjphi);
          }

          //Numerator fill
          if (numpassed) {
            //
            double dRmin = 99999.;
            double dPhimin = 9999.;
            if (v.getPath().find("L1") != std::string::npos) {
              if (l1Index >= triggerObj_->sizeFilters()) {
                edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt" << hltIndex << " of that name ";
              } else {
                const trigger::Keys& kl1 = triggerObj_->filterKeys(l1Index);
                for (unsigned short ki : kl1) {
                  double dR = deltaR(toc[ki].eta(), toc[ki].phi(), leadjeta, leadjphi);
                  if (dR < dRmin) {
                    dRmin = dR;
                  }
                }
              }
            } else {
              if (hltIndex >= triggerObj_->sizeFilters()) {
                edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt" << hltIndex << " of that name ";
              } else {
                const trigger::Keys& khlt = triggerObj_->filterKeys(hltIndex);
                auto kj = khlt.begin();
                for (; kj != khlt.end(); ++kj) {
                  double dR = deltaR(toc[*kj].eta(), toc[*kj].phi(), leadjeta, leadjphi);
                  if (dR < dRmin) {
                    dRmin = dR;
                  }
                  double dPhi = deltaPhi(toc[*kj].phi(), leadjphi);
                  if (dPhi < dPhimin) {
                    dPhimin = dPhi;
                  }
                }
                //v->getMEhisto_DeltaPhi()->Fill(dPhimin);
                v.getMEhisto_DeltaPhi()->Fill(dPhimin);
                v.getMEhisto_DeltaR()->Fill(dRmin);
              }
            }
            if (dRmin < 0.1 || (v.getPath().find("L1") != std::string::npos && dRmin < 0.4)) {
              v.getMEhisto_NumeratorPt()->Fill(leadjpt);
              if (isBarrel(leadjeta))
                v.getMEhisto_NumeratorPtBarrel()->Fill(leadjpt);
              if (isEndCap(leadjeta))
                v.getMEhisto_NumeratorPtEndcap()->Fill(leadjpt);
              if (isForward(leadjeta))
                v.getMEhisto_NumeratorPtForward()->Fill(leadjpt);
              v.getMEhisto_NumeratorEta()->Fill(leadjeta);
              v.getMEhisto_NumeratorPhi()->Fill(leadjphi);
              v.getMEhisto_NumeratorEtaPhi()->Fill(leadjeta, leadjphi);
              if (isBarrel(leadjeta)) {
                v.getMEhisto_NumeratorEtaBarrel()->Fill(leadjeta);
                v.getMEhisto_NumeratorPhiBarrel()->Fill(leadjphi);
              }
              if (isEndCap(leadjeta)) {
                v.getMEhisto_NumeratorEtaEndcap()->Fill(leadjeta);
                v.getMEhisto_NumeratorPhiEndcap()->Fill(leadjphi);
              }
              if (isForward(leadjeta)) {
                v.getMEhisto_NumeratorEtaForward()->Fill(leadjeta);
                v.getMEhisto_NumeratorPhiForward()->Fill(leadjphi);
              }
              if ((leadjpt > trigLowpTcut && !isForward(leadjeta)) ||
                  (leadjpt > trigLowpTcutFwd && isForward(leadjeta))) {
                v.getMEhisto_NumeratorEta_LowpTcut()->Fill(leadjeta);
                v.getMEhisto_NumeratorPhi_LowpTcut()->Fill(leadjphi);
                v.getMEhisto_NumeratorEtaPhi_LowpTcut()->Fill(leadjeta, leadjphi);
              }
              if ((leadjpt > trigMedpTcut && !isForward(leadjeta)) ||
                  (leadjpt > trigMedpTcutFwd && isForward(leadjeta))) {
                v.getMEhisto_NumeratorEta_MedpTcut()->Fill(leadjeta);
                v.getMEhisto_NumeratorPhi_MedpTcut()->Fill(leadjphi);
                v.getMEhisto_NumeratorEtaPhi_MedpTcut()->Fill(leadjeta, leadjphi);
              }
              if ((leadjpt > trigHighpTcut && !isForward(leadjeta)) ||
                  (leadjpt > trigHighpTcutFwd && isForward(leadjeta))) {
                v.getMEhisto_NumeratorEta_HighpTcut()->Fill(leadjeta);
                v.getMEhisto_NumeratorPhi_HighpTcut()->Fill(leadjphi);
                v.getMEhisto_NumeratorEtaPhi_HighpTcut()->Fill(leadjeta, leadjphi);
              }
            }
          }  //numpassed
        }    //CalojetID filter
      }

      if (jetIDbool == true && (v.getTriggerType() == "DiJet_Trigger") && calojet.size() > 1) {
        if (((CaloJetEMF[1] > _fEMF || std::abs(CaloJetEta[1]) > _feta) && CaloJetfHPD[0] < _fHPD &&
             CaloJetn90[0] > _n90Hits)) {
          v.getMEhisto_DenominatorPt()->Fill((CaloJetPt[0] + CaloJetPt[1]) / 2.);
          v.getMEhisto_DenominatorEta()->Fill((CaloJetEta[0] + CaloJetEta[1]) / 2.);
          if (numpassed == true) {
            v.getMEhisto_NumeratorPt()->Fill((CaloJetPt[0] + CaloJetPt[1]) / 2.);
            v.getMEhisto_NumeratorEta()->Fill((CaloJetEta[0] + CaloJetEta[1]) / 2.);
          }
        }
      }
    }  // Jet trigger and valid jet collection

    //PFJet paths
    if (verbose_)
      std::cout << "fillMEforEffAllTrigger: PFJet -------------------" << std::endl;
    if (pfjetColl_.isValid() && (v.getObjectType() == trigger::TriggerJet)) {
      //cout<<"   - PFJet "<<endl;
      //&& (v->getPath().find("HLT_PFJet")!=std::string::npos)
      //&& (v->getPath().find("HLT_DiPFJet")!=std::string::npos)){
      bool jetIDbool = false;
      double leadjpt = PFJetPt[0];
      double leadjeta = PFJetEta[0];
      double leadjphi = PFJetPhi[0];
      double ljNHEF = PFJetNHEF[0];
      double ljCHEF = PFJetCHEF[0];
      double ljNEMF = PFJetNEMF[0];
      double ljCEMF = PFJetCEMF[0];
      //double sleadjpt  = PFJetPt[1];
      //double sleadjeta = PFJetEta[1];
      //double sleadjphi = PFJetPhi[1];
      double sljNHEF = PFJetNHEF[1];
      double sljCHEF = PFJetCHEF[1];
      double sljNEMF = PFJetNEMF[1];
      double sljCEMF = PFJetCEMF[1];
      //
      double pfMHTx = pfMHTx_All;
      double pfMHTy = pfMHTy_All;
      //
      if ((v.getTriggerType() == "SingleJet_Trigger") && !pfjet.empty()) {  //this line stops the central jets

        //======get pfmht
        _pfMHT = sqrt(pfMHTx * pfMHTx + pfMHTy * pfMHTy);
        v.getMEhisto_DenominatorPFMHT()->Fill(_pfMHT);

        if (ljNHEF >= _min_NHEF && ljNHEF <= _max_NHEF && ljCHEF >= _min_CHEF && ljCHEF <= _max_CHEF &&
            ljNEMF >= _min_NEMF && ljNEMF <= _max_NEMF && ljCEMF >= _min_CEMF && ljCEMF <= _max_CEMF) {
          if (verbose_)
            cout << "Passed PFJet ID -------------------" << endl;
          jetIDbool = true;
          v.getMEhisto_DenominatorPFPt()->Fill(leadjpt);
          if (isBarrel(leadjeta))
            v.getMEhisto_DenominatorPFPtBarrel()->Fill(leadjpt);
          if (isEndCap(leadjeta))
            v.getMEhisto_DenominatorPFPtEndcap()->Fill(leadjpt);
          if (isForward(leadjeta))
            v.getMEhisto_DenominatorPFPtForward()->Fill(leadjpt);
          v.getMEhisto_DenominatorPFEta()->Fill(leadjeta);
          v.getMEhisto_DenominatorPFPhi()->Fill(leadjphi);
          v.getMEhisto_DenominatorPFEtaPhi()->Fill(leadjeta, leadjphi);
          if (isBarrel(leadjeta)) {
            v.getMEhisto_DenominatorPFEtaBarrel()->Fill(leadjeta);
            v.getMEhisto_DenominatorPFPhiBarrel()->Fill(leadjphi);
          }
          if (isEndCap(leadjeta)) {
            v.getMEhisto_DenominatorPFEtaEndcap()->Fill(leadjeta);
            v.getMEhisto_DenominatorPFPhiEndcap()->Fill(leadjphi);
          }
          if (isForward(leadjeta)) {
            v.getMEhisto_DenominatorPFEtaForward()->Fill(leadjeta);
            v.getMEhisto_DenominatorPFPhiForward()->Fill(leadjphi);
          }
          if ((leadjpt > trigLowpTPFcut && !isForward(leadjeta)) ||
              (leadjpt > trigLowpTPFcutFwd && isForward(leadjeta))) {
            v.getMEhisto_DenominatorPFEta_LowpTcut()->Fill(leadjeta);
            v.getMEhisto_DenominatorPFPhi_LowpTcut()->Fill(leadjphi);
            v.getMEhisto_DenominatorPFEtaPhi_LowpTcut()->Fill(leadjeta, leadjphi);
          }
          if ((leadjpt > trigMedpTPFcut && !isForward(leadjeta)) ||
              (leadjpt > trigMedpTPFcutFwd && isForward(leadjeta))) {
            v.getMEhisto_DenominatorPFEta_MedpTcut()->Fill(leadjeta);
            v.getMEhisto_DenominatorPFPhi_MedpTcut()->Fill(leadjphi);
            v.getMEhisto_DenominatorPFEtaPhi_MedpTcut()->Fill(leadjeta, leadjphi);
          }
          if ((leadjpt > trigHighpTPFcut && !isForward(leadjeta)) ||
              (leadjpt > trigHighpTPFcutFwd && isForward(leadjeta))) {
            v.getMEhisto_DenominatorPFEta_HighpTcut()->Fill(leadjeta);
            v.getMEhisto_DenominatorPFPhi_HighpTcut()->Fill(leadjphi);
            v.getMEhisto_DenominatorPFEtaPhi_HighpTcut()->Fill(leadjeta, leadjphi);
          }

          //Numerator fill
          if (numpassed) {
            double dRmin = 99999.;
            double dPhimin = 9999.;
            if (v.getPath().find("L1") != std::string::npos) {
              if (l1Index >= triggerObj_->sizeFilters()) {
                edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt" << hltIndex << " of that name ";
              } else {
                const trigger::Keys& kl1 = triggerObj_->filterKeys(l1Index);
                for (unsigned short ki : kl1) {
                  double dR = deltaR(toc[ki].eta(), toc[ki].phi(), leadjeta, leadjphi);
                  if (dR < dRmin) {
                    dRmin = dR;
                  }
                }
              }
            } else {
              if (hltIndex >= triggerObj_->sizeFilters()) {
                edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt" << hltIndex << " of that name ";
              } else {
                const trigger::Keys& khlt = triggerObj_->filterKeys(hltIndex);
                for (unsigned short kj : khlt) {
                  double dR = deltaR(toc[kj].eta(), toc[kj].phi(), leadjeta, leadjphi);
                  if (dR < dRmin) {
                    dRmin = dR;
                  }
                  double dPhi = deltaPhi(toc[kj].phi(), leadjphi);
                  if (dPhi < dPhimin) {
                    dPhimin = dPhi;
                  }
                }
                v.getMEhisto_PFDeltaPhi()->Fill(dPhimin);
                v.getMEhisto_PFDeltaR()->Fill(dRmin);
              }
            }
            if (dRmin < 0.1 || (v.getPath().find("L1") != std::string::npos && dRmin < 0.4)) {
              v.getMEhisto_NumeratorPFPt()->Fill(leadjpt);
              if (isBarrel(leadjeta))
                v.getMEhisto_NumeratorPFPtBarrel()->Fill(leadjpt);
              if (isEndCap(leadjeta))
                v.getMEhisto_NumeratorPFPtEndcap()->Fill(leadjpt);
              if (isForward(leadjeta))
                v.getMEhisto_NumeratorPFPtForward()->Fill(leadjpt);
              v.getMEhisto_NumeratorPFEta()->Fill(leadjeta);
              v.getMEhisto_NumeratorPFPhi()->Fill(leadjphi);
              v.getMEhisto_NumeratorPFEtaPhi()->Fill(leadjeta, leadjphi);
              if (isBarrel(leadjeta)) {
                v.getMEhisto_NumeratorPFEtaBarrel()->Fill(leadjeta);
                v.getMEhisto_NumeratorPFPhiBarrel()->Fill(leadjphi);
              }
              if (isEndCap(leadjeta)) {
                v.getMEhisto_NumeratorPFEtaEndcap()->Fill(leadjeta);
                v.getMEhisto_NumeratorPFPhiEndcap()->Fill(leadjphi);
              }
              if (isForward(leadjeta)) {
                v.getMEhisto_NumeratorPFEtaForward()->Fill(leadjeta);
                v.getMEhisto_NumeratorPFPhiForward()->Fill(leadjphi);
              }
              if ((leadjpt > trigLowpTPFcut && !isForward(leadjeta)) ||
                  (leadjpt > trigLowpTPFcutFwd && isForward(leadjeta))) {
                v.getMEhisto_NumeratorPFEta_LowpTcut()->Fill(leadjeta);
                v.getMEhisto_NumeratorPFPhi_LowpTcut()->Fill(leadjphi);
                v.getMEhisto_NumeratorPFEtaPhi_LowpTcut()->Fill(leadjeta, leadjphi);
              }
              if ((leadjpt > trigMedpTPFcut && !isForward(leadjeta)) ||
                  (leadjpt > trigMedpTPFcutFwd && isForward(leadjeta))) {
                v.getMEhisto_NumeratorPFEta_MedpTcut()->Fill(leadjeta);
                v.getMEhisto_NumeratorPFPhi_MedpTcut()->Fill(leadjphi);
                v.getMEhisto_NumeratorPFEtaPhi_MedpTcut()->Fill(leadjeta, leadjphi);
              }
              if ((leadjpt > trigHighpTPFcut && !isForward(leadjeta)) ||
                  (leadjpt > trigHighpTPFcutFwd && isForward(leadjeta))) {
                v.getMEhisto_NumeratorPFEta_HighpTcut()->Fill(leadjeta);
                v.getMEhisto_NumeratorPFPhi_HighpTcut()->Fill(leadjphi);
                v.getMEhisto_NumeratorPFEtaPhi_HighpTcut()->Fill(leadjeta, leadjphi);
              }
            }
          }
        }
      }
      if (jetIDbool == true && (v.getTriggerType() == "DiJet_Trigger") && pfjet.size() > 1) {
        if (ljNHEF >= _min_NHEF && ljNHEF <= _max_NHEF && ljCHEF >= _min_CHEF && ljCHEF <= _max_CHEF &&
            ljNEMF >= _min_NEMF && ljNEMF <= _max_NEMF && ljCEMF >= _min_CEMF && ljCEMF <= _max_CEMF &&
            sljNHEF >= _min_NHEF && sljNHEF <= _max_NHEF && sljCHEF >= _min_CHEF && sljCHEF <= _max_CHEF &&
            sljNEMF >= _min_NEMF && sljNEMF <= _max_NEMF && sljCEMF >= _min_CEMF && sljCEMF <= _max_CEMF) {
          v.getMEhisto_DenominatorPFPt()->Fill((PFJetPt[0] + PFJetPt[1]) / 2.);
          v.getMEhisto_DenominatorPFEta()->Fill((PFJetEta[0] + PFJetEta[1]) / 2.);
          if (numpassed) {
            v.getMEhisto_NumeratorPFPt()->Fill((PFJetPt[0] + PFJetPt[1]) / 2.);
            v.getMEhisto_NumeratorPFEta()->Fill((PFJetEta[0] + PFJetEta[1]) / 2.);
          }
        }
      }
    }  // PF Jet trigger and valid jet collection

    //CaloMET path
    if (verbose_)
      std::cout << "fillMEforEffAllTrigger: CaloMET -------------------" << std::endl;
    if (calometColl_.isValid() &&
        ((v.getObjectType() == trigger::TriggerMET) || (v.getObjectType() == trigger::TriggerTET)) &&
        (v.getPath().find("HLT_PFMET") == std::string::npos)) {
      const CaloMETCollection* calometcol = calometColl_.product();
      const CaloMET met = calometcol->front();
      v.getMEhisto_DenominatorPt()->Fill(met.et());
      v.getMEhisto_DenominatorPhi()->Fill(met.phi());
      if (numpassed) {
        v.getMEhisto_NumeratorPt()->Fill(met.et());
        v.getMEhisto_NumeratorPhi()->Fill(met.phi());
        if (hltIndex >= triggerObj_->sizeFilters()) {
          edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt" << hltIndex << " of that name ";
        } else {
          double dPhimin = 9999.;  //
          const trigger::Keys& khlt = triggerObj_->filterKeys(hltIndex);
          for (unsigned short kj : khlt) {
            double dPhi = deltaPhi(toc[kj].phi(), met.phi());
            if (dPhi < dPhimin) {
              dPhimin = dPhi;
            }
          }
          v.getMEhisto_DeltaPhi()->Fill(dPhimin);
        }
      }
    }

    //PFMET
    if (verbose_)
      std::cout << "fillMEforEffAllTrigger: PFMET -------------------" << std::endl;
    if (pfmetColl_.isValid() &&
        ((v.getObjectType() == trigger::TriggerMET) || (v.getObjectType() == trigger::TriggerTET)) &&
        (v.getPath().find("HLT_PFMET") != std::string::npos)) {
      const PFMETCollection* pfmetcol = pfmetColl_.product();
      const PFMET met = pfmetcol->front();
      v.getMEhisto_DenominatorPt()->Fill(met.et());
      v.getMEhisto_DenominatorPhi()->Fill(met.phi());
      if (numpassed) {
        v.getMEhisto_NumeratorPt()->Fill(met.et());
        v.getMEhisto_NumeratorPhi()->Fill(met.phi());
        if (hltIndex >= triggerObj_->sizeFilters()) {
          edm::LogInfo("JetMETHLTOfflineSource") << "no index hlt" << hltIndex << " of that name ";
        } else {
          double dPhimin = 9999.;  //
          const trigger::Keys& khlt = triggerObj_->filterKeys(hltIndex);
          for (unsigned short kj : khlt) {
            double dPhi = deltaPhi(toc[kj].phi(), met.phi());
            if (dPhi < dPhimin) {
              dPhimin = dPhi;
            }
          }
          v.getMEhisto_DeltaPhi()->Fill(dPhimin);
        }
      }
    }

    /*
      if(pfmhtColl_.isValid() && ((v->getObjectType() == trigger::TriggerMET)|| (v->getObjectType() == trigger::TriggerTET))){
      const PFMHTCollection *pfmhtcol = pfmhtColl_.product();
      const PFMHT met = pfmhtcol->front();
      v->getMEhisto_DenominatorPFPt()->Fill(met.pt());
      v->getMEhisto_DenominatorPFPhi()->Fill(met.phi());
      }// PFMHT  trigger and valid MET collection 
    */
  }  // trigger under study
}

//------------------------------------------------------------------------//
//This method is called before the booking action in the DQMStore is triggered.
void JetMETHLTOfflineSource::dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) {
  if (!isSetup_) {
    //--- htlConfig_
    bool changed(true);
    if (!hltConfig_.init(run, c, processname_, changed)) {
      LogDebug("HLTJetMETDQMSource") << "HLTConfigProvider failed to initialize.";
    }

    /*
      Here we select the Single Jet, DiJet, MET trigger. SingleJet and DiJet trigger are saved under same object type "TriggerJet". 
      We can easily separate out single and di jet trigger later. For the first trigger in the list, denominator trigger is dummy 
      (empty) whereas for other triggers denom is previous trigger of same type. e.g. SingleJet50 has singleJet30 as denominator.
    */

    const unsigned int n(hltConfig_.size());
    int singleJet = 0;
    int diJet = 0;
    int met = 0;
    int tet = 0;
    for (unsigned int i = 0; i != n; ++i) {
      bool denomFound = false;
      bool numFound = false;
      bool checkPath = false;

      //Look for paths if "path name fraction" is found in the pathname
      std::string pathname = hltConfig_.triggerName(i);
      //Filter only paths JetMET triggers are interested in
      auto controlPathname = pathFilter_.begin();
      for (; controlPathname != pathFilter_.end(); ++controlPathname) {
        if (pathname.find((*controlPathname)) != std::string::npos) {
          checkPath = true;
          break;
        }
      }
      if (checkPath == false)
        continue;

      //Reject if keyword(s) is found in the pathname
      auto rejectPathname = pathRejectKeyword_.begin();
      for (; rejectPathname != pathRejectKeyword_.end(); ++rejectPathname) {
        if (pathname.find((*rejectPathname)) != std::string::npos) {
          checkPath = false;
          break;
        }
      }
      if (checkPath == false)
        continue;

      //
      if (verbose_)
        cout << "==pathname==" << pathname << endl;
      std::string dpathname = MuonTrigPaths_[0];
      std::string l1pathname = "dummy";
      std::string denompathname = "";
      unsigned int usedPrescale = 1;
      unsigned int objectType = 0;
      std::string triggerType = "";
      std::string filtername("dummy");
      std::string Denomfiltername("denomdummy");

      if (pathname.find("Jet") != std::string::npos && !(pathname.find("DoubleJet") != std::string::npos) &&
          !(pathname.find("DiJet") != std::string::npos) && !(pathname.find("DiPFJet") != std::string::npos) &&
          !(pathname.find("BTag") != std::string::npos) && !(pathname.find("Mu") != std::string::npos) &&
          !(pathname.find("Fwd") != std::string::npos)) {
        triggerType = "SingleJet_Trigger";
        objectType = trigger::TriggerJet;
      }
      if (pathname.find("DiJet") != std::string::npos || pathname.find("DiPFJet") != std::string::npos ||
          pathname.find("DoubleJet") != std::string::npos) {
        triggerType = "DiJet_Trigger";
        objectType = trigger::TriggerJet;
      }
      if (pathname.find("MET") != std::string::npos) {
        triggerType = "MET_Trigger";
        objectType = trigger::TriggerMET;
      }
      if (pathname.find("HT") != std::string::npos) {
        triggerType = "TET_Trigger";
        objectType = trigger::TriggerTET;
      }

      //
      if (objectType == trigger::TriggerJet && !(pathname.find("DiJet") != std::string::npos) &&
          !(pathname.find("DiPFJet") != std::string::npos) && !(pathname.find("DoubleJet") != std::string::npos)) {
        singleJet++;
        if (singleJet > 1)
          dpathname = dpathname = hltConfig_.triggerName(i - 1);
        if (singleJet == 1)
          dpathname = MuonTrigPaths_[0];
      }
      if (objectType == trigger::TriggerJet &&
          ((pathname.find("DiJet") != std::string::npos) || (pathname.find("DiPFJet") != std::string::npos))) {
        diJet++;
        if (diJet > 1)
          dpathname = dpathname = hltConfig_.triggerName(i - 1);
        if (diJet == 1)
          dpathname = MuonTrigPaths_[0];
      }
      if (objectType == trigger::TriggerMET) {
        met++;
        if (met > 1)
          dpathname = dpathname = hltConfig_.triggerName(i - 1);
        if (met == 1)
          dpathname = MuonTrigPaths_[0];
      }
      if (objectType == trigger::TriggerTET) {
        tet++;
        if (tet > 1)
          dpathname = dpathname = hltConfig_.triggerName(i - 1);
        if (tet == 1)
          dpathname = MuonTrigPaths_[0];
      }

      // find L1 condition for numpath with numpath objecttype
      // find PSet for L1 global seed for numpath,sss
      // list module labels for numpath

      // Checking if the trigger exist in HLT table or not
      for (unsigned int i = 0; i != n; ++i) {
        std::string HLTname = hltConfig_.triggerName(i);
        if (HLTname == pathname)
          numFound = true;
        if (HLTname == dpathname)
          denomFound = true;
      }

      if (numFound) {  //make trigger exist in the menu
        //ml needs change l1pathname
        l1pathname = getL1ConditionModuleName(pathname);  //ml added L1conditionmodulename
        //ml added
        std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
        for (auto& numpathmodule : numpathmodules) {
          edm::InputTag testTag(numpathmodule, "", processname_);
          if ((hltConfig_.moduleType(numpathmodule) == "HLT1CaloJet") ||
              (hltConfig_.moduleType(numpathmodule) == "HLT1PFJet") ||
              (hltConfig_.moduleType(numpathmodule) == "HLTDiJetAveFilter") ||
              (hltConfig_.moduleType(numpathmodule) == "HLTDiPFJetAveFilter") ||
              (hltConfig_.moduleType(numpathmodule) == "HLT1CaloMET") ||
              (hltConfig_.moduleType(numpathmodule) == "HLTMhtFilter") ||
              (hltConfig_.moduleType(numpathmodule) == "HLTPrescaler"))
            filtername = numpathmodule;
        }
      }

      if (objectType != 0 && denomFound) {
        std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(dpathname);
        for (auto& numpathmodule : numpathmodules) {
          edm::InputTag testTag(numpathmodule, "", processname_);
          if ((hltConfig_.moduleType(numpathmodule) == "HLT1CaloJet") ||
              (hltConfig_.moduleType(numpathmodule) == "HLT1PFJet") ||
              (hltConfig_.moduleType(numpathmodule) == "HLTDiJetAveFilter") ||
              (hltConfig_.moduleType(numpathmodule) == "HLTDiPFJetAveFilter") ||
              (hltConfig_.moduleType(numpathmodule) == "HLT1CaloMET") ||
              (hltConfig_.moduleType(numpathmodule) == "HLTMhtFilter") ||
              (hltConfig_.moduleType(numpathmodule) == "HLTPrescaler"))
            Denomfiltername = numpathmodule;
        }
      }

      if (objectType != 0 && numFound) {
        if (verbose_)
          cout << "=Pathname= " << pathname << " | =Denompath= " << dpathname << " | =Filtername= " << filtername
               << " | =Denomfiltername= " << Denomfiltername << " | =L1pathname= " << l1pathname
               << " | =ObjectType= " << objectType << endl;
        if (!((pathname.find("HT") != std::string::npos) || (pathname.find("Quad") != std::string::npos))) {
          hltPathsAll_.push_back(PathInfo(usedPrescale,
                                          dpathname,
                                          pathname,
                                          l1pathname,
                                          filtername,
                                          Denomfiltername,
                                          processname_,
                                          objectType,
                                          triggerType));
          if (!nameForEff_ && denomFound)
            hltPathsEff_.push_back(PathInfo(usedPrescale,
                                            dpathname,
                                            pathname,
                                            l1pathname,
                                            filtername,
                                            Denomfiltername,
                                            processname_,
                                            objectType,
                                            triggerType));
        }
        hltPathsAllTriggerSummary_.push_back(PathInfo(usedPrescale,
                                                      dpathname,
                                                      pathname,
                                                      l1pathname,
                                                      filtername,
                                                      Denomfiltername,
                                                      processname_,
                                                      objectType,
                                                      triggerType));
      }
    }  //Loop over paths

    if (verbose_)
      cout << "get names for efficicncy------------------" << endl;
    //---------bool to pick trigger names pair from config file-------------
    if (nameForEff_) {
      std::string l1pathname = "dummy";
      std::string denompathname = "";
      unsigned int usedPrescale = 1;
      unsigned int objectType = 0;
      std::string triggerType = "";
      std::string filtername("dummy");
      std::string Denomfiltername("denomdummy");
      for (auto& custompathnamepair : custompathnamepairs_) {
        std::string pathname = custompathnamepair.first;
        std::string dpathname = custompathnamepair.second;
        bool numFound = false;
        bool denomFound = false;
        // Checking if the trigger exist in HLT table or not
        for (unsigned int i = 0; i != n; ++i) {
          std::string HLTname = hltConfig_.triggerName(i);
          if (HLTname.find(pathname) != std::string::npos) {
            numFound = true;
            pathname = HLTname;
          }  //changed to get versions
          if (HLTname.find(dpathname) != std::string::npos) {
            denomFound = true;
            dpathname = HLTname;
          }
        }
        if (numFound && denomFound) {
          if (pathname.find("Jet") != std::string::npos && !(pathname.find("DiJet") != std::string::npos) &&
              !(pathname.find("DiPFJet") != std::string::npos) && !(pathname.find("DoubleJet") != std::string::npos) &&
              !(pathname.find("BTag") != std::string::npos) && !(pathname.find("Mu") != std::string::npos) &&
              !(pathname.find("Fwd") != std::string::npos)) {
            triggerType = "SingleJet_Trigger";
            objectType = trigger::TriggerJet;
          }
          if (pathname.find("DiJet") != std::string::npos || pathname.find("DiPFJet") != std::string::npos ||
              pathname.find("DoubleJet") != std::string::npos) {
            triggerType = "DiJet_Trigger";
            objectType = trigger::TriggerJet;
          }
          if (pathname.find("MET") != std::string::npos) {
            triggerType = "MET_Trigger";
            objectType = trigger::TriggerMET;
          }
          if (pathname.find("TET") != std::string::npos) {
            triggerType = "TET_Trigger";
            objectType = trigger::TriggerTET;
          }

          l1pathname = getL1ConditionModuleName(pathname);  //ml added L1conditionmodulename
          std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
          for (auto& numpathmodule : numpathmodules) {
            edm::InputTag testTag(numpathmodule, "", processname_);
            if ((hltConfig_.moduleType(numpathmodule) == "HLT1CaloJet") ||
                (hltConfig_.moduleType(numpathmodule) == "HLT1PFJet") ||
                (hltConfig_.moduleType(numpathmodule) == "HLTDiJetAveFilter") ||
                (hltConfig_.moduleType(numpathmodule) == "HLTDiPFJetAveFilter") ||
                (hltConfig_.moduleType(numpathmodule) == "HLT1CaloMET") ||
                (hltConfig_.moduleType(numpathmodule) == "HLTMhtFilter") ||
                (hltConfig_.moduleType(numpathmodule) == "HLTPrescaler"))
              filtername = numpathmodule;
          }

          if (objectType != 0) {
            std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(dpathname);
            for (auto& numpathmodule : numpathmodules) {
              edm::InputTag testTag(numpathmodule, "", processname_);
              if ((hltConfig_.moduleType(numpathmodule) == "HLT1CaloJet") ||
                  (hltConfig_.moduleType(numpathmodule) == "HLT1PFJet") ||
                  (hltConfig_.moduleType(numpathmodule) == "HLTDiJetAveFilter") ||
                  (hltConfig_.moduleType(numpathmodule) == "HLTDiPFJetAveFilter") ||
                  (hltConfig_.moduleType(numpathmodule) == "HLT1CaloMET") ||
                  (hltConfig_.moduleType(numpathmodule) == "HLTMhtFilter") ||
                  (hltConfig_.moduleType(numpathmodule) == "HLTPrescaler"))
                Denomfiltername = numpathmodule;
            }

            if (verbose_)
              cout << "==pathname==" << pathname << "==denompath==" << dpathname << "==filtername==" << filtername
                   << "==denomfiltername==" << Denomfiltername << "==l1pathname==" << l1pathname
                   << "==objectType==" << objectType << endl;
            hltPathsEff_.push_back(PathInfo(usedPrescale,
                                            dpathname,
                                            pathname,
                                            l1pathname,
                                            filtername,
                                            Denomfiltername,
                                            processname_,
                                            objectType,
                                            triggerType));
          }
        }
      }
    }

    if (verbose_)
      cout << "== end hltPathsEff_.push_back ======" << endl;
  }
}

//------------------------------------------------------------------------//
void JetMETHLTOfflineSource::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& c) {
  if (!isSetup_) {
    iBooker.setCurrentFolder(dirname_);

    //-----------------------------------------------------------------
    //---book trigger summary histos
    if (!isSetup_) {
      std::string foldernm = "/TriggerSummary/";
      iBooker.setCurrentFolder(dirname_ + foldernm);

      int TrigBins_ = hltPathsAllTriggerSummary_.size();
      double TrigMin_ = -0.5;
      double TrigMax_ = hltPathsAllTriggerSummary_.size() - 0.5;

      std::string histonm = "JetMET_TriggerRate";
      std::string histot = "JetMET TriggerRate Summary";
      rate_All = iBooker.book1D(histonm.c_str(), histot.c_str(), TrigBins_, TrigMin_, TrigMax_);

      histonm = "JetMET_TriggerRate_Correlation";
      histot = "JetMET TriggerRate Correlation Summary;y&&!x;x&&y";
      correlation_All =
          iBooker.book2D(histonm.c_str(), histot.c_str(), TrigBins_, TrigMin_, TrigMax_, TrigBins_, TrigMin_, TrigMax_);

      histonm = "JetMET_NVertices";
      histot = "No. of vertices";
      NVertices = iBooker.book1D(histonm.c_str(), histot.c_str(), 100, 0, 50);

      histonm = "JetMET_PVZ";
      histot = "Primary Vertex Z pos";
      PVZ = iBooker.book1D(histonm.c_str(), histot.c_str(), 100, -50., 50.);

      if (runStandalone_) {
        histonm = "JetMET_TriggerRate_WrtMuTrigger";
        histot = "JetMET TriggerRate Summary Wrt Muon Trigger ";
        rate_AllWrtMu = iBooker.book1D(histonm.c_str(), histot.c_str(), TrigBins_, TrigMin_, TrigMax_);

        histonm = "JetMET_TriggerRate_Correlation_WrtMuTrigger";
        histot = "JetMET TriggerRate Correlation Summary Wrt Muon Trigger;y&&!x;x&&y";
        correlation_AllWrtMu = iBooker.book2D(
            histonm.c_str(), histot.c_str(), TrigBins_, TrigMin_, TrigMax_, TrigBins_, TrigMin_, TrigMax_);

        histonm = "JetMET_TriggerRate_WrtMBTrigger";
        histot = "JetMET TriggerRate Summary Wrt MB Trigger";
        rate_AllWrtMB = iBooker.book1D(histonm.c_str(), histot.c_str(), TrigBins_, TrigMin_, TrigMax_);

        histonm = "JetMET_TriggerRate_Correlation_WrtMBTrigger";
        histot = "JetMET TriggerRate Correlation Wrt MB Trigger;y&&!x;x&&y";
        correlation_AllWrtMB = iBooker.book2D(
            histonm.c_str(), histot.c_str(), TrigBins_, TrigMin_, TrigMax_, TrigBins_, TrigMin_, TrigMax_);
      }
      isSetup_ = true;
    }
    //---Set bin label

    for (auto& v : hltPathsAllTriggerSummary_) {
      std::string labelnm("dummy");
      labelnm = v.getPath();
      int nbins = rate_All->getTH1()->GetNbinsX();
      for (int ibin = 1; ibin < nbins + 1; ibin++) {
        const char* binLabel = rate_All->getTH1()->GetXaxis()->GetBinLabel(ibin);
        std::string binLabel_str = string(binLabel);
        if (binLabel_str == labelnm)
          break;
        if (binLabel[0] == '\0') {
          rate_All->setBinLabel(ibin, labelnm);
          correlation_All->setBinLabel(ibin, labelnm, 1);
          correlation_All->setBinLabel(ibin, labelnm, 2);
          if (runStandalone_) {
            rate_AllWrtMu->setBinLabel(ibin, labelnm);
            rate_AllWrtMB->setBinLabel(ibin, labelnm);
            correlation_AllWrtMu->setBinLabel(ibin, labelnm, 1);
            correlation_AllWrtMB->setBinLabel(ibin, labelnm, 1);
            correlation_AllWrtMu->setBinLabel(ibin, labelnm, 2);
            correlation_AllWrtMB->setBinLabel(ibin, labelnm, 2);
          }
          break;
        }
      }
    }

    // Now define histos for All triggers
    if (plotAll_) {
      //
      int Nbins_ = 10;
      double Nmin_ = -0.5;
      double Nmax_ = 9.5;
      //
      int Ptbins_ = 100;
      if (runStandalone_)
        Ptbins_ = 1000;
      double PtMin_ = 0.;
      double PtMax_ = 1000.;
      //
      int Etabins_ = 50;
      if (runStandalone_)
        Etabins_ = 100;
      double EtaMin_ = -5.;
      double EtaMax_ = 5.;
      //
      int Phibins_ = 35;
      double PhiMin_ = -3.5;
      double PhiMax_ = 3.5;

      int Resbins_ = 30;
      double ResMin_ = -1.5;
      double ResMax_ = 1.5;
      //
      std::string dirName = dirname_ + "/MonitorAllTriggers/";
      for (auto& v : hltPathsAll_) {
        //
        std::string trgPathName = HLTConfigProvider::removeVersion(v.getPath());
        std::string subdirName = dirName + trgPathName;
        std::string trigPath = "(" + trgPathName + ")";
        iBooker.setCurrentFolder(subdirName);

        std::string labelname("ME");
        std::string histoname(labelname + "");
        std::string title(labelname + "");

        MonitorElement* dummy;
        dummy = iBooker.bookFloat("dummy");

        if (v.getObjectType() == trigger::TriggerJet && v.getTriggerType() == "SingleJet_Trigger") {
          histoname = labelname + "_recObjN";
          title = labelname + "_recObjN;Reco multiplicity()" + trigPath;
          MonitorElement* N = iBooker.book1D(histoname.c_str(), title.c_str(), Nbins_, Nmin_, Nmax_);
          N->getTH1();

          histoname = labelname + "_recObjPt";
          title = labelname + "_recObjPt; Reco Pt[GeV/c]" + trigPath;
          MonitorElement* Pt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          Pt->getTH1();

          histoname = labelname + "_recObjPtBarrel";
          title = labelname + "_recObjPtBarrel;Reco Pt[GeV/c]" + trigPath;
          MonitorElement* PtBarrel = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          PtBarrel->getTH1();

          histoname = labelname + "_recObjPtEndcap";
          title = labelname + "_recObjPtEndcap;Reco Pt[GeV/c]" + trigPath;
          MonitorElement* PtEndcap = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          PtEndcap->getTH1();

          histoname = labelname + "_recObjPtForward";
          title = labelname + "_recObjPtForward;Reco Pt[GeV/c]" + trigPath;
          MonitorElement* PtForward = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          PtForward->getTH1();

          histoname = labelname + "_recObjEta";
          title = labelname + "_recObjEta;Reco #eta" + trigPath;
          MonitorElement* Eta = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          Eta->getTH1();

          histoname = labelname + "_recObjPhi";
          title = labelname + "_recObjPhi;Reco #Phi" + trigPath;
          MonitorElement* Phi = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          Phi->getTH1();

          histoname = labelname + "_recObjEtaPhi";
          title = labelname + "_recObjEtaPhi;Reco #eta;Reco #Phi" + trigPath;
          MonitorElement* EtaPhi =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          EtaPhi->getTH1();

          histoname = labelname + "_l1ObjPt";
          title = labelname + "_l1ObjPt;L1 Pt[GeV/c]" + trigPath;
          MonitorElement* Pt_L1 = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          Pt_L1->getTH1();

          histoname = labelname + "_l1ObjEta";
          title = labelname + "_l1ObjEta;L1 #eta" + trigPath;
          MonitorElement* Eta_L1 = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          Eta_L1->getTH1();

          histoname = labelname + "_l1ObjPhi";
          title = labelname + "_l1ObjPhi;L1 #Phi" + trigPath;
          MonitorElement* Phi_L1 = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          Phi_L1->getTH1();

          histoname = labelname + "_l1ObjEtaPhi";
          title = labelname + "_l1ObjEtaPhi;L1 #eta;L1 #Phi" + trigPath;
          MonitorElement* EtaPhi_L1 =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          EtaPhi_L1->getTH1();

          histoname = labelname + "_l1ObjN";
          title = labelname + "_l1ObjN;L1 multiplicity" + trigPath;
          MonitorElement* N_L1 = iBooker.book1D(histoname.c_str(), title.c_str(), Nbins_, Nmin_, Nmax_);
          N_L1->getTH1();

          histoname = labelname + "_l1ObjPtBarrel";
          title = labelname + "_l1ObjPtBarrel;L1 Pt[GeV/c]" + trigPath;
          MonitorElement* PtBarrel_L1 = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          PtBarrel_L1->getTH1();

          histoname = labelname + "_l1ObjPtEndcap";
          title = labelname + "_l1ObjPtEndcap;L1 Pt[GeV/c]" + trigPath;
          MonitorElement* PtEndcap_L1 = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          PtEndcap_L1->getTH1();

          histoname = labelname + "_l1ObjPtForward";
          title = labelname + "_l1ObjPtForward;L1 Pt[GeV/c]" + trigPath;
          MonitorElement* PtForward_L1 = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          PtForward_L1->getTH1();

          histoname = labelname + "_hltObjN";
          title = labelname + "_hltObjN;HLT multiplicity" + trigPath;
          MonitorElement* N_HLT = iBooker.book1D(histoname.c_str(), title.c_str(), Nbins_, Nmin_, Nmax_);
          N_HLT->getTH1();

          histoname = labelname + "_hltObjPtBarrel";
          title = labelname + "_hltObjPtBarrel;HLT Pt[GeV/c]" + trigPath;
          MonitorElement* PtBarrel_HLT = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          PtBarrel_HLT->getTH1();

          histoname = labelname + "_hltObjPtEndcap";
          title = labelname + "_hltObjPtEndcap;HLT Pt[GeV/c]" + trigPath;
          MonitorElement* PtEndcap_HLT = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          PtEndcap_HLT->getTH1();

          histoname = labelname + "_hltObjPtForward";
          title = labelname + "_hltObjPtForward;HLT Pt[GeV/c]" + trigPath;
          MonitorElement* PtForward_HLT = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          PtForward_HLT->getTH1();

          histoname = labelname + "_hltObjPt";
          title = labelname + "_hltObjPt;HLT Pt[GeV/c]" + trigPath;
          MonitorElement* Pt_HLT = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          Pt_HLT->getTH1();

          histoname = labelname + "_hltObjEta";
          title = labelname + "_hltObjEta;HLT #eta" + trigPath;
          MonitorElement* Eta_HLT = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          Eta_HLT->getTH1();

          histoname = labelname + "_hltObjPhi";
          title = labelname + "_hltObjPhi;HLT #Phi" + trigPath;
          MonitorElement* Phi_HLT = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          Phi_HLT->getTH1();

          histoname = labelname + "_hltObjEtaPhi";
          title = labelname + "_hltObjEtaPhi;HLT #eta;HLT #Phi" + trigPath;
          MonitorElement* EtaPhi_HLT =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          EtaPhi_HLT->getTH1();

          histoname = labelname + "_l1HLTPtResolution";
          title = labelname + "_l1HLTPtResolution;(Pt(L1)-Pt(HLT))/Pt(L1)" + trigPath;
          MonitorElement* PtResolution_L1HLT =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          PtResolution_L1HLT->getTH1();

          histoname = labelname + "_l1HLTEtaResolution";
          title = labelname + "_l1HLTEtaResolution;#eta(L1)-#eta(HLT)" + trigPath;
          MonitorElement* EtaResolution_L1HLT =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          EtaResolution_L1HLT->getTH1();

          histoname = labelname + "_l1HLTPhiResolution";
          title = labelname + "_l1HLTPhiResolution;#Phi(L1)-#Phi(HLT)" + trigPath;
          MonitorElement* PhiResolution_L1HLT =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          PhiResolution_L1HLT->getTH1();

          histoname = labelname + "_l1HLTPtCorrelation";
          title = labelname + "_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]" + trigPath;
          MonitorElement* PtCorrelation_L1HLT =
              iBooker.book2D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_, Ptbins_, PtMin_, PtMax_);
          PtCorrelation_L1HLT->getTH1();

          histoname = labelname + "_l1HLTEtaCorrelation";
          title = labelname + "_l1HLTEtaCorrelation;#eta(L1);#eta(HLT)" + trigPath;
          MonitorElement* EtaCorrelation_L1HLT =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Etabins_, EtaMin_, EtaMax_);
          EtaCorrelation_L1HLT->getTH1();

          histoname = labelname + "_l1HLTPhiCorrelation";
          title = labelname + "_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)" + trigPath;
          MonitorElement* PhiCorrelation_L1HLT =
              iBooker.book2D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_, Phibins_, PhiMin_, PhiMax_);
          PhiCorrelation_L1HLT->getTH1();

          histoname = labelname + "_hltRecObjPtResolution";
          title = labelname + "_hltRecObjPtResolution;(Pt(HLT)-Pt(Reco))/Pt(HLT)" + trigPath;
          MonitorElement* PtResolution_HLTRecObj =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          PtResolution_HLTRecObj->getTH1();

          histoname = labelname + "_hltRecObjEtaResolution";
          title = labelname + "_hltRecObjEtaResolution;#eta(HLT)-#eta(Reco)" + trigPath;
          MonitorElement* EtaResolution_HLTRecObj =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          EtaResolution_HLTRecObj->getTH1();

          histoname = labelname + "_hltRecObjPhiResolution";
          title = labelname + "_hltRecObjPhiResolution;#Phi(HLT)-#Phi(Reco)" + trigPath;
          MonitorElement* PhiResolution_HLTRecObj =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          PhiResolution_HLTRecObj->getTH1();

          histoname = labelname + "_hltRecObjPtCorrelation";
          title = labelname + "_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]" + trigPath;
          MonitorElement* PtCorrelation_HLTRecObj =
              iBooker.book2D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_, Ptbins_, PtMin_, PtMax_);
          PtCorrelation_HLTRecObj->getTH1();

          histoname = labelname + "_hltRecObjEtaCorrelation";
          title = labelname + "_hltRecObjEtaCorrelation;#eta(HLT);#eta(Reco)" + trigPath;
          MonitorElement* EtaCorrelation_HLTRecObj =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Etabins_, EtaMin_, EtaMax_);
          EtaCorrelation_HLTRecObj->getTH1();

          histoname = labelname + "_hltRecObjPhiCorrelation";
          title = labelname + "_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)" + trigPath;
          MonitorElement* PhiCorrelation_HLTRecObj =
              iBooker.book2D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_, Phibins_, PhiMin_, PhiMax_);
          PhiCorrelation_HLTRecObj->getTH1();

          v.setHistos(N,
                      Pt,
                      PtBarrel,
                      PtEndcap,
                      PtForward,
                      Eta,
                      Phi,
                      EtaPhi,
                      N_L1,
                      Pt_L1,
                      PtBarrel_L1,
                      PtEndcap_L1,
                      PtForward_L1,
                      Eta_L1,
                      Phi_L1,
                      EtaPhi_L1,
                      N_HLT,
                      Pt_HLT,
                      PtBarrel_HLT,
                      PtEndcap_HLT,
                      PtForward_HLT,
                      Eta_HLT,
                      Phi_HLT,
                      EtaPhi_HLT,
                      PtResolution_L1HLT,
                      EtaResolution_L1HLT,
                      PhiResolution_L1HLT,
                      PtResolution_HLTRecObj,
                      EtaResolution_HLTRecObj,
                      PhiResolution_HLTRecObj,
                      PtCorrelation_L1HLT,
                      EtaCorrelation_L1HLT,
                      PhiCorrelation_L1HLT,
                      PtCorrelation_HLTRecObj,
                      EtaCorrelation_HLTRecObj,
                      PhiCorrelation_HLTRecObj,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy);
        }  // histos for SingleJet Triggers

        if (v.getObjectType() == trigger::TriggerJet && v.getTriggerType() == "DiJet_Trigger") {
          histoname = labelname + "_RecObjAveragePt";
          title = labelname + "_RecObjAveragePt;Reco Average Pt[GeV/c]" + trigPath;
          MonitorElement* jetAveragePt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          jetAveragePt->getTH1();

          histoname = labelname + "_RecObjAverageEta";
          title = labelname + "_RecObjAverageEta;Reco Average #eta" + trigPath;
          MonitorElement* jetAverageEta = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          jetAverageEta->getTH1();

          histoname = labelname + "_RecObjPhiDifference";
          title = labelname + "_RecObjPhiDifference;Reco #Delta#Phi" + trigPath;
          MonitorElement* jetPhiDifference =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          jetPhiDifference->getTH1();

          histoname = labelname + "_hltObjAveragePt";
          title = labelname + "_hltObjAveragePt;HLT Average Pt[GeV/c]" + trigPath;
          MonitorElement* hltAveragePt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          hltAveragePt->getTH1();

          histoname = labelname + "_hltObjAverageEta";
          title = labelname + "_hltObjAverageEta;HLT Average #eta" + trigPath;
          MonitorElement* hltAverageEta = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          hltAverageEta->getTH1();

          histoname = labelname + "_hltObjPhiDifference";
          title = labelname + "_hltObjPhiDifference;Reco #Delta#Phi" + trigPath;
          MonitorElement* hltPhiDifference =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          hltPhiDifference->getTH1();

          v.setHistos(dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      jetAveragePt,
                      jetAverageEta,
                      jetPhiDifference,
                      hltAveragePt,
                      hltAverageEta,
                      hltPhiDifference,
                      dummy,
                      dummy,
                      dummy);
        }  // histos for DiJet Triggers

        if (v.getObjectType() == trigger::TriggerMET || (v.getObjectType() == trigger::TriggerTET)) {
          histoname = labelname + "_recObjPt";
          title = labelname + "_recObjPt;Reco Pt[GeV/c]" + trigPath;
          MonitorElement* Pt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          Pt->getTH1();

          histoname = labelname + "_recObjPhi";
          title = labelname + "_recObjPhi;Reco #Phi" + trigPath;
          MonitorElement* Phi = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          Phi->getTH1();

          histoname = labelname + "_l1ObjPt";
          title = labelname + "_l1ObjPt;L1 Pt[GeV/c]" + trigPath;
          MonitorElement* Pt_L1 = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          Pt_L1->getTH1();

          histoname = labelname + "_l1ObjPhi";
          title = labelname + "_l1ObjPhi;L1 #Phi" + trigPath;
          MonitorElement* Phi_L1 = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          Phi_L1->getTH1();

          histoname = labelname + "_hltObjPt";
          title = labelname + "_hltObjPt;HLT Pt[GeV/c]" + trigPath;
          MonitorElement* Pt_HLT = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          Pt_HLT->getTH1();

          histoname = labelname + "_hltObjPhi";
          title = labelname + "_hltObjPhi;HLT #Phi" + trigPath;
          MonitorElement* Phi_HLT = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          Phi_HLT->getTH1();

          histoname = labelname + "_l1HLTPtResolution";
          title = labelname + "_l1HLTPtResolution;(Pt(L1)-Pt(HLT))/Pt(L1)" + trigPath;
          MonitorElement* PtResolution_L1HLT =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          PtResolution_L1HLT->getTH1();

          histoname = labelname + "_l1HLTPhiResolution";
          title = labelname + "_l1HLTPhiResolution;#Phi(L1)-#Phi(HLT)" + trigPath;
          MonitorElement* PhiResolution_L1HLT =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          PhiResolution_L1HLT->getTH1();

          histoname = labelname + "_l1HLTPtCorrelation";
          title = labelname + "_l1HLTPtCorrelation;Pt(L1)[GeV/c];Pt(HLT)[GeV/c]" + trigPath;
          MonitorElement* PtCorrelation_L1HLT =
              iBooker.book2D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_, Ptbins_, PtMin_, PtMax_);
          PtCorrelation_L1HLT->getTH1();

          histoname = labelname + "_l1HLTPhiCorrelation";
          title = labelname + "_l1HLTPhiCorrelation;#Phi(L1);#Phi(HLT)" + trigPath;
          MonitorElement* PhiCorrelation_L1HLT =
              iBooker.book2D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_, Phibins_, PhiMin_, PhiMax_);
          PhiCorrelation_L1HLT->getTH1();

          histoname = labelname + "_hltRecObjPtResolution";
          title = labelname + "_hltRecObjPtResolution;(Pt(HLT)-Pt(Reco))/Pt(HLT)" + trigPath;
          MonitorElement* PtResolution_HLTRecObj =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          PtResolution_HLTRecObj->getTH1();

          histoname = labelname + "_hltRecObjPhiResolution";
          title = labelname + "_hltRecObjPhiResolution;#Phi(HLT)-#Phi(Reco)" + trigPath;
          MonitorElement* PhiResolution_HLTRecObj =
              iBooker.book1D(histoname.c_str(), title.c_str(), Resbins_, ResMin_, ResMax_);
          PhiResolution_HLTRecObj->getTH1();

          histoname = labelname + "_hltRecObjPtCorrelation";
          title = labelname + "_hltRecObjPtCorrelation;Pt(HLT)[GeV/c];Pt(Reco)[GeV/c]" + trigPath;
          MonitorElement* PtCorrelation_HLTRecObj =
              iBooker.book2D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_, Ptbins_, PtMin_, PtMax_);
          PtCorrelation_HLTRecObj->getTH1();

          histoname = labelname + "_hltRecObjPhiCorrelation";
          title = labelname + "_hltRecObjPhiCorrelation;#Phi(HLT);#Phi(Reco)" + trigPath;
          MonitorElement* PhiCorrelation_HLTRecObj =
              iBooker.book2D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_, Phibins_, PhiMin_, PhiMax_);
          PhiCorrelation_HLTRecObj->getTH1();

          v.setHistos(dummy,
                      Pt,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      Phi,
                      dummy,
                      dummy,
                      Pt_L1,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      Phi_L1,
                      dummy,
                      dummy,
                      Pt_HLT,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      Phi_HLT,
                      dummy,
                      PtResolution_L1HLT,
                      dummy,
                      PhiResolution_L1HLT,
                      PtResolution_HLTRecObj,
                      dummy,
                      PhiResolution_HLTRecObj,
                      PtCorrelation_L1HLT,
                      dummy,
                      PhiCorrelation_L1HLT,
                      PtCorrelation_HLTRecObj,
                      dummy,
                      PhiCorrelation_HLTRecObj,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy,
                      dummy);
        }  // histos for MET Triggers
      }
    }  //plotAll_

    //-------Now Efficiency histos--------
    if (plotEff_) {
      int Ptbins_ = 100;
      if (runStandalone_)
        Ptbins_ = 1000;
      double PtMin_ = 0.;
      double PtMax_ = 1000.;
      //
      int Etabins_ = 50;
      double EtaMin_ = -5.;
      double EtaMax_ = 5.;
      //
      int Phibins_ = 35;
      double PhiMin_ = -3.5;
      double PhiMax_ = 3.5;
      // Now define histos wrt lower threshold trigger
      std::string dirName1 = dirname_ + "/RelativeTriggerEff/";
      for (auto& v : hltPathsEff_) {
        //
        std::string trgPathName = HLTConfigProvider::removeVersion(v.getPath());
        std::string trgPathNameD = HLTConfigProvider::removeVersion(v.getDenomPath());
        //
        std::string labelname("ME");
        std::string subdirName = dirName1 + trgPathName + "_wrt_" + trgPathNameD;
        iBooker.setCurrentFolder(subdirName);
        //
        std::string histoname(labelname + "");
        std::string title(labelname + "");

        MonitorElement* dummy;
        dummy = iBooker.bookFloat("dummy");

        if ((v.getObjectType() == trigger::TriggerJet) && (v.getTriggerType() == "SingleJet_Trigger")) {
          histoname = labelname + "_NumeratorPt";
          title = labelname + "NumeratorPt;Calo Pt[GeV/c]";
          MonitorElement* NumeratorPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPt->getTH1();

          histoname = labelname + "_NumeratorPtBarrel";
          title = labelname + "NumeratorPtBarrel;Calo Pt[GeV/c] ";
          MonitorElement* NumeratorPtBarrel = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPtBarrel->getTH1();

          histoname = labelname + "_NumeratorPtEndcap";
          title = labelname + "NumeratorPtEndcap;Calo Pt[GeV/c]";
          MonitorElement* NumeratorPtEndcap = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPtEndcap->getTH1();

          histoname = labelname + "_NumeratorPtForward";
          title = labelname + "NumeratorPtForward;Calo Pt[GeV/c]";
          MonitorElement* NumeratorPtForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPtForward->getTH1();

          histoname = labelname + "_NumeratorEta";
          title = labelname + "NumeratorEta;Calo #eta ";
          MonitorElement* NumeratorEta = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorEta->getTH1();

          histoname = labelname + "_NumeratorPhi";
          title = labelname + "NumeratorPhi;Calo #Phi";
          MonitorElement* NumeratorPhi = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPhi->getTH1();

          histoname = labelname + "_NumeratorEtaPhi";
          title = labelname + "NumeratorEtaPhi;Calo #eta;Calo #Phi";
          MonitorElement* NumeratorEtaPhi =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          NumeratorEtaPhi->getTH1();

          histoname = labelname + "_NumeratorEtaBarrel";
          title = labelname + "NumeratorEtaBarrel;Calo #eta ";
          MonitorElement* NumeratorEtaBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorEtaBarrel->getTH1();

          histoname = labelname + "_NumeratorPhiBarrel";
          title = labelname + "NumeratorPhiBarrel;Calo #Phi";
          MonitorElement* NumeratorPhiBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPhiBarrel->getTH1();

          histoname = labelname + "_NumeratorEtaEndcap";
          title = labelname + "NumeratorEtaEndcap;Calo #eta ";
          MonitorElement* NumeratorEtaEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorEtaEndcap->getTH1();

          histoname = labelname + "_NumeratorPhiEndcap";
          title = labelname + "NumeratorPhiEndcap;Calo #Phi";
          MonitorElement* NumeratorPhiEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPhiEndcap->getTH1();

          histoname = labelname + "_NumeratorEtaForward";
          title = labelname + "NumeratorEtaForward;Calo #eta ";
          MonitorElement* NumeratorEtaForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorEtaForward->getTH1();

          histoname = labelname + "_NumeratorPhiForward";
          title = labelname + "NumeratorPhiForward;Calo #Phi";
          MonitorElement* NumeratorPhiForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPhiForward->getTH1();

          histoname = labelname + "_NumeratorEta_LowpTcut";
          title = labelname + "NumeratorEta_LowpTcut;Calo #eta ";
          MonitorElement* NumeratorEta_LowpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorEta_LowpTcut->getTH1();

          histoname = labelname + "_NumeratorPhi_LowpTcut";
          title = labelname + "NumeratorPhi_LowpTcut;Calo #Phi";
          MonitorElement* NumeratorPhi_LowpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPhi_LowpTcut->getTH1();

          histoname = labelname + "_NumeratorEtaPhi_LowpTcut";
          title = labelname + "NumeratorEtaPhi_LowpTcut;Calo #eta;Calo #Phi";
          MonitorElement* NumeratorEtaPhi_LowpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          NumeratorEtaPhi_LowpTcut->getTH1();

          histoname = labelname + "_NumeratorEta_MedpTcut";
          title = labelname + "NumeratorEta_MedpTcut;Calo #eta ";
          MonitorElement* NumeratorEta_MedpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorEta_MedpTcut->getTH1();

          histoname = labelname + "_NumeratorPhi_MedpTcut";
          title = labelname + "NumeratorPhi_MedpTcut;Calo #Phi";
          MonitorElement* NumeratorPhi_MedpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPhi_MedpTcut->getTH1();

          histoname = labelname + "_NumeratorEtaPhi_MedpTcut";
          title = labelname + "NumeratorEtaPhi_MedpTcut;Calo #eta;Calo #Phi";
          MonitorElement* NumeratorEtaPhi_MedpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          NumeratorEtaPhi_MedpTcut->getTH1();

          histoname = labelname + "_NumeratorEta_HighpTcut";
          title = labelname + "NumeratorEta_HighpTcut;Calo #eta ";
          MonitorElement* NumeratorEta_HighpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorEta_HighpTcut->getTH1();

          histoname = labelname + "_NumeratorPhi_HighpTcut";
          title = labelname + "NumeratorPhi_HighpTcut;Calo #Phi";
          MonitorElement* NumeratorPhi_HighpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPhi_HighpTcut->getTH1();

          histoname = labelname + "_NumeratorEtaPhi_HighpTcut";
          title = labelname + "NumeratorEtaPhi_HighpTcut;Calo #eta;Calo #Phi";
          MonitorElement* NumeratorEtaPhi_HighpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          NumeratorEtaPhi_HighpTcut->getTH1();

          histoname = labelname + "_DenominatorPt";
          title = labelname + "DenominatorPt;Calo Pt[GeV/c]";
          MonitorElement* DenominatorPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPt->getTH1();

          histoname = labelname + "_DenominatorPtBarrel";
          title = labelname + "DenominatorPtBarrel;Calo Pt[GeV/c]";
          MonitorElement* DenominatorPtBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPtBarrel->getTH1();

          histoname = labelname + "_DenominatorPtEndcap";
          title = labelname + "DenominatorPtEndcap;Calo Pt[GeV/c]";
          MonitorElement* DenominatorPtEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPtEndcap->getTH1();

          histoname = labelname + "_DenominatorPtForward";
          title = labelname + "DenominatorPtForward;Calo Pt[GeV/c] ";
          MonitorElement* DenominatorPtForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPtForward->getTH1();

          histoname = labelname + "_DenominatorEta";
          title = labelname + "DenominatorEta;Calo #eta ";
          MonitorElement* DenominatorEta = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorEta->getTH1();

          histoname = labelname + "_DenominatorPhi";
          title = labelname + "DenominatorPhi;Calo #Phi";
          MonitorElement* DenominatorPhi = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPhi->getTH1();

          histoname = labelname + "_DenominatorEtaPhi";
          title = labelname + "DenominatorEtaPhi;Calo #eta; Calo #Phi";
          MonitorElement* DenominatorEtaPhi =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          DenominatorEtaPhi->getTH1();

          histoname = labelname + "_DenominatorEtaBarrel";
          title = labelname + "DenominatorEtaBarrel;Calo #eta ";
          MonitorElement* DenominatorEtaBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorEtaBarrel->getTH1();

          histoname = labelname + "_DenominatorPhiBarrel";
          title = labelname + "DenominatorPhiBarrel;Calo #Phi";
          MonitorElement* DenominatorPhiBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPhiBarrel->getTH1();

          histoname = labelname + "_DenominatorEtaEndcap";
          title = labelname + "DenominatorEtaEndcap;Calo #eta ";
          MonitorElement* DenominatorEtaEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorEtaEndcap->getTH1();

          histoname = labelname + "_DenominatorPhiEndcap";
          title = labelname + "DenominatorPhiEndcap;Calo #Phi";
          MonitorElement* DenominatorPhiEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPhiEndcap->getTH1();

          histoname = labelname + "_DenominatorEtaForward";
          title = labelname + "DenominatorEtaForward;Calo #eta ";
          MonitorElement* DenominatorEtaForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorEtaForward->getTH1();

          histoname = labelname + "_DenominatorPhiForward";
          title = labelname + "DenominatorPhiForward;Calo #Phi";
          MonitorElement* DenominatorPhiForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPhiForward->getTH1();

          histoname = labelname + "_DenominatorEta_LowpTcut";
          title = labelname + "DenominatorEta_LowpTcut;Calo #eta ";
          MonitorElement* DenominatorEta_LowpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorEta_LowpTcut->getTH1();

          histoname = labelname + "_DenominatorPhi_LowpTcut";
          title = labelname + "DenominatorPhi_LowpTcut;Calo #Phi";
          MonitorElement* DenominatorPhi_LowpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPhi_LowpTcut->getTH1();

          histoname = labelname + "_DenominatorEtaPhi_LowpTcut";
          title = labelname + "DenominatorEtaPhi_LowpTcut;Calo #eta;Calo #Phi";
          MonitorElement* DenominatorEtaPhi_LowpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          DenominatorEtaPhi_LowpTcut->getTH1();

          histoname = labelname + "_DenominatorEta_MedpTcut";
          title = labelname + "DenominatorEta_MedpTcut;Calo #eta ";
          MonitorElement* DenominatorEta_MedpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorEta_MedpTcut->getTH1();

          histoname = labelname + "_DenominatorPhi_MedpTcut";
          title = labelname + "DenominatorPhi_MedpTcut;Calo #Phi";
          MonitorElement* DenominatorPhi_MedpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPhi_MedpTcut->getTH1();

          histoname = labelname + "_DenominatorEtaPhi_MedpTcut";
          title = labelname + "DenominatorEtaPhi_MedpTcut;Calo #eta;Calo #Phi";
          MonitorElement* DenominatorEtaPhi_MedpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          DenominatorEtaPhi_MedpTcut->getTH1();

          histoname = labelname + "_DenominatorEta_HighpTcut";
          title = labelname + "DenominatorEta_HighpTcut;Calo #eta ";
          MonitorElement* DenominatorEta_HighpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorEta_HighpTcut->getTH1();

          histoname = labelname + "_DenominatorPhi_HighpTcut";
          title = labelname + "DenominatorPhi_HighpTcut;Calo #Phi";
          MonitorElement* DenominatorPhi_HighpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPhi_HighpTcut->getTH1();

          histoname = labelname + "_DenominatorEtaPhi_HighpTcut";
          title = labelname + "DenominatorEtaPhi_HighpTcut;Calo #eta;Calo #Phi";
          MonitorElement* DenominatorEtaPhi_HighpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          DenominatorEtaPhi_HighpTcut->getTH1();

          histoname = labelname + "_DeltaR";
          title = labelname + "DeltaR;";
          MonitorElement* DeltaR = iBooker.book1D(histoname.c_str(), title.c_str(), 100, 0., 1.5);
          DeltaR->getTH1();

          histoname = labelname + "_DeltaPhi";
          title = labelname + "DeltaPhi;";
          MonitorElement* DeltaPhi = iBooker.book1D(histoname.c_str(), title.c_str(), 500, -5.0, 5.0);
          DeltaPhi->getTH1();

          histoname = labelname + "_NumeratorPFMHT";
          title = labelname + "NumeratorPFMHT;PFMHT[GeV/c]";
          MonitorElement* NumeratorPFMHT = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPFMHT->getTH1();

          histoname = labelname + "_NumeratorPFPt";
          title = labelname + "NumeratorPFPt;PF Pt[GeV/c]";
          MonitorElement* NumeratorPFPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPFPt->getTH1();

          histoname = labelname + "_NumeratorPFPtBarrel";
          title = labelname + "NumeratorPFPtBarrel;PF Pt[GeV/c] ";
          MonitorElement* NumeratorPFPtBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPFPtBarrel->getTH1();

          histoname = labelname + "_NumeratorPFPtEndcap";
          title = labelname + "NumeratorPFPtEndcap;PF Pt[GeV/c]";
          MonitorElement* NumeratorPFPtEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPFPtEndcap->getTH1();

          histoname = labelname + "_NumeratorPFPtForward";
          title = labelname + "NumeratorPFPtForward;PF Pt[GeV/c]";
          MonitorElement* NumeratorPFPtForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPFPtForward->getTH1();

          histoname = labelname + "_NumeratorPFEta";
          title = labelname + "NumeratorPFEta;PF #eta ";
          MonitorElement* NumeratorPFEta = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorPFEta->getTH1();

          histoname = labelname + "_NumeratorPFPhi";
          title = labelname + "NumeratorPFPhi;Calo #Phi";
          MonitorElement* NumeratorPFPhi = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPFPhi->getTH1();

          histoname = labelname + "_NumeratorPFEtaPhi";
          title = labelname + "NumeratorPFEtaPhi;PF #eta;Calo #Phi";
          MonitorElement* NumeratorPFEtaPhi =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          NumeratorPFEtaPhi->getTH1();

          histoname = labelname + "_NumeratorPFEtaBarrel";
          title = labelname + "NumeratorPFEtaBarrel;PF #eta ";
          MonitorElement* NumeratorPFEtaBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorPFEtaBarrel->getTH1();

          histoname = labelname + "_NumeratorPFPhiBarrel";
          title = labelname + "NumeratorPFPhiBarrel;PF #Phi";
          MonitorElement* NumeratorPFPhiBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPFPhiBarrel->getTH1();

          histoname = labelname + "_NumeratorPFEtaEndcap";
          title = labelname + "NumeratorPFEtaEndcap;Calo #eta ";
          MonitorElement* NumeratorPFEtaEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorPFEtaEndcap->getTH1();

          histoname = labelname + "_NumeratorPFPhiEndcap";
          title = labelname + "NumeratorPFPhiEndcap;PF #Phi";
          MonitorElement* NumeratorPFPhiEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPFPhiEndcap->getTH1();

          histoname = labelname + "_NumeratorPFEtaForward";
          title = labelname + "NumeratorPFEtaForward;Calo #eta ";
          MonitorElement* NumeratorPFEtaForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorPFEtaForward->getTH1();

          histoname = labelname + "_NumeratorPFPhiForward";
          title = labelname + "NumeratorPFPhiForward;PF #Phi";
          MonitorElement* NumeratorPFPhiForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPFPhiForward->getTH1();

          histoname = labelname + "_NumeratorPFEta_LowpTcut";
          title = labelname + "NumeratorPFEta_LowpTcut;PF #eta ";
          MonitorElement* NumeratorPFEta_LowpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorPFEta_LowpTcut->getTH1();

          histoname = labelname + "_NumeratorPFPhi_LowpTcut";
          title = labelname + "NumeratorPFPhi_LowpTcut;PF #Phi";
          MonitorElement* NumeratorPFPhi_LowpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPFPhi_LowpTcut->getTH1();

          histoname = labelname + "_NumeratorPFEtaPhi_LowpTcut";
          title = labelname + "NumeratorPFEtaPhi_LowpTcut;PF #eta;Calo #Phi";
          MonitorElement* NumeratorPFEtaPhi_LowpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          NumeratorPFEtaPhi_LowpTcut->getTH1();

          histoname = labelname + "_NumeratorPFEta_MedpTcut";
          title = labelname + "NumeratorPFEta_MedpTcut;PF #eta ";
          MonitorElement* NumeratorPFEta_MedpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorPFEta_MedpTcut->getTH1();

          histoname = labelname + "_NumeratorPFPhi_MedpTcut";
          title = labelname + "NumeratorPFPhi_MedpTcut;PF #Phi";
          MonitorElement* NumeratorPFPhi_MedpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPFPhi_MedpTcut->getTH1();

          histoname = labelname + "_NumeratorPFEtaPhi_MedpTcut";
          title = labelname + "NumeratorPFEtaPhi_MedpTcut;PF #eta;PF #Phi";
          MonitorElement* NumeratorPFEtaPhi_MedpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          NumeratorPFEtaPhi_MedpTcut->getTH1();

          histoname = labelname + "_NumeratorPFEta_HighpTcut";
          title = labelname + "NumeratorPFEta_HighpTcut;Calo #eta ";
          MonitorElement* NumeratorPFEta_HighpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorPFEta_HighpTcut->getTH1();

          histoname = labelname + "_NumeratorPFPhi_HighpTcut";
          title = labelname + "NumeratorPFPhi_HighpTcut;PF #Phi";
          MonitorElement* NumeratorPFPhi_HighpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPFPhi_HighpTcut->getTH1();

          histoname = labelname + "_NumeratorPFEtaPhi_HighpTcut";
          title = labelname + "NumeratorPFEtaPhi_HighpTcut;PF #eta;PF #Phi";
          MonitorElement* NumeratorPFEtaPhi_HighpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          NumeratorPFEtaPhi_HighpTcut->getTH1();

          histoname = labelname + "_DenominatorPFMHT";
          title = labelname + "DenominatorPFMHT;PF Pt[GeV/c]";
          MonitorElement* DenominatorPFMHT = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPFMHT->getTH1();

          histoname = labelname + "_DenominatorPFPt";
          title = labelname + "DenominatorPFPt;PF Pt[GeV/c]";
          MonitorElement* DenominatorPFPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPFPt->getTH1();

          histoname = labelname + "_DenominatorPFPtBarrel";
          title = labelname + "DenominatorPFPtBarrel;Calo Pt[GeV/c]";
          MonitorElement* DenominatorPFPtBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPFPtBarrel->getTH1();

          histoname = labelname + "_DenominatorPFPtEndcap";
          title = labelname + "DenominatorPFPtEndcap;PF Pt[GeV/c]";
          MonitorElement* DenominatorPFPtEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPFPtEndcap->getTH1();

          histoname = labelname + "_DenominatorPFPtForward";
          title = labelname + "DenominatorPFPtForward;PF Pt[GeV/c] ";
          MonitorElement* DenominatorPFPtForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPFPtForward->getTH1();

          histoname = labelname + "_DenominatorPFEta";
          title = labelname + "DenominatorPFEta;PF #eta ";
          MonitorElement* DenominatorPFEta =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorPFEta->getTH1();

          histoname = labelname + "_DenominatorPFPhi";
          title = labelname + "DenominatorPFPhi;PF #Phi";
          MonitorElement* DenominatorPFPhi =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPFPhi->getTH1();

          histoname = labelname + "_DenominatorPFEtaPhi";
          title = labelname + "DenominatorPFEtaPhi;PF #eta; Calo #Phi";
          MonitorElement* DenominatorPFEtaPhi =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          DenominatorPFEtaPhi->getTH1();

          histoname = labelname + "_DenominatorPFEtaBarrel";
          title = labelname + "DenominatorPFEtaBarrel;Calo #eta ";
          MonitorElement* DenominatorPFEtaBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorPFEtaBarrel->getTH1();

          histoname = labelname + "_DenominatorPFPhiBarrel";
          title = labelname + "DenominatorPFPhiBarrel;PF #Phi";
          MonitorElement* DenominatorPFPhiBarrel =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPFPhiBarrel->getTH1();

          histoname = labelname + "_DenominatorPFEtaEndcap";
          title = labelname + "DenominatorPFEtaEndcap;PF #eta ";
          MonitorElement* DenominatorPFEtaEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorPFEtaEndcap->getTH1();

          histoname = labelname + "_DenominatorPFPhiEndcap";
          title = labelname + "DenominatorPFPhiEndcap;Calo #Phi";
          MonitorElement* DenominatorPFPhiEndcap =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPFPhiEndcap->getTH1();

          histoname = labelname + "_DenominatorPFEtaForward";
          title = labelname + "DenominatorPFEtaForward;PF #eta ";
          MonitorElement* DenominatorPFEtaForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorPFEtaForward->getTH1();

          histoname = labelname + "_DenominatorPFPhiForward";
          title = labelname + "DenominatorPFPhiForward;PF #Phi";
          MonitorElement* DenominatorPFPhiForward =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPFPhiForward->getTH1();

          histoname = labelname + "_DenominatorPFEta_LowpTcut";
          title = labelname + "DenominatorPFEta_LowpTcut;PF #eta ";
          MonitorElement* DenominatorPFEta_LowpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorPFEta_LowpTcut->getTH1();

          histoname = labelname + "_DenominatorPFPhi_LowpTcut";
          title = labelname + "DenominatorPFPhi_LowpTcut;PF #Phi";
          MonitorElement* DenominatorPFPhi_LowpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPFPhi_LowpTcut->getTH1();

          histoname = labelname + "_DenominatorPFEtaPhi_LowpTcut";
          title = labelname + "DenominatorPFEtaPhi_LowpTcut;PF #eta;Calo #Phi";
          MonitorElement* DenominatorPFEtaPhi_LowpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          DenominatorPFEtaPhi_LowpTcut->getTH1();

          histoname = labelname + "_DenominatorPFEta_MedpTcut";
          title = labelname + "DenominatorPFEta_MedpTcut;PF #eta ";
          MonitorElement* DenominatorPFEta_MedpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorPFEta_MedpTcut->getTH1();

          histoname = labelname + "_DenominatorPFPhi_MedpTcut";
          title = labelname + "DenominatorPFPhi_MedpTcut;PF #Phi";
          MonitorElement* DenominatorPFPhi_MedpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPFPhi_MedpTcut->getTH1();

          histoname = labelname + "_DenominatorPFEtaPhi_MedpTcut";
          title = labelname + "DenominatorPFEtaPhi_MedpTcut;PF #eta;Calo #Phi";
          MonitorElement* DenominatorPFEtaPhi_MedpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          DenominatorPFEtaPhi_MedpTcut->getTH1();

          histoname = labelname + "_DenominatorPFEta_HighpTcut";
          title = labelname + "DenominatorPFEta_HighpTcut;PF #eta ";
          MonitorElement* DenominatorPFEta_HighpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorPFEta_HighpTcut->getTH1();

          histoname = labelname + "_DenominatorPFPhi_HighpTcut";
          title = labelname + "DenominatorPFPhi_HighpTcut;PF #Phi";
          MonitorElement* DenominatorPFPhi_HighpTcut =
              iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPFPhi_HighpTcut->getTH1();

          histoname = labelname + "_DenominatorPFEtaPhi_HighpTcut";
          title = labelname + "DenominatorPFEtaPhi_HighpTcut;PF #eta;Calo #Phi";
          MonitorElement* DenominatorPFEtaPhi_HighpTcut =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Phibins_, PhiMin_, PhiMax_);
          DenominatorPFEtaPhi_HighpTcut->getTH1();

          histoname = labelname + "_PFDeltaR";
          title = labelname + "PFDeltaR;";
          MonitorElement* PFDeltaR = iBooker.book1D(histoname.c_str(), title.c_str(), 100, 0., 1.5);
          PFDeltaR->getTH1();

          histoname = labelname + "_PFDeltaPhi";
          title = labelname + "PFDeltaPhi;";
          MonitorElement* PFDeltaPhi = iBooker.book1D(histoname.c_str(), title.c_str(), 500, -5.0, 5.0);
          PFDeltaPhi->getTH1();

          v.setEffHistos(NumeratorPt,
                         NumeratorPtBarrel,
                         NumeratorPtEndcap,
                         NumeratorPtForward,
                         NumeratorEta,
                         NumeratorPhi,
                         NumeratorEtaPhi,
                         //
                         NumeratorEtaBarrel,
                         NumeratorPhiBarrel,
                         NumeratorEtaEndcap,
                         NumeratorPhiEndcap,
                         NumeratorEtaForward,
                         NumeratorPhiForward,
                         NumeratorEta_LowpTcut,
                         NumeratorPhi_LowpTcut,
                         NumeratorEtaPhi_LowpTcut,
                         NumeratorEta_MedpTcut,
                         NumeratorPhi_MedpTcut,
                         NumeratorEtaPhi_MedpTcut,
                         NumeratorEta_HighpTcut,
                         NumeratorPhi_HighpTcut,
                         NumeratorEtaPhi_HighpTcut,
                         //
                         DenominatorPt,
                         DenominatorPtBarrel,
                         DenominatorPtEndcap,
                         DenominatorPtForward,
                         DenominatorEta,
                         DenominatorPhi,
                         DenominatorEtaPhi,
                         //
                         DenominatorEtaBarrel,
                         DenominatorPhiBarrel,
                         DenominatorEtaEndcap,
                         DenominatorPhiEndcap,
                         DenominatorEtaForward,
                         DenominatorPhiForward,
                         DenominatorEta_LowpTcut,
                         DenominatorPhi_LowpTcut,
                         DenominatorEtaPhi_LowpTcut,
                         DenominatorEta_MedpTcut,
                         DenominatorPhi_MedpTcut,
                         DenominatorEtaPhi_MedpTcut,
                         DenominatorEta_HighpTcut,
                         DenominatorPhi_HighpTcut,
                         DenominatorEtaPhi_HighpTcut,
                         DeltaR,
                         DeltaPhi,
                         //
                         NumeratorPFPt,
                         NumeratorPFMHT,
                         NumeratorPFPtBarrel,
                         NumeratorPFPtEndcap,
                         NumeratorPFPtForward,
                         NumeratorPFEta,
                         NumeratorPFPhi,
                         NumeratorPFEtaPhi,
                         NumeratorPFEtaBarrel,
                         NumeratorPFPhiBarrel,
                         NumeratorPFEtaEndcap,
                         NumeratorPFPhiEndcap,
                         NumeratorPFEtaForward,
                         NumeratorPFPhiForward,
                         NumeratorPFEta_LowpTcut,
                         NumeratorPFPhi_LowpTcut,
                         NumeratorPFEtaPhi_LowpTcut,
                         NumeratorPFEta_MedpTcut,
                         NumeratorPFPhi_MedpTcut,
                         NumeratorPFEtaPhi_MedpTcut,
                         NumeratorPFEta_HighpTcut,
                         NumeratorPFPhi_HighpTcut,
                         NumeratorPFEtaPhi_HighpTcut,
                         DenominatorPFPt,
                         DenominatorPFMHT,
                         DenominatorPFPtBarrel,
                         DenominatorPFPtEndcap,
                         DenominatorPFPtForward,
                         DenominatorPFEta,
                         DenominatorPFPhi,
                         DenominatorPFEtaPhi,
                         DenominatorPFEtaBarrel,
                         DenominatorPFPhiBarrel,
                         DenominatorPFEtaEndcap,
                         DenominatorPFPhiEndcap,
                         DenominatorPFEtaForward,
                         DenominatorPFPhiForward,
                         DenominatorPFEta_LowpTcut,
                         DenominatorPFPhi_LowpTcut,
                         DenominatorPFEtaPhi_LowpTcut,
                         DenominatorPFEta_MedpTcut,
                         DenominatorPFPhi_MedpTcut,
                         DenominatorPFEtaPhi_MedpTcut,
                         DenominatorPFEta_HighpTcut,
                         DenominatorPFPhi_HighpTcut,
                         DenominatorPFEtaPhi_HighpTcut,
                         PFDeltaR,
                         PFDeltaPhi);

        }  // Loop over Jet Trigger

        if ((v.getObjectType() == trigger::TriggerJet) && (v.getTriggerType() == "DiJet_Trigger")) {
          histoname = labelname + "_NumeratorAvrgPt";
          title = labelname + "NumeratorAvrgPt;Calo Pt[GeV/c]";
          MonitorElement* NumeratorPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPt->getTH1();

          histoname = labelname + "_NumeratorAvrgEta";
          title = labelname + "NumeratorAvrgEta;Calo #eta";
          MonitorElement* NumeratorEta = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorEta->getTH1();

          histoname = labelname + "_DenominatorAvrgPt";
          title = labelname + "DenominatorAvrgPt;Calo Pt[GeV/c] ";
          MonitorElement* DenominatorPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPt->getTH1();

          histoname = labelname + "_DenominatorAvrgEta";
          title = labelname + "DenominatorAvrgEta;Calo #eta";
          MonitorElement* DenominatorEta = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorEta->getTH1();

          histoname = labelname + "_DeltaR";
          title = labelname + "DeltaR;";
          MonitorElement* DeltaR = iBooker.book1D(histoname.c_str(), title.c_str(), 100, 0., 1.5);
          DeltaR->getTH1();

          histoname = labelname + "_DeltaPhi";
          title = labelname + "DeltaPhi;";
          MonitorElement* DeltaPhi = iBooker.book1D(histoname.c_str(), title.c_str(), 500, -5., 5.);
          DeltaPhi->getTH1();

          //add PF histo: SJ
          histoname = labelname + "_NumeratorAvrgPFPt";
          title = labelname + "NumeratorAvrgPFPt;PF Pt[GeV/c]";
          MonitorElement* NumeratorPFPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPFPt->getTH1();

          histoname = labelname + "_NumeratorAvrgPFEta";
          title = labelname + "NumeratorAvrgPFEta;PF #eta";
          MonitorElement* NumeratorPFEta = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          NumeratorPFEta->getTH1();

          histoname = labelname + "_DenominatorAvrgPFPt";
          title = labelname + "DenominatorAvrgPFPt;PF Pt[GeV/c] ";
          MonitorElement* DenominatorPFPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPFPt->getTH1();

          histoname = labelname + "_DenominatorAvrgPFEta";
          title = labelname + "DenominatorAvrgPFEta;PF #eta";
          MonitorElement* DenominatorPFEta =
              iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          DenominatorPFEta->getTH1();

          histoname = labelname + "_PFDeltaR";
          title = labelname + "PFDeltaR;";
          MonitorElement* PFDeltaR = iBooker.book1D(histoname.c_str(), title.c_str(), 100, 0., 1.5);
          PFDeltaR->getTH1();

          histoname = labelname + "_PFDeltaPhi";
          title = labelname + "PFDeltaPhi;";
          MonitorElement* PFDeltaPhi = iBooker.book1D(histoname.c_str(), title.c_str(), 500, -5., 5.);
          PFDeltaPhi->getTH1();

          v.setEffHistos(dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy);
        }

        if (v.getObjectType() == trigger::TriggerMET || (v.getObjectType() == trigger::TriggerTET)) {
          histoname = labelname + "_NumeratorPt";
          if (v.getPath().find("HLT_PFMET") == std::string::npos)
            title = labelname + "NumeratorPt; CaloMET[GeV/c]";
          else
            title = labelname + "NumeratorPt; PFMET[GeV/c]";
          MonitorElement* NumeratorPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          NumeratorPt->getTH1();

          histoname = labelname + "_NumeratorPhi";
          title = labelname + "NumeratorPhi; #Phi";
          MonitorElement* NumeratorPhi = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          NumeratorPhi->getTH1();

          histoname = labelname + "_DenominatorPt";
          if (v.getPath().find("HLT_PFMET") == std::string::npos)
            title = labelname + "DenominatorPt; CaloMET[GeV/c]";
          else
            title = labelname + "DenominatorPt; PFMET[GeV/c]";
          MonitorElement* DenominatorPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          DenominatorPt->getTH1();

          histoname = labelname + "_DenominatorPhi";
          title = labelname + "DenominatorPhi; #Phi";
          MonitorElement* DenominatorPhi = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          DenominatorPhi->getTH1();

          v.setEffHistos(NumeratorPt,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         NumeratorPhi,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         DenominatorPt,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         DenominatorPhi,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy,
                         dummy);
        }  // Loop over MET trigger
      }
    }  //plotEff_

    if (runStandalone_) {  //runStandalone
      //--------Histos to see WHY trigger is NOT fired----------
      int Nbins_ = 10;
      int Nmin_ = 0;
      int Nmax_ = 10;
      int Ptbins_ = 1000;
      int Etabins_ = 40;
      int Phibins_ = 35;
      double PtMin_ = 0.;
      double PtMax_ = 1000.;
      double EtaMin_ = -5.;
      double EtaMax_ = 5.;
      double PhiMin_ = -3.14159;
      double PhiMax_ = 3.14159;

      std::string dirName4_ = dirname_ + "/TriggerNotFired/";
      //      iBooker.setCurrentFolder(dirName4);

      for (auto& v : hltPathsAll_) {
        MonitorElement* dummy;
        dummy = iBooker.bookFloat("dummy");

        std::string labelname("ME");
        std::string histoname(labelname + "");
        std::string title(labelname + "");
        iBooker.setCurrentFolder(dirName4_ + v.getPath());

        histoname = labelname + "_TriggerSummary";
        title = labelname + "Summary of trigger levels";
        MonitorElement* TriggerSummary = iBooker.book1D(histoname.c_str(), title.c_str(), 7, -0.5, 6.5);

        std::vector<std::string> trigger;
        trigger.emplace_back("Nevt");
        trigger.emplace_back("L1 failed");
        trigger.emplace_back("L1 & HLT failed");
        trigger.emplace_back("L1 failed but not HLT");
        trigger.emplace_back("L1 passed");
        trigger.emplace_back("L1 & HLT passed");
        trigger.emplace_back("L1 passed but not HLT");

        for (unsigned int i = 0; i < trigger.size(); i++)
          TriggerSummary->setBinLabel(i + 1, trigger[i]);

        if ((v.getTriggerType() == "SingleJet_Trigger")) {
          histoname = labelname + "_JetPt";
          title = labelname + "Leading jet pT;Pt[GeV/c]";
          MonitorElement* JetPt = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          JetPt->getTH1();

          histoname = labelname + "_JetEtaVsPt";
          title = labelname + "Leading jet #eta vs pT;#eta;Pt[GeV/c]";
          MonitorElement* JetEtaVsPt =
              iBooker.book2D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_, Ptbins_, PtMin_, PtMax_);
          JetEtaVsPt->getTH1();

          histoname = labelname + "_JetPhiVsPt";
          title = labelname + "Leading jet #Phi vs pT;#Phi;Pt[GeV/c]";
          MonitorElement* JetPhiVsPt =
              iBooker.book2D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_, Ptbins_, PtMin_, PtMax_);
          JetPhiVsPt->getTH1();

          v.setDgnsHistos(
              TriggerSummary, dummy, JetPt, JetEtaVsPt, JetPhiVsPt, dummy, dummy, dummy, dummy, dummy, dummy);
        }  // single Jet trigger

        if ((v.getTriggerType() == "DiJet_Trigger")) {
          histoname = labelname + "_JetSize";
          title = labelname + "Jet Size;multiplicity";
          MonitorElement* JetSize = iBooker.book1D(histoname.c_str(), title.c_str(), Nbins_, Nmin_, Nmax_);
          JetSize->getTH1();

          histoname = labelname + "_AvergPt";
          title = labelname + "Average Pt;Pt[GeV/c]";
          MonitorElement* Pt12 = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          Pt12->getTH1();

          histoname = labelname + "_AvergEta";
          title = labelname + "Average Eta;#eta";
          MonitorElement* Eta12 = iBooker.book1D(histoname.c_str(), title.c_str(), Etabins_, EtaMin_, EtaMax_);
          Eta12->getTH1();

          histoname = labelname + "_PhiDifference";
          title = labelname + "#Delta#Phi;#Delta#Phi";
          MonitorElement* Phi12 = iBooker.book1D(histoname.c_str(), title.c_str(), Phibins_, PhiMin_, PhiMax_);
          Phi12->getTH1();

          histoname = labelname + "_Pt3Jet";
          title = labelname + "Pt of 3rd Jet;Pt[GeV/c]";
          MonitorElement* Pt3 = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          Pt3->getTH1();

          histoname = labelname + "_Pt12VsPt3Jet";
          title = labelname + "Pt of 3rd Jet vs Average Pt of leading jets;Avergage Pt[GeV/c]; Pt of 3rd Jet [GeV/c]";
          MonitorElement* Pt12Pt3 =
              iBooker.book2D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_, Ptbins_, PtMin_, PtMax_);
          Pt12Pt3->getTH1();

          histoname = labelname + "_Pt12VsPhi12";
          title = labelname +
                  "Average Pt of leading jets vs #Delta#Phi between leading jets;Avergage Pt[GeV/c]; #Delta#Phi";
          MonitorElement* Pt12Phi12 =
              iBooker.book2D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_, Phibins_, PhiMin_, PhiMax_);
          Pt12Phi12->getTH1();

          v.setDgnsHistos(TriggerSummary, JetSize, dummy, dummy, dummy, Pt12, Eta12, Phi12, Pt3, Pt12Pt3, Pt12Phi12);
        }  // Dijet Jet trigger

        if ((v.getTriggerType() == "MET_Trigger")) {
          histoname = labelname + "_MET";
          title = labelname + "MET;Pt[GeV/c]";
          MonitorElement* MET = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          MET->getTH1();

          v.setDgnsHistos(TriggerSummary, dummy, MET, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
        }  // MET trigger

        if ((v.getTriggerType() == "TET_Trigger")) {
          histoname = labelname + "_TET";
          title = labelname + "TET;Pt[GeV/c]";
          MonitorElement* TET = iBooker.book1D(histoname.c_str(), title.c_str(), Ptbins_, PtMin_, PtMax_);
          TET->getTH1();

          v.setDgnsHistos(TriggerSummary, dummy, TET, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
        }  // TET trigger
      }
    }  //runStandalone
  }
}
//------------------------------------------------------------------------//
const std::string JetMETHLTOfflineSource::getL1ConditionModuleName(const std::string& pathname) {
  // find L1 condition for numpath with numpath objecttype
  // find PSet for L1 global seed for numpath,
  // list module labels for numpath
  std::string l1pathname = "dummy";

  std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

  for (auto& numpathmodule : numpathmodules) {
    if (hltConfig_.moduleType(numpathmodule) == "HLTLevel1GTSeed") {
      l1pathname = numpathmodule;
      break;
    }
  }
  return l1pathname;
}

//------------------------------------------------------------------------//
bool JetMETHLTOfflineSource::isBarrel(double eta) {
  bool output = false;
  if (fabs(eta) <= 1.3)
    output = true;
  return output;
}

//------------------------------------------------------------------------//
bool JetMETHLTOfflineSource::isEndCap(double eta) {
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3)
    output = true;
  return output;
}

//------------------------------------------------------------------------//
bool JetMETHLTOfflineSource::isForward(double eta) {
  bool output = false;
  if (fabs(eta) > 3.0)
    output = true;
  return output;
}

//------------------------------------------------------------------------//
bool JetMETHLTOfflineSource::validPathHLT(std::string pathname) {
  // hltConfig_ has to be defined first before calling this method
  bool output = false;
  for (unsigned int j = 0; j != hltConfig_.size(); ++j) {
    if (hltConfig_.triggerName(j) == pathname)
      output = true;
  }
  return output;
}

//------------------------------------------------------------------------//
bool JetMETHLTOfflineSource::isHLTPathAccepted(std::string pathName) {
  // triggerResults_, triggerNames_ has to be defined first before calling this method
  bool output = false;
  if (triggerResults_.isValid()) {
    unsigned index = triggerNames_.triggerIndex(pathName);
    if (index < triggerNames_.size() && triggerResults_->accept(index))
      output = true;
  }
  return output;
}

//------------------------------------------------------------------------//
// This returns the position of trigger name defined in summary histograms
double JetMETHLTOfflineSource::TriggerPosition(std::string trigName) {
  int nbins = rate_All->getTH1()->GetNbinsX();
  double binVal = -100;
  for (int ibin = 1; ibin < nbins + 1; ibin++) {
    const char* binLabel = rate_All->getTH1()->GetXaxis()->GetBinLabel(ibin);
    if (binLabel[0] == '\0')
      continue;
    //       std::string binLabel_str = string(binLabel);
    //       if(binLabel_str.compare(trigName)!=0)continue;
    if (trigName != binLabel)
      continue;

    if (trigName == binLabel) {
      binVal = rate_All->getTH1()->GetBinCenter(ibin);
      break;
    }
  }
  return binVal;
}

//------------------------------------------------------------------------//
bool JetMETHLTOfflineSource::isTriggerObjectFound(std::string objectName) {
  // processname_, triggerObj_ has to be defined before calling this method
  bool output = false;
  edm::InputTag testTag(objectName, "", processname_);
  const int index = triggerObj_->filterIndex(testTag);
  if (index >= triggerObj_->sizeFilters()) {
    edm::LogInfo("JetMETHLTOfflineSource") << "no index " << index << " of that name ";
  } else {
    const trigger::Keys& k = triggerObj_->filterKeys(index);
    if (!k.empty())
      output = true;
  }
  return output;
}
//------------------------------------------------------------------------//
