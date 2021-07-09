#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimTracker/TrackTriggerAssociation/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <TProfile.h>
#include <TH1F.h>

using namespace std;
using namespace edm;
using namespace tt;

namespace trklet {

  /*! \class  trklet::AnalyzerTT
   *  \brief  Class to analyze TTTracks
   *  \author Thomas Schuh
   *  \date   2020, Oct
   */
  class AnalyzerTT : public one::EDAnalyzer<one::WatchRuns, one::SharedResources> {
  public:
    AnalyzerTT(const ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const Run& iEvent, const EventSetup& iSetup) override;
    void analyze(const Event& iEvent, const EventSetup& iSetup) override;
    void endRun(const Run& iEvent, const EventSetup& iSetup) override {}
    void endJob() override {}

  private:
    // ED input token of TTTrackRefMap
    EDGetTokenT<TTTrackRefMap> edGetTokenTTTrackMap_;
    // ED input token of TTStubRef to TPPtr association for tracking efficiency
    EDGetTokenT<StubAssociation> edGetTokenStubAssociation_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // stores, calculates and provides run-time constants
    const Setup* setup_;

    // histos

    TH1F* hisQoverPt_;
    TH1F* hisPhi0_;
    TH1F* hisEta_;
    TH1F* hisZ0_;
    TProfile* profResQoverPtOverEta_;
    TProfile* profResPhi0OverEta_;
    TProfile* profResEtaOverEta_;
    TProfile* profResZ0OverEta_;
  };

  AnalyzerTT::AnalyzerTT(const ParameterSet& iConfig) {
    usesResource("TFileService");
    // book in- and output ED products
    const auto& label = iConfig.getParameter<string>("LabelAS");
    const auto& branch = iConfig.getParameter<string>("BranchAcceptedTracks");
    const auto& inputTag = iConfig.getParameter<InputTag>("InputTagSelection");
    edGetTokenTTTrackMap_ = consumes<TTTrackRefMap>(InputTag(label, branch));
    edGetTokenStubAssociation_ = consumes<StubAssociation>(inputTag);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
    // initial ES products
    setup_ = nullptr;
  }

  void AnalyzerTT::beginRun(const Run& iEvent, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
    // book histograms
    Service<TFileService> fs;
    TFileDirectory dir;
    dir = fs->mkdir("TT");
    hisQoverPt_ = dir.make<TH1F>("His qOverPt", ";", 100, -1., 1.);
    hisPhi0_ = dir.make<TH1F>("His phi0", ";", 100, -M_PI, M_PI);
    hisEta_ = dir.make<TH1F>("His eta", ";", 100, -2.5, 2.5);
    hisZ0_ = dir.make<TH1F>("His z0", ";", 100, -20., 20.);
    profResQoverPtOverEta_ = dir.make<TProfile>("Prof Res qOverPt over |eta|", ";", 100, 0., 2.5);
    profResPhi0OverEta_ = dir.make<TProfile>("Prof Res phi0 over |eta|", ";", 100, 0., 2.5);
    profResEtaOverEta_ = dir.make<TProfile>("Prof Res eta over |eta|", ";", 100, 0., 2.5);
    profResZ0OverEta_ = dir.make<TProfile>("Prof Res z0 over |eta|", ";", 100, 0., 2.5);
  }

  void AnalyzerTT::analyze(const Event& iEvent, const EventSetup& iSetup) {
    Handle<TTTrackRefMap> handleTTTrackMap;
    iEvent.getByToken<TTTrackRefMap>(edGetTokenTTTrackMap_, handleTTTrackMap);
    Handle<StubAssociation> handleStubAssociation;
    iEvent.getByToken<StubAssociation>(edGetTokenStubAssociation_, handleStubAssociation);
    for (const pair<TTTrackRef, TTTrackRef>& p : *handleTTTrackMap) {
      const TTTrackRef& ttTrackRef = p.first;
      //const TTTrackRef& ttTrackRef = p.second;
      const vector<TTStubRef>& ttStubRefs = ttTrackRef->getStubRefs();
      const vector<TPPtr>& tpPtrs = handleStubAssociation->associate(ttStubRefs);
      if (tpPtrs.empty())
        continue;
      const TPPtr& tpPtr = tpPtrs.front();
      const math::XYZPointD& v = tpPtr->vertex();
      const double qOverPtTT = ttTrackRef->rInv() / setup_->invPtToDphi() / 2.0;
      const double qOverPtTP = tpPtr->charge() / tpPtr->pt();
      const double qOverPtDiff = qOverPtTP - qOverPtTT;
      const double phi0TT = deltaPhi(ttTrackRef->phi() + ttTrackRef->phiSector() * setup_->baseRegion());
      //const double phi0TT = ttTrackRef->phi();
      const double phi0TP = tpPtr->phi();
      const double phi0Diff = phi0TP - phi0TT;
      const double etaTT = asinh(ttTrackRef->tanL());
      const double etaTP = tpPtr->eta();
      const double etaDiff = etaTP - etaTT;
      const double z0TT = ttTrackRef->z0();
      const double z0TP = v.z() - sinh(tpPtr->eta()) * (v.x() * cos(tpPtr->phi()) + v.y() * sin(tpPtr->phi()));
      const double z0Diff = z0TP - z0TT;
      hisQoverPt_->Fill(qOverPtTT);
      hisPhi0_->Fill(phi0TT);
      hisEta_->Fill(etaTT);
      hisZ0_->Fill(z0TT);
      profResQoverPtOverEta_->Fill(abs(etaTP), abs(qOverPtDiff));
      profResPhi0OverEta_->Fill(abs(etaTP), abs(phi0Diff));
      profResEtaOverEta_->Fill(abs(etaTP), abs(etaDiff));
      profResZ0OverEta_->Fill(abs(etaTP), abs(z0Diff));
      /*cout << etaTT << " " << etaTP << " " << etaDiff << " " << ttTrackRef->etaSector() << endl;
      cout << "m0TT = " << -.5 * ttTrackRef->rInv() << endl;
      cout << "m0TP = " << -qOverPtTP * setup_->invPtToDphi() << endl;
      cout << "c0TT = " << phi0TT << endl;
      cout << "c0TP = " << phi0TP << endl;
      cout << "m1TT = " << ttTrackRef->tanL() << endl;
      cout << "m1TP = " << sinh(tpPtr->eta()) << endl;
      cout << "c1TT = " << z0TT << endl;
      cout << "c1TP = " << z0TP << endl;
      const vector<TTStubRef>& ttStubRefsTP = handleStubAssociation->findTTStubRefs(tpPtr);
      for (const TTStubRef stub : ttStubRefs) {
        const GlobalPoint& gp = setup_->stubPos(stub);
        cout << gp.perp() << " " << gp.phi() << " " << gp.z() << endl;
      }
      cout << endl;
      for (const TTStubRef stub : ttStubRefsTP) {
        const GlobalPoint& gp = setup_->stubPos(stub);
        cout << gp.perp() << " " << gp.phi() << " " << gp.z() << " " << setup_->dPhi(stub, 0.5*ttTrackRef->rInv()) << " " << setup_->dZ(stub, ttTrackRef->tanL()) << endl;
      }*/
    }
  }

}  // namespace trklet

DEFINE_FWK_MODULE(trklet::AnalyzerTT);