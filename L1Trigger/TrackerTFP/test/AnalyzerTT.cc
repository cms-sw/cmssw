#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimTracker/TrackTriggerAssociation/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::AnalyzerTT
   *  \brief  Class to analyze TTTracks
   *  \author Thomas Schuh
   *  \date   2020, Oct
   */
  class AnalyzerTT : public one::EDAnalyzer<one::WatchRuns> {
  public:
    AnalyzerTT(const ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const Run& iEvent, const EventSetup& iSetup) override;
    void analyze(const Event& iEvent, const EventSetup& iSetup) override;
    void endRun(const Run& iEvent, const EventSetup& iSetup) override {}
    void endJob() override {}

  private:
    // ED input token of tt::TTTrackRefMap
    EDGetTokenT<tt::TTTrackRefMap> edGetTokenTTTrackMap_;
    // ED input token of TTStubRef to TPPtr association for tracking efficiency
    EDGetTokenT<StubAssociation> edGetTokenStubAssociation_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // stores, calculates and provides run-time constants
    const Setup* setup_ = nullptr;
  };

  AnalyzerTT::AnalyzerTT(const ParameterSet& iConfig) {
    // book in- and output ED products
    const auto& label = iConfig.getParameter<string>("LabelAS");
    const auto& branch = iConfig.getParameter<string>("BranchAcceptedTracks");
    const auto& inputTag = iConfig.getParameter<InputTag>("InputTagSelection");
    edGetTokenTTTrackMap_ = consumes<tt::TTTrackRefMap>(InputTag(label, branch));
    edGetTokenStubAssociation_ = consumes<StubAssociation>(inputTag);
    // book ES products
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, Transition::BeginRun>();
  }

  void AnalyzerTT::beginRun(const Run& iEvent, const EventSetup& iSetup) {
    // helper class to store configurations
    setup_ = &iSetup.getData(esGetTokenSetup_);
  }

  void AnalyzerTT::analyze(const Event& iEvent, const EventSetup& iSetup) {
    Handle<tt::TTTrackRefMap> handleTTTrackMap;
    iEvent.getByToken<tt::TTTrackRefMap>(edGetTokenTTTrackMap_, handleTTTrackMap);
    Handle<StubAssociation> handleStubAssociation;
    iEvent.getByToken<StubAssociation>(edGetTokenStubAssociation_, handleStubAssociation);
    if (false)
      return;
    for (const auto& p : *handleTTTrackMap) {
      const TTTrackRef& found = p.second;
      const TTTrackRef& fitted = p.first;
      const vector<TTStubRef>& ttStubRefsFound = found->getStubRefs();
      const vector<TPPtr>& tpPtrsFound = handleStubAssociation->associate(ttStubRefsFound);
      if (tpPtrsFound.empty())
        continue;
      const vector<TTStubRef>& ttStubRefsFitted = fitted->getStubRefs();
      const vector<TPPtr>& tpPtrsFitted = handleStubAssociation->associate(ttStubRefsFitted);
      if (!tpPtrsFitted.empty())
        continue;
      const TPPtr& tpPtr = tpPtrsFound.front();
      const double off = (found->phiSector() - .5) * 2. * M_PI / setup_->numRegions() / setup_->numSectorsPhi();
      cout << setprecision(8);
      cout << found->phiSector() << " " << found->etaSector() << " " << endl;
      cout << "Found" << endl;
      for (const TTStubRef& ttStubRef : ttStubRefsFound) {
        const GlobalPoint& gp = setup_->stubPos(ttStubRef);
        cout << gp.perp() << " " << gp.phi() << " " << gp.z() << " " << setup_->layerId(ttStubRef) << endl;
      }
      cout << "Fitted" << endl;
      for (const TTStubRef& ttStubRef : ttStubRefsFitted) {
        const GlobalPoint& gp = setup_->stubPos(ttStubRef);
        cout << gp.perp() << " " << gp.phi() << " " << gp.z() << " " << setup_->layerId(ttStubRef) << endl;
      }
      cout << "TP" << endl;
      for (const TTStubRef& ttStubRef : handleStubAssociation->findTTStubRefs(tpPtr)) {
        const GlobalPoint& gp = setup_->stubPos(ttStubRef);
        cout << gp.perp() << " " << gp.phi() << " " << gp.z() << " " << setup_->layerId(ttStubRef) << endl;
      }
      cout << found->hitPattern() << " " << found->trackSeedType() << endl;
      cout << "m0SF = "
           << " " << -found->rInv() << endl;
      cout << "c0SF = "
           << " " << deltaPhi(found->phi() + found->rInv() * setup_->chosenRofPhi() + off) << endl;
      cout << "m1SF = "
           << " " << found->tanL() + setup_->sectorCot(found->etaSector()) << endl;
      cout << "c1SF = "
           << " " << found->z0() - found->tanL() * setup_->chosenRofZ() << endl;
      cout << "m0KF = "
           << " " << -fitted->rInv() * setup_->invPtToDphi() << endl;
      cout << "c0KF = "
           << " " << fitted->phi() << endl;
      cout << "m1KF = "
           << " " << fitted->tanL() << endl;
      cout << "c1KF = "
           << " " << fitted->z0() << endl;
      cout << "m0TP = "
           << " " << -tpPtr->charge() / tpPtr->pt() * setup_->invPtToDphi() << endl;
      cout << "c0TP = "
           << " " << tpPtr->phi() << endl;
      cout << "m1TP = "
           << " " << sinh(tpPtr->eta()) << endl;
      const math::XYZPointD& v = tpPtr->vertex();
      cout << "c1TP = "
           << " " << v.z() - sinh(tpPtr->eta()) * (v.x() * cos(tpPtr->phi()) + v.y() * sin(tpPtr->phi())) << endl;
      throw cms::Exception("...");
    }
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::AnalyzerTT);