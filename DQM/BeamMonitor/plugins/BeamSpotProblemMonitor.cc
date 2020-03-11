/*
 * \file BeamSpotProblemMonitor.cc
 * \author Sushil S. Chauhan/UC Davis
 *
 */

#include "DQM/BeamMonitor/plugins/BeamSpotProblemMonitor.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoVertex/BeamSpotProducer/interface/BeamSpotOnlineProducer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <numeric>
#include <cmath>

using namespace std;
using namespace edm;

//
// constructors and destructor
//
BeamSpotProblemMonitor::BeamSpotProblemMonitor(const ParameterSet& ps)
    : dcsStatus_{consumes<DcsStatusCollection>(ps.getUntrackedParameter<InputTag>("DCSStatus"))},
      scalertag_{consumes<BeamSpotOnlineCollection>(ps.getUntrackedParameter<InputTag>("scalarBSCollection"))},
      trkSrc_{consumes<reco::TrackCollection>(ps.getUntrackedParameter<InputTag>("pixelTracks"))},
      nTracks_{0},
      nCosmicTrk_{ps.getUntrackedParameter<int>("nCosmicTrk")},
      fitNLumi_{1},
      debug_{ps.getUntrackedParameter<bool>("Debug")},
      onlineMode_{ps.getUntrackedParameter<bool>("OnlineMode")},
      doTest_{ps.getUntrackedParameter<bool>("doTest")},
      alarmONThreshold_{ps.getUntrackedParameter<int>("AlarmONThreshold")},
      alarmOFFThreshold_{ps.getUntrackedParameter<int>("AlarmOFFThreshold")},
      lastlumi_{0},
      nextlumi_{0},
      processed_{false},
      alarmOn_{false},
      beamSpotStatus_{0},
      beamSpotFromDB_{0} {
  monitorName_ = ps.getUntrackedParameter<string>("monitorName");

  if (not monitorName_.empty())
    monitorName_ += "/";
}

void BeamSpotProblemMonitor::fillDescriptions(ConfigurationDescriptions& oDesc) {
  ParameterSetDescription desc;
  desc.addUntracked<string>("monitorName", "BeamSpotProblemMonitor");
  desc.addUntracked<InputTag>("DCSStatus", edm::InputTag("scalersRawToDigi"));
  desc.addUntracked<InputTag>("scalarBSCollection", edm::InputTag("scalersRawToDigi"));
  desc.addUntracked<InputTag>("pixelTracks", edm::InputTag("pixelTracks"));
  desc.addUntracked<int>("nCosmicTrk", 10);
  desc.addUntracked<bool>("Debug", false);
  desc.addUntracked<bool>("OnlineMode", true);
  desc.addUntracked<bool>("doTest", false);
  desc.addUntracked<int>("AlarmONThreshold", 10);
  desc.addUntracked<int>("AlarmOFFThreshold", 40);

  oDesc.add("dqmBeamSpotProblemMonitor", desc);
}

//--------------------------------------------------------
void BeamSpotProblemMonitor::bookHistograms(DQMStore::IBooker& iB, const edm::Run&, const edm::EventSetup&) {
  // create and cd into new folder
  iB.setCurrentFolder(monitorName_ + "FitFromScalars");

  const string coord{"BeamSpotStatus"};

  string histName(coord + "_lumi");
  string histTitle(coord);
  const string ytitle("Problem (-1)  /  OK (1)");
  const string xtitle("Lumisection");

  beamSpotStatusLumi_ = iB.book1D(histName, histTitle, 40, 0.5, 40.5);
  beamSpotStatusLumi_->setAxisTitle(xtitle, 1);
  beamSpotStatusLumi_->setAxisTitle(ytitle, 2);

  histName += "_all";
  histTitle += " all";
  beamSpotStatusLumiAll_ = iB.book1D(histName, histTitle, 40, 0.5, 40.5);
  beamSpotStatusLumiAll_->getTH1()->SetCanExtend(TH1::kAllAxes);
  beamSpotStatusLumiAll_->setAxisTitle(xtitle, 1);
  beamSpotStatusLumiAll_->setAxisTitle(ytitle, 2);

  //NOTE: This in principal should be a Lumi only histogram since it gets reset at every
  // dqmBeginLuminosityBlock call. However, it is also filled at that time and the DQMStore
  // clears all lumi histograms at postGlobalBeginLuminosityBlock!
  beamSpotError_ = iB.book1D("BeamSpotError", "ERROR: Beamspot missing from scalars", 20, 0.5, 20.5);
  beamSpotError_->setAxisTitle("# of consecutive LSs with problem", 1);
  beamSpotError_->setAxisTitle("Problem with scalar BeamSpot", 2);
}

//--------------------------------------------------------
void BeamSpotProblemMonitor::dqmBeginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  const int nthlumi = lumiSeg.luminosityBlock();

  if (onlineMode_) {
    if (nthlumi > nextlumi_) {
      fillPlots(lastlumi_, nextlumi_, nthlumi);
      nextlumi_ = nthlumi;
      edm::LogInfo("BeamSpotProblemMonitor") << "dqmBeginLuminosityBlock:: Next Lumi to Fit: " << nextlumi_ << endl;
    }
  } else {
    if (processed_)
      fillPlots(lastlumi_, nextlumi_, nthlumi);
    nextlumi_ = nthlumi;
    edm::LogInfo("BeamSpotProblemMonitor") << " dqmBeginLuminosityBlock:: Next Lumi to Fit: " << nextlumi_ << endl;
  }

  if (processed_)
    processed_ = false;
  edm::LogInfo("BeamSpotProblemMonitor") << " dqmBeginLuminosityBlock::  Begin of Lumi: " << nthlumi << endl;
}

// ----------------------------------------------------------
void BeamSpotProblemMonitor::analyze(const Event& iEvent, const EventSetup& iSetup) {
  const int nthlumi = iEvent.luminosityBlock();

  if (onlineMode_ && (nthlumi < nextlumi_)) {
    edm::LogInfo("BeamSpotProblemMonitor") << "analyze::  Spilt event from previous lumi section!" << std::endl;
    return;
  }
  if (onlineMode_ && (nthlumi > nextlumi_)) {
    edm::LogInfo("BeamSpotProblemMonitor") << "analyze::  Spilt event from next lumi section!!!" << std::endl;
    return;
  }

  beamSpotStatus_ = 0.;

  // Checking TK status
  Handle<DcsStatusCollection> dcsStatus;
  iEvent.getByToken(dcsStatus_, dcsStatus);
  std::array<bool, 6> dcsTk;
  for (auto& e : dcsTk) {
    e = true;
  }
  for (auto const& status : *dcsStatus) {
    if (!status.ready(DcsStatus::BPIX))
      dcsTk[0] = false;
    if (!status.ready(DcsStatus::FPIX))
      dcsTk[1] = false;
    if (!status.ready(DcsStatus::TIBTID))
      dcsTk[2] = false;
    if (!status.ready(DcsStatus::TOB))
      dcsTk[3] = false;
    if (!status.ready(DcsStatus::TECp))
      dcsTk[4] = false;
    if (!status.ready(DcsStatus::TECm))
      dcsTk[5] = false;
  }

  bool allTkOn = true;
  for (auto status : dcsTk) {
    if (!status) {
      allTkOn = false;
      break;
    }
  }

  //If tracker is ON and collision is going on then must be few track ther
  edm::Handle<reco::TrackCollection> trackCollection;
  iEvent.getByToken(trkSrc_, trackCollection);
  for (auto const& track : *trackCollection) {
    if (track.pt() > 1.0)
      nTracks_++;
    if (nTracks_ > 200)
      break;
  }

  // get scalar collection and BeamSpot
  Handle<BeamSpotOnlineCollection> handleScaler;
  iEvent.getByToken(scalertag_, handleScaler);

  // beam spot scalar object
  BeamSpotOnline spotOnline;

  bool fallBackToDB = false;
  alarmOn_ = false;

  if (!handleScaler->empty()) {
    spotOnline = *(handleScaler->begin());

    // check if we have a valid beam spot fit result from online DQM thrugh scalars
    if (spotOnline.x() == 0. && spotOnline.y() == 0. && spotOnline.z() == 0. && spotOnline.width_x() == 0. &&
        spotOnline.width_y() == 0.) {
      fallBackToDB = true;
    }
  }

  //For testing set it false for every LSs
  if (doTest_)
    fallBackToDB = true;

  //based on last event of this lumi only as it overwrite it
  if (allTkOn && fallBackToDB) {
    beamSpotStatus_ = -1.;
  }  //i.e,from DB
  if (allTkOn && (!fallBackToDB)) {
    beamSpotStatus_ = 1.;
  }  //i.e,from online DQM

  //when collision at least few tracks should be there otherwise it give false ALARM
  if (allTkOn && nTracks_ < nCosmicTrk_)
    beamSpotStatus_ = 0.;

  processed_ = true;
}

//--------------------------------------------------------
void BeamSpotProblemMonitor::fillPlots(int& lastlumi, int& nextlumi, int nthlumi) {
  if (onlineMode_ && (nthlumi <= nextlumi))
    return;

  int currentlumi = nextlumi;
  lastlumi = currentlumi;

  //Chcek status and if lumi are in succession when fall to DB
  if (beamSpotStatus_ == -1. && (lastlumi + 1) == nthlumi) {
    beamSpotFromDB_++;
  } else {
    beamSpotFromDB_ = 0;  //if not in succesion or status is ok then set zero
  }

  if (beamSpotFromDB_ >= alarmONThreshold_) {
    alarmOn_ = true;  //set the audio alarm true after N successive LSs
  }

  if (beamSpotFromDB_ > alarmOFFThreshold_) {
    alarmOn_ = false;     //set the audio alarm true after 10 successive LSs
    beamSpotFromDB_ = 0;  //reset it for new incident
  }

  if (onlineMode_) {  // filling LS gap For status plot

    const int countLS_bs = beamSpotStatusLumi_->getTH1()->GetEntries();
    int LSgap_bs = currentlumi / fitNLumi_ - countLS_bs;
    if (currentlumi % fitNLumi_ == 0)
      LSgap_bs--;

    // filling previous fits if LS gap ever exists
    for (int ig = 0; ig < LSgap_bs; ig++) {
      beamSpotStatusLumi_->ShiftFillLast(0., 0., fitNLumi_);  //x0 , x0err, fitNLumi_;  see DQMCore....
    }

    beamSpotStatusLumi_->ShiftFillLast(
        beamSpotStatus_,
        0.,
        fitNLumi_);  //beamSpotStatus_ =>0. (no collision, no tracks); =>1 (OK from scaler), =>-1 (No scalar results)
    beamSpotStatusLumiAll_->setBinContent(currentlumi, beamSpotStatus_);

  } else {
    beamSpotStatusLumi_->ShiftFillLast(0., 0., fitNLumi_);
  }  //onlineMode_

  //Reset it here for next lumi
  beamSpotError_->Reset();
  if (alarmOn_)
    beamSpotError_->Fill(beamSpotFromDB_);

  //Get quality report
  const QReport* beamSpotQReport = beamSpotError_->getQReport("BeamSpotOnlineTest");

  if (beamSpotQReport) {
    /* S.Dutta : Commenting out these variables are not used and giving error with "-Werror=unused-variable" option
    float qtresult = BeamSpotQReport->getQTresult();
    int qtstatus   = BeamSpotQReport->getStatus() ; // get QT status value (see table below) */
  }

  nTracks_ = 0;
}

//--------------------------------------------------------
void BeamSpotProblemMonitor::dqmEndLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& iSetup) {
  const int nthlumi = lumiSeg.id().luminosityBlock();
  edm::LogInfo("BeamSpotProblemMonitor")
      << "dqmEndLuminosityBlock:: Lumi of the last event before dqmEndLuminosityBlock: " << nthlumi << endl;
}
//-------------------------------------------------------

void BeamSpotProblemMonitor::dqmEndRun(const Run& r, const EventSetup& context) {
  if (debug_)
    edm::LogInfo("BeamSpotProblemMonitor") << "dqmEndRun:: Clearing all the Maps " << endl;
  //Reset it end of job
  beamSpotError_->Reset();
}

DEFINE_FWK_MODULE(BeamSpotProblemMonitor);

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
