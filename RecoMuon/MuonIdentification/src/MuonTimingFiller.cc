//
// Package:    MuonTimingFiller
// Class:      MuonTimingFiller
//
/**\class MuonTimingFiller MuonTimingFiller.cc RecoMuon/MuonIdentification/src/MuonTimingFiller.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Piotr Traczyk, CERN
//         Created:  Mon Mar 16 12:27:22 CET 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"

#include "RecoMuon/MuonIdentification/interface/MuonTimingFiller.h"
#include "RecoMuon/MuonIdentification/interface/TimeMeasurementSequence.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

//
// constructors and destructor
//
MuonTimingFiller::MuonTimingFiller(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC) {
  // Load parameters for the DTTimingExtractor
  edm::ParameterSet dtTimingParameters = iConfig.getParameter<edm::ParameterSet>("DTTimingParameters");

  // Load parameters for the CSCTimingExtractor
  edm::ParameterSet cscTimingParameters = iConfig.getParameter<edm::ParameterSet>("CSCTimingParameters");

  // Fallback mechanism for old configs (there the segment matcher was built inside the timing extractors)
  edm::ParameterSet matchParameters;
  if (iConfig.existsAs<edm::ParameterSet>("MatchParameters"))
    matchParameters = iConfig.getParameter<edm::ParameterSet>("MatchParameters");
  else
    matchParameters = dtTimingParameters.getParameter<edm::ParameterSet>("MatchParameters");

  theMatcher_ = std::make_unique<MuonSegmentMatcher>(matchParameters, iC);
  theDTTimingExtractor_ = std::make_unique<DTTimingExtractor>(dtTimingParameters, theMatcher_.get(), iC);
  theCSCTimingExtractor_ = std::make_unique<CSCTimingExtractor>(cscTimingParameters, theMatcher_.get(), iC);

  errorEB_ = iConfig.getParameter<double>("ErrorEB");
  errorEE_ = iConfig.getParameter<double>("ErrorEE");
  ecalEcut_ = iConfig.getParameter<double>("EcalEnergyCut");

  useDT_ = iConfig.getParameter<bool>("UseDT");
  useCSC_ = iConfig.getParameter<bool>("UseCSC");
  useECAL_ = iConfig.getParameter<bool>("UseECAL");
}

MuonTimingFiller::~MuonTimingFiller() {}

//
// member functions
//

void MuonTimingFiller::fillTiming(const reco::Muon& muon,
                                  reco::MuonTimeExtra& dtTime,
                                  reco::MuonTimeExtra& cscTime,
                                  reco::MuonTime& rpcTime,
                                  reco::MuonTimeExtra& combinedTime,
                                  edm::Event& iEvent,
                                  const edm::EventSetup& iSetup) {
  TimeMeasurementSequence dtTmSeq, cscTmSeq;

  if (!(muon.combinedMuon().isNull())) {
    theDTTimingExtractor_->fillTiming(dtTmSeq, muon.combinedMuon(), iEvent, iSetup);
    theCSCTimingExtractor_->fillTiming(cscTmSeq, muon.combinedMuon(), iEvent, iSetup);
  } else {
    if (!(muon.standAloneMuon().isNull())) {
      theDTTimingExtractor_->fillTiming(dtTmSeq, muon.standAloneMuon(), iEvent, iSetup);
      theCSCTimingExtractor_->fillTiming(cscTmSeq, muon.standAloneMuon(), iEvent, iSetup);
    } else {
      if (muon.isTrackerMuon()) {
        std::vector<const DTRecSegment4D*> dtSegments;
        std::vector<const CSCSegment*> cscSegments;
        for (auto& chamber : muon.matches()) {
          for (auto& segment : chamber.segmentMatches) {
            // Use only the segments that passed arbitration to avoid mixing
            // segments from in-time and out-of-time muons that may bias the result
            // SegmentAndTrackArbitration
            if (segment.isMask(reco::MuonSegmentMatch::BestInStationByDR) &&
                segment.isMask(reco::MuonSegmentMatch::BelongsToTrackByDR)) {
              if (!(segment.dtSegmentRef.isNull()))
                dtSegments.push_back(segment.dtSegmentRef.get());
              if (!(segment.cscSegmentRef.isNull()))
                cscSegments.push_back(segment.cscSegmentRef.get());
            }
          }
        }
        theDTTimingExtractor_->fillTiming(dtTmSeq, dtSegments, muon.innerTrack(), iEvent, iSetup);
        theCSCTimingExtractor_->fillTiming(cscTmSeq, cscSegments, muon.innerTrack(), iEvent, iSetup);
      }
    }
  }

  // Fill DT-specific timing information block
  fillTimeFromMeasurements(dtTmSeq, dtTime);

  // Fill CSC-specific timing information block
  fillTimeFromMeasurements(cscTmSeq, cscTime);

  // Fill RPC-specific timing information block
  fillRPCTime(muon, rpcTime, iEvent);

  // Combine the TimeMeasurementSequences from DT/CSC subdetectors
  TimeMeasurementSequence combinedTmSeq;
  combineTMSequences(muon, dtTmSeq, cscTmSeq, combinedTmSeq);
  // add ECAL info
  if (useECAL_)
    addEcalTime(muon, combinedTmSeq);

  // Fill the master timing block
  fillTimeFromMeasurements(combinedTmSeq, combinedTime);

  LogTrace("MuonTime") << "Global 1/beta: " << combinedTime.inverseBeta() << " +/- " << combinedTime.inverseBetaErr()
                       << std::endl;
  LogTrace("MuonTime") << "  Free 1/beta: " << combinedTime.freeInverseBeta() << " +/- "
                       << combinedTime.freeInverseBetaErr() << std::endl;
  LogTrace("MuonTime") << "  Vertex time (in-out): " << combinedTime.timeAtIpInOut() << " +/- "
                       << combinedTime.timeAtIpInOutErr() << "  # of points: " << combinedTime.nDof() << std::endl;
  LogTrace("MuonTime") << "  Vertex time (out-in): " << combinedTime.timeAtIpOutIn() << " +/- "
                       << combinedTime.timeAtIpOutInErr() << std::endl;
  LogTrace("MuonTime") << "  direction: " << combinedTime.direction() << std::endl;
}

void MuonTimingFiller::fillTimeFromMeasurements(const TimeMeasurementSequence& tmSeq, reco::MuonTimeExtra& muTime) {
  std::vector<double> x, y;
  double invbeta(0), invbetaerr(0);
  double vertexTime(0), vertexTimeErr(0), vertexTimeR(0), vertexTimeRErr(0);
  double freeBeta(0), freeBetaErr(0), freeTime(0), freeTimeErr(0);

  if (tmSeq.dstnc.size() <= 1)
    return;

  for (unsigned int i = 0; i < tmSeq.dstnc.size(); i++) {
    invbeta +=
        (1. + tmSeq.local_t0.at(i) / tmSeq.dstnc.at(i) * 30.) * tmSeq.weightInvbeta.at(i) / tmSeq.totalWeightInvbeta;
    x.push_back(tmSeq.dstnc.at(i) / 30.);
    y.push_back(tmSeq.local_t0.at(i) + tmSeq.dstnc.at(i) / 30.);
    vertexTime += tmSeq.local_t0.at(i) * tmSeq.weightTimeVtx.at(i) / tmSeq.totalWeightTimeVtx;
    vertexTimeR +=
        (tmSeq.local_t0.at(i) + 2 * tmSeq.dstnc.at(i) / 30.) * tmSeq.weightTimeVtx.at(i) / tmSeq.totalWeightTimeVtx;
  }

  double diff;
  for (unsigned int i = 0; i < tmSeq.dstnc.size(); i++) {
    diff = (1. + tmSeq.local_t0.at(i) / tmSeq.dstnc.at(i) * 30.) - invbeta;
    invbetaerr += diff * diff * tmSeq.weightInvbeta.at(i);
    diff = tmSeq.local_t0.at(i) - vertexTime;
    vertexTimeErr += diff * diff * tmSeq.weightTimeVtx.at(i);
    diff = tmSeq.local_t0.at(i) + 2 * tmSeq.dstnc.at(i) / 30. - vertexTimeR;
    vertexTimeRErr += diff * diff * tmSeq.weightTimeVtx.at(i);
  }

  double cf = 1. / (tmSeq.dstnc.size() - 1);
  invbetaerr = sqrt(invbetaerr / tmSeq.totalWeightInvbeta * cf);
  vertexTimeErr = sqrt(vertexTimeErr / tmSeq.totalWeightTimeVtx * cf);
  vertexTimeRErr = sqrt(vertexTimeRErr / tmSeq.totalWeightTimeVtx * cf);

  muTime.setInverseBeta(invbeta);
  muTime.setInverseBetaErr(invbetaerr);
  muTime.setTimeAtIpInOut(vertexTime);
  muTime.setTimeAtIpInOutErr(vertexTimeErr);
  muTime.setTimeAtIpOutIn(vertexTimeR);
  muTime.setTimeAtIpOutInErr(vertexTimeRErr);

  rawFit(freeBeta, freeBetaErr, freeTime, freeTimeErr, x, y);

  muTime.setFreeInverseBeta(freeBeta);
  muTime.setFreeInverseBetaErr(freeBetaErr);

  muTime.setNDof(tmSeq.dstnc.size());
}

void MuonTimingFiller::fillRPCTime(const reco::Muon& muon, reco::MuonTime& rpcTime, edm::Event& iEvent) {
  double trpc = 0, trpc2 = 0;

  reco::TrackRef staTrack = muon.standAloneMuon();
  if (staTrack.isNull())
    return;

  const std::vector<const RPCRecHit*> rpcHits = theMatcher_->matchRPC(*staTrack, iEvent);
  const int nrpc = rpcHits.size();
  for (const auto& hitRPC : rpcHits) {
    const double time = hitRPC->timeError() < 0 ? hitRPC->BunchX() * 25. : hitRPC->time();
    trpc += time;
    trpc2 += time * time;
  }

  if (nrpc == 0)
    return;

  trpc2 = trpc2 / nrpc;
  trpc = trpc / nrpc;
  const double trpcerr = sqrt(std::max(0., trpc2 - trpc * trpc));

  rpcTime.timeAtIpInOut = trpc;
  rpcTime.timeAtIpInOutErr = trpcerr;
  rpcTime.nDof = nrpc;

  // currently unused
  rpcTime.timeAtIpOutIn = 0.;
  rpcTime.timeAtIpOutInErr = 0.;
}

void MuonTimingFiller::combineTMSequences(const reco::Muon& muon,
                                          const TimeMeasurementSequence& dtSeq,
                                          const TimeMeasurementSequence& cscSeq,
                                          TimeMeasurementSequence& cmbSeq) {
  if (useDT_)
    for (unsigned int i = 0; i < dtSeq.dstnc.size(); i++) {
      cmbSeq.dstnc.push_back(dtSeq.dstnc.at(i));
      cmbSeq.local_t0.push_back(dtSeq.local_t0.at(i));
      cmbSeq.weightTimeVtx.push_back(dtSeq.weightTimeVtx.at(i));
      cmbSeq.weightInvbeta.push_back(dtSeq.weightInvbeta.at(i));

      cmbSeq.totalWeightTimeVtx += dtSeq.weightTimeVtx.at(i);
      cmbSeq.totalWeightInvbeta += dtSeq.weightInvbeta.at(i);
    }

  if (useCSC_)
    for (unsigned int i = 0; i < cscSeq.dstnc.size(); i++) {
      cmbSeq.dstnc.push_back(cscSeq.dstnc.at(i));
      cmbSeq.local_t0.push_back(cscSeq.local_t0.at(i));
      cmbSeq.weightTimeVtx.push_back(cscSeq.weightTimeVtx.at(i));
      cmbSeq.weightInvbeta.push_back(cscSeq.weightInvbeta.at(i));

      cmbSeq.totalWeightTimeVtx += cscSeq.weightTimeVtx.at(i);
      cmbSeq.totalWeightInvbeta += cscSeq.weightInvbeta.at(i);
    }
}

void MuonTimingFiller::addEcalTime(const reco::Muon& muon, TimeMeasurementSequence& cmbSeq) {
  reco::MuonEnergy muonE;
  if (muon.isEnergyValid())
    muonE = muon.calEnergy();

  // Cut on the crystal energy and restrict to the ECAL barrel for now
  //  if (muonE.emMax<ecalEcut_ || fabs(muon.eta())>1.5) return;
  if (muonE.emMax < ecalEcut_)
    return;

  // A simple parametrization of the error on the ECAL time measurement
  double emErr;
  if (muonE.ecal_id.subdetId() == EcalBarrel)
    emErr = errorEB_ / muonE.emMax;
  else
    emErr = errorEE_ / muonE.emMax;
  double hitWeight = 1 / (emErr * emErr);
  double hitDist = muonE.ecal_position.r();

  cmbSeq.local_t0.push_back(muonE.ecal_time);
  cmbSeq.weightTimeVtx.push_back(hitWeight);
  cmbSeq.weightInvbeta.push_back(hitDist * hitDist * hitWeight / (30. * 30.));

  cmbSeq.dstnc.push_back(hitDist);

  cmbSeq.totalWeightTimeVtx += hitWeight;
  cmbSeq.totalWeightInvbeta += hitDist * hitDist * hitWeight / (30. * 30.);
}

void MuonTimingFiller::rawFit(double& freeBeta,
                              double& freeBetaErr,
                              double& freeTime,
                              double& freeTimeErr,
                              const std::vector<double>& hitsx,
                              const std::vector<double>& hitsy) {
  double s = 0, sx = 0, sy = 0, x, y;
  double sxx = 0, sxy = 0;

  freeBeta = 0;
  freeBetaErr = 0;
  freeTime = 0;
  freeTimeErr = 0;

  if (hitsx.empty())
    return;
  if (hitsx.size() == 1) {
    freeTime = hitsy[0];
  } else {
    for (unsigned int i = 0; i != hitsx.size(); i++) {
      x = hitsx[i];
      y = hitsy[i];
      sy += y;
      sxy += x * y;
      s += 1.;
      sx += x;
      sxx += x * x;
    }

    double d = s * sxx - sx * sx;
    freeTime = (sxx * sy - sx * sxy) / d;
    freeBeta = (s * sxy - sx * sy) / d;
    freeBetaErr = sqrt(sxx / d);
    freeTimeErr = sqrt(s / d);
  }
}
