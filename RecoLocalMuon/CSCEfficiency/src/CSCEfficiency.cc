/*
 *  Routine to calculate CSC efficiencies 
 *  Comments about the program logic are denoted by //----
 * 
 *  Stoyan Stoynev, Northwestern University.
 */

#include "RecoLocalMuon/CSCEfficiency/src/CSCEfficiency.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

using namespace std;

bool CSCEfficiency::filter(edm::Event &event, const edm::EventSetup &eventSetup) {
  passTheEvent = false;
  DataFlow->Fill(0.);
  MuonPatternRecoDumper debug;

  //---- increment counter
  nEventsAnalyzed++;
  // printalot debug output
  printalot = (nEventsAnalyzed < int(printout_NEvents));  //
  edm::RunNumber_t const iRun = event.id().run();
  edm::EventNumber_t const iEvent = event.id().event();
  if (0 == fmod(double(nEventsAnalyzed), double(1000))) {
    if (printalot) {
      printf("\n==enter==CSCEfficiency===== run %u\tevent %llu\tn Analyzed %i\n", iRun, iEvent, nEventsAnalyzed);
    }
  }
  theService->update(eventSetup);
  //---- These declarations create handles to the types of records that you want
  //---- to retrieve from event "e".
  if (printalot)
    printf("\tget handles for digi collections\n");

  //---- Pass the handle to the method "getByType", which is used to retrieve
  //---- one and only one instance of the type in question out of event "e". If
  //---- zero or more than one instance exists in the event an exception is thrown.
  if (printalot)
    printf("\tpass handles\n");
  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedlcts;
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  edm::Handle<CSCRecHit2DCollection> rechits;
  edm::Handle<CSCSegmentCollection> segments;
  edm::Handle<edm::View<reco::Track> > trackCollectionH;
  edm::Handle<edm::PSimHitContainer> simhits;

  if (useDigis) {
    event.getByToken(wd_token, wires);
    event.getByToken(sd_token, strips);
    event.getByToken(al_token, alcts);
    event.getByToken(cl_token, clcts);
    event.getByToken(co_token, correlatedlcts);
  }
  if (!isData) {
    event.getByToken(sh_token, simhits);
  }
  event.getByToken(rh_token, rechits);
  event.getByToken(se_token, segments);
  event.getByToken(tk_token, trackCollectionH);
  const edm::View<reco::Track> trackCollection = *(trackCollectionH.product());

  //---- Get the CSC Geometry :
  if (printalot)
    printf("\tget the CSC geometry.\n");
  edm::ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);

  // use theTrackingGeometry instead of cscGeom?
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  bool triggerPassed = true;
  if (useTrigger) {
    // access the trigger information
    // trigger names can be find in HLTrigger/Configuration/python/HLT_2E30_cff.py (or?)
    // get hold of TriggerResults
    edm::Handle<edm::TriggerResults> hltR;
    event.getByToken(ht_token, hltR);
    const edm::TriggerNames &triggerNames = event.triggerNames(*hltR);
    triggerPassed = applyTrigger(hltR, triggerNames);
  }
  if (!triggerPassed) {
    return triggerPassed;
  }
  DataFlow->Fill(1.);
  GlobalPoint gpZero(0., 0., 0.);
  if (theService->magneticField()->inTesla(gpZero).mag2() < 0.1) {
    magField = false;
  } else {
    magField = true;
  }

  //---- store info from digis
  fillDigiInfo(alcts, clcts, correlatedlcts, wires, strips, simhits, rechits, segments, cscGeom);
  //
  edm::Handle<reco::MuonCollection> muons;
  edm::InputTag muonTag_("muons");
  event.getByLabel(muonTag_, muons);

  edm::Handle<reco::BeamSpot> beamSpotHandle;
  event.getByLabel("offlineBeamSpot", beamSpotHandle);
  reco::BeamSpot vertexBeamSpot = *beamSpotHandle;
  //
  std::vector<reco::MuonCollection::const_iterator> goodMuons_it;
  unsigned int nPositiveZ = 0;
  unsigned int nNegativeZ = 0;
  float muonOuterZPosition = -99999.;
  if (isIPdata) {
    if (printalot)
      std::cout << " muons.size() = " << muons->size() << std::endl;
    for (reco::MuonCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++muon) {
      DataFlow->Fill(31.);
      if (printalot) {
        std::cout << "  iMuon = " << muon - muons->begin() << " charge = " << muon->charge() << " p = " << muon->p()
                  << " pt = " << muon->pt() << " eta = " << muon->eta() << " phi = " << muon->phi()
                  << " matches = " << muon->matches().size()
                  << " matched Seg = " << muon->numberOfMatches(reco::Muon::SegmentAndTrackArbitration)
                  << " GLB/TR/STA = " << muon->isGlobalMuon() << "/" << muon->isTrackerMuon() << "/"
                  << muon->isStandAloneMuon() << std::endl;
      }
      if (!(muon->isTrackerMuon() && muon->isGlobalMuon())) {
        continue;
      }
      DataFlow->Fill(32.);
      double relISO =
          (muon->isolationR03().sumPt + muon->isolationR03().emEt + muon->isolationR03().hadEt) / muon->track()->pt();
      if (printalot) {
        std::cout << " relISO = " << relISO << " emVetoEt = " << muon->isolationR03().emVetoEt
                  << " caloComp = " << muon::caloCompatibility(*(muon))
                  << " dxy = " << fabs(muon->track()->dxy(vertexBeamSpot.position())) << std::endl;
      }
      if (
          //relISO>0.1 || muon::caloCompatibility(*(muon))<.90 ||
          fabs(muon->track()->dxy(vertexBeamSpot.position())) > 0.2 || muon->pt() < 6.) {
        continue;
      }
      DataFlow->Fill(33.);
      if (muon->track()->hitPattern().numberOfValidPixelHits() < 1 ||
          muon->track()->hitPattern().numberOfValidTrackerHits() < 11 ||
          muon->combinedMuon()->hitPattern().numberOfValidMuonHits() < 1 ||
          muon->combinedMuon()->normalizedChi2() > 10. || muon->numberOfMatches() < 2) {
        continue;
      }
      DataFlow->Fill(34.);
      float zOuter = muon->combinedMuon()->outerPosition().z();
      float rhoOuter = muon->combinedMuon()->outerPosition().rho();
      bool passDepth = true;
      // barrel region
      //if ( fabs(zOuter) < 660. && rhoOuter > 400. && rhoOuter < 480.){
      if (fabs(zOuter) < 660. && rhoOuter > 400. && rhoOuter < 540.) {
        passDepth = false;
      }
      // endcap region
      //else if( fabs(zOuter) > 550. && fabs(zOuter) < 650. && rhoOuter < 300.){
      else if (fabs(zOuter) > 550. && fabs(zOuter) < 650. && rhoOuter < 300.) {
        passDepth = false;
      }
      // overlap region
      //else if ( fabs(zOuter) > 680. && fabs(zOuter) < 730. && rhoOuter < 480.){
      else if (fabs(zOuter) > 680. && fabs(zOuter) < 880. && rhoOuter < 540.) {
        passDepth = false;
      }
      if (!passDepth) {
        continue;
      }
      DataFlow->Fill(35.);
      goodMuons_it.push_back(muon);
      if (muon->track()->momentum().z() > 0.) {
        ++nPositiveZ;
      }
      if (muon->track()->momentum().z() < 0.) {
        ++nNegativeZ;
      }
    }
  }

  //

  if (printalot)
    std::cout << "Start track loop over " << trackCollection.size() << " tracks" << std::endl;
  for (edm::View<reco::Track>::size_type i = 0; i < trackCollection.size(); ++i) {
    DataFlow->Fill(2.);
    edm::RefToBase<reco::Track> track(trackCollectionH, i);
    //std::cout<<" iTR = "<<i<<" eta = "<<track->eta()<<" phi = "<<track->phi()<<std::cout<<" pt = "<<track->pt()<<std::endl;
    if (isIPdata) {
      if (printalot) {
        std::cout << " nNegativeZ = " << nNegativeZ << " nPositiveZ = " << nPositiveZ << std::endl;
      }
      if (nNegativeZ > 1 || nPositiveZ > 1) {
        break;
      }
      bool trackOK = false;
      if (printalot) {
        std::cout << " goodMuons_it.size() = " << goodMuons_it.size() << std::endl;
      }
      for (size_t iM = 0; iM < goodMuons_it.size(); ++iM) {
        //std::cout<<" iM = "<<iM<<" eta = "<<goodMuons_it[iM]->track()->eta()<<
        //" phi = "<<goodMuons_it[iM]->track()->phi()<<
        //" pt = "<<goodMuons_it[iM]->track()->pt()<<std::endl;
        float deltaR = pow(track->phi() - goodMuons_it[iM]->track()->phi(), 2) +
                       pow(track->eta() - goodMuons_it[iM]->track()->eta(), 2);
        deltaR = sqrt(deltaR);
        if (printalot) {
          std::cout << " TR mu match to a tr: deltaR = " << deltaR
                    << " dPt = " << track->pt() - goodMuons_it[iM]->track()->pt() << std::endl;
        }
        if (deltaR > 0.01 || fabs(track->pt() - goodMuons_it[iM]->track()->pt()) > 0.1) {
          continue;
        } else {
          trackOK = true;
          if (printalot) {
            std::cout << " trackOK " << std::endl;
          }
          muonOuterZPosition = goodMuons_it[iM]->combinedMuon()->outerPosition().z();
          break;
          //++nChosenTracks;
        }
      }
      if (!trackOK) {
        if (printalot) {
          std::cout << " failed: trackOK " << std::endl;
        }
        continue;
      }
    } else {
      //---- Do we need a better "clean track" definition?
      if (trackCollection.size() > 2) {
        break;
      }
      DataFlow->Fill(3.);
      if (!i && 2 == trackCollection.size()) {
        edm::View<reco::Track>::size_type tType = 1;
        edm::RefToBase<reco::Track> trackTwo(trackCollectionH, tType);
        if (track->outerPosition().z() * trackTwo->outerPosition().z() > 0) {  // in one and the same "endcap"
          break;
        }
      }
    }
    DataFlow->Fill(4.);
    if (printalot) {
      std::cout << "i track = " << i << " P = " << track->p() << " chi2/ndf = " << track->normalizedChi2()
                << " nSeg = " << segments->size() << std::endl;
      std::cout << "quality undef/loose/tight/high/confirmed/goodIt/size " << track->quality(reco::Track::undefQuality)
                << "/" << track->quality(reco::Track::loose) << "/" << track->quality(reco::Track::tight) << "/"
                << track->quality(reco::Track::highPurity) << "/" << track->quality(reco::Track::confirmed) << "/"
                << track->quality(reco::Track::goodIterative) << "/" << track->quality(reco::Track::qualitySize)
                << std::endl;
      std::cout << " pt = " << track->pt() << " +-" << track->ptError() << " q/pt = " << track->qoverp() << " +- "
                << track->qoverpError() << std::endl;
      //std::cout<<" const Pmin = "<<minTrackMomentum<<" pMax = "<<maxTrackMomentum<<" maxNormChi2 = "<<maxNormChi2<<std::endl;
      std::cout << " track inner position = " << track->innerPosition()
                << " outer position = " << track->outerPosition() << std::endl;
      std::cout << "track eta (outer) = " << track->outerPosition().eta()
                << " phi (outer) = " << track->outerPosition().phi() << std::endl;
      if (fabs(track->innerPosition().z()) > 500.) {
        DetId innerDetId(track->innerDetId());
        std::cout << " dump inner state MUON detid  = " << debug.dumpMuonId(innerDetId) << std::endl;
      }
      if (fabs(track->outerPosition().z()) > 500.) {
        DetId outerDetId(track->outerDetId());
        std::cout << " dump outer state MUON detid  = " << debug.dumpMuonId(outerDetId) << std::endl;
      }

      std::cout << " nHits = " << track->found() << std::endl;
      /*
      trackingRecHit_iterator rhbegin = track->recHitsBegin();
      trackingRecHit_iterator rhend = track->recHitsEnd();
      int iRH = 0;
      for(trackingRecHit_iterator recHit = rhbegin; recHit != rhend; ++recHit){
        const GeomDet* geomDet = theTrackingGeometry->idToDet((*recHit)->geographicalId());
	std::cout<<"hit "<<iRH<<" loc pos = " <<(*recHit)->localPosition()<<
	  " glob pos = " <<geomDet->toGlobal((*recHit)->localPosition())<<std::endl;
        ++iRH;
      }
      */
    }
    float dpT_ov_pT = 0.;
    if (fabs(track->pt()) > 0.001) {
      dpT_ov_pT = track->ptError() / track->pt();
    }
    //---- These define a "good" track
    if (track->normalizedChi2() > maxNormChi2) {  // quality
      break;
    }
    DataFlow->Fill(5.);
    if (track->found() < minTrackHits) {  // enough data points
      break;
    }
    DataFlow->Fill(6.);
    if (!segments->size()) {  // better have something in the CSC
      break;
    }
    DataFlow->Fill(7.);
    if (magField && (track->p() < minP || track->p() > maxP)) {  // proper energy range
      break;
    }
    DataFlow->Fill(8.);
    if (magField && (dpT_ov_pT > 0.5)) {  // not too crazy uncertainty
      break;
    }
    DataFlow->Fill(9.);

    passTheEvent = true;
    if (printalot)
      std::cout << "good Track" << std::endl;
    CLHEP::Hep3Vector r3T_inner(track->innerPosition().x(), track->innerPosition().y(), track->innerPosition().z());
    CLHEP::Hep3Vector r3T(track->outerPosition().x(), track->outerPosition().y(), track->outerPosition().z());
    chooseDirection(r3T_inner, r3T);  // for non-IP

    CLHEP::Hep3Vector p3T(track->outerMomentum().x(), track->outerMomentum().y(), track->outerMomentum().z());
    CLHEP::Hep3Vector p3_propagated, r3_propagated;
    AlgebraicSymMatrix66 cov_propagated, covT;
    covT *= 1e-20;
    cov_propagated *= 1e-20;
    int charge = track->charge();
    FreeTrajectoryState ftsStart = getFromCLHEP(p3T, r3T, charge, covT, &*(theService->magneticField()));
    if (printalot) {
      std::cout << " p = " << track->p() << " norm chi2 = " << track->normalizedChi2() << std::endl;
      std::cout << " dump the very first FTS  = " << debug.dumpFTS(ftsStart) << std::endl;
    }
    TrajectoryStateOnSurface tSOSDest;
    int endcap = 0;
    //---- which endcap to look at
    if (track->outerPosition().z() > 0) {
      endcap = 1;
    } else {
      endcap = 2;
    }
    int chamber = 1;
    //---- a "refference" CSCDetId for each ring
    std::vector<CSCDetId> refME;
    for (int iS = 1; iS < 5; ++iS) {
      for (int iR = 1; iR < 4; ++iR) {
        if (1 != iS && iR > 2) {
          continue;
        } else if (4 == iS && iR > 1) {
          continue;
        }
        refME.push_back(CSCDetId(endcap, iS, iR, chamber));
      }
    }
    //---- loop over the "refference" CSCDetIds
    for (size_t iSt = 0; iSt < refME.size(); ++iSt) {
      if (printalot) {
        std::cout << "loop iStatation = " << iSt << std::endl;
        std::cout << "refME[iSt]: st = " << refME[iSt].station() << " rg = " << refME[iSt].ring() << std::endl;
      }
      std::map<std::string, bool> chamberTypes;
      chamberTypes["ME11"] = false;
      chamberTypes["ME12"] = false;
      chamberTypes["ME13"] = false;
      chamberTypes["ME21"] = false;
      chamberTypes["ME22"] = false;
      chamberTypes["ME31"] = false;
      chamberTypes["ME32"] = false;
      chamberTypes["ME41"] = false;
      const CSCChamber *cscChamber_base = cscGeom->chamber(refME[iSt].chamberId());
      DetId detId = cscChamber_base->geographicalId();
      if (printalot) {
        std::cout << " base iStation : eta = " << cscGeom->idToDet(detId)->surface().position().eta()
                  << " phi = " << cscGeom->idToDet(detId)->surface().position().phi()
                  << " y = " << cscGeom->idToDet(detId)->surface().position().y() << std::endl;
        std::cout << " dump base iStation detid  = " << debug.dumpMuonId(detId) << std::endl;
        std::cout << " dump FTS start  = " << debug.dumpFTS(ftsStart) << std::endl;
      }
      //---- propagate to this ME
      tSOSDest = propagate(ftsStart, cscGeom->idToDet(detId)->surface());
      if (tSOSDest.isValid()) {
        ftsStart = *tSOSDest.freeState();
        if (printalot)
          std::cout << "  dump FTS end   = " << debug.dumpFTS(ftsStart) << std::endl;
        getFromFTS(ftsStart, p3_propagated, r3_propagated, charge, cov_propagated);
        float feta = fabs(r3_propagated.eta());
        float phi = r3_propagated.phi();
        //---- which rings are (possibly) penetrated
        ringCandidates(refME[iSt].station(), feta, chamberTypes);

        map<std::string, bool>::iterator iter;
        int iterations = 0;
        //---- loop over ring candidates
        for (iter = chamberTypes.begin(); iter != chamberTypes.end(); iter++) {
          ++iterations;
          //---- is this ME a machinig candidate station
          if (iter->second && (iterations - 1) == int(iSt)) {
            if (printalot) {
              std::cout << " Chamber type " << iter->first << " is a candidate..." << std::endl;
              std::cout << " station() = " << refME[iSt].station() << " ring() = " << refME[iSt].ring()
                        << " iSt = " << iSt << std::endl;
            }
            std::vector<int> coupleOfChambers;
            //---- which chamber (and its closes neighbor) is penetrated by the track - candidates
            chamberCandidates(refME[iSt].station(), refME[iSt].ring(), phi, coupleOfChambers);
            //---- loop over the two chamber candidates
            for (size_t iCh = 0; iCh < coupleOfChambers.size(); ++iCh) {
              DataFlow->Fill(11.);
              if (printalot)
                std::cout << " Check chamber N = " << coupleOfChambers.at(iCh) << std::endl;
              ;
              if ((!getAbsoluteEfficiency) &&
                  (true == emptyChambers[refME[iSt].endcap() - 1][refME[iSt].station() - 1][refME[iSt].ring() - 1]
                                        [coupleOfChambers.at(iCh) - FirstCh])) {
                continue;
              }
              CSCDetId theCSCId(refME[iSt].endcap(), refME[iSt].station(), refME[iSt].ring(), coupleOfChambers.at(iCh));
              const CSCChamber *cscChamber = cscGeom->chamber(theCSCId.chamberId());
              const BoundPlane bpCh = cscGeom->idToDet(cscChamber->geographicalId())->surface();
              float zFTS = ftsStart.position().z();
              float dz = fabs(bpCh.position().z() - zFTS);
              float zDistInner = track->innerPosition().z() - bpCh.position().z();
              float zDistOuter = track->outerPosition().z() - bpCh.position().z();
              //---- only detectors between the inner and outer points of the track are considered for non IP-data
              if (printalot) {
                std::cout << " zIn = " << track->innerPosition().z() << " zOut = " << track->outerPosition().z()
                          << " zSurf = " << bpCh.position().z() << std::endl;
              }
              if (!isIPdata && (zDistInner * zDistOuter > 0. || fabs(zDistInner) < 15. ||
                                fabs(zDistOuter) < 15.)) {  // for non IP-data
                if (printalot) {
                  std::cout << " Not an intermediate (as defined) point... Skip." << std::endl;
                }
                continue;
              }
              if (isIPdata && fabs(track->eta()) < 1.8) {
                if (fabs(muonOuterZPosition) - fabs(bpCh.position().z()) < 0 ||
                    fabs(muonOuterZPosition - bpCh.position().z()) < 15.) {
                  continue;
                }
              }
              DataFlow->Fill(13.);
              //---- propagate to the chamber (from this ME) if it is a different surface (odd/even chambers)
              if (dz > 0.1) {  // i.e. non-zero (float 0 check is bad)
                //if(fabs(zChanmber - zFTS ) > 0.1){
                tSOSDest = propagate(ftsStart, cscGeom->idToDet(cscChamber->geographicalId())->surface());
                if (tSOSDest.isValid()) {
                  ftsStart = *tSOSDest.freeState();
                } else {
                  if (printalot)
                    std::cout << "TSOS not valid! Break." << std::endl;
                  break;
                }
              } else {
                if (printalot)
                  std::cout << " info: dz<0.1" << std::endl;
              }
              DataFlow->Fill(15.);
              FreeTrajectoryState ftsInit = ftsStart;
              bool inDeadZone = false;
              //---- loop over the 6 layers
              for (int iLayer = 0; iLayer < 6; ++iLayer) {
                bool extrapolationPassed = true;
                if (printalot) {
                  std::cout << " iLayer = " << iLayer << "   dump FTS init  = " << debug.dumpFTS(ftsInit) << std::endl;
                  std::cout << " dump detid  = " << debug.dumpMuonId(cscChamber->geographicalId()) << std::endl;
                  std::cout << "Surface to propagate to:  pos = " << cscChamber->layer(iLayer + 1)->surface().position()
                            << " eta = " << cscChamber->layer(iLayer + 1)->surface().position().eta()
                            << " phi = " << cscChamber->layer(iLayer + 1)->surface().position().phi() << std::endl;
                }
                //---- propagate to this layer
                tSOSDest = propagate(ftsInit, cscChamber->layer(iLayer + 1)->surface());
                if (tSOSDest.isValid()) {
                  ftsInit = *tSOSDest.freeState();
                  if (printalot)
                    std::cout << " Propagation between layers successful:  dump FTS end  = " << debug.dumpFTS(ftsInit)
                              << std::endl;
                  getFromFTS(ftsInit, p3_propagated, r3_propagated, charge, cov_propagated);
                } else {
                  if (printalot)
                    std::cout << "Propagation between layers not successful - notValid TSOS" << std::endl;
                  extrapolationPassed = false;
                  inDeadZone = true;
                }
                //}
                //---- Extrapolation passed? For each layer?
                if (extrapolationPassed) {
                  GlobalPoint theExtrapolationPoint(r3_propagated.x(), r3_propagated.y(), r3_propagated.z());
                  LocalPoint theLocalPoint = cscChamber->layer(iLayer + 1)->toLocal(theExtrapolationPoint);
                  //std::cout<<" Candidate chamber: extrapolated LocalPoint = "<<theLocalPoint<<std::endl;
                  inDeadZone = (inDeadZone ||
                                !inSensitiveLocalRegion(
                                    theLocalPoint.x(), theLocalPoint.y(), refME[iSt].station(), refME[iSt].ring()));
                  if (printalot) {
                    std::cout << " Candidate chamber: extrapolated LocalPoint = " << theLocalPoint
                              << "inDeadZone = " << inDeadZone << std::endl;
                  }
                  //---- break if in dead zone for any layer ("clean" tracks)
                  if (inDeadZone) {
                    break;
                  }
                } else {
                  break;
                }
              }
              DataFlow->Fill(17.);
              //---- Is a track in a sensitive area for each layer?
              if (!inDeadZone) {  //---- for any layer
                DataFlow->Fill(19.);
                if (printalot)
                  std::cout << "Do efficiencies..." << std::endl;
                //---- Do efficiencies
                // angle cuts applied (if configured)
                bool angle_flag = true;
                angle_flag = efficienciesPerChamber(theCSCId, cscChamber, ftsStart);
                if (useDigis && angle_flag) {
                  stripWire_Efficiencies(theCSCId, ftsStart);
                }
                if (angle_flag) {
                  recHitSegment_Efficiencies(theCSCId, cscChamber, ftsStart);
                  if (!isData) {
                    recSimHitEfficiency(theCSCId, ftsStart);
                  }
                }
              } else {
                if (printalot)
                  std::cout << " Not in active area for all layers" << std::endl;
              }
            }
            if (tSOSDest.isValid()) {
              ftsStart = *tSOSDest.freeState();
            }
          }
        }
      } else {
        if (printalot)
          std::cout << " TSOS not valid..." << std::endl;
      }
    }
  }
  //---- End
  if (printalot)
    printf("==exit===CSCEfficiency===== run %u\tevent %llu\n\n", iRun, iEvent);
  return passTheEvent;
}

//
bool CSCEfficiency::inSensitiveLocalRegion(double xLocal, double yLocal, int station, int ring) {
  //---- Good region means sensitive area of a chamber. "Local" stands for the local system
  bool pass = false;
  std::vector<double> chamberBounds(3);  // the sensitive area
  float y_center = 99999.;
  //---- hardcoded... not good
  if (station > 1 && station < 5) {
    if (2 == ring) {
      chamberBounds[0] = 66.46 / 2;   // (+-)x1 shorter
      chamberBounds[1] = 127.15 / 2;  // (+-)x2 longer
      chamberBounds[2] = 323.06 / 2;
      y_center = -0.95;
    } else {
      if (2 == station) {
        chamberBounds[0] = 54.00 / 2;   // (+-)x1 shorter
        chamberBounds[1] = 125.71 / 2;  // (+-)x2 longer
        chamberBounds[2] = 189.66 / 2;
        y_center = -0.955;
      } else if (3 == station) {
        chamberBounds[0] = 61.40 / 2;   // (+-)x1 shorter
        chamberBounds[1] = 125.71 / 2;  // (+-)x2 longer
        chamberBounds[2] = 169.70 / 2;
        y_center = -0.97;
      } else if (4 == station) {
        chamberBounds[0] = 69.01 / 2;   // (+-)x1 shorter
        chamberBounds[1] = 125.65 / 2;  // (+-)x2 longer
        chamberBounds[2] = 149.42 / 2;
        y_center = -0.94;
      }
    }
  } else if (1 == station) {
    if (3 == ring) {
      chamberBounds[0] = 63.40 / 2;  // (+-)x1 shorter
      chamberBounds[1] = 92.10 / 2;  // (+-)x2 longer
      chamberBounds[2] = 164.16 / 2;
      y_center = -1.075;
    } else if (2 == ring) {
      chamberBounds[0] = 51.00 / 2;  // (+-)x1 shorter
      chamberBounds[1] = 83.74 / 2;  // (+-)x2 longer
      chamberBounds[2] = 174.49 / 2;
      y_center = -0.96;
    } else {                        // to be investigated
      chamberBounds[0] = 30. / 2;   //40./2; // (+-)x1 shorter
      chamberBounds[1] = 60. / 2;   //100./2; // (+-)x2 longer
      chamberBounds[2] = 160. / 2;  //142./2;
      y_center = 0.;
    }
  }
  double yUp = chamberBounds[2] + y_center;
  double yDown = -chamberBounds[2] + y_center;
  double xBound1Shifted = chamberBounds[0] - distanceFromDeadZone;  //
  double xBound2Shifted = chamberBounds[1] - distanceFromDeadZone;  //
  double lineSlope = (yUp - yDown) / (xBound2Shifted - xBound1Shifted);
  double lineConst = yUp - lineSlope * xBound2Shifted;
  double yBoundary = lineSlope * abs(xLocal) + lineConst;
  pass = checkLocal(yLocal, yBoundary, station, ring);
  return pass;
}

bool CSCEfficiency::checkLocal(double yLocal, double yBoundary, int station, int ring) {
  //---- check if it is in a good local region (sensitive area - geometrical and HV boundaries excluded)
  bool pass = false;
  std::vector<float> deadZoneCenter(6);
  const float deadZoneHalf = 0.32 * 7 / 2;              // wire spacing * (wires missing + 1)/2
  float cutZone = deadZoneHalf + distanceFromDeadZone;  //cm
  //---- hardcoded... not good
  if (station > 1 && station < 5) {
    if (2 == ring) {
      deadZoneCenter[0] = -162.48;
      deadZoneCenter[1] = -81.8744;
      deadZoneCenter[2] = -21.18165;
      deadZoneCenter[3] = 39.51105;
      deadZoneCenter[4] = 100.2939;
      deadZoneCenter[5] = 160.58;

      if (yLocal > yBoundary && ((yLocal > deadZoneCenter[0] + cutZone && yLocal < deadZoneCenter[1] - cutZone) ||
                                 (yLocal > deadZoneCenter[1] + cutZone && yLocal < deadZoneCenter[2] - cutZone) ||
                                 (yLocal > deadZoneCenter[2] + cutZone && yLocal < deadZoneCenter[3] - cutZone) ||
                                 (yLocal > deadZoneCenter[3] + cutZone && yLocal < deadZoneCenter[4] - cutZone) ||
                                 (yLocal > deadZoneCenter[4] + cutZone && yLocal < deadZoneCenter[5] - cutZone))) {
        pass = true;
      }
    } else if (1 == ring) {
      if (2 == station) {
        deadZoneCenter[0] = -95.94;
        deadZoneCenter[1] = -27.47;
        deadZoneCenter[2] = 33.67;
        deadZoneCenter[3] = 93.72;
      } else if (3 == station) {
        deadZoneCenter[0] = -85.97;
        deadZoneCenter[1] = -36.21;
        deadZoneCenter[2] = 23.68;
        deadZoneCenter[3] = 84.04;
      } else if (4 == station) {
        deadZoneCenter[0] = -75.82;
        deadZoneCenter[1] = -26.14;
        deadZoneCenter[2] = 23.85;
        deadZoneCenter[3] = 73.91;
      }
      if (yLocal > yBoundary && ((yLocal > deadZoneCenter[0] + cutZone && yLocal < deadZoneCenter[1] - cutZone) ||
                                 (yLocal > deadZoneCenter[1] + cutZone && yLocal < deadZoneCenter[2] - cutZone) ||
                                 (yLocal > deadZoneCenter[2] + cutZone && yLocal < deadZoneCenter[3] - cutZone))) {
        pass = true;
      }
    }
  } else if (1 == station) {
    if (3 == ring) {
      deadZoneCenter[0] = -83.155;
      deadZoneCenter[1] = -22.7401;
      deadZoneCenter[2] = 27.86665;
      deadZoneCenter[3] = 81.005;
      if (yLocal > yBoundary && ((yLocal > deadZoneCenter[0] + cutZone && yLocal < deadZoneCenter[1] - cutZone) ||
                                 (yLocal > deadZoneCenter[1] + cutZone && yLocal < deadZoneCenter[2] - cutZone) ||
                                 (yLocal > deadZoneCenter[2] + cutZone && yLocal < deadZoneCenter[3] - cutZone))) {
        pass = true;
      }
    } else if (2 == ring) {
      deadZoneCenter[0] = -86.285;
      deadZoneCenter[1] = -32.88305;
      deadZoneCenter[2] = 32.867423;
      deadZoneCenter[3] = 88.205;
      if (yLocal > (yBoundary) && ((yLocal > deadZoneCenter[0] + cutZone && yLocal < deadZoneCenter[1] - cutZone) ||
                                   (yLocal > deadZoneCenter[1] + cutZone && yLocal < deadZoneCenter[2] - cutZone) ||
                                   (yLocal > deadZoneCenter[2] + cutZone && yLocal < deadZoneCenter[3] - cutZone))) {
        pass = true;
      }
    } else {
      deadZoneCenter[0] = -81.0;
      deadZoneCenter[1] = 81.0;
      if (yLocal > (yBoundary) && ((yLocal > deadZoneCenter[0] + cutZone && yLocal < deadZoneCenter[1] - cutZone))) {
        pass = true;
      }
    }
  }
  return pass;
}

void CSCEfficiency::fillDigiInfo(edm::Handle<CSCALCTDigiCollection> &alcts,
                                 edm::Handle<CSCCLCTDigiCollection> &clcts,
                                 edm::Handle<CSCCorrelatedLCTDigiCollection> &correlatedlcts,
                                 edm::Handle<CSCWireDigiCollection> &wires,
                                 edm::Handle<CSCStripDigiCollection> &strips,
                                 edm::Handle<edm::PSimHitContainer> &simhits,
                                 edm::Handle<CSCRecHit2DCollection> &rechits,
                                 edm::Handle<CSCSegmentCollection> &segments,
                                 edm::ESHandle<CSCGeometry> &cscGeom) {
  for (int iE = 0; iE < 2; iE++) {
    for (int iS = 0; iS < 4; iS++) {
      for (int iR = 0; iR < 4; iR++) {
        for (int iC = 0; iC < NumCh; iC++) {
          allSegments[iE][iS][iR][iC].clear();
          allCLCT[iE][iS][iR][iC] = allALCT[iE][iS][iR][iC] = allCorrLCT[iE][iS][iR][iC] = false;
          for (int iL = 0; iL < 6; iL++) {
            allStrips[iE][iS][iR][iC][iL].clear();
            allWG[iE][iS][iR][iC][iL].clear();
            allRechits[iE][iS][iR][iC][iL].clear();
            allSimhits[iE][iS][iR][iC][iL].clear();
          }
        }
      }
    }
  }
  //
  if (useDigis) {
    fillLCT_info(alcts, clcts, correlatedlcts);
    fillWG_info(wires, cscGeom);
    fillStrips_info(strips);
  }
  fillRechitsSegments_info(rechits, segments, cscGeom);
  if (!isData) {
    fillSimhit_info(simhits);
  }
}

void CSCEfficiency::fillLCT_info(edm::Handle<CSCALCTDigiCollection> &alcts,
                                 edm::Handle<CSCCLCTDigiCollection> &clcts,
                                 edm::Handle<CSCCorrelatedLCTDigiCollection> &correlatedlcts) {
  //---- ALCTDigis
  int nSize = 0;
  for (CSCALCTDigiCollection::DigiRangeIterator j = alcts->begin(); j != alcts->end(); j++) {
    ++nSize;
    const CSCDetId &id = (*j).first;
    const CSCALCTDigiCollection::Range &range = (*j).second;
    for (CSCALCTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
      // Valid digi in the chamber (or in neighbouring chamber)
      if ((*digiIt).isValid()) {
        allALCT[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - FirstCh] = true;
      }
    }  // for digis in layer
  }    // end of for (j=...
  ALCTPerEvent->Fill(nSize);
  //---- CLCTDigis
  nSize = 0;
  for (CSCCLCTDigiCollection::DigiRangeIterator j = clcts->begin(); j != clcts->end(); j++) {
    ++nSize;
    const CSCDetId &id = (*j).first;
    std::vector<CSCCLCTDigi>::const_iterator digiIt = (*j).second.first;
    std::vector<CSCCLCTDigi>::const_iterator last = (*j).second.second;
    for (; digiIt != last; ++digiIt) {
      // Valid digi in the chamber (or in neighbouring chamber)
      if ((*digiIt).isValid()) {
        allCLCT[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - FirstCh] = true;
      }
    }
  }
  CLCTPerEvent->Fill(nSize);
  //---- CorrLCTDigis
  for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator j = correlatedlcts->begin(); j != correlatedlcts->end(); j++) {
    const CSCDetId &id = (*j).first;
    std::vector<CSCCorrelatedLCTDigi>::const_iterator digiIt = (*j).second.first;
    std::vector<CSCCorrelatedLCTDigi>::const_iterator last = (*j).second.second;
    for (; digiIt != last; ++digiIt) {
      // Valid digi in the chamber (or in neighbouring chamber)
      if ((*digiIt).isValid()) {
        allCorrLCT[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - FirstCh] = true;
      }
    }
  }
}
//
void CSCEfficiency::fillWG_info(edm::Handle<CSCWireDigiCollection> &wires, edm::ESHandle<CSCGeometry> &cscGeom) {
  //---- WIRE GROUPS
  for (CSCWireDigiCollection::DigiRangeIterator j = wires->begin(); j != wires->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    const CSCLayer *layer_p = cscGeom->layer(id);
    const CSCLayerGeometry *layerGeom = layer_p->geometry();
    //
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    //
    for (; digiItr != last; ++digiItr) {
      std::pair<int, float> WG_pos(digiItr->getWireGroup(), layerGeom->yOfWireGroup(digiItr->getWireGroup()));
      std::pair<std::pair<int, float>, int> LayerSignal(WG_pos, digiItr->getTimeBin());

      //---- AllWG contains basic information about WG (WG number and Y-position, time bin)
      allWG[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - FirstCh][id.layer() - 1].push_back(
          LayerSignal);
      if (printalot) {
        //std::cout<<" WG check : "<<std::endl;
        //printf("\t\tendcap/station/ring/chamber/layer: %i/%i/%i/%i/%i\n",id.endcap(),id.station(),id.ring(),id.chamber(),id.layer());
        //std::cout<<" WG size = "<<allWG[id.endcap()-1][id.station()-1][id.ring()-1][id.chamber()-FirstCh]
        //[id.layer()-1].size()<<std::endl;
      }
    }
  }
}
void CSCEfficiency::fillStrips_info(edm::Handle<CSCStripDigiCollection> &strips) {
  //---- STRIPS
  for (CSCStripDigiCollection::DigiRangeIterator j = strips->begin(); j != strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int largestADCValue = -1;
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for (; digiItr != last; ++digiItr) {
      int maxADC = largestADCValue;
      int myStrip = digiItr->getStrip();
      std::vector<int> myADCVals = digiItr->getADCCounts();
      float thisPedestal = 0.5 * (float)(myADCVals[0] + myADCVals[1]);
      float threshold = 13.3;
      float diff = 0.;
      float peakADC = -1000.;
      for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
        diff = (float)myADCVals[iCount] - thisPedestal;
        if (diff > threshold) {
          if (myADCVals[iCount] > largestADCValue) {
            largestADCValue = myADCVals[iCount];
          }
        }
        if (diff > threshold && diff > peakADC) {
          peakADC = diff;
        }
      }
      if (largestADCValue > maxADC) {  // FIX IT!!!
        maxADC = largestADCValue;
        std::pair<int, float> LayerSignal(myStrip, peakADC);

        //---- AllStrips contains basic information about strips
        //---- (strip number and peak signal for most significant strip in the layer)
        allStrips[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - 1][id.layer() - 1].clear();
        allStrips[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - 1][id.layer() - 1].push_back(
            LayerSignal);
      }
    }
  }
}
void CSCEfficiency::fillSimhit_info(edm::Handle<edm::PSimHitContainer> &simhits) {
  //---- SIMHITS
  edm::PSimHitContainer::const_iterator dSHsimIter;
  for (dSHsimIter = simhits->begin(); dSHsimIter != simhits->end(); dSHsimIter++) {
    // Get DetID for this simHit:
    CSCDetId sId = (CSCDetId)(*dSHsimIter).detUnitId();
    std::pair<LocalPoint, int> simHitPos((*dSHsimIter).localPosition(), (*dSHsimIter).particleType());
    allSimhits[sId.endcap() - 1][sId.station() - 1][sId.ring() - 1][sId.chamber() - FirstCh][sId.layer() - 1].push_back(
        simHitPos);
  }
}
//
void CSCEfficiency::fillRechitsSegments_info(edm::Handle<CSCRecHit2DCollection> &rechits,
                                             edm::Handle<CSCSegmentCollection> &segments,
                                             edm::ESHandle<CSCGeometry> &cscGeom) {
  //---- RECHITS AND SEGMENTS
  //---- Loop over rechits
  if (printalot) {
    //printf("\tGet the recHits collection.\t ");
    printf("  The size of the rechit collection is %i\n", int(rechits->size()));
    //printf("\t...start loop over rechits...\n");
  }
  recHitsPerEvent->Fill(rechits->size());
  //---- Build iterator for rechits and loop :
  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = rechits->begin(); recIt != rechits->end(); recIt++) {
    //---- Find chamber with rechits in CSC
    CSCDetId id = (CSCDetId)(*recIt).cscDetId();
    if (printalot) {
      const CSCLayer *csclayer = cscGeom->layer(id);
      LocalPoint rhitlocal = (*recIt).localPosition();
      LocalError rerrlocal = (*recIt).localPositionError();
      GlobalPoint rhitglobal = csclayer->toGlobal(rhitlocal);
      printf("\t\tendcap/station/ring/chamber/layer: %i/%i/%i/%i/%i\n",
             id.endcap(),
             id.station(),
             id.ring(),
             id.chamber(),
             id.layer());
      printf("\t\tx,y,z: %f, %f, %f\texx,eey,exy: %f, %f, %f\tglobal x,y,z: %f, %f, %f \n",
             rhitlocal.x(),
             rhitlocal.y(),
             rhitlocal.z(),
             rerrlocal.xx(),
             rerrlocal.yy(),
             rerrlocal.xy(),
             rhitglobal.x(),
             rhitglobal.y(),
             rhitglobal.z());
    }
    std::pair<LocalPoint, bool> recHitPos((*recIt).localPosition(), false);
    allRechits[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - FirstCh][id.layer() - 1].push_back(
        recHitPos);
  }
  //---- "Empty" chambers
  for (int iE = 0; iE < 2; iE++) {
    for (int iS = 0; iS < 4; iS++) {
      for (int iR = 0; iR < 4; iR++) {
        for (int iC = 0; iC < NumCh; iC++) {
          int numLayers = 0;
          for (int iL = 0; iL < 6; iL++) {
            if (!allRechits[iE][iS][iR][iC][iL].empty()) {
              ++numLayers;
            }
          }
          if (numLayers > 1) {
            emptyChambers[iE][iS][iR][iC] = false;
          } else {
            emptyChambers[iE][iS][iR][iC] = true;
          }
        }
      }
    }
  }

  //
  if (printalot) {
    printf("  The size of the segment collection is %i\n", int(segments->size()));
    //printf("\t...start loop over segments...\n");
  }
  segmentsPerEvent->Fill(segments->size());
  for (CSCSegmentCollection::const_iterator it = segments->begin(); it != segments->end(); it++) {
    CSCDetId id = (CSCDetId)(*it).cscDetId();
    StHist[id.endcap() - 1][id.station() - 1].segmentChi2_ndf->Fill((*it).chi2() / (*it).degreesOfFreedom());
    StHist[id.endcap() - 1][id.station() - 1].hitsInSegment->Fill((*it).nRecHits());
    if (printalot) {
      printf("\tendcap/station/ring/chamber: %i %i %i %i\n", id.endcap(), id.station(), id.ring(), id.chamber());
      std::cout << "\tposition(loc) = " << (*it).localPosition() << " error(loc) = " << (*it).localPositionError()
                << std::endl;
      std::cout << "\t chi2/ndf = " << (*it).chi2() / (*it).degreesOfFreedom() << " nhits = " << (*it).nRecHits()
                << std::endl;
    }
    allSegments[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - FirstCh].push_back(
        make_pair((*it).localPosition(), (*it).localDirection()));

    //---- try to get the CSC recHits that contribute to this segment.
    //if (printalot) printf("\tGet the recHits for this segment.\t");
    std::vector<CSCRecHit2D> theseRecHits = (*it).specificRecHits();
    int nRH = (*it).nRecHits();
    if (printalot) {
      printf("\tGet the recHits for this segment.\t");
      printf("    nRH = %i\n", nRH);
    }
    //---- Find which of the rechits in the chamber is in the segment
    int layerRH = 0;
    for (vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
      ++layerRH;
      CSCDetId idRH = (CSCDetId)(*iRH).cscDetId();
      if (printalot) {
        printf("\t%i RH\tendcap/station/ring/chamber/layer: %i/%i/%i/%i/%i\n",
               layerRH,
               idRH.endcap(),
               idRH.station(),
               idRH.ring(),
               idRH.chamber(),
               idRH.layer());
      }
      for (size_t jRH = 0;
           jRH <
           allRechits[idRH.endcap() - 1][idRH.station() - 1][idRH.ring() - 1][idRH.chamber() - FirstCh][idRH.layer() - 1]
               .size();
           ++jRH) {
        float xDiff = iRH->localPosition().x() - allRechits[idRH.endcap() - 1][idRH.station() - 1][idRH.ring() - 1]
                                                           [idRH.chamber() - FirstCh][idRH.layer() - 1][jRH]
                                                               .first.x();
        float yDiff = iRH->localPosition().y() - allRechits[idRH.endcap() - 1][idRH.station() - 1][idRH.ring() - 1]
                                                           [idRH.chamber() - FirstCh][idRH.layer() - 1][jRH]
                                                               .first.y();
        if (fabs(xDiff) < 0.0001 && fabs(yDiff) < 0.0001) {
          std::pair<LocalPoint, bool> recHitPos(allRechits[idRH.endcap() - 1][idRH.station() - 1][idRH.ring() - 1]
                                                          [idRH.chamber() - FirstCh][idRH.layer() - 1][jRH]
                                                              .first,
                                                true);
          allRechits[idRH.endcap() - 1][idRH.station() - 1][idRH.ring() - 1][idRH.chamber() - FirstCh][idRH.layer() - 1]
                    [jRH] = recHitPos;
          if (printalot) {
            std::cout << " number of the rechit (from zero) in the segment = " << jRH << std::endl;
          }
        }
      }
    }
  }
}
//

void CSCEfficiency::ringCandidates(int station, float feta, std::map<std::string, bool> &chamberTypes) {
  // yeah, hardcoded again...
  switch (station) {
    case 1:
      if (feta > 0.85 && feta < 1.18) {  //ME13
        chamberTypes["ME13"] = true;
      }
      if (feta > 1.18 && feta < 1.7) {  //ME12
        chamberTypes["ME12"] = true;
      }
      if (feta > 1.5 && feta < 2.45) {  //ME11
        chamberTypes["ME11"] = true;
      }
      break;
    case 2:
      if (feta > 0.95 && feta < 1.6) {  //ME22
        chamberTypes["ME22"] = true;
      }
      if (feta > 1.55 && feta < 2.45) {  //ME21
        chamberTypes["ME21"] = true;
      }
      break;
    case 3:
      if (feta > 1.08 && feta < 1.72) {  //ME32
        chamberTypes["ME32"] = true;
      }
      if (feta > 1.69 && feta < 2.45) {  //ME31
        chamberTypes["ME31"] = true;
      }
      break;
    case 4:
      if (feta > 1.78 && feta < 2.45) {  //ME41
        chamberTypes["ME41"] = true;
      }
      break;
    default:
      break;
  }
}
//
void CSCEfficiency::chamberCandidates(int station, int ring, float phi, std::vector<int> &coupleOfChambers) {
  coupleOfChambers.clear();
  // -pi< phi<+pi
  float phi_zero = 0.;  // check! the phi at the "edge" of Ch 1
  float phi_const = 2. * M_PI / 36.;
  int last_chamber = 36;
  int first_chamber = 1;
  if (1 != station && 1 == ring) {  // 18 chambers in the ring
    phi_const *= 2;
    last_chamber /= 2;
  }
  if (phi < 0.) {
    if (printalot)
      std::cout << " info: negative phi = " << phi << std::endl;
    phi += 2 * M_PI;
  }
  float chamber_float = (phi - phi_zero) / phi_const;
  int chamber_int = int(chamber_float);
  if (chamber_float - float(chamber_int) - 0.5 < 0.) {
    if (0 != chamber_int) {
      coupleOfChambers.push_back(chamber_int);
    } else {
      coupleOfChambers.push_back(last_chamber);
    }
    coupleOfChambers.push_back(chamber_int + 1);

  } else {
    coupleOfChambers.push_back(chamber_int + 1);
    if (last_chamber != chamber_int + 1) {
      coupleOfChambers.push_back(chamber_int + 2);
    } else {
      coupleOfChambers.push_back(first_chamber);
    }
  }
  if (printalot)
    std::cout << " phi = " << phi << " phi_zero = " << phi_zero << " phi_const = " << phi_const
              << " candidate chambers: first ch = " << coupleOfChambers[0] << " second ch = " << coupleOfChambers[1]
              << std::endl;
}

//
bool CSCEfficiency::efficienciesPerChamber(CSCDetId &id,
                                           const CSCChamber *cscChamber,
                                           FreeTrajectoryState &ftsChamber) {
  int ec, st, rg, ch, secondRing;
  returnTypes(id, ec, st, rg, ch, secondRing);

  LocalVector localDir = cscChamber->toLocal(ftsChamber.momentum());
  if (printalot) {
    std::cout << " global dir = " << ftsChamber.momentum() << std::endl;
    std::cout << " local dir = " << localDir << std::endl;
    std::cout << " local theta = " << localDir.theta() << std::endl;
  }
  float dxdz = localDir.x() / localDir.z();
  float dydz = localDir.y() / localDir.z();
  if (2 == st || 3 == st) {
    if (printalot) {
      std::cout << "st 3 or 4 ... flip dy/dz" << std::endl;
    }
    dydz = -dydz;
  }
  if (printalot) {
    std::cout << "dy/dz = " << dydz << std::endl;
  }
  // Apply angle cut
  bool out = true;
  if (applyIPangleCuts) {
    if (dydz > local_DY_DZ_Max || dydz < local_DY_DZ_Min || fabs(dxdz) > local_DX_DZ_Max) {
      out = false;
    }
  }

  // Segments
  bool firstCondition = !allSegments[ec][st][rg][ch].empty() ? true : false;
  bool secondCondition = false;
  //---- ME1 is special as usual - ME1a and ME1b are actually one chamber
  if (secondRing > -1) {
    secondCondition = !allSegments[ec][st][secondRing][ch].empty() ? true : false;
  }
  if (firstCondition || secondCondition) {
    if (out) {
      ChHist[ec][st][rg][ch].digiAppearanceCount->Fill(1);
    }
  } else {
    if (out) {
      ChHist[ec][st][rg][ch].digiAppearanceCount->Fill(0);
    }
  }

  if (useDigis) {
    // ALCTs
    firstCondition = allALCT[ec][st][rg][ch];
    secondCondition = false;
    if (secondRing > -1) {
      secondCondition = allALCT[ec][st][secondRing][ch];
    }
    if (firstCondition || secondCondition) {
      if (out) {
        ChHist[ec][st][rg][ch].digiAppearanceCount->Fill(3);
      }
      // always apply partial angle cuts for this kind of histos
      if (fabs(dxdz) < local_DX_DZ_Max) {
        StHist[ec][st].EfficientALCT_momTheta->Fill(ftsChamber.momentum().theta());
        ChHist[ec][st][rg][ch].EfficientALCT_dydz->Fill(dydz);
      }
    } else {
      if (out) {
        ChHist[ec][st][rg][ch].digiAppearanceCount->Fill(2);
      }
      if (fabs(dxdz) < local_DX_DZ_Max) {
        StHist[ec][st].InefficientALCT_momTheta->Fill(ftsChamber.momentum().theta());
        ChHist[ec][st][rg][ch].InefficientALCT_dydz->Fill(dydz);
      }
      if (printalot) {
        std::cout << " missing ALCT (dy/dz = " << dydz << ")";
        printf("\t\tendcap/station/ring/chamber: %i/%i/%i/%i\n", ec + 1, st + 1, rg + 1, ch + 1);
      }
    }

    // CLCTs
    firstCondition = allCLCT[ec][st][rg][ch];
    secondCondition = false;
    if (secondRing > -1) {
      secondCondition = allCLCT[ec][st][secondRing][ch];
    }
    if (firstCondition || secondCondition) {
      if (out) {
        ChHist[ec][st][rg][ch].digiAppearanceCount->Fill(5);
      }
      if (dydz < local_DY_DZ_Max && dydz > local_DY_DZ_Min) {
        StHist[ec][st].EfficientCLCT_momPhi->Fill(ftsChamber.momentum().phi());  // - phi chamber...
        ChHist[ec][st][rg][ch].EfficientCLCT_dxdz->Fill(dxdz);
      }
    } else {
      if (out) {
        ChHist[ec][st][rg][ch].digiAppearanceCount->Fill(4);
      }
      if (dydz < local_DY_DZ_Max && dydz > local_DY_DZ_Min) {
        StHist[ec][st].InefficientCLCT_momPhi->Fill(ftsChamber.momentum().phi());  // - phi chamber...
        ChHist[ec][st][rg][ch].InefficientCLCT_dxdz->Fill(dxdz);
      }
      if (printalot) {
        std::cout << " missing CLCT  (dx/dz = " << dxdz << ")";
        printf("\t\tendcap/station/ring/chamber: %i/%i/%i/%i\n", ec + 1, st + 1, rg + 1, ch + 1);
      }
    }
    if (out) {
      // CorrLCTs
      firstCondition = allCorrLCT[ec][st][rg][ch];
      secondCondition = false;
      if (secondRing > -1) {
        secondCondition = allCorrLCT[ec][st][secondRing][ch];
      }
      if (firstCondition || secondCondition) {
        ChHist[ec][st][rg][ch].digiAppearanceCount->Fill(7);
      } else {
        ChHist[ec][st][rg][ch].digiAppearanceCount->Fill(6);
      }
    }
  }
  return out;
}

//
bool CSCEfficiency::stripWire_Efficiencies(CSCDetId &id, FreeTrajectoryState &ftsChamber) {
  int ec, st, rg, ch, secondRing;
  returnTypes(id, ec, st, rg, ch, secondRing);

  bool firstCondition, secondCondition;
  int missingLayers_s = 0;
  int missingLayers_wg = 0;
  for (int iLayer = 0; iLayer < 6; iLayer++) {
    //----Strips
    if (printalot) {
      printf("\t%i swEff: \tendcap/station/ring/chamber/layer: %i/%i/%i/%i/%i\n",
             iLayer + 1,
             id.endcap(),
             id.station(),
             id.ring(),
             id.chamber(),
             iLayer + 1);
      std::cout << " size S = "
                << allStrips[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - FirstCh][iLayer].size()
                << "size W = "
                << allWG[id.endcap() - 1][id.station() - 1][id.ring() - 1][id.chamber() - FirstCh][iLayer].size()
                << std::endl;
    }
    firstCondition = !allStrips[ec][st][rg][ch][iLayer].empty() ? true : false;
    //allSegments[ec][st][rg][ch].size() ? true : false;
    secondCondition = false;
    if (secondRing > -1) {
      secondCondition = !allStrips[ec][st][secondRing][ch][iLayer].empty() ? true : false;
    }
    if (firstCondition || secondCondition) {
      ChHist[ec][st][rg][ch].EfficientStrips->Fill(iLayer + 1);
    } else {
      if (printalot) {
        std::cout << "missing strips ";
        printf("\t\tendcap/station/ring/chamber/layer: %i/%i/%i/%i/%i\n",
               id.endcap(),
               id.station(),
               id.ring(),
               id.chamber(),
               iLayer + 1);
      }
    }
    // Wires
    firstCondition = !allWG[ec][st][rg][ch][iLayer].empty() ? true : false;
    secondCondition = false;
    if (secondRing > -1) {
      secondCondition = !allWG[ec][st][secondRing][ch][iLayer].empty() ? true : false;
    }
    if (firstCondition || secondCondition) {
      ChHist[ec][st][rg][ch].EfficientWireGroups->Fill(iLayer + 1);
    } else {
      if (printalot) {
        std::cout << "missing wires ";
        printf("\t\tendcap/station/ring/chamber/layer: %i/%i/%i/%i/%i\n",
               id.endcap(),
               id.station(),
               id.ring(),
               id.chamber(),
               iLayer + 1);
      }
    }
  }
  // Normalization
  if (6 != missingLayers_s) {
    ChHist[ec][st][rg][ch].EfficientStrips->Fill(8);
  }
  if (6 != missingLayers_wg) {
    ChHist[ec][st][rg][ch].EfficientWireGroups->Fill(8);
  }
  ChHist[ec][st][rg][ch].EfficientStrips->Fill(9);
  ChHist[ec][st][rg][ch].EfficientWireGroups->Fill(9);
  //
  ChHist[ec][st][rg][ch].StripWiresCorrelations->Fill(1);
  if (missingLayers_s != missingLayers_wg) {
    ChHist[ec][st][rg][ch].StripWiresCorrelations->Fill(2);
    if (6 == missingLayers_wg) {
      ChHist[ec][st][rg][ch].StripWiresCorrelations->Fill(3);
      ChHist[ec][st][rg][ch].NoWires_momTheta->Fill(ftsChamber.momentum().theta());
    }
    if (6 == missingLayers_s) {
      ChHist[ec][st][rg][ch].StripWiresCorrelations->Fill(4);
      ChHist[ec][st][rg][ch].NoStrips_momPhi->Fill(ftsChamber.momentum().theta());
    }
  } else if (6 == missingLayers_s) {
    ChHist[ec][st][rg][ch].StripWiresCorrelations->Fill(5);
  }

  return true;
}
//
bool CSCEfficiency::recSimHitEfficiency(CSCDetId &id, FreeTrajectoryState &ftsChamber) {
  int ec, st, rg, ch, secondRing;
  returnTypes(id, ec, st, rg, ch, secondRing);
  bool firstCondition, secondCondition;
  for (int iLayer = 0; iLayer < 6; iLayer++) {
    firstCondition = !allSimhits[ec][st][rg][ch][iLayer].empty() ? true : false;
    secondCondition = false;
    int thisRing = rg;
    if (secondRing > -1) {
      secondCondition = !allSimhits[ec][st][secondRing][ch][iLayer].empty() ? true : false;
      if (secondCondition) {
        thisRing = secondRing;
      }
    }
    if (firstCondition || secondCondition) {
      for (size_t iSH = 0; iSH < allSimhits[ec][st][thisRing][ch][iLayer].size(); ++iSH) {
        if (13 == fabs(allSimhits[ec][st][thisRing][ch][iLayer][iSH].second)) {
          ChHist[ec][st][rg][ch].SimSimhits->Fill(iLayer + 1);
          if (!allRechits[ec][st][thisRing][ch][iLayer].empty()) {
            ChHist[ec][st][rg][ch].SimRechits->Fill(iLayer + 1);
          }
          break;
        }
      }
      //---- Next is not too usefull...
      /*
      for(unsigned int iSimHits=0;
	  iSimHits<allSimhits[id.endcap()-1][id.station()-1][id.ring()-1][id.chamber()-FirstCh][iLayer].size();
	  iSimHits++){
	ChHist[ec][st][rg][id.chamber()-FirstCh].SimSimhits_each->Fill(iLayer+1);
      }
      for(unsigned int iRecHits=0;
	  iRecHits<allRechits[id.endcap()-1][id.station()-1][id.ring()-1][id.chamber()-FirstCh][iLayer].size();
	  iRecHits++){
	ChHist[ec][st][rg][id.chamber()-FirstCh].SimRechits_each->Fill(iLayer+1);
      }
      */
      //
    }
  }
  return true;
}

//
bool CSCEfficiency::recHitSegment_Efficiencies(CSCDetId &id,
                                               const CSCChamber *cscChamber,
                                               FreeTrajectoryState &ftsChamber) {
  int ec, st, rg, ch, secondRing;
  returnTypes(id, ec, st, rg, ch, secondRing);
  bool firstCondition, secondCondition;

  std::vector<bool> missingLayers_rh(6);
  std::vector<int> usedInSegment(6);
  // Rechits
  if (printalot)
    std::cout << "RecHits eff" << std::endl;
  for (int iLayer = 0; iLayer < 6; ++iLayer) {
    firstCondition = !allRechits[ec][st][rg][ch][iLayer].empty() ? true : false;
    secondCondition = false;
    int thisRing = rg;
    if (secondRing > -1) {
      secondCondition = !allRechits[ec][st][secondRing][ch][iLayer].empty() ? true : false;
      if (secondCondition) {
        thisRing = secondRing;
      }
    }
    if (firstCondition || secondCondition) {
      ChHist[ec][st][rg][ch].EfficientRechits_good->Fill(iLayer + 1);
      for (size_t iR = 0; iR < allRechits[ec][st][thisRing][ch][iLayer].size(); ++iR) {
        if (allRechits[ec][st][thisRing][ch][iLayer][iR].second) {
          usedInSegment[iLayer] = 1;
          break;
        } else {
          usedInSegment[iLayer] = -1;
        }
      }
    } else {
      missingLayers_rh[iLayer] = true;
      if (printalot) {
        std::cout << "missing rechits ";
        printf("\t\tendcap/station/ring/chamber/layer: %i/%i/%i/%i/%i\n",
               id.endcap(),
               id.station(),
               id.ring(),
               id.chamber(),
               iLayer + 1);
      }
    }
  }
  GlobalVector globalDir;
  GlobalPoint globalPos;
  // Segments
  firstCondition = !allSegments[ec][st][rg][ch].empty() ? true : false;
  secondCondition = false;
  int secondSize = 0;
  int thisRing = rg;
  if (secondRing > -1) {
    secondCondition = !allSegments[ec][st][secondRing][ch].empty() ? true : false;
    secondSize = allSegments[ec][st][secondRing][ch].size();
    if (secondCondition) {
      thisRing = secondRing;
    }
  }
  if (firstCondition || secondCondition) {
    if (printalot)
      std::cout << "segments - start ec = " << ec << " st = " << st << " rg = " << rg << " ch = " << ch << std::endl;
    StHist[ec][st].EfficientSegments_XY->Fill(ftsChamber.position().x(), ftsChamber.position().y());
    if (1 == allSegments[ec][st][rg][ch].size() + secondSize) {
      globalDir = cscChamber->toGlobal(allSegments[ec][st][thisRing][ch][0].second);
      globalPos = cscChamber->toGlobal(allSegments[ec][st][thisRing][ch][0].first);
      StHist[ec][st].EfficientSegments_eta->Fill(fabs(ftsChamber.position().eta()));
      double residual =
          sqrt(pow(ftsChamber.position().x() - globalPos.x(), 2) + pow(ftsChamber.position().y() - globalPos.y(), 2) +
               pow(ftsChamber.position().z() - globalPos.z(), 2));
      if (printalot)
        std::cout << " fts.position() = " << ftsChamber.position() << " segPos = " << globalPos << " res = " << residual
                  << std::endl;
      StHist[ec][st].ResidualSegments->Fill(residual);
    }
    for (int iLayer = 0; iLayer < 6; ++iLayer) {
      if (printalot)
        std::cout << " iLayer = " << iLayer << " usedInSegment = " << usedInSegment[iLayer] << std::endl;
      if (0 != usedInSegment[iLayer]) {
        if (-1 == usedInSegment[iLayer]) {
          ChHist[ec][st][rg][ch].InefficientSingleHits->Fill(iLayer + 1);
        }
        ChHist[ec][st][rg][ch].AllSingleHits->Fill(iLayer + 1);
      }
      firstCondition = !allRechits[ec][st][rg][ch][iLayer].empty() ? true : false;
      secondCondition = false;
      if (secondRing > -1) {
        secondCondition = !allRechits[ec][st][secondRing][ch][iLayer].empty() ? true : false;
      }
      float stripAngle = 99999.;
      std::vector<float> posXY(2);
      bool oneSegment = false;
      if (1 == allSegments[ec][st][rg][ch].size() + secondSize) {
        oneSegment = true;
        const BoundPlane bp = cscChamber->layer(iLayer + 1)->surface();
        linearExtrapolation(globalPos, globalDir, bp.position().z(), posXY);
        GlobalPoint gp_extrapol(posXY.at(0), posXY.at(1), bp.position().z());
        const LocalPoint lp_extrapol = cscChamber->layer(iLayer + 1)->toLocal(gp_extrapol);
        posXY.at(0) = lp_extrapol.x();
        posXY.at(1) = lp_extrapol.y();
        int nearestStrip = cscChamber->layer(iLayer + 1)->geometry()->nearestStrip(lp_extrapol);
        stripAngle = cscChamber->layer(iLayer + 1)->geometry()->stripAngle(nearestStrip) - M_PI / 2.;
      }
      if (firstCondition || secondCondition) {
        ChHist[ec][st][rg][ch].EfficientRechits_inSegment->Fill(iLayer + 1);
        if (oneSegment) {
          ChHist[ec][st][rg][ch].Y_EfficientRecHits_inSegment[iLayer]->Fill(posXY.at(1));
          ChHist[ec][st][rg][ch].Phi_EfficientRecHits_inSegment[iLayer]->Fill(stripAngle);
        }
      } else {
        if (oneSegment) {
          ChHist[ec][st][rg][ch].Y_InefficientRecHits_inSegment[iLayer]->Fill(posXY.at(1));
          ChHist[ec][st][rg][ch].Phi_InefficientRecHits_inSegment[iLayer]->Fill(stripAngle);
        }
      }
    }
  } else {
    StHist[ec][st].InefficientSegments_XY->Fill(ftsChamber.position().x(), ftsChamber.position().y());
    if (printalot) {
      std::cout << "missing segment " << std::endl;
      printf("\t\tendcap/station/ring/chamber: %i/%i/%i/%i\n", id.endcap(), id.station(), id.ring(), id.chamber());
      std::cout << " fts.position() = " << ftsChamber.position() << std::endl;
    }
  }
  // Normalization
  ChHist[ec][st][rg][ch].EfficientRechits_good->Fill(8);
  if (allSegments[ec][st][rg][ch].size() + secondSize < 2) {
    StHist[ec][st].AllSegments_eta->Fill(fabs(ftsChamber.position().eta()));
  }
  ChHist[ec][st][rg][id.chamber() - FirstCh].EfficientRechits_inSegment->Fill(9);

  return true;
}
//
void CSCEfficiency::returnTypes(CSCDetId &id, int &ec, int &st, int &rg, int &ch, int &secondRing) {
  ec = id.endcap() - 1;
  st = id.station() - 1;
  rg = id.ring() - 1;
  secondRing = -1;
  if (1 == id.station() && (4 == id.ring() || 1 == id.ring())) {
    rg = 0;
    secondRing = 3;
  }
  ch = id.chamber() - FirstCh;
}

//
void CSCEfficiency::getFromFTS(const FreeTrajectoryState &fts,
                               CLHEP::Hep3Vector &p3,
                               CLHEP::Hep3Vector &r3,
                               int &charge,
                               AlgebraicSymMatrix66 &cov) {
  GlobalVector p3GV = fts.momentum();
  GlobalPoint r3GP = fts.position();

  p3.set(p3GV.x(), p3GV.y(), p3GV.z());
  r3.set(r3GP.x(), r3GP.y(), r3GP.z());

  charge = fts.charge();
  cov = fts.hasError() ? fts.cartesianError().matrix() : AlgebraicSymMatrix66();
}

FreeTrajectoryState CSCEfficiency::getFromCLHEP(const CLHEP::Hep3Vector &p3,
                                                const CLHEP::Hep3Vector &r3,
                                                int charge,
                                                const AlgebraicSymMatrix66 &cov,
                                                const MagneticField *field) {
  GlobalVector p3GV(p3.x(), p3.y(), p3.z());
  GlobalPoint r3GP(r3.x(), r3.y(), r3.z());
  GlobalTrajectoryParameters tPars(r3GP, p3GV, charge, field);

  CartesianTrajectoryError tCov(cov);

  return cov.kRows == 6 ? FreeTrajectoryState(tPars, tCov) : FreeTrajectoryState(tPars);
}

void CSCEfficiency::linearExtrapolation(GlobalPoint initialPosition,
                                        GlobalVector initialDirection,
                                        float zSurface,
                                        std::vector<float> &posZY) {
  double paramLine = lineParameter(initialPosition.z(), zSurface, initialDirection.z());
  double xPosition = extrapolate1D(initialPosition.x(), initialDirection.x(), paramLine);
  double yPosition = extrapolate1D(initialPosition.y(), initialDirection.y(), paramLine);
  posZY.clear();
  posZY.push_back(xPosition);
  posZY.push_back(yPosition);
}
//
double CSCEfficiency::extrapolate1D(double initPosition, double initDirection, double parameterOfTheLine) {
  double extrapolatedPosition = initPosition + initDirection * parameterOfTheLine;
  return extrapolatedPosition;
}
//
double CSCEfficiency::lineParameter(double initZPosition, double destZPosition, double initZDirection) {
  double paramLine = (destZPosition - initZPosition) / initZDirection;
  return paramLine;
}
//
void CSCEfficiency::chooseDirection(CLHEP::Hep3Vector &innerPosition, CLHEP::Hep3Vector &outerPosition) {
  //---- Be careful with trigger conditions too
  if (!isIPdata) {
    float dy = outerPosition.y() - innerPosition.y();
    float dz = outerPosition.z() - innerPosition.z();
    if (isBeamdata) {
      if (dz > 0) {
        alongZ = true;
      } else {
        alongZ = false;
      }
    } else {  //cosmics
      if (dy / dz > 0) {
        alongZ = false;
      } else {
        alongZ = true;
      }
    }
  }
}
//
const Propagator *CSCEfficiency::propagator(std::string propagatorName) const {
  return &*theService->propagator(propagatorName);
}

//
TrajectoryStateOnSurface CSCEfficiency::propagate(FreeTrajectoryState &ftsStart, const BoundPlane &bpDest) {
  TrajectoryStateOnSurface tSOSDest;
  std::string propagatorName;
  /*
// it would work if cosmic muons had properly assigned direction...
  bool dzPositive = bpDest.position().z() - ftsStart.position().z() > 0 ? true : false;
 //---- Be careful with trigger conditions too
  if(!isIPdata){
    bool rightDirection = !(alongZ^dzPositive);
    if(rightDirection){
      if(printalot) std::cout<<" propagate along momentum"<<std::endl;
      propagatorName = "SteppingHelixPropagatorAlong";
    }
    else{
      if(printalot) std::cout<<" propagate opposite momentum"<<std::endl;
      propagatorName = "SteppingHelixPropagatorOpposite";
    }
  }
  else{
    if(printalot) std::cout<<" propagate any (momentum)"<<std::endl;
    propagatorName = "SteppingHelixPropagatorAny";
  }
*/
  propagatorName = "SteppingHelixPropagatorAny";
  tSOSDest = propagator(propagatorName)->propagate(ftsStart, bpDest);
  return tSOSDest;
}
//
bool CSCEfficiency::applyTrigger(edm::Handle<edm::TriggerResults> &hltR, const edm::TriggerNames &triggerNames) {
  bool triggerPassed = true;
  std::vector<std::string> hlNames = triggerNames.triggerNames();
  pointToTriggers.clear();
  for (size_t imyT = 0; imyT < myTriggers.size(); ++imyT) {
    for (size_t iT = 0; iT < hlNames.size(); ++iT) {
      //std::cout<<" iT = "<<iT<<" hlNames[iT] = "<<hlNames[iT]<<
      //" : wasrun = "<<hltR->wasrun(iT)<<" accept = "<<
      //	 hltR->accept(iT)<<" !error = "<<
      //	!hltR->error(iT)<<std::endl;
      if (!imyT) {
        if (hltR->wasrun(iT) && hltR->accept(iT) && !hltR->error(iT)) {
          TriggersFired->Fill(iT);
        }
      }
      if (hlNames[iT] == myTriggers[imyT]) {
        pointToTriggers.push_back(iT);
        if (imyT) {
          break;
        }
      }
    }
  }
  if (pointToTriggers.size() != myTriggers.size()) {
    pointToTriggers.clear();
    if (printalot) {
      std::cout << " Not all trigger names found - all trigger specifications will be ignored. Check your cfg file!"
                << std::endl;
    }
  } else {
    if (!pointToTriggers.empty()) {
      if (printalot) {
        std::cout << "The following triggers will be required in the event: " << std::endl;
        for (size_t imyT = 0; imyT < pointToTriggers.size(); ++imyT) {
          std::cout << "  " << hlNames[pointToTriggers[imyT]];
        }
        std::cout << std::endl;
        std::cout << " in condition (AND/OR) : " << !andOr << "/" << andOr << std::endl;
      }
    }
  }

  if (hltR.isValid()) {
    if (pointToTriggers.empty()) {
      if (printalot) {
        std::cout
            << " No triggers specified in the configuration or all ignored - no trigger information will be considered"
            << std::endl;
      }
    }
    for (size_t imyT = 0; imyT < pointToTriggers.size(); ++imyT) {
      if (hltR->wasrun(pointToTriggers[imyT]) && hltR->accept(pointToTriggers[imyT]) &&
          !hltR->error(pointToTriggers[imyT])) {
        triggerPassed = true;
        if (andOr) {
          break;
        }
      } else {
        triggerPassed = false;
        if (!andOr) {
          triggerPassed = false;
          break;
        }
      }
    }
  } else {
    if (printalot) {
      std::cout << " TriggerResults handle returns invalid state?! No trigger information will be considered"
                << std::endl;
    }
  }
  if (printalot) {
    std::cout << " Trigger passed: " << triggerPassed << std::endl;
  }
  return triggerPassed;
}
//

// Constructor
CSCEfficiency::CSCEfficiency(const edm::ParameterSet &pset) {
  // const float Xmin = -70;
  //const float Xmax = 70;
  //const int nXbins = int(4.*(Xmax - Xmin));
  const float Ymin = -165;
  const float Ymax = 165;
  const int nYbins = int((Ymax - Ymin) / 2);
  const float Layer_min = -0.5;
  const float Layer_max = 9.5;
  const int nLayer_bins = int(Layer_max - Layer_min);
  //

  //---- Get the input parameters
  printout_NEvents = pset.getUntrackedParameter<unsigned int>("printout_NEvents", 0);
  rootFileName = pset.getUntrackedParameter<string>("rootFileName", "cscHists.root");

  isData = pset.getUntrackedParameter<bool>("runOnData", true);                             //
  isIPdata = pset.getUntrackedParameter<bool>("IPdata", false);                             //
  isBeamdata = pset.getUntrackedParameter<bool>("Beamdata", false);                         //
  getAbsoluteEfficiency = pset.getUntrackedParameter<bool>("getAbsoluteEfficiency", true);  //
  useDigis = pset.getUntrackedParameter<bool>("useDigis", true);                            //
  distanceFromDeadZone = pset.getUntrackedParameter<double>("distanceFromDeadZone", 10.);   //
  minP = pset.getUntrackedParameter<double>("minP", 20.);                                   //
  maxP = pset.getUntrackedParameter<double>("maxP", 100.);                                  //
  maxNormChi2 = pset.getUntrackedParameter<double>("maxNormChi2", 3.);                      //
  minTrackHits = pset.getUntrackedParameter<unsigned int>("minTrackHits", 10);              //

  applyIPangleCuts = pset.getUntrackedParameter<bool>("applyIPangleCuts", false);  //
  local_DY_DZ_Max = pset.getUntrackedParameter<double>("local_DY_DZ_Max", -0.1);   //
  local_DY_DZ_Min = pset.getUntrackedParameter<double>("local_DY_DZ_Min", -0.8);   //
  local_DX_DZ_Max = pset.getUntrackedParameter<double>("local_DX_DZ_Max", 0.2);    //

  sd_token = consumes<CSCStripDigiCollection>(pset.getParameter<edm::InputTag>("stripDigiTag"));
  wd_token = consumes<CSCWireDigiCollection>(pset.getParameter<edm::InputTag>("wireDigiTag"));
  al_token = consumes<CSCALCTDigiCollection>(pset.getParameter<edm::InputTag>("alctDigiTag"));
  cl_token = consumes<CSCCLCTDigiCollection>(pset.getParameter<edm::InputTag>("clctDigiTag"));
  co_token = consumes<CSCCorrelatedLCTDigiCollection>(pset.getParameter<edm::InputTag>("corrlctDigiTag"));
  rh_token = consumes<CSCRecHit2DCollection>(pset.getParameter<edm::InputTag>("rechitTag"));
  se_token = consumes<CSCSegmentCollection>(pset.getParameter<edm::InputTag>("segmentTag"));
  tk_token = consumes<edm::View<reco::Track> >(pset.getParameter<edm::InputTag>("tracksTag"));
  sh_token = consumes<edm::PSimHitContainer>(pset.getParameter<edm::InputTag>("simHitTag"));

  edm::ParameterSet serviceParameters = pset.getParameter<edm::ParameterSet>("ServiceParameters");
  // maybe use the service for getting magnetic field, propagators, etc. ...
  theService = new MuonServiceProxy(serviceParameters, consumesCollector());

  // Trigger
  useTrigger = pset.getUntrackedParameter<bool>("useTrigger", false);

  ht_token = consumes<edm::TriggerResults>(pset.getParameter<edm::InputTag>("HLTriggerResults"));

  myTriggers = pset.getParameter<std::vector<std::string> >("myTriggers");
  andOr = pset.getUntrackedParameter<bool>("andOr");
  pointToTriggers.clear();

  //---- set counter to zero
  nEventsAnalyzed = 0;
  //---- set presence of magnetic field
  magField = true;
  //
  std::string Path = "AllChambers/";
  std::string FullName;
  //---- File with output histograms
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();
  //---- Book histograms for the analysis
  char SpecName[60];

  sprintf(SpecName, "DataFlow");
  DataFlow = new TH1F(SpecName, "Data flow;condition number;entries", 40, -0.5, 39.5);
  //
  sprintf(SpecName, "TriggersFired");
  TriggersFired = new TH1F(SpecName, "Triggers fired;trigger number;entries", 140, -0.5, 139.5);
  //
  int Chan = 50;
  float minChan = -0.5;
  float maxChan = 49.5;
  //
  sprintf(SpecName, "ALCTPerEvent");
  ALCTPerEvent = new TH1F(SpecName, "ALCTs per event;N digis;entries", Chan, minChan, maxChan);
  //
  sprintf(SpecName, "CLCTPerEvent");
  CLCTPerEvent = new TH1F(SpecName, "CLCTs per event;N digis;entries", Chan, minChan, maxChan);
  //
  sprintf(SpecName, "recHitsPerEvent");
  recHitsPerEvent = new TH1F(SpecName, "RecHits per event;N digis;entries", 150, -0.5, 149.5);
  //
  sprintf(SpecName, "segmentsPerEvent");
  segmentsPerEvent = new TH1F(SpecName, "segments per event;N digis;entries", Chan, minChan, maxChan);
  //
  //---- Book groups of histograms (for any chamber)

  map<std::string, bool>::iterator iter;
  for (int ec = 0; ec < 2; ++ec) {
    for (int st = 0; st < 4; ++st) {
      theFile->cd();
      sprintf(SpecName, "Stations__E%d_S%d", ec + 1, st + 1);
      theFile->mkdir(SpecName);
      theFile->cd(SpecName);

      //
      sprintf(SpecName, "segmentChi2_ndf_St%d", st + 1);
      StHist[ec][st].segmentChi2_ndf = new TH1F(SpecName, "Chi2/ndf of a segment;chi2/ndf;entries", 100, 0., 20.);
      //
      sprintf(SpecName, "hitsInSegment_St%d", st + 1);
      StHist[ec][st].hitsInSegment = new TH1F(SpecName, "Number of hits in a segment;nHits;entries", 7, -0.5, 6.5);
      //
      Chan = 170;
      minChan = 0.85;
      maxChan = 2.55;
      //
      sprintf(SpecName, "AllSegments_eta_St%d", st + 1);
      StHist[ec][st].AllSegments_eta = new TH1F(SpecName, "All segments in eta;eta;entries", Chan, minChan, maxChan);
      //
      sprintf(SpecName, "EfficientSegments_eta_St%d", st + 1);
      StHist[ec][st].EfficientSegments_eta =
          new TH1F(SpecName, "Efficient segments in eta;eta;entries", Chan, minChan, maxChan);
      //
      sprintf(SpecName, "ResidualSegments_St%d", st + 1);
      StHist[ec][st].ResidualSegments = new TH1F(SpecName, "Residual (segments);residual,cm;entries", 75, 0., 15.);
      //
      Chan = 200;
      minChan = -800.;
      maxChan = 800.;
      int Chan2 = 200;
      float minChan2 = -800.;
      float maxChan2 = 800.;

      sprintf(SpecName, "EfficientSegments_XY_St%d", st + 1);
      StHist[ec][st].EfficientSegments_XY =
          new TH2F(SpecName, "Efficient segments in XY;X;Y", Chan, minChan, maxChan, Chan2, minChan2, maxChan2);
      sprintf(SpecName, "InefficientSegments_XY_St%d", st + 1);
      StHist[ec][st].InefficientSegments_XY =
          new TH2F(SpecName, "Inefficient segments in XY;X;Y", Chan, minChan, maxChan, Chan2, minChan2, maxChan2);
      //
      Chan = 80;
      minChan = 0;
      maxChan = 3.2;
      sprintf(SpecName, "EfficientALCT_momTheta_St%d", st + 1);
      StHist[ec][st].EfficientALCT_momTheta =
          new TH1F(SpecName, "Efficient ALCT in theta (momentum);theta, rad;entries", Chan, minChan, maxChan);
      //
      sprintf(SpecName, "InefficientALCT_momTheta_St%d", st + 1);
      StHist[ec][st].InefficientALCT_momTheta =
          new TH1F(SpecName, "Inefficient ALCT in theta (momentum);theta, rad;entries", Chan, minChan, maxChan);
      //
      Chan = 160;
      minChan = -3.2;
      maxChan = 3.2;
      sprintf(SpecName, "EfficientCLCT_momPhi_St%d", st + 1);
      StHist[ec][st].EfficientCLCT_momPhi =
          new TH1F(SpecName, "Efficient CLCT in phi (momentum);phi, rad;entries", Chan, minChan, maxChan);
      //
      sprintf(SpecName, "InefficientCLCT_momPhi_St%d", st + 1);
      StHist[ec][st].InefficientCLCT_momPhi =
          new TH1F(SpecName, "Inefficient CLCT in phi (momentum);phi, rad;entries", Chan, minChan, maxChan);
      //
      theFile->cd();
      for (int rg = 0; rg < 3; ++rg) {
        if (0 != st && rg > 1) {
          continue;
        } else if (1 == rg && 3 == st) {
          continue;
        }
        for (int iChamber = FirstCh; iChamber < FirstCh + NumCh; iChamber++) {
          if (0 != st && 0 == rg && iChamber > 18) {
            continue;
          }
          theFile->cd();
          sprintf(SpecName, "Chambers__E%d_S%d_R%d_Chamber_%d", ec + 1, st + 1, rg + 1, iChamber);
          theFile->mkdir(SpecName);
          theFile->cd(SpecName);
          //

          sprintf(SpecName, "EfficientRechits_inSegment_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientRechits_inSegment = new TH1F(
              SpecName, "Existing RecHit given a segment;layers (1-6);entries", nLayer_bins, Layer_min, Layer_max);
          //
          sprintf(SpecName, "InefficientSingleHits_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].InefficientSingleHits = new TH1F(
              SpecName, "Single RecHits not in the segment;layers (1-6);entries ", nLayer_bins, Layer_min, Layer_max);
          //
          sprintf(SpecName, "AllSingleHits_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].AllSingleHits = new TH1F(
              SpecName, "Single RecHits given a segment; layers (1-6);entries", nLayer_bins, Layer_min, Layer_max);
          //
          sprintf(SpecName, "digiAppearanceCount_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].digiAppearanceCount =
              new TH1F(SpecName,
                       "Digi appearance (no-yes): segment(0,1), ALCT(2,3), CLCT(4,5), CorrLCT(6,7); digi type;entries",
                       8,
                       -0.5,
                       7.5);
          //
          Chan = 100;
          minChan = -1.1;
          maxChan = 0.9;
          sprintf(SpecName, "EfficientALCT_dydz_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientALCT_dydz =
              new TH1F(SpecName, "Efficient ALCT; local dy/dz (ME 3 and 4 flipped);entries", Chan, minChan, maxChan);
          //
          sprintf(SpecName, "InefficientALCT_dydz_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].InefficientALCT_dydz =
              new TH1F(SpecName, "Inefficient ALCT; local dy/dz (ME 3 and 4 flipped);entries", Chan, minChan, maxChan);
          //
          Chan = 100;
          minChan = -1.;
          maxChan = 1.0;
          sprintf(SpecName, "EfficientCLCT_dxdz_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientCLCT_dxdz =
              new TH1F(SpecName, "Efficient CLCT; local dxdz;entries", Chan, minChan, maxChan);
          //
          sprintf(SpecName, "InefficientCLCT_dxdz_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].InefficientCLCT_dxdz =
              new TH1F(SpecName, "Inefficient CLCT; local dxdz;entries", Chan, minChan, maxChan);
          //
          sprintf(SpecName, "EfficientRechits_good_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientRechits_good = new TH1F(
              SpecName, "Existing RecHit - sensitive area only;layers (1-6);entries", nLayer_bins, Layer_min, Layer_max);
          //
          sprintf(SpecName, "EfficientStrips_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientStrips =
              new TH1F(SpecName, "Existing strip;layer (1-6); entries", nLayer_bins, Layer_min, Layer_max);
          //
          sprintf(SpecName, "EfficientWireGroups_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientWireGroups =
              new TH1F(SpecName, "Existing WireGroups;layer (1-6); entries ", nLayer_bins, Layer_min, Layer_max);
          //
          sprintf(SpecName, "StripWiresCorrelations_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].StripWiresCorrelations =
              new TH1F(SpecName, "StripWire correlations;; entries ", 5, 0.5, 5.5);
          //
          Chan = 80;
          minChan = 0;
          maxChan = 3.2;
          sprintf(SpecName, "NoWires_momTheta_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].NoWires_momTheta =
              new TH1F(SpecName,
                       "No wires (all strips present) - in theta (momentum);theta, rad;entries",
                       Chan,
                       minChan,
                       maxChan);
          //
          Chan = 160;
          minChan = -3.2;
          maxChan = 3.2;
          sprintf(SpecName, "NoStrips_momPhi_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].NoStrips_momPhi = new TH1F(
              SpecName, "No strips (all wires present) - in phi (momentum);phi, rad;entries", Chan, minChan, maxChan);
          //
          for (int iLayer = 0; iLayer < 6; iLayer++) {
            sprintf(SpecName, "Y_InefficientRecHits_inSegment_Ch%d_L%d", iChamber, iLayer);
            ChHist[ec][st][rg][iChamber - FirstCh].Y_InefficientRecHits_inSegment.push_back(
                new TH1F(SpecName,
                         "Missing RecHit/layer in a segment (local system, whole chamber);Y, cm; entries",
                         nYbins,
                         Ymin,
                         Ymax));
            //
            sprintf(SpecName, "Y_EfficientRecHits_inSegment_Ch%d_L%d", iChamber, iLayer);
            ChHist[ec][st][rg][iChamber - FirstCh].Y_EfficientRecHits_inSegment.push_back(
                new TH1F(SpecName,
                         "Efficient (extrapolated from the segment) RecHit/layer in a segment (local system, whole "
                         "chamber);Y, cm; entries",
                         nYbins,
                         Ymin,
                         Ymax));
            //
            Chan = 200;
            minChan = -0.2;
            maxChan = 0.2;
            sprintf(SpecName, "Phi_InefficientRecHits_inSegment_Ch%d_L%d", iChamber, iLayer);
            ChHist[ec][st][rg][iChamber - FirstCh].Phi_InefficientRecHits_inSegment.push_back(
                new TH1F(SpecName,
                         "Missing RecHit/layer in a segment (local system, whole chamber);Phi, rad; entries",
                         Chan,
                         minChan,
                         maxChan));
            //
            sprintf(SpecName, "Phi_EfficientRecHits_inSegment_Ch%d_L%d", iChamber, iLayer);
            ChHist[ec][st][rg][iChamber - FirstCh].Phi_EfficientRecHits_inSegment.push_back(
                new TH1F(SpecName,
                         "Efficient (extrapolated from the segment) in a segment (local system, whole chamber);Phi, "
                         "rad; entries",
                         Chan,
                         minChan,
                         maxChan));
          }
          //
          sprintf(SpecName, "Sim_Rechits_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].SimRechits =
              new TH1F(SpecName, "Existing RecHit (Sim);layers (1-6);entries", nLayer_bins, Layer_min, Layer_max);
          //
          sprintf(SpecName, "Sim_Simhits_Ch%d", iChamber);
          ChHist[ec][st][rg][iChamber - FirstCh].SimSimhits =
              new TH1F(SpecName, "Existing SimHit (Sim);layers (1-6);entries", nLayer_bins, Layer_min, Layer_max);
          //
          /*
	  sprintf(SpecName,"Sim_Rechits_each_Ch%d",iChamber);
	  ChHist[ec][st][rg][iChamber-FirstCh].SimRechits_each = 
	    new TH1F(SpecName,"Existing RecHit (Sim), each;layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
	  //
	  sprintf(SpecName,"Sim_Simhits_each_Ch%d",iChamber);
	  ChHist[ec][st][rg][iChamber-FirstCh].SimSimhits_each = 
	    new TH1F(SpecName,"Existing SimHit (Sim), each;layers (1-6);entries",nLayer_bins,Layer_min,Layer_max);
	  */
          theFile->cd();
        }
      }
    }
  }
}

// Destructor
CSCEfficiency::~CSCEfficiency() {
  if (theService)
    delete theService;
  // Write the histos to a file
  theFile->cd();
  //
  char SpecName[60];
  std::vector<float> bins, Efficiency, EffError;
  std::vector<float> eff(2);

  //---- loop over chambers
  std::map<std::string, bool> chamberTypes;
  chamberTypes["ME11"] = false;
  chamberTypes["ME12"] = false;
  chamberTypes["ME13"] = false;
  chamberTypes["ME21"] = false;
  chamberTypes["ME22"] = false;
  chamberTypes["ME31"] = false;
  chamberTypes["ME32"] = false;
  chamberTypes["ME41"] = false;

  map<std::string, bool>::iterator iter;
  std::cout << " Writing proper histogram structure (patience)..." << std::endl;
  for (int ec = 0; ec < 2; ++ec) {
    for (int st = 0; st < 4; ++st) {
      snprintf(SpecName, sizeof(SpecName), "Stations__E%d_S%d", ec + 1, st + 1);
      theFile->cd(SpecName);
      StHist[ec][st].segmentChi2_ndf->Write();
      StHist[ec][st].hitsInSegment->Write();
      StHist[ec][st].AllSegments_eta->Write();
      StHist[ec][st].EfficientSegments_eta->Write();
      StHist[ec][st].ResidualSegments->Write();
      StHist[ec][st].EfficientSegments_XY->Write();
      StHist[ec][st].InefficientSegments_XY->Write();
      StHist[ec][st].EfficientALCT_momTheta->Write();
      StHist[ec][st].InefficientALCT_momTheta->Write();
      StHist[ec][st].EfficientCLCT_momPhi->Write();
      StHist[ec][st].InefficientCLCT_momPhi->Write();
      for (int rg = 0; rg < 3; ++rg) {
        if (0 != st && rg > 1) {
          continue;
        } else if (1 == rg && 3 == st) {
          continue;
        }
        for (int iChamber = FirstCh; iChamber < FirstCh + NumCh; iChamber++) {
          if (0 != st && 0 == rg && iChamber > 18) {
            continue;
          }
          snprintf(SpecName, sizeof(SpecName), "Chambers__E%d_S%d_R%d_Chamber_%d", ec + 1, st + 1, rg + 1, iChamber);
          theFile->cd(SpecName);

          ChHist[ec][st][rg][iChamber - FirstCh].EfficientRechits_inSegment->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].AllSingleHits->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].digiAppearanceCount->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientALCT_dydz->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].InefficientALCT_dydz->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientCLCT_dxdz->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].InefficientCLCT_dxdz->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].InefficientSingleHits->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientRechits_good->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientStrips->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].StripWiresCorrelations->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].NoWires_momTheta->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].NoStrips_momPhi->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].EfficientWireGroups->Write();
          for (unsigned int iLayer = 0; iLayer < 6; iLayer++) {
            ChHist[ec][st][rg][iChamber - FirstCh].Y_InefficientRecHits_inSegment[iLayer]->Write();
            ChHist[ec][st][rg][iChamber - FirstCh].Y_EfficientRecHits_inSegment[iLayer]->Write();
            ChHist[ec][st][rg][iChamber - FirstCh].Phi_InefficientRecHits_inSegment[iLayer]->Write();
            ChHist[ec][st][rg][iChamber - FirstCh].Phi_EfficientRecHits_inSegment[iLayer]->Write();
          }
          ChHist[ec][st][rg][iChamber - FirstCh].SimRechits->Write();
          ChHist[ec][st][rg][iChamber - FirstCh].SimSimhits->Write();
          /*
	  ChHist[ec][st][rg][iChamber-FirstCh].SimRechits_each->Write();
	  ChHist[ec][st][rg][iChamber-FirstCh].SimSimhits_each->Write();
	  */
          //
          theFile->cd(SpecName);
          theFile->cd();
        }
      }
    }
  }
  //
  snprintf(SpecName, sizeof(SpecName), "AllChambers");
  theFile->mkdir(SpecName);
  theFile->cd(SpecName);
  DataFlow->Write();
  TriggersFired->Write();
  ALCTPerEvent->Write();
  CLCTPerEvent->Write();
  recHitsPerEvent->Write();
  segmentsPerEvent->Write();
  //
  theFile->cd(SpecName);
  //---- Close the file
  theFile->Close();
}

// ------------ method called once each job just before starting event loop  ------------
void CSCEfficiency::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void CSCEfficiency::endJob() {}

DEFINE_FWK_MODULE(CSCEfficiency);
