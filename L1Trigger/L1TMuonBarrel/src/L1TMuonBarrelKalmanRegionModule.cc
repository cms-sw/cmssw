#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanRegionModule.h"

L1TMuonBarrelKalmanRegionModule::L1TMuonBarrelKalmanRegionModule(const edm::ParameterSet& iConfig,
                                                                 int wheel,
                                                                 int sector)
    : verbose_(iConfig.getParameter<int>("verbose")), sector_(sector), wheel_(wheel) {
  if (sector == 11) {
    nextSector_ = 0;
    previousSector_ = 10;
  } else if (sector == 0) {
    nextSector_ = 1;
    previousSector_ = 11;
  } else {
    nextSector_ = sector + 1;
    previousSector_ = sector - 1;
  }

  switch (wheel) {
    case -2:
      nextWheel_ = -1;
      break;

    case -1:
      nextWheel_ = 0;
      break;

    case 0:
      nextWheel_ = 999;
      break;

    case 1:
      nextWheel_ = 0;
      break;

    case 2:
      nextWheel_ = 1;
      break;

    default:
      nextWheel_ = 999;
      break;
  }
}

L1TMuonBarrelKalmanRegionModule::~L1TMuonBarrelKalmanRegionModule() {}

L1MuKBMTrackCollection L1TMuonBarrelKalmanRegionModule::process(L1TMuonBarrelKalmanAlgo* trackMaker,
                                                                const L1MuKBMTCombinedStubRefVector& stubsAll,
                                                                int bx) {
  L1MuKBMTCombinedStubRefVector stubs;
  L1MuKBMTCombinedStubRefVector seeds;
  L1MuKBMTrackCollection pretracks2;
  L1MuKBMTrackCollection pretracks3;
  L1MuKBMTrackCollection pretracks4;
  for (const auto& stub : stubsAll) {
    if (stub->bxNum() != bx)
      continue;

    if ((stub->scNum() == nextSector_ && stub->phi() >= -112) ||
        (stub->scNum() == previousSector_ && stub->phi() <= 111))
      continue;

    if (stub->whNum() == wheel_ && stub->scNum() == sector_) {
      seeds.push_back(stub);
      stubs.push_back(stub);
    } else if (stub->whNum() == wheel_ && (stub->scNum() == nextSector_ || stub->scNum() == previousSector_)) {
      stubs.push_back(stub);
    } else if (stub->whNum() == nextWheel_ &&
               (stub->scNum() == nextSector_ || stub->scNum() == previousSector_ || stub->scNum() == sector_)) {
      stubs.push_back(stub);
    }
  }

  //Sort the seeds by tag so that the emulator is aligned like the firmware

  SeedSorter sorter;
  if (seeds.size() > 1) {
    std::sort(seeds.begin(), seeds.end(), sorter);
  }

  for (const auto& seed : seeds) {
    std::pair<bool, L1MuKBMTrack> trackInfo = trackMaker->chain(seed, stubs);
    //printf("Kalman Track %d valid=%d tag=%d rank=%d charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f pattern=%d\n",seed->stNum(),trackInfo.first, trackInfo.second.stubs()[0]->tag(),trackInfo.second.rank(),trackInfo.second.charge(),trackInfo.second.pt(),trackInfo.second.eta(),trackInfo.second.phi(),trackInfo.second.curvatureAtVertex(),trackInfo.second.curvatureAtMuon(),int(trackInfo.second.stubs().size()),trackInfo.second.approxChi2(),trackInfo.second.pt(),trackInfo.second.ptUnconstrained(),trackInfo.second.hitPattern());

    L1MuKBMTrack nullTrack(seed, seed->phi(), 8 * seed->phiB());
    nullTrack.setPtEtaPhi(0, 0, 0);
    nullTrack.setRank(0);
    if (trackInfo.first) {
      if (seed->stNum() == 2)
        pretracks2.push_back(trackInfo.second);
      if (seed->stNum() == 3)
        pretracks3.push_back(trackInfo.second);
      if (seed->stNum() == 4)
        pretracks4.push_back(trackInfo.second);
    } else {
      if (seed->stNum() == 2)
        pretracks2.push_back(nullTrack);
      if (seed->stNum() == 3)
        pretracks3.push_back(nullTrack);
      if (seed->stNum() == 4)
        pretracks4.push_back(nullTrack);
    }
  }
  L1MuKBMTrack nullTrack;
  nullTrack.setPtEtaPhi(0, 0, 0);
  nullTrack.setRank(0);
  // All pretracks must have trackL and trackH like firmware
  // Swap trackH and trackL for seeds 2/3 to mimic firmware
  if (pretracks2.size() < 2) {
    if (pretracks2.empty()) {  // if no tracks, set trackH and trackL to null
      pretracks2.push_back(nullTrack);
      pretracks2.push_back(nullTrack);
    } else {  // otherwise add nulltrack for trackH or trackL
      if (pretracks2[0].stubs()[0]->tag() == 0)
        pretracks2.push_back(nullTrack);
      else
        pretracks2.insert(pretracks2.begin(), nullTrack);
    }
  }
  std::swap(pretracks2[0], pretracks2[1]);

  if (pretracks3.size() < 2) {
    if (pretracks3.empty()) {
      pretracks3.push_back(nullTrack);
      pretracks3.push_back(nullTrack);
    } else {
      if (pretracks3[0].stubs()[0]->tag() == 0)
        pretracks3.push_back(nullTrack);
      else
        pretracks3.insert(pretracks3.begin(), nullTrack);
    }
  }
  std::swap(pretracks3[0], pretracks3[1]);

  if (pretracks4.size() < 2) {
    if (pretracks4.empty()) {
      pretracks4.push_back(nullTrack);
      pretracks4.push_back(nullTrack);
    } else {
      if (pretracks4[0].stubs()[0]->tag() == 0)
        pretracks4.push_back(nullTrack);
      else
        pretracks4.insert(pretracks4.begin(), nullTrack);
    }
  }

  /*
  printf("SEED 2\n");
  for (const auto& track1 :pretracks2){
    printf("   Kalman Track rank=%d charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f pattern=%d\n",track1.rank(),track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained(),track1.hitPattern());
  }
  printf("SEED 3\n");
  for (const auto& track1 :pretracks3){
    printf("   Kalman Track rank=%d charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f pattern=%d\n",track1.rank(),track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained(),track1.hitPattern());
  }
  printf("SEED 4\n");
  for (const auto& track1 :pretracks4){
    printf("   Kalman Track rank=%d charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f pattern=%d\n",track1.rank(),track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained(),track1.hitPattern());
  }
  */

  L1MuKBMTrackCollection out = cleanRegion(pretracks2, pretracks3, pretracks4);
  if (verbose_) {
    printf(" -----Sector Processor Kalman Tracks-----\n");
    for (const auto& track1 : out)
      printf("Kalman Track charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f\n",
             track1.charge(),
             track1.pt(),
             track1.eta(),
             track1.phi(),
             track1.curvatureAtVertex(),
             track1.curvatureAtMuon(),
             int(track1.stubs().size()),
             track1.approxChi2(),
             track1.pt(),
             track1.ptUnconstrained());
  }

  return out;
}

L1MuKBMTrackCollection L1TMuonBarrelKalmanRegionModule::selfClean(const L1MuKBMTrackCollection& tracks) {
  L1MuKBMTrackCollection out;

  for (uint i = 0; i < tracks.size(); ++i) {
    //bool keep = true;
    L1MuKBMTrack temp = tracks[i];
    for (uint j = 0; j < tracks.size(); ++j) {
      if (i == j)
        continue;

      if (tracks[i].overlapTrack(tracks[j])) {
        if (tracks[i].rank() < tracks[j].rank()) {
          //keep = false;
          temp.setPtEtaPhi(0, 0, 0);
          temp.setRank(0);
        } else if (tracks[i].rank() == tracks[j].rank()) {  //if same rank prefer seed that is high
          if (!tracks[j].stubs()[0]->tag()) {
            //keep = false;
            temp.setPtEtaPhi(0, 0, 0);
            temp.setRank(0);
          }
        }
      }
    }
    out.push_back(temp);
    //if (keep)
    //out.push_back(tracks[i]);
  }

  return out;
}

L1MuKBMTrackCollection L1TMuonBarrelKalmanRegionModule::cleanHigher(const L1MuKBMTrackCollection& tracks1,
                                                                    const L1MuKBMTrackCollection& tracks2) {
  L1MuKBMTrackCollection out;

  for (uint i = 0; i < tracks1.size(); ++i) {
    //bool keep = true;
    L1MuKBMTrack temp = tracks1[i];
    for (uint j = 0; j < tracks2.size(); ++j) {
      if (tracks1[i].overlapTrack(tracks2[j])) {
        if (tracks1[i].rank() <= tracks2[j].rank()) {
          //keep = false;
          temp.setPtEtaPhi(0, 0, 0);
          temp.setRank(0);
        }
      }
    }
    out.push_back(temp);
    //if (keep)
    //  out.push_back(tracks1[i]);
  }

  return out;
}

L1MuKBMTrackCollection L1TMuonBarrelKalmanRegionModule::cleanLower(const L1MuKBMTrackCollection& tracks1,
                                                                   const L1MuKBMTrackCollection& tracks2) {
  L1MuKBMTrackCollection out;

  for (uint i = 0; i < tracks1.size(); ++i) {
    //bool keep = true;
    L1MuKBMTrack temp = tracks1[i];
    for (uint j = 0; j < tracks2.size(); ++j) {
      if (tracks1[i].overlapTrack(tracks2[j])) {
        if (tracks1[i].rank() < tracks2[j].rank()) {
          //keep = false;
          temp.setPtEtaPhi(0, 0, 0);
          temp.setRank(0);
        }
      }
    }
    out.push_back(temp);
    //if (keep)
    // out.push_back(tracks1[i]);
  }

  return out;
}

L1MuKBMTrackCollection L1TMuonBarrelKalmanRegionModule::sort4(const L1MuKBMTrackCollection& in) {
  L1MuKBMTrackCollection out;
  //partial sort like in firmwarE (bitonic)

  if (in.size() <= 2)
    return in;
  else if (in.size() == 3) {
    //Step 1
    L1MuKBMTrack s2_1;
    L1MuKBMTrack s2_3;
    if (in[2].pt() >= in[0].pt()) {
      s2_1 = in[2];
      s2_3 = in[0];
    } else {
      s2_1 = in[0];
      s2_3 = in[2];
    }

    L1MuKBMTrack s2_2 = in[1];
    //Step 2;
    L1MuKBMTrack s3_1 = s2_1;
    L1MuKBMTrack s3_2;
    L1MuKBMTrack s3_3;

    if (s2_3.pt() >= s2_2.pt()) {
      s3_2 = s2_3;
      s3_3 = s2_2;
    } else {
      s3_2 = s2_2;
      s3_3 = s2_3;
    }

    out.push_back(s3_1);
    out.push_back(s3_2);

  } else {
    //Step 1
    L1MuKBMTrack s2_1;
    L1MuKBMTrack s2_2;
    L1MuKBMTrack s2_3;
    L1MuKBMTrack s2_4;

    if (in[2].pt() >= in[0].pt()) {
      s2_1 = in[2];
      s2_3 = in[0];
    } else {
      s2_1 = in[0];
      s2_3 = in[2];
    }
    if (in[3].pt() >= in[1].pt()) {
      s2_2 = in[3];
      s2_4 = in[1];
    } else {
      s2_2 = in[1];
      s2_4 = in[3];
    }
    //Step 2
    L1MuKBMTrack s3_1;
    L1MuKBMTrack s3_2;
    L1MuKBMTrack s3_3;
    L1MuKBMTrack s3_4;

    if (s2_4.pt() >= s2_1.pt()) {
      s3_1 = s2_4;
      s3_4 = s2_1;
    } else {
      s3_1 = s2_1;
      s3_4 = s2_4;
    }

    if (s2_3.pt() >= s2_2.pt()) {
      s3_2 = s2_3;
      s3_3 = s2_2;
    } else {
      s3_2 = s2_2;
      s3_3 = s2_3;
    }

    out.push_back(s3_1);
    out.push_back(s3_2);
  }
  return out;
}

L1MuKBMTrackCollection L1TMuonBarrelKalmanRegionModule::cleanRegion(const L1MuKBMTrackCollection& tracks2,
                                                                    const L1MuKBMTrackCollection& tracks3,
                                                                    const L1MuKBMTrackCollection& tracks4) {
  L1MuKBMTrackCollection cleaned2 = selfClean(tracks2);
  L1MuKBMTrackCollection cleaned3 = selfClean(tracks3);

  L1MuKBMTrackCollection cleaned23 = cleanHigher(cleaned2, tracks3);
  L1MuKBMTrackCollection cleaned32 = cleanLower(cleaned3, tracks2);

  //merge 2,3
  L1MuKBMTrackCollection step1;
  if (!cleaned23.empty())
    step1.insert(step1.end(), cleaned23.begin(), cleaned23.end());
  if (!cleaned32.empty())
    step1.insert(step1.end(), cleaned32.begin(), cleaned32.end());

  //take the best 2
  L1MuKBMTrackCollection sorted23 = sort4(step1);

  //Now clean the tracks 4 between them
  L1MuKBMTrackCollection cleaned4 = selfClean(tracks4);

  //Now clean the 23 tracks from tracks4
  L1MuKBMTrackCollection cleanedSorted23 = cleanHigher(sorted23, tracks4);

  //Now clean the  tracks4 from sorted 23
  L1MuKBMTrackCollection cleanedSorted4 = cleanLower(cleaned4, sorted23);

  //Now merge all of those
  L1MuKBMTrackCollection step2;

  if (!cleanedSorted4.empty())
    step2.insert(step2.end(), cleanedSorted4.begin(), cleanedSorted4.end());
  if (!cleanedSorted23.empty())
    step2.insert(step2.end(), cleanedSorted23.begin(), cleanedSorted23.end());

  L1MuKBMTrackCollection out = sort4(step2);
  // Verbose statements:
  /*
  printf("tracks 1-4\n");
  for (const auto& track1 :step1)
    printf("   rank=%d charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f pattern=%d\n",track1.rank(),track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained(),track1.hitPattern());

  printf("sorted1\n");
  for (const auto& track1 :sorted23)
    printf("   rank=%d charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f pattern=%d\n",track1.rank(),track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained(),track1.hitPattern());

  printf("track 5-8\n");
  for (const auto& track1 :step2)
    printf("   rank=%d charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f pattern=%d\n",track1.rank(),track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained(),track1.hitPattern());
  
  printf("OUTPUT\n");
  for (const auto& track1 :out)
    printf("   rank=%d charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f pattern=%d\n",track1.rank(),track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained(),track1.hitPattern());
  */
  return out;
}
