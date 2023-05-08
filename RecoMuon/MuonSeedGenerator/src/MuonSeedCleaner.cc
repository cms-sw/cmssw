/**
 *  See header file for a description of this class.
 *  
 *  \author Shih-Chuan Kao, Dominique Fortin - UCR
 */

#include <RecoMuon/MuonSeedGenerator/src/MuonSeedCleaner.h>

// Data Formats
#include <DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h>
#include <DataFormats/TrajectorySeed/interface/TrajectorySeed.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegment.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4D.h>

// Geometry
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <TrackingTools/DetLayers/interface/DetLayer.h>
#include <RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h>
#include <RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h>
#include <RecoMuon/Records/interface/MuonRecoGeometryRecord.h>

// muon service
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>
#include <DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h>

// Framework
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <DataFormats/Common/interface/Handle.h>

// C++
#include <vector>
#include <deque>
#include <utility>

//typedef std::pair<double, TrajectorySeed> seedpr ;
//static bool ptDecreasing(const seedpr s1, const seedpr s2) { return ( s1.first > s2.first ); }
static bool lengthSorting(const TrajectorySeed& s1, const TrajectorySeed& s2) { return (s1.nHits() > s2.nHits()); }

/*
 * Constructor
 */
MuonSeedCleaner::MuonSeedCleaner(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) {
  // Local Debug flag
  debug = pset.getParameter<bool>("DebugMuonSeed");

  // muon service
  edm::ParameterSet serviceParameters = pset.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters, std::move(iC));
}

/*
 * Destructor
 */
MuonSeedCleaner::~MuonSeedCleaner() {
  if (theService)
    delete theService;
}

/*********************************** 
 *
 *  Seed Cleaner
 *
 ***********************************/

std::vector<TrajectorySeed> MuonSeedCleaner::seedCleaner(const edm::EventSetup& eventSetup,
                                                         std::vector<TrajectorySeed>& seeds) {
  theService->update(eventSetup);

  std::vector<TrajectorySeed> FinalSeeds;

  // group the seeds
  std::vector<SeedContainer> theCollection = GroupSeeds(seeds);

  // ckeck each group and pick the good one
  for (size_t i = 0; i < theCollection.size(); i++) {
    // separate seeds w/ more than 1 segments and w/ 1st layer segment information
    SeedContainer goodSeeds = SeedCandidates(theCollection[i], true);
    SeedContainer otherSeeds = SeedCandidates(theCollection[i], false);
    if (MomentumFilter(goodSeeds)) {
      //std::cout<<" == type1 "<<std::endl;
      TrajectorySeed bestSeed = Chi2LengthSelection(goodSeeds);
      FinalSeeds.push_back(bestSeed);

      GlobalPoint seedgp = SeedPosition(bestSeed);
      double eta = fabs(seedgp.eta());
      if (goodSeeds.size() > 2 && eta > 1.5) {
        TrajectorySeed anotherSeed = MoreRecHits(goodSeeds);
        FinalSeeds.push_back(anotherSeed);
      }
    } else if (MomentumFilter(otherSeeds)) {
      //std::cout<<" == type2 "<<std::endl;
      TrajectorySeed bestSeed = MoreRecHits(otherSeeds);
      FinalSeeds.push_back(bestSeed);

      GlobalPoint seedgp = SeedPosition(bestSeed);
      double eta = fabs(seedgp.eta());
      if (otherSeeds.size() > 2 && eta > 1.5) {
        TrajectorySeed anotherSeed = LeanHighMomentum(otherSeeds);
        FinalSeeds.push_back(anotherSeed);
      }
    } else {
      //std::cout<<" == type3 "<<std::endl;
      TrajectorySeed bestSeed = LeanHighMomentum(theCollection[i]);
      FinalSeeds.push_back(bestSeed);

      GlobalPoint seedgp = SeedPosition(bestSeed);
      double eta = fabs(seedgp.eta());
      if (theCollection.size() > 2 && eta > 1.5) {
        TrajectorySeed anotherSeed = BiggerCone(theCollection[i]);
        FinalSeeds.push_back(anotherSeed);
      }
    }
  }
  return FinalSeeds;
}

TrajectorySeed MuonSeedCleaner::Chi2LengthSelection(std::vector<TrajectorySeed>& seeds) {
  if (seeds.size() == 1)
    return seeds[0];

  int winner = 0;
  int moreHits = 0;
  double bestChi2 = 99999.;
  for (size_t i = 0; i < seeds.size(); i++) {
    // 1. fill out the Nchi2 of segments of the seed
    //GlobalVector mom = SeedMomentum( seeds[i] ); // temporary use for debugging
    //double pt = sqrt( (mom.x()*mom.x()) + (mom.y()*mom.y()) );
    //std::cout<<" > SEED"<<i<<"  pt:"<<pt<< std::endl;

    double theChi2 = SeedChi2(seeds[i]);
    double dChi2 = fabs(1. - (theChi2 / bestChi2));
    int theHits = seeds[i].nHits();
    int dHits = theHits - moreHits;
    //std::cout<<" -----  "<<std::endl;

    // 2. better chi2
    if (theChi2 < bestChi2 && dChi2 > 0.05) {
      winner = static_cast<int>(i);
      bestChi2 = theChi2;
      moreHits = theHits;
    }
    // 3. if chi2 is not much better, pick more rechits one
    if (theChi2 >= bestChi2 && dChi2 < 0.05 && dHits > 0) {
      winner = static_cast<int>(i);
      bestChi2 = theChi2;
      moreHits = theHits;
    }
  }
  //std::cout<<" Winner is "<< winner <<std::endl;
  TrajectorySeed theSeed = seeds[winner];
  seeds.erase(seeds.begin() + winner);
  return theSeed;
}

TrajectorySeed MuonSeedCleaner::BiggerCone(std::vector<TrajectorySeed>& seeds) {
  if (seeds.size() == 1)
    return seeds[0];

  float biggerProjErr = 9999.;
  int winner = 0;
  AlgebraicSymMatrix mat(5, 0);
  for (size_t i = 0; i < seeds.size(); i++) {
    auto r1 = seeds[i].recHits().begin();
    mat = r1->parametersError().similarityT(r1->projectionMatrix());

    int NRecHits = NRecHitsFromSegment(*r1);

    float ddx = mat[1][1];
    float ddy = mat[2][2];
    float dxx = mat[3][3];
    float dyy = mat[4][4];
    float projectErr = sqrt((ddx * 10000.) + (ddy * 10000.) + dxx + dyy);

    if (NRecHits < 5)
      continue;
    if (projectErr < biggerProjErr)
      continue;

    winner = static_cast<int>(i);
    biggerProjErr = projectErr;
  }
  TrajectorySeed theSeed = seeds[winner];
  seeds.erase(seeds.begin() + winner);
  return theSeed;
}

TrajectorySeed MuonSeedCleaner::LeanHighMomentum(std::vector<TrajectorySeed>& seeds) {
  if (seeds.size() == 1)
    return seeds[0];

  double highestPt = 0.;
  int winner = 0;
  for (size_t i = 0; i < seeds.size(); i++) {
    GlobalVector mom = SeedMomentum(seeds[i]);
    double pt = sqrt((mom.x() * mom.x()) + (mom.y() * mom.y()));
    if (pt > highestPt) {
      winner = static_cast<int>(i);
      highestPt = pt;
    }
  }
  TrajectorySeed theSeed = seeds[winner];
  seeds.erase(seeds.begin() + winner);
  return theSeed;
}

TrajectorySeed MuonSeedCleaner::MoreRecHits(std::vector<TrajectorySeed>& seeds) {
  if (seeds.size() == 1)
    return seeds[0];

  int winner = 0;
  int moreHits = 0;
  double betterChi2 = 99999.;
  for (size_t i = 0; i < seeds.size(); i++) {
    int theHits = 0;
    for (auto const& r1 : seeds[i].recHits()) {
      theHits += NRecHitsFromSegment(r1);
    }

    double theChi2 = SeedChi2(seeds[i]);

    if (theHits == moreHits && theChi2 < betterChi2) {
      betterChi2 = theChi2;
      winner = static_cast<int>(i);
    }
    if (theHits > moreHits) {
      moreHits = theHits;
      betterChi2 = theChi2;
      winner = static_cast<int>(i);
    }
  }
  TrajectorySeed theSeed = seeds[winner];
  seeds.erase(seeds.begin() + winner);
  return theSeed;
}

SeedContainer MuonSeedCleaner::LengthFilter(std::vector<TrajectorySeed>& seeds) {
  SeedContainer longSeeds;
  int NSegs = 0;
  for (size_t i = 0; i < seeds.size(); i++) {
    int theLength = static_cast<int>(seeds[i].nHits());
    if (theLength > NSegs) {
      NSegs = theLength;
      longSeeds.clear();
      longSeeds.push_back(seeds[i]);
    } else if (theLength == NSegs) {
      longSeeds.push_back(seeds[i]);
    } else {
      continue;
    }
  }
  //std::cout<<" final Length :"<<NSegs<<std::endl;

  return longSeeds;
}

bool MuonSeedCleaner::MomentumFilter(std::vector<TrajectorySeed>& seeds) {
  bool findgoodMomentum = false;
  SeedContainer goodMomentumSeeds = seeds;
  seeds.clear();
  for (size_t i = 0; i < goodMomentumSeeds.size(); i++) {
    GlobalVector mom = SeedMomentum(goodMomentumSeeds[i]);
    double pt = sqrt((mom.x() * mom.x()) + (mom.y() * mom.y()));
    if (pt < 6. || pt > 2000.)
      continue;
    //if ( pt < 6. ) continue;
    //std::cout<<" passed momentum :"<< pt <<std::endl;
    seeds.push_back(goodMomentumSeeds[i]);
    findgoodMomentum = true;
  }
  if (seeds.empty())
    seeds = goodMomentumSeeds;

  return findgoodMomentum;
}

SeedContainer MuonSeedCleaner::SeedCandidates(std::vector<TrajectorySeed>& seeds, bool good) {
  SeedContainer theCandidate;
  theCandidate.clear();

  bool longSeed = false;
  bool withFirstLayer = false;

  //std::cout<<"***** Seed Classification *****"<< seeds.size() <<std::endl;
  for (size_t i = 0; i < seeds.size(); i++) {
    if (seeds[i].nHits() > 1)
      longSeed = true;
    //std::cout<<"  Seed: "<<i<<" w/"<<seeds[i].nHits()<<" segs "<<std::endl;
    // looking for 1st layer segment
    for (auto const& r1 : seeds[i].recHits()) {
      const GeomDet* gdet = theService->trackingGeometry()->idToDet(r1.geographicalId());
      DetId geoId = gdet->geographicalId();

      if (geoId.subdetId() == MuonSubdetId::DT) {
        DTChamberId DT_Id(r1.geographicalId());
        //std::cout<<" ID:"<<DT_Id <<" pos:"<< r1->localPosition()  <<std::endl;
        if (DT_Id.station() != 1)
          continue;
        withFirstLayer = true;
      }
      if (geoId.subdetId() == MuonSubdetId::CSC) {
        CSCDetId CSC_Id = CSCDetId(r1.geographicalId());
        //std::cout<<" ID:"<<CSC_Id <<" pos:"<< r1->localPosition()  <<std::endl;
        if (CSC_Id.station() != 1)
          continue;
        withFirstLayer = true;
      }
    }
    bool goodseed = (longSeed && withFirstLayer) ? true : false;

    if (goodseed == good)
      theCandidate.push_back(seeds[i]);
  }
  return theCandidate;
}

std::vector<SeedContainer> MuonSeedCleaner::GroupSeeds(std::vector<TrajectorySeed>& seeds) {
  std::vector<SeedContainer> seedCollection;
  seedCollection.clear();
  std::vector<TrajectorySeed> theGroup;
  std::vector<bool> usedSeed(seeds.size(), false);

  // categorize seeds by comparing overlapping segments or a certian eta-phi cone
  for (size_t i = 0; i < seeds.size(); i++) {
    if (usedSeed[i])
      continue;
    theGroup.push_back(seeds[i]);
    usedSeed[i] = true;

    GlobalPoint pos1 = SeedPosition(seeds[i]);

    for (size_t j = i + 1; j < seeds.size(); j++) {
      // 1.1 seeds with overlaaping segments will be grouped together
      unsigned int overlapping = OverlapSegments(seeds[i], seeds[j]);
      if (!usedSeed[j] && overlapping > 0) {
        // reject the identical seeds
        if (seeds[i].nHits() == overlapping && seeds[j].nHits() == overlapping) {
          usedSeed[j] = true;
          continue;
        }
        theGroup.push_back(seeds[j]);
        usedSeed[j] = true;
      }
      if (usedSeed[j])
        continue;

      // 1.2 seeds in a certain cone are grouped together
      GlobalPoint pos2 = SeedPosition(seeds[j]);
      double dh = pos1.eta() - pos2.eta();
      double df = pos1.phi() - pos2.phi();
      double dR = sqrt((dh * dh) + (df * df));

      if (dR > 0.3 && seeds[j].nHits() == 1)
        continue;
      if (dR > 0.2 && seeds[j].nHits() > 1)
        continue;
      theGroup.push_back(seeds[j]);
      usedSeed[j] = true;
    }
    sort(theGroup.begin(), theGroup.end(), lengthSorting);
    seedCollection.push_back(theGroup);
    //std::cout<<" group "<<seedCollection.size() <<" w/"<< theGroup.size() <<" seeds"<<std::endl;
    theGroup.clear();
  }
  return seedCollection;
}

unsigned int MuonSeedCleaner::OverlapSegments(const TrajectorySeed& seed1, const TrajectorySeed& seed2) {
  unsigned int overlapping = 0;
  for (auto const& r1 : seed1.recHits()) {
    DetId id1 = r1.geographicalId();
    const GeomDet* gdet1 = theService->trackingGeometry()->idToDet(id1);
    GlobalPoint gp1 = gdet1->toGlobal(r1.localPosition());

    for (auto const& r2 : seed2.recHits()) {
      DetId id2 = r2.geographicalId();
      if (id1 != id2)
        continue;

      const GeomDet* gdet2 = theService->trackingGeometry()->idToDet(id2);
      GlobalPoint gp2 = gdet2->toGlobal(r2.localPosition());

      double dx = gp1.x() - gp2.x();
      double dy = gp1.y() - gp2.y();
      double dz = gp1.z() - gp2.z();
      double dL = sqrt(dx * dx + dy * dy + dz * dz);

      if (dL < 1.)
        overlapping++;
    }
  }
  return overlapping;
}

double MuonSeedCleaner::SeedChi2(const TrajectorySeed& seed) {
  double theChi2 = 0.;
  for (auto const& r1 : seed.recHits()) {
    //std::cout<<"    segmet : "<<it <<std::endl;
    theChi2 += NChi2OfSegment(r1);
  }
  theChi2 = theChi2 / seed.nHits();

  //std::cout<<" final Length :"<<NSegs<<std::endl;
  return theChi2;
}

int MuonSeedCleaner::SeedLength(const TrajectorySeed& seed) {
  int theHits = 0;
  for (auto const& recHit : seed.recHits()) {
    //std::cout<<"    segmet : "<<it <<std::endl;
    theHits += NRecHitsFromSegment(recHit);
  }

  //std::cout<<" final Length :"<<NSegs<<std::endl;
  return theHits;
}

GlobalPoint MuonSeedCleaner::SeedPosition(const TrajectorySeed& seed) {
  PTrajectoryStateOnDet pTSOD = seed.startingState();
  DetId SeedDetId(pTSOD.detId());
  const GeomDet* geoDet = theService->trackingGeometry()->idToDet(SeedDetId);
  TrajectoryStateOnSurface SeedTSOS =
      trajectoryStateTransform::transientState(pTSOD, &(geoDet->surface()), &*theService->magneticField());
  GlobalPoint pos = SeedTSOS.globalPosition();

  return pos;
}

GlobalVector MuonSeedCleaner::SeedMomentum(const TrajectorySeed& seed) {
  PTrajectoryStateOnDet pTSOD = seed.startingState();
  DetId SeedDetId(pTSOD.detId());
  const GeomDet* geoDet = theService->trackingGeometry()->idToDet(SeedDetId);
  TrajectoryStateOnSurface SeedTSOS =
      trajectoryStateTransform::transientState(pTSOD, &(geoDet->surface()), &*theService->magneticField());
  GlobalVector mom = SeedTSOS.globalMomentum();

  return mom;
}

int MuonSeedCleaner::NRecHitsFromSegment(const TrackingRecHit& rhit) {
  int NRechits = 0;
  const GeomDet* gdet = theService->trackingGeometry()->idToDet(rhit.geographicalId());
  MuonTransientTrackingRecHit::MuonRecHitPointer theSeg =
      MuonTransientTrackingRecHit::specificBuild(gdet, rhit.clone());

  DetId geoId = gdet->geographicalId();
  if (geoId.subdetId() == MuonSubdetId::DT) {
    DTChamberId DT_Id(rhit.geographicalId());
    std::vector<TrackingRecHit*> DThits = theSeg->recHits();
    int dt1DHits = 0;
    for (size_t j = 0; j < DThits.size(); j++) {
      dt1DHits += (DThits[j]->recHits()).size();
    }
    NRechits = dt1DHits;
  }

  if (geoId.subdetId() == MuonSubdetId::CSC) {
    NRechits = (theSeg->recHits()).size();
  }
  return NRechits;
}

int MuonSeedCleaner::NRecHitsFromSegment(MuonTransientTrackingRecHit* rhit) {
  int NRechits = 0;
  DetId geoId = rhit->geographicalId();
  if (geoId.subdetId() == MuonSubdetId::DT) {
    DTChamberId DT_Id(geoId);
    std::vector<TrackingRecHit*> DThits = rhit->recHits();
    int dt1DHits = 0;
    for (size_t j = 0; j < DThits.size(); j++) {
      dt1DHits += (DThits[j]->recHits()).size();
    }
    NRechits = dt1DHits;
    //std::cout<<" D_rh("<< dt1DHits  <<") " ;
  }
  if (geoId.subdetId() == MuonSubdetId::CSC) {
    NRechits = (rhit->recHits()).size();
    //std::cout<<" C_rh("<<(rhit->recHits()).size() <<") " ;
  }
  return NRechits;
}

double MuonSeedCleaner::NChi2OfSegment(const TrackingRecHit& rhit) {
  double NChi2 = 999999.;
  const GeomDet* gdet = theService->trackingGeometry()->idToDet(rhit.geographicalId());
  MuonTransientTrackingRecHit::MuonRecHitPointer theSeg =
      MuonTransientTrackingRecHit::specificBuild(gdet, rhit.clone());

  double dof = static_cast<double>(theSeg->degreesOfFreedom());
  NChi2 = theSeg->chi2() / dof;
  //std::cout<<" Chi2 = "<< NChi2  <<" |" ;

  return NChi2;
}
