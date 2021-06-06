#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoMET/METAlgorithms/interface/HcalHaloAlgo.h"
#include <map>

/*
  [class]:  HcalHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: See HcalHaloAlgo.h
  [date]: October 15, 2009
*/
namespace {
  constexpr float c_cm_per_ns = 29.9792458;
  constexpr float zseparation_HBHE = 380.;
};  // namespace

using namespace reco;

#include <iomanip>
bool CompareTime(const HBHERecHit* x, const HBHERecHit* y) { return x->time() < y->time(); }
bool CompareTowers(const CaloTower* x, const CaloTower* y) {
  return x->iphi() * 1000 + x->ieta() < y->iphi() * 1000 + y->ieta();
}

HcalHaloAlgo::HcalHaloAlgo(edm::ConsumesCollector iC) : geoToken_(iC.esConsumes()), geo_(nullptr), hgeo_(nullptr) {
  HBRecHitEnergyThreshold = 0.;
  HERecHitEnergyThreshold = 0.;
  SumEnergyThreshold = 0.;
  NHitsThreshold = 0;
}

HcalHaloData HcalHaloAlgo::Calculate(const CaloGeometry& TheCaloGeometry,
                                     edm::Handle<HBHERecHitCollection>& TheHBHERecHits,
                                     edm::Handle<EBRecHitCollection>& TheEBRecHits,
                                     edm::Handle<EERecHitCollection>& TheEERecHits,
                                     const edm::EventSetup& TheSetup) {
  edm::Handle<CaloTowerCollection> TheCaloTowers;
  return Calculate(TheCaloGeometry, TheHBHERecHits, TheCaloTowers, TheEBRecHits, TheEERecHits, TheSetup);
}

HcalHaloData HcalHaloAlgo::Calculate(const CaloGeometry& TheCaloGeometry,
                                     edm::Handle<HBHERecHitCollection>& TheHBHERecHits,
                                     edm::Handle<CaloTowerCollection>& TheCaloTowers,
                                     edm::Handle<EBRecHitCollection>& TheEBRecHits,
                                     edm::Handle<EERecHitCollection>& TheEERecHits,
                                     const edm::EventSetup& TheSetup) {
  HcalHaloData TheHcalHaloData;
  // ieta overlap geometrically w/ HB
  const int iEtaOverlap = 22;
  const int nPhiMax = 73;
  // Store Energy sum of rechits as a function of iPhi (iPhi goes from 1 to 72)
  float SumE[nPhiMax];
  // Store Number of rechits as a function of iPhi
  int NumHits[nPhiMax];
  // Store minimum time of rechit as a function of iPhi
  float MinTimeHits[nPhiMax];
  // Store maximum time of rechit as a function of iPhi
  float MaxTimeHits[nPhiMax];
  for (unsigned int i = 0; i < nPhiMax; i++) {
    SumE[i] = 0;
    NumHits[i] = 0;
    MinTimeHits[i] = 0.;
    MaxTimeHits[i] = 0.;
  }

  for (const auto& hit : (*TheHBHERecHits)) {
    HcalDetId id = HcalDetId(hit.id());
    switch (id.subdet()) {
      case HcalBarrel:
        if (hit.energy() < HBRecHitEnergyThreshold)
          continue;
        break;
      case HcalEndcap:
        if (hit.energy() < HERecHitEnergyThreshold)
          continue;
        break;
      default:
        continue;
    }

    int iEta = id.ieta();
    int iPhi = id.iphi();
    if (iPhi < nPhiMax && std::abs(iEta) <= iEtaOverlap) {
      SumE[iPhi] += hit.energy();
      NumHits[iPhi]++;

      float time = hit.time();
      MinTimeHits[iPhi] = time < MinTimeHits[iPhi] ? time : MinTimeHits[iPhi];
      MaxTimeHits[iPhi] = time > MaxTimeHits[iPhi] ? time : MaxTimeHits[iPhi];
    }
  }

  for (int iPhi = 1; iPhi < nPhiMax; iPhi++) {
    if (SumE[iPhi] >= SumEnergyThreshold && NumHits[iPhi] > NHitsThreshold) {
      // Build PhiWedge and store to HcalHaloData if energy or #hits pass thresholds
      PhiWedge wedge(SumE[iPhi], iPhi, NumHits[iPhi], MinTimeHits[iPhi], MaxTimeHits[iPhi]);

      // Loop over rechits again to calculate direction based on timing info
      std::vector<const HBHERecHit*> Hits;
      for (const auto& hit : (*TheHBHERecHits)) {
        HcalDetId id = HcalDetId(hit.id());
        if (id.iphi() != iPhi)
          continue;
        if (std::abs(id.ieta()) > iEtaOverlap)
          continue;  // has to overlap geometrically w/ HB
        switch (id.subdet()) {
          case HcalBarrel:
            if (hit.energy() < HBRecHitEnergyThreshold)
              continue;
            break;
          case HcalEndcap:
            if (hit.energy() < HERecHitEnergyThreshold)
              continue;
            break;
          default:
            continue;
        }
        Hits.push_back(&(hit));
      }

      std::sort(Hits.begin(), Hits.end(), CompareTime);
      float MinusToPlus = 0.;
      float PlusToMinus = 0.;
      for (unsigned int i = 0; i < Hits.size(); i++) {
        HcalDetId id_i = HcalDetId(Hits[i]->id());
        int ieta_i = id_i.ieta();
        for (unsigned int j = (i + 1); j < Hits.size(); j++) {
          HcalDetId id_j = HcalDetId(Hits[j]->id());
          int ieta_j = id_j.ieta();
          if (ieta_i > ieta_j)
            PlusToMinus += std::abs(ieta_i - ieta_j);
          else
            MinusToPlus += std::abs(ieta_i - ieta_j);
        }
      }
      float PlusZOriginConfidence = (PlusToMinus + MinusToPlus) ? PlusToMinus / (PlusToMinus + MinusToPlus) : -1.;
      wedge.SetPlusZOriginConfidence(PlusZOriginConfidence);
      TheHcalHaloData.GetPhiWedges().push_back(wedge);
    }
  }

  // Don't use HF.
  int maxAbsIEta = 29;

  std::map<int, float> iPhiHadEtMap;
  std::vector<const CaloTower*> sortedCaloTowers;
  for (const auto& tower : (*TheCaloTowers)) {
    if (std::abs(tower.ieta()) > maxAbsIEta)
      continue;

    int iPhi = tower.iphi();
    if (!iPhiHadEtMap.count(iPhi))
      iPhiHadEtMap[iPhi] = 0.0;
    iPhiHadEtMap[iPhi] += tower.hadEt();

    if (tower.numProblematicHcalCells() > 0)
      sortedCaloTowers.push_back(&(tower));
  }

  // Sort towers such that lowest iphi and ieta are first, highest last, and towers
  // with same iphi value are consecutive. Then we can do everything else in one loop.
  std::sort(sortedCaloTowers.begin(), sortedCaloTowers.end(), CompareTowers);

  HaloTowerStrip strip;

  int prevIEta = -99, prevIPhi = -99;
  float prevHadEt = 0.;
  float prevEmEt = 0.;
  std::pair<uint8_t, CaloTowerDetId> prevPair, towerPair;
  bool wasContiguous = true;

  // Loop through and store a vector of pairs (problematicCells, DetId) for each contiguous strip we find
  for (unsigned int i = 0; i < sortedCaloTowers.size(); i++) {
    const CaloTower* tower = sortedCaloTowers[i];

    towerPair = std::make_pair((uint8_t)tower->numProblematicHcalCells(), tower->id());

    bool newIPhi = tower->iphi() != prevIPhi;
    bool isContiguous = tower->ieta() == 1 ? tower->ieta() - 2 == prevIEta : tower->ieta() - 1 == prevIEta;

    isContiguous = isContiguous || (tower->ieta() == -maxAbsIEta);
    if (newIPhi)
      isContiguous = false;

    if (!wasContiguous && isContiguous) {
      strip.cellTowerIds.push_back(prevPair);
      strip.cellTowerIds.push_back(towerPair);
      strip.hadEt += prevHadEt + tower->hadEt();
      strip.emEt += prevEmEt + tower->emEt();
    }

    if (wasContiguous && isContiguous) {
      strip.cellTowerIds.push_back(towerPair);
      strip.hadEt += tower->hadEt();
      strip.emEt += tower->emEt();
    }

    if ((wasContiguous && !isContiguous) || i == sortedCaloTowers.size() - 1) {  //ended the strip, so flush it

      if (strip.cellTowerIds.size() > 3) {
        int iPhi = strip.cellTowerIds.at(0).second.iphi();
        int iPhiLower = (iPhi == 1) ? 72 : iPhi - 1;
        int iPhiUpper = (iPhi == 72) ? 1 : iPhi + 1;

        float energyRatio = 0.0;
        if (iPhiHadEtMap.count(iPhiLower))
          energyRatio += iPhiHadEtMap[iPhiLower];
        if (iPhiHadEtMap.count(iPhiUpper))
          energyRatio += iPhiHadEtMap[iPhiUpper];
        iPhiHadEtMap[iPhi] = std::max(iPhiHadEtMap[iPhi], 0.001F);

        energyRatio /= iPhiHadEtMap[iPhi];
        strip.energyRatio = energyRatio;

        TheHcalHaloData.getProblematicStrips().push_back(strip);
      }
      strip = HaloTowerStrip();
    }

    wasContiguous = isContiguous;
    prevPair = towerPair;
    prevEmEt = tower->emEt();
    prevIPhi = tower->iphi();
    prevIEta = tower->ieta();
    prevHadEt = tower->hadEt();
  }

  geo_ = &TheSetup.getData(geoToken_);
  hgeo_ = dynamic_cast<const HcalGeometry*>(geo_->getSubdetectorGeometry(DetId::Hcal, 1));

  //Halo cluster building:
  //Various clusters are built, depending on the subdetector.
  //In barrel, one looks for deposits narrow in phi.
  //In endcaps, one looks for localized deposits (dr condition in EE where r =sqrt(dphi*dphi+deta*deta)
  //E/H condition is also applied.
  //The halo cluster building step targets a large efficiency (ideally >99%) for beam halo deposits.
  //These clusters are used as input for the halo pattern finding methods in HcalHaloAlgo and for the CSC-calo matching methods in GlobalHaloAlgo.

  //Et threshold hardcoded for now. Might one to get it from config

  std::vector<HaloClusterCandidateHCAL> haloclustercands_HB;
  haloclustercands_HB = GetHaloClusterCandidateHB(TheEBRecHits, TheHBHERecHits, 5);

  std::vector<HaloClusterCandidateHCAL> haloclustercands_HE;
  haloclustercands_HE = GetHaloClusterCandidateHE(TheEERecHits, TheHBHERecHits, 10);

  TheHcalHaloData.setHaloClusterCandidatesHB(haloclustercands_HB);
  TheHcalHaloData.setHaloClusterCandidatesHE(haloclustercands_HE);

  return TheHcalHaloData;
}

std::vector<HaloClusterCandidateHCAL> HcalHaloAlgo::GetHaloClusterCandidateHB(
    edm::Handle<EcalRecHitCollection>& ecalrechitcoll,
    edm::Handle<HBHERecHitCollection>& hbherechitcoll,
    float et_thresh_seedrh) {
  std::vector<HaloClusterCandidateHCAL> TheHaloClusterCandsHB;

  reco::Vertex::Point vtx(0, 0, 0);

  for (size_t ihit = 0; ihit < hbherechitcoll->size(); ++ihit) {
    HaloClusterCandidateHCAL clustercand;

    const HBHERecHit& rechit = (*hbherechitcoll)[ihit];
    math::XYZPoint rhpos = getPosition(rechit.id(), vtx);
    //Et condition
    double rhet = rechit.energy() * sqrt(rhpos.perp2() / rhpos.mag2());
    if (rhet < et_thresh_seedrh)
      continue;
    if (std::abs(rhpos.z()) > zseparation_HBHE)
      continue;
    double eta = rhpos.eta();
    double phi = rhpos.phi();

    bool isiso = true;
    double etcluster(0);
    int nbtowerssameeta(0);
    double timediscriminatorITBH(0), timediscriminatorOTBH(0);
    double etstrip_phiseedplus1(0), etstrip_phiseedminus1(0);

    //Building the cluster
    edm::RefVector<HBHERecHitCollection> bhrhcandidates;
    for (size_t jhit = 0; jhit < hbherechitcoll->size(); ++jhit) {
      const HBHERecHit& rechitj = (*hbherechitcoll)[jhit];
      HBHERecHitRef rhRef(hbherechitcoll, jhit);
      math::XYZPoint rhposj = getPosition(rechitj.id(), vtx);
      double rhetj = rechitj.energy() * sqrt(rhposj.perp2() / rhposj.mag2());
      if (rhetj < 2)
        continue;
      if (std::abs(rhposj.z()) > zseparation_HBHE)
        continue;
      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      double deta = eta - etaj;
      double dphi = deltaPhi(phi, phij);
      if (std::abs(deta) > 0.4)
        continue;  //This means +/-4 towers in eta
      if (std::abs(dphi) > 0.2)
        continue;  //This means +/-2 towers in phi
      if (std::abs(dphi) > 0.1 && std::abs(deta) < 0.2) {
        isiso = false;
        break;
      }  //The strip should be isolated
      if (std::abs(dphi) > 0.1)
        continue;
      if (std::abs(dphi) < 0.05)
        nbtowerssameeta++;
      if (dphi > 0.05)
        etstrip_phiseedplus1 += rhetj;
      if (dphi < -0.05)
        etstrip_phiseedminus1 += rhetj;

      etcluster += rhetj;
      //Timing discriminator
      //We assign a weight to the rechit defined as:
      //Log10(Et)*f(T,R,Z)
      //where f(T,R,Z) is the separation curve between halo-like and IP-like times.
      //The time difference between a deposit from a outgoing IT halo and a deposit coming from a particle emitted at the IP is given by:
      //dt= ( - sqrt(R^2+z^2) + |z| )/c
      // For OT beam halo, the time difference is:
      //dt= ( 25 + sqrt(R^2+z^2) + |z| )/c
      //only consider the central part of HB as things get hard at large z.
      //The best fitted value for R leads to 240 cm (IT) and 330 cm (OT)
      double rhtj = rechitj.time();
      timediscriminatorITBH +=
          std::log10(rhetj) *
          (rhtj + 0.5 * (sqrt(240. * 240. + rhposj.z() * rhposj.z()) - std::abs(rhposj.z())) / c_cm_per_ns);
      if (std::abs(rhposj.z()) < 300)
        timediscriminatorOTBH +=
            std::log10(rhetj) *
            (rhtj - 0.5 * (25 - (sqrt(330. * 330. + rhposj.z() * rhposj.z()) + std::abs(rhposj.z())) / c_cm_per_ns));
      bhrhcandidates.push_back(rhRef);
    }
    //Isolation conditions
    if (!isiso)
      continue;
    if (etstrip_phiseedplus1 / etcluster > 0.2 && etstrip_phiseedminus1 / etcluster > 0.2)
      continue;

    //Calculate E/H
    double eoh(0);
    for (size_t jhit = 0; jhit < ecalrechitcoll->size(); ++jhit) {
      const EcalRecHit& rechitj = (*ecalrechitcoll)[jhit];
      math::XYZPoint rhposj = getPosition(rechitj.id(), vtx);
      double rhetj = rechitj.energy() * sqrt(rhposj.perp2() / rhposj.mag2());
      if (rhetj < 2)
        continue;
      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      if (std::abs(eta - etaj) > 0.2)
        continue;
      if (std::abs(deltaPhi(phi, phij)) > 0.2)
        continue;
      eoh += rhetj / etcluster;
    }
    //E/H condition
    if (eoh > 0.1)
      continue;

    clustercand.setClusterEt(etcluster);
    clustercand.setSeedEt(rhet);
    clustercand.setSeedEta(eta);
    clustercand.setSeedPhi(phi);
    clustercand.setSeedZ(rhpos.Z());
    clustercand.setSeedR(sqrt(rhpos.perp2()));
    clustercand.setSeedTime(rechit.time());
    clustercand.setEoverH(eoh);
    clustercand.setNbTowersInEta(nbtowerssameeta);
    clustercand.setEtStripPhiSeedPlus1(etstrip_phiseedplus1);
    clustercand.setEtStripPhiSeedMinus1(etstrip_phiseedminus1);
    clustercand.setTimeDiscriminatorITBH(timediscriminatorITBH);
    clustercand.setTimeDiscriminatorOTBH(timediscriminatorOTBH);
    clustercand.setBeamHaloRecHitsCandidates(bhrhcandidates);

    bool isbeamhalofrompattern = HBClusterShapeandTimeStudy(clustercand, false);
    clustercand.setIsHaloFromPattern(isbeamhalofrompattern);
    bool isbeamhalofrompattern_hlt = HBClusterShapeandTimeStudy(clustercand, true);
    clustercand.setIsHaloFromPattern_HLT(isbeamhalofrompattern_hlt);

    TheHaloClusterCandsHB.push_back(clustercand);
  }

  return TheHaloClusterCandsHB;
}

std::vector<HaloClusterCandidateHCAL> HcalHaloAlgo::GetHaloClusterCandidateHE(
    edm::Handle<EcalRecHitCollection>& ecalrechitcoll,
    edm::Handle<HBHERecHitCollection>& hbherechitcoll,
    float et_thresh_seedrh) {
  std::vector<HaloClusterCandidateHCAL> TheHaloClusterCandsHE;

  reco::Vertex::Point vtx(0, 0, 0);

  for (size_t ihit = 0; ihit < hbherechitcoll->size(); ++ihit) {
    HaloClusterCandidateHCAL clustercand;

    const HBHERecHit& rechit = (*hbherechitcoll)[ihit];
    math::XYZPoint rhpos = getPosition(rechit.id(), vtx);
    //Et condition
    double rhet = rechit.energy() * sqrt(rhpos.perp2() / rhpos.mag2());
    if (rhet < et_thresh_seedrh)
      continue;
    if (std::abs(rhpos.z()) < zseparation_HBHE)
      continue;
    double eta = rhpos.eta();
    double phi = rhpos.phi();
    double rhr = sqrt(rhpos.perp2());
    bool isiso = true;
    double etcluster(0), hdepth1(0);
    int clustersize(0);
    double etstrip_phiseedplus1(0), etstrip_phiseedminus1(0);

    //Building the cluster
    edm::RefVector<HBHERecHitCollection> bhrhcandidates;
    for (size_t jhit = 0; jhit < hbherechitcoll->size(); ++jhit) {
      const HBHERecHit& rechitj = (*hbherechitcoll)[jhit];
      HBHERecHitRef rhRef(hbherechitcoll, jhit);
      math::XYZPoint rhposj = getPosition(rechitj.id(), vtx);
      double rhetj = rechitj.energy() * sqrt(rhposj.perp2() / rhposj.mag2());
      if (rhetj < 2)
        continue;
      if (std::abs(rhposj.z()) < zseparation_HBHE)
        continue;
      if (rhpos.z() * rhposj.z() < 0)
        continue;
      double phij = rhposj.phi();
      double dphi = deltaPhi(phi, phij);
      if (std::abs(dphi) > 0.4)
        continue;
      double rhrj = sqrt(rhposj.perp2());
      if (std::abs(rhr - rhrj) > 50)
        continue;
      if (std::abs(dphi) > 0.2 || std::abs(rhr - rhrj) > 20) {
        isiso = false;
        break;
      }  //The deposit should be isolated
      if (dphi > 0.05)
        etstrip_phiseedplus1 += rhetj;
      if (dphi < -0.05)
        etstrip_phiseedminus1 += rhetj;
      clustersize++;
      etcluster += rhetj;
      if (std::abs(rhposj.z()) < 405)
        hdepth1 += rhetj;
      //No timing condition for now in HE
      bhrhcandidates.push_back(rhRef);
    }
    //Isolation conditions
    if (!isiso)
      continue;
    if (etstrip_phiseedplus1 / etcluster > 0.1 && etstrip_phiseedminus1 / etcluster > 0.1)
      continue;

    //Calculate E/H
    double eoh(0);
    for (size_t jhit = 0; jhit < ecalrechitcoll->size(); ++jhit) {
      const EcalRecHit& rechitj = (*ecalrechitcoll)[jhit];
      math::XYZPoint rhposj = getPosition(rechitj.id(), vtx);
      double rhetj = rechitj.energy() * sqrt(rhposj.perp2() / rhposj.mag2());
      if (rhetj < 2)
        continue;
      if (rhpos.z() * rhposj.z() < 0)
        continue;
      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      double dr = sqrt((eta - etaj) * (eta - etaj) + deltaPhi(phi, phij) * deltaPhi(phi, phij));
      if (dr > 0.3)
        continue;

      eoh += rhetj / etcluster;
    }
    //E/H condition
    if (eoh > 0.1)
      continue;

    clustercand.setClusterEt(etcluster);
    clustercand.setSeedEt(rhet);
    clustercand.setSeedEta(eta);
    clustercand.setSeedPhi(phi);
    clustercand.setSeedZ(rhpos.Z());
    clustercand.setSeedR(sqrt(rhpos.perp2()));
    clustercand.setSeedTime(rechit.time());
    clustercand.setEoverH(eoh);
    clustercand.setH1overH123(hdepth1 / etcluster);
    clustercand.setClusterSize(clustersize);
    clustercand.setEtStripPhiSeedPlus1(etstrip_phiseedplus1);
    clustercand.setEtStripPhiSeedMinus1(etstrip_phiseedminus1);
    clustercand.setTimeDiscriminator(0);
    clustercand.setBeamHaloRecHitsCandidates(bhrhcandidates);

    bool isbeamhalofrompattern = HEClusterShapeandTimeStudy(clustercand, false);
    clustercand.setIsHaloFromPattern(isbeamhalofrompattern);
    bool isbeamhalofrompattern_hlt = HEClusterShapeandTimeStudy(clustercand, true);
    clustercand.setIsHaloFromPattern_HLT(isbeamhalofrompattern_hlt);

    TheHaloClusterCandsHE.push_back(clustercand);
  }

  return TheHaloClusterCandsHE;
}

bool HcalHaloAlgo::HBClusterShapeandTimeStudy(HaloClusterCandidateHCAL hcand, bool ishlt) {
  //Conditions on the central strip size in eta.
  //For low size, extra conditions on seed et, isolation and cluster timing
  //Here we target both IT and OT beam halo. Two separate discriminators were built for the two cases.

  if (hcand.getSeedEt() < 10)
    return false;

  if (hcand.getNbTowersInEta() < 3)
    return false;
  //Isolation criteria for very short eta strips
  if (hcand.getNbTowersInEta() == 3 && (hcand.getEtStripPhiSeedPlus1() > 0.1 || hcand.getEtStripPhiSeedMinus1() > 0.1))
    return false;
  if (hcand.getNbTowersInEta() <= 5 && (hcand.getEtStripPhiSeedPlus1() > 0.1 && hcand.getEtStripPhiSeedMinus1() > 0.1))
    return false;

  //Timing conditions for short eta strips
  if (hcand.getNbTowersInEta() == 3 && hcand.getTimeDiscriminatorITBH() >= 0.)
    return false;
  if (hcand.getNbTowersInEta() <= 6 && hcand.getTimeDiscriminatorITBH() >= 5. && hcand.getTimeDiscriminatorOTBH() < 0.)
    return false;

  //For HLT, only use conditions without timing
  if (ishlt && hcand.getNbTowersInEta() < 7)
    return false;

  hcand.setIsHaloFromPattern(true);

  return true;
}

bool HcalHaloAlgo::HEClusterShapeandTimeStudy(HaloClusterCandidateHCAL hcand, bool ishlt) {
  //Conditions on H1/H123 to spot halo interacting only in one HCAL layer.
  //For R> about 170cm, HE has only one layer and this condition cannot be applied
  //Note that for R>170 cm, the halo is in CSC acceptance and will most likely be spotted by the CSC-calo matching method
  //A method to identify halos interacting in both H1 and H2/H3 at low R is still missing.

  if (hcand.getSeedEt() < 20)
    return false;
  if (hcand.getSeedR() > 170)
    return false;

  if (hcand.getH1overH123() > 0.02 && hcand.getH1overH123() < 0.98)
    return false;

  //This method is one of the ones with the highest fake rate: in JetHT dataset, it happens in around 0.1% of the cases that a low pt jet (pt= 20) leaves all of its energy in only one HCAL layer.
  //At HLT, one only cares about large deposits from BH that would lead to a MET/SinglePhoton trigger to be fired.
  //Rising the seed Et threshold at HLT has therefore little impact on the HLT performances but ensures that possible controversial events are still recorded.
  if (ishlt && hcand.getSeedEt() < 50)
    return false;

  hcand.setIsHaloFromPattern(true);

  return true;
}

math::XYZPoint HcalHaloAlgo::getPosition(const DetId& id, reco::Vertex::Point vtx) {
  const GlobalPoint pos = ((id.det() == DetId::Hcal) ? hgeo_->getPosition(id) : GlobalPoint(geo_->getPosition(id)));
  math::XYZPoint posV(pos.x() - vtx.x(), pos.y() - vtx.y(), pos.z() - vtx.z());
  return posV;
}
