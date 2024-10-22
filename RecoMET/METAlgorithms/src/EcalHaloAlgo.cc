#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoMET/METAlgorithms/interface/EcalHaloAlgo.h"
#include "DataFormats/Common/interface/ValueMap.h"

/*
  [class]:  EcalHaloAlgo
  [authors]: R. Remington, The University of Florida
  [description]: See EcalHaloAlgo.h
  [date]: October 15, 2009
*/
namespace {
  constexpr float c_cm_per_ns = 29.9792458;
};
using namespace std;
using namespace reco;
using namespace edm;

bool CompareTime(const EcalRecHit* x, const EcalRecHit* y) { return x->time() < y->time(); }

EcalHaloAlgo::EcalHaloAlgo(edm::ConsumesCollector iC) : geoToken_(iC.esConsumes()) {
  RoundnessCut = 0;
  AngleCut = 0;
  EBRecHitEnergyThreshold = 0.;
  EERecHitEnergyThreshold = 0.;
  ESRecHitEnergyThreshold = 0.;
  SumEnergyThreshold = 0.;
  NHitsThreshold = 0;

  geo = nullptr;
}

EcalHaloData EcalHaloAlgo::Calculate(const CaloGeometry& TheCaloGeometry,
                                     edm::Handle<reco::PhotonCollection>& ThePhotons,
                                     edm::Handle<reco::SuperClusterCollection>& TheSuperClusters,
                                     edm::Handle<EBRecHitCollection>& TheEBRecHits,
                                     edm::Handle<EERecHitCollection>& TheEERecHits,
                                     edm::Handle<ESRecHitCollection>& TheESRecHits,
                                     edm::Handle<HBHERecHitCollection>& TheHBHERecHits,
                                     const edm::EventSetup& TheSetup) {
  EcalHaloData TheEcalHaloData;

  // Store energy sum of rechits as a function of iPhi (iphi goes from 1 to 72)
  float SumE[361];
  // Store number of rechits as a function of iPhi
  int NumHits[361];
  // Store minimum time of rechit as a function of iPhi
  float MinTimeHits[361];
  // Store maximum time of rechit as a function of iPhi
  float MaxTimeHits[361];

  // initialize
  for (int i = 0; i < 361; i++) {
    SumE[i] = 0.;
    NumHits[i] = 0;
    MinTimeHits[i] = 9999.;
    MaxTimeHits[i] = -9999.;
  }

  // Loop over EB RecHits
  for (EBRecHitCollection::const_iterator hit = TheEBRecHits->begin(); hit != TheEBRecHits->end(); hit++) {
    // Arbitrary threshold to kill noise (needs to be optimized with data)
    if (hit->energy() < EBRecHitEnergyThreshold)
      continue;

    // Get Det Id of the rechit
    DetId id = DetId(hit->id());

    // Get EB geometry
    const CaloSubdetectorGeometry* TheSubGeometry = TheCaloGeometry.getSubdetectorGeometry(DetId::Ecal, 1);
    EBDetId EcalID(id.rawId());
    auto cell = (TheSubGeometry) ? (TheSubGeometry->getGeometry(id)) : nullptr;

    if (cell) {
      // GlobalPoint globalpos = cell->getPosition();
      //	  float r = TMath::Sqrt ( globalpos.y()*globalpos.y() + globalpos.x()*globalpos.x());
      int iPhi = EcalID.iphi();

      if (iPhi < 361)  // just to be safe
      {
        //iPhi = (iPhi-1)/5 +1;  // convert ecal iphi to phiwedge iphi  (e.g. there are 5 crystal per phi wedge, as in calotowers )
        SumE[iPhi] += hit->energy();
        NumHits[iPhi]++;

        float time = hit->time();
        MinTimeHits[iPhi] = time < MinTimeHits[iPhi] ? time : MinTimeHits[iPhi];
        MaxTimeHits[iPhi] = time > MaxTimeHits[iPhi] ? time : MaxTimeHits[iPhi];
      }
    }
  }

  //for( int iPhi = 1 ; iPhi < 73; iPhi++ )
  for (int iPhi = 1; iPhi < 361; iPhi++) {
    if (SumE[iPhi] >= SumEnergyThreshold && NumHits[iPhi] > NHitsThreshold) {
      // Build PhiWedge and store to EcalHaloData if energy or #hits pass thresholds
      PhiWedge wedge(SumE[iPhi], iPhi, NumHits[iPhi], MinTimeHits[iPhi], MaxTimeHits[iPhi]);

      // Loop over rechits again to calculate direction based on timing info

      // Loop over EB RecHits
      std::vector<const EcalRecHit*> Hits;
      for (EBRecHitCollection::const_iterator hit = TheEBRecHits->begin(); hit != TheEBRecHits->end(); hit++) {
        if (hit->energy() < EBRecHitEnergyThreshold)
          continue;

        // Get Det Id of the rechit
        DetId id = DetId(hit->id());
        EBDetId EcalID(id.rawId());
        int Hit_iPhi = EcalID.iphi();
        //Hit_iPhi = (Hit_iPhi-1)/5 +1; // convert ecal iphi to phiwedge iphi
        if (Hit_iPhi != iPhi)
          continue;
        Hits.push_back(&(*hit));
      }
      std::sort(Hits.begin(), Hits.end(), CompareTime);
      float MinusToPlus = 0.;
      float PlusToMinus = 0.;
      for (unsigned int i = 0; i < Hits.size(); i++) {
        DetId id_i = DetId(Hits[i]->id());
        EBDetId EcalID_i(id_i.rawId());
        int ieta_i = EcalID_i.ieta();
        for (unsigned int j = (i + 1); j < Hits.size(); j++) {
          DetId id_j = DetId(Hits[j]->id());
          EBDetId EcalID_j(id_j.rawId());
          int ieta_j = EcalID_j.ieta();
          if (ieta_i > ieta_j)
            PlusToMinus += TMath::Abs(ieta_j - ieta_i);
          else
            MinusToPlus += TMath::Abs(ieta_j - ieta_i);
        }
      }

      float PlusZOriginConfidence = (PlusToMinus + MinusToPlus) ? PlusToMinus / (PlusToMinus + MinusToPlus) : -1.;
      wedge.SetPlusZOriginConfidence(PlusZOriginConfidence);
      TheEcalHaloData.GetPhiWedges().push_back(wedge);
    }
  }

  std::vector<float> vShowerShapes_Roundness;
  std::vector<float> vShowerShapes_Angle;
  if (TheSuperClusters.isValid()) {
    for (reco::SuperClusterCollection::const_iterator cluster = TheSuperClusters->begin();
         cluster != TheSuperClusters->end();
         cluster++) {
      if (abs(cluster->eta()) <= 1.48) {
        vector<float> shapes = EcalClusterTools::roundnessBarrelSuperClusters(*cluster, (*TheEBRecHits.product()));
        float roundness = shapes[0];
        float angle = shapes[1];

        // Check if supercluster belongs to photon and passes the cuts on roundness and angle, if so store the reference to it
        if ((roundness >= 0 && roundness < GetRoundnessCut()) && angle >= 0 && angle < GetAngleCut()) {
          edm::Ref<SuperClusterCollection> TheClusterRef(TheSuperClusters, cluster - TheSuperClusters->begin());
          bool BelongsToPhoton = false;
          if (ThePhotons.isValid()) {
            for (reco::PhotonCollection::const_iterator iPhoton = ThePhotons->begin(); iPhoton != ThePhotons->end();
                 iPhoton++) {
              if (iPhoton->isEB())
                if (TheClusterRef == iPhoton->superCluster()) {
                  BelongsToPhoton = true;
                  break;
                }
            }
          }
          //Only store refs to suspicious EB SuperClusters which belong to Photons
          //Showershape variables are more discriminating for these cases
          if (BelongsToPhoton) {
            TheEcalHaloData.GetSuperClusters().push_back(TheClusterRef);
          }
        }
        vShowerShapes_Roundness.push_back(shapes[0]);
        vShowerShapes_Angle.push_back(shapes[1]);
      } else {
        vShowerShapes_Roundness.push_back(-1.);
        vShowerShapes_Angle.push_back(-1.);
      }
    }

    edm::ValueMap<float>::Filler TheRoundnessFiller(TheEcalHaloData.GetShowerShapesRoundness());
    TheRoundnessFiller.insert(TheSuperClusters, vShowerShapes_Roundness.begin(), vShowerShapes_Roundness.end());
    TheRoundnessFiller.fill();

    edm::ValueMap<float>::Filler TheAngleFiller(TheEcalHaloData.GetShowerShapesAngle());
    TheAngleFiller.insert(TheSuperClusters, vShowerShapes_Angle.begin(), vShowerShapes_Angle.end());
    TheAngleFiller.fill();
  }

  geo = &TheSetup.getData(geoToken_);

  //Halo cluster building:
  //Various clusters are built, depending on the subdetector.
  //In barrel, one looks for deposits narrow in phi.
  //In endcaps, one looks for localized deposits (dr condition in EE where r =sqrt(dphi*dphi+deta*deta)
  //H/E condition is also applied in EB.
  //The halo cluster building step targets a large efficiency (ideally >99%) for beam halo deposits.
  //These clusters are used as input for the halo pattern finding methods in EcalHaloAlgo and for the CSC-calo matching methods in GlobalHaloAlgo.

  //Et threshold hardcoded for now. Might one to get it from config
  std::vector<HaloClusterCandidateECAL> haloclustercands_EB;
  haloclustercands_EB = GetHaloClusterCandidateEB(TheEBRecHits, TheHBHERecHits, 5);

  std::vector<HaloClusterCandidateECAL> haloclustercands_EE;
  haloclustercands_EE = GetHaloClusterCandidateEE(TheEERecHits, TheHBHERecHits, 10);

  TheEcalHaloData.setHaloClusterCandidatesEB(haloclustercands_EB);
  TheEcalHaloData.setHaloClusterCandidatesEE(haloclustercands_EE);

  return TheEcalHaloData;
}

std::vector<HaloClusterCandidateECAL> EcalHaloAlgo::GetHaloClusterCandidateEB(
    edm::Handle<EcalRecHitCollection>& ecalrechitcoll,
    edm::Handle<HBHERecHitCollection>& hbherechitcoll,
    float et_thresh_seedrh) {
  std::vector<HaloClusterCandidateECAL> TheHaloClusterCandsEB;
  reco::Vertex::Point vtx(0, 0, 0);

  for (size_t ihit = 0; ihit < ecalrechitcoll->size(); ++ihit) {
    HaloClusterCandidateECAL clustercand;

    const EcalRecHit& rechit = (*ecalrechitcoll)[ihit];
    math::XYZPoint rhpos = getPosition(rechit.id(), vtx);
    //Et condition

    double rhet = rechit.energy() * sqrt(rhpos.perp2() / rhpos.mag2());
    if (rhet < et_thresh_seedrh)
      continue;
    double eta = rhpos.eta();
    double phi = rhpos.phi();

    bool isiso = true;
    double etcluster(0);
    int nbcrystalsameeta(0);
    double timediscriminator(0);
    double etstrip_iphiseedplus1(0), etstrip_iphiseedminus1(0);

    //Building the cluster
    edm::RefVector<EcalRecHitCollection> bhrhcandidates;
    for (size_t jhit = 0; jhit < ecalrechitcoll->size(); ++jhit) {
      const EcalRecHit& rechitj = (*ecalrechitcoll)[jhit];
      EcalRecHitRef rhRef(ecalrechitcoll, jhit);
      math::XYZPoint rhposj = getPosition(rechitj.id(), vtx);

      double etaj = rhposj.eta();
      double phij = rhposj.phi();

      double deta = eta - etaj;
      double dphi = deltaPhi(phi, phij);
      if (std::abs(deta) > 0.2)
        continue;  //This means +/-11 crystals in eta
      if (std::abs(dphi) > 0.08)
        continue;  //This means +/-4 crystals in phi

      double rhetj = rechitj.energy() * sqrt(rhposj.perp2() / rhposj.mag2());
      //Rechits with et between 1 and 2 GeV are saved in the rh list but not used in the calculation of the halocluster variables
      if (rhetj < 1)
        continue;
      bhrhcandidates.push_back(rhRef);
      if (rhetj < 2)
        continue;

      if (std::abs(dphi) > 0.03) {
        isiso = false;
        break;
      }  //The strip should be isolated
      if (std::abs(dphi) < 0.01)
        nbcrystalsameeta++;
      if (dphi > 0.01)
        etstrip_iphiseedplus1 += rhetj;
      if (dphi < -0.01)
        etstrip_iphiseedminus1 += rhetj;
      etcluster += rhetj;
      //Timing discriminator
      //We assign a weight to the rechit defined as:
      //Log10(Et)*f(T,R,Z)
      //where f(T,R,Z) is the separation curve between halo-like and IP-like times.
      //The time difference between a deposit from a outgoing IT halo and a deposit coming from a particle emitted at the IP is given by:
      //dt= ( - sqrt(R^2+z^2) + |z| )/c
      //Here we take R to be 130 cm.
      //For EB, the function was parametrized as a function of ieta instead of Z.
      double rhtj = rechitj.time();
      EBDetId detj = rechitj.id();
      int rhietaj = detj.ieta();
      timediscriminator += std::log10(rhetj) *
                           (rhtj + 0.5 * (sqrt(16900 + 9 * rhietaj * rhietaj) - 3 * std::abs(rhietaj)) / c_cm_per_ns);
    }
    //Isolation condition
    if (!isiso)
      continue;

    //Calculate H/E
    double hoe(0);
    for (size_t jhit = 0; jhit < hbherechitcoll->size(); ++jhit) {
      const HBHERecHit& rechitj = (*hbherechitcoll)[jhit];
      math::XYZPoint rhposj = getPosition(rechitj.id(), vtx);
      double rhetj = rechitj.energy() * sqrt(rhposj.perp2() / rhposj.mag2());
      if (rhetj < 2)
        continue;
      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      double deta = eta - etaj;
      double dphi = deltaPhi(phi, phij);
      if (std::abs(deta) > 0.2)
        continue;
      if (std::abs(dphi) > 0.2)
        continue;
      hoe += rhetj / etcluster;
    }
    //H/E condition
    if (hoe > 0.1)
      continue;

    clustercand.setClusterEt(etcluster);
    clustercand.setSeedEt(rhet);
    clustercand.setSeedEta(eta);
    clustercand.setSeedPhi(phi);
    clustercand.setSeedZ(rhpos.Z());
    clustercand.setSeedR(sqrt(rhpos.perp2()));
    clustercand.setSeedTime(rechit.time());
    clustercand.setHoverE(hoe);
    clustercand.setNbofCrystalsInEta(nbcrystalsameeta);
    clustercand.setEtStripIPhiSeedPlus1(etstrip_iphiseedplus1);
    clustercand.setEtStripIPhiSeedMinus1(etstrip_iphiseedminus1);
    clustercand.setTimeDiscriminator(timediscriminator);
    clustercand.setBeamHaloRecHitsCandidates(bhrhcandidates);

    bool isbeamhalofrompattern = EBClusterShapeandTimeStudy(clustercand, false);
    clustercand.setIsHaloFromPattern(isbeamhalofrompattern);

    bool isbeamhalofrompattern_hlt = EBClusterShapeandTimeStudy(clustercand, true);
    clustercand.setIsHaloFromPattern_HLT(isbeamhalofrompattern_hlt);

    TheHaloClusterCandsEB.push_back(clustercand);
  }

  return TheHaloClusterCandsEB;
}

std::vector<HaloClusterCandidateECAL> EcalHaloAlgo::GetHaloClusterCandidateEE(
    edm::Handle<EcalRecHitCollection>& ecalrechitcoll,
    edm::Handle<HBHERecHitCollection>& hbherechitcoll,
    float et_thresh_seedrh) {
  std::vector<HaloClusterCandidateECAL> TheHaloClusterCandsEE;

  reco::Vertex::Point vtx(0, 0, 0);

  for (size_t ihit = 0; ihit < ecalrechitcoll->size(); ++ihit) {
    HaloClusterCandidateECAL clustercand;

    const EcalRecHit& rechit = (*ecalrechitcoll)[ihit];
    math::XYZPoint rhpos = getPosition(rechit.id(), vtx);
    //Et condition
    double rhet = rechit.energy() * sqrt(rhpos.perp2() / rhpos.mag2());
    if (rhet < et_thresh_seedrh)
      continue;
    double eta = rhpos.eta();
    double phi = rhpos.phi();
    double rhr = sqrt(rhpos.perp2());

    bool isiso = true;
    double etcluster(0);
    double timediscriminator(0);
    int clustersize(0);
    int nbcrystalssmallt(0);
    int nbcrystalshight(0);
    //Building the cluster
    edm::RefVector<EcalRecHitCollection> bhrhcandidates;
    for (size_t jhit = 0; jhit < ecalrechitcoll->size(); ++jhit) {
      const EcalRecHit& rechitj = (*ecalrechitcoll)[jhit];
      EcalRecHitRef rhRef(ecalrechitcoll, jhit);
      math::XYZPoint rhposj = getPosition(rechitj.id(), vtx);

      //Ask the hits to be in the same endcap
      if (rhposj.z() * rhpos.z() < 0)
        continue;

      double etaj = rhposj.eta();
      double phij = rhposj.phi();
      double dr = sqrt((eta - etaj) * (eta - etaj) + deltaPhi(phi, phij) * deltaPhi(phi, phij));

      //Outer cone
      if (dr > 0.3)
        continue;

      double rhetj = rechitj.energy() * sqrt(rhposj.perp2() / rhposj.mag2());
      //Rechits with et between 1 and 2 GeV are saved in the rh list but not used in the calculation of the halocluster variables
      if (rhetj < 1)
        continue;
      bhrhcandidates.push_back(rhRef);
      if (rhetj < 2)
        continue;

      //Isolation between outer and inner cone
      if (dr > 0.05) {
        isiso = false;
        break;
      }  //The deposit should be isolated

      etcluster += rhetj;

      //Timing infos:
      //Here we target both IT and OT beam halo
      double rhtj = rechitj.time();

      //Discriminating variables for OT beam halo:
      if (rhtj > 1)
        nbcrystalshight++;
      if (rhtj < 0)
        nbcrystalssmallt++;
      //Timing test (likelihood ratio), only for seeds with large R (100 cm) and for crystals with et>5,
      //This targets IT beam halo (t around - 1ns)
      if (rhtj > 5) {
        double corrt_j = rhtj + sqrt(rhposj.x() * rhposj.x() + rhposj.y() * rhposj.y() + 320. * 320.) / c_cm_per_ns -
                         320. / c_cm_per_ns;
        //BH is modeled by a Gaussian peaking at 0.
        //Collisions is modeled by a Gaussian peaking at 0.3
        //The width is similar and taken to be 0.4
        timediscriminator += 0.5 * (pow((corrt_j - 0.3) / 0.4, 2) - pow((corrt_j - 0.) / 0.4, 2));
        clustersize++;
      }
    }
    //Isolation condition
    if (!isiso)
      continue;

    //Calculate H2/E
    //Only second hcal layer is considered as it can happen that a shower initiated in EE reaches HCAL first layer
    double h2oe(0);
    for (size_t jhit = 0; jhit < hbherechitcoll->size(); ++jhit) {
      const HBHERecHit& rechitj = (*hbherechitcoll)[jhit];
      math::XYZPoint rhposj = getPosition(rechitj.id(), vtx);

      //Ask the hits to be in the same endcap
      if (rhposj.z() * rhpos.z() < 0)
        continue;
      //Selects only second HCAL layer
      if (std::abs(rhposj.z()) < 425)
        continue;

      double rhetj = rechitj.energy() * sqrt(rhposj.perp2() / rhposj.mag2());
      if (rhetj < 2)
        continue;

      double phij = rhposj.phi();
      if (std::abs(deltaPhi(phi, phij)) > 0.4)
        continue;

      double rhrj = sqrt(rhposj.perp2());
      if (std::abs(rhr - rhrj) > 50)
        continue;

      h2oe += rhetj / etcluster;
    }
    //H/E condition
    if (h2oe > 0.1)
      continue;

    clustercand.setClusterEt(etcluster);
    clustercand.setSeedEt(rhet);
    clustercand.setSeedEta(eta);
    clustercand.setSeedPhi(phi);
    clustercand.setSeedZ(rhpos.Z());
    clustercand.setSeedR(sqrt(rhpos.perp2()));
    clustercand.setSeedTime(rechit.time());
    clustercand.setH2overE(h2oe);
    clustercand.setNbEarlyCrystals(nbcrystalssmallt);
    clustercand.setNbLateCrystals(nbcrystalshight);
    clustercand.setClusterSize(clustersize);
    clustercand.setTimeDiscriminator(timediscriminator);
    clustercand.setBeamHaloRecHitsCandidates(bhrhcandidates);

    bool isbeamhalofrompattern =
        EEClusterShapeandTimeStudy_ITBH(clustercand, false) || EEClusterShapeandTimeStudy_OTBH(clustercand, false);
    clustercand.setIsHaloFromPattern(isbeamhalofrompattern);

    bool isbeamhalofrompattern_hlt =
        EEClusterShapeandTimeStudy_ITBH(clustercand, true) || EEClusterShapeandTimeStudy_OTBH(clustercand, true);
    clustercand.setIsHaloFromPattern_HLT(isbeamhalofrompattern_hlt);

    TheHaloClusterCandsEE.push_back(clustercand);
  }

  return TheHaloClusterCandsEE;
}

bool EcalHaloAlgo::EBClusterShapeandTimeStudy(HaloClusterCandidateECAL hcand, bool ishlt) {
  //Conditions on the central strip size in eta.
  //For low size, extra conditions on seed et, isolation and cluster timing
  //The time condition only targets IT beam halo.
  //EB rechits from OT beam halos are typically too late (around 5 ns or more) and seem therefore already cleaned by the reconstruction.

  if (hcand.getSeedEt() < 5)
    return false;
  if (hcand.getNbofCrystalsInEta() < 4)
    return false;
  if (hcand.getNbofCrystalsInEta() == 4 && hcand.getSeedEt() < 10)
    return false;
  if (hcand.getNbofCrystalsInEta() == 4 && hcand.getEtStripIPhiSeedPlus1() > 0.1 &&
      hcand.getEtStripIPhiSeedMinus1() > 0.1)
    return false;
  if (hcand.getNbofCrystalsInEta() <= 5 && hcand.getTimeDiscriminator() >= 0.)
    return false;

  //For HLT, only use conditions without timing and tighten seed et condition
  if (ishlt && hcand.getNbofCrystalsInEta() <= 5)
    return false;
  if (ishlt && hcand.getSeedEt() < 10)
    return false;

  hcand.setIsHaloFromPattern(true);

  return true;
}

bool EcalHaloAlgo::EEClusterShapeandTimeStudy_OTBH(HaloClusterCandidateECAL hcand, bool ishlt) {
  //Separate conditions targeting IT and OT beam halos
  //For OT beam halos, just require enough crystals with large T
  if (hcand.getSeedEt() < 20)
    return false;
  if (hcand.getSeedTime() < 0.5)
    return false;
  if (hcand.getNbLateCrystals() - hcand.getNbEarlyCrystals() < 2)
    return false;

  //The use of time information does not allow this method to work at HLT
  if (ishlt)
    return false;

  hcand.setIsHaloFromPattern(true);

  return true;
}

bool EcalHaloAlgo::EEClusterShapeandTimeStudy_ITBH(HaloClusterCandidateECAL hcand, bool ishlt) {
  //Separate conditions targeting IT and OT beam halos
  //For IT beam halos, fakes from collisions are higher => require the cluster size to be small.
  //Only halos with R>100 cm are considered here.
  //For lower values, the time difference with particles from collisions is too small
  //IT outgoing beam halos that interact in EE at low R is probably the most difficult category to deal with:
  //Their signature is very close to the one of photon from collisions (similar cluster shape and timing)
  if (hcand.getSeedEt() < 20)
    return false;
  if (hcand.getSeedR() < 100)
    return false;
  if (hcand.getTimeDiscriminator() < 1)
    return false;
  if (hcand.getClusterSize() < 2)
    return false;
  if (hcand.getClusterSize() > 4)
    return false;

  //The use of time information does not allow this method to work at HLT
  if (ishlt)
    return false;

  hcand.setIsHaloFromPattern(true);

  return true;
}

math::XYZPoint EcalHaloAlgo::getPosition(const DetId& id, reco::Vertex::Point vtx) {
  const GlobalPoint& pos = geo->getPosition(id);
  math::XYZPoint posV(pos.x() - vtx.x(), pos.y() - vtx.y(), pos.z() - vtx.z());
  return posV;
}
