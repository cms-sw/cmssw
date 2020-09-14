//
// ** class l1t::Stage2Layer2TauAlgorithmFirmwareImp1
// ** authors: J. Brooke, L. Cadamuro, L. Mastrolorenzo, J.B. Sauvan, T. Strebler, O. Davignon, etc.
// ** date:   2 Oct 2015
// ** Description: version of tau algorithm matching the jet-eg-tau merged implementation

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2TauAlgorithmFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"
#include "L1Trigger/L1TCalorimeter/interface/BitonicSort.h"
#include "L1Trigger/L1TCalorimeter/interface/AccumulatingSort.h"

namespace l1t {
  bool operator>(const l1t::Tau& a, const l1t::Tau& b) { return a.pt() > b.pt(); }
}  // namespace l1t

l1t::Stage2Layer2TauAlgorithmFirmwareImp1::Stage2Layer2TauAlgorithmFirmwareImp1(CaloParamsHelper const* params)
    : params_(params) {
  loadCalibrationLuts();
}

l1t::Stage2Layer2TauAlgorithmFirmwareImp1::~Stage2Layer2TauAlgorithmFirmwareImp1() {}

void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster>& clusters,
                                                             const std::vector<l1t::CaloTower>& towers,
                                                             std::vector<l1t::Tau>& taus) {
  // fill L1 candidates collections from clusters, merging neighbour clusters
  merging(clusters, towers, taus);
  //FIXME: TO DO
  // isolation   (taus);
  dosorting(taus);
}

void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::merging(const std::vector<l1t::CaloCluster>& clusters,
                                                        const std::vector<l1t::CaloTower>& towers,
                                                        std::vector<l1t::Tau>& taus) {
  // navigator
  l1t::CaloStage2Nav caloNav;

  // this is common to all taus in this event
  const int nrTowers = CaloTools::calNrTowers(-1 * params_->tauPUSParam(1),
                                              params_->tauPUSParam(1),
                                              1,
                                              72,
                                              towers,
                                              1 + params_->pileUpTowerThreshold(),
                                              999,
                                              CaloTools::CALO);

  for (const auto& mainCluster : clusters) {
    // loop only on valid clusters
    // by construction of the clustering, they are local maxima in the 9x3 jet window
    if (mainCluster.isValid()) {
      if (abs(mainCluster.hwEta()) > params_->isoTauEtaMax())
        continue;  // limit in main seed position in firmware

      taus.emplace_back(mainCluster);
      l1t::Tau& tau = taus.back();

      int iEta = mainCluster.hwEta();
      int iPhi = mainCluster.hwPhi();
      int iEtaP = caloNav.offsetIEta(iEta, 1);
      int iEtaM = caloNav.offsetIEta(iEta, -1);
      int iPhiP = caloNav.offsetIPhi(iPhi, 1);
      int iPhiM = caloNav.offsetIPhi(iPhi, -1);
      int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
      int iPhiP3 = caloNav.offsetIPhi(iPhi, 3);
      int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
      int iPhiM3 = caloNav.offsetIPhi(iPhi, -3);

      // get list of neighbor seeds and determine the highest E one
      std::vector<l1t::CaloTower> satellites;
      const l1t::CaloTower& towerN2 = l1t::CaloTools::getTower(towers, iEta, iPhiM2);
      const l1t::CaloTower& towerN3 = l1t::CaloTools::getTower(towers, iEta, iPhiM3);
      const l1t::CaloTower& towerN2W = l1t::CaloTools::getTower(towers, iEtaM, iPhiM2);
      const l1t::CaloTower& towerN2E = l1t::CaloTools::getTower(towers, iEtaP, iPhiM2);
      const l1t::CaloTower& towerS2 = l1t::CaloTools::getTower(towers, iEta, iPhiP2);
      const l1t::CaloTower& towerS3 = l1t::CaloTools::getTower(towers, iEta, iPhiP3);
      const l1t::CaloTower& towerS2W = l1t::CaloTools::getTower(towers, iEtaM, iPhiP2);
      const l1t::CaloTower& towerS2E = l1t::CaloTools::getTower(towers, iEtaP, iPhiP2);

      int seedThreshold = floor(params_->egSeedThreshold() / params_->towerLsbSum());

      std::vector<int> sites;  // numbering of the secondary cluster sites (seed positions)
      // get only local max, also ask that they are above seed threshold
      // FIXME : in firmware N --> larger phi but apparently only for these secondaries ... sigh ...
      // might need to revert ; ALSO check EAST / WEST
      // or at least check that everything is coherent here
      if (is3x3Maximum(towerN2, towers, caloNav) && towerN2.hwPt() >= seedThreshold &&
          !mainCluster.checkClusterFlag(CaloCluster::INCLUDE_NN))
        sites.push_back(5);
      if (is3x3Maximum(towerN3, towers, caloNav) && towerN3.hwPt() >= seedThreshold)
        sites.push_back(7);
      if (is3x3Maximum(towerN2W, towers, caloNav) && towerN2W.hwPt() >= seedThreshold)
        sites.push_back(4);
      if (is3x3Maximum(towerN2E, towers, caloNav) && towerN2E.hwPt() >= seedThreshold)
        sites.push_back(6);
      if (is3x3Maximum(towerS2, towers, caloNav) && towerS2.hwPt() >= seedThreshold &&
          !mainCluster.checkClusterFlag(CaloCluster::INCLUDE_SS))
        sites.push_back(2);
      if (is3x3Maximum(towerS3, towers, caloNav) && towerS3.hwPt() >= seedThreshold)
        sites.push_back(0);
      if (is3x3Maximum(towerS2W, towers, caloNav) && towerS2W.hwPt() >= seedThreshold)
        sites.push_back(1);
      if (is3x3Maximum(towerS2E, towers, caloNav) && towerS2E.hwPt() >= seedThreshold)
        sites.push_back(3);

      // find neighbor with highest energy, with some preference that is defined as in the firmware
      // Remember: the maxima requirement is already applied
      // For the four towers in a T
      // If cluster 0 is on a maxima, use that
      // Else if cluster 2 is on a maxima, use that
      // Else if both clusters 1 & 3 are both on maxima: Use the highest energy cluster
      // Else if cluster 1 is on a maxima, use that
      // Else if cluster 3 is on a maxima, use that
      // Else there is no candidate
      // Similarly for south
      // Then if N>S, use N
      // else S
      auto secClusters = makeSecClusters(towers, sites, mainCluster, caloNav);
      l1t::CaloCluster* secMaxN = nullptr;
      l1t::CaloCluster* secMaxS = nullptr;
      l1t::CaloCluster* secondaryCluster = nullptr;

      std::vector<int>::iterator isNeigh0 = find(sites.begin(), sites.end(), 0);
      std::vector<int>::iterator isNeigh1 = find(sites.begin(), sites.end(), 1);
      std::vector<int>::iterator isNeigh2 = find(sites.begin(), sites.end(), 2);
      std::vector<int>::iterator isNeigh3 = find(sites.begin(), sites.end(), 3);
      std::vector<int>::iterator isNeigh4 = find(sites.begin(), sites.end(), 4);
      std::vector<int>::iterator isNeigh5 = find(sites.begin(), sites.end(), 5);
      std::vector<int>::iterator isNeigh6 = find(sites.begin(), sites.end(), 6);
      std::vector<int>::iterator isNeigh7 = find(sites.begin(), sites.end(), 7);

      // N neighbor --------------------------------------------------
      if (isNeigh0 != sites.end())
        secMaxN = secClusters.at(isNeigh0 - sites.begin()).get();
      else if (isNeigh2 != sites.end())
        secMaxN = secClusters.at(isNeigh2 - sites.begin()).get();

      else if (isNeigh1 != sites.end() && isNeigh3 != sites.end()) {
        if ((secClusters.at(isNeigh1 - sites.begin()))->hwPt() == (secClusters.at(isNeigh3 - sites.begin()))->hwPt()) {
          // same E --> take 1
          if (mainCluster.hwEta() > 0)
            secMaxN = secClusters.at(isNeigh1 - sites.begin()).get();
          else
            secMaxN = secClusters.at(isNeigh3 - sites.begin()).get();
        }

        else {
          if ((secClusters.at(isNeigh1 - sites.begin()))->hwPt() > (secClusters.at(isNeigh3 - sites.begin()))->hwPt()) {
            secMaxN = secClusters.at(isNeigh1 - sites.begin()).get();
          } else
            secMaxN = secClusters.at(isNeigh3 - sites.begin()).get();
        }

      }

      else if (isNeigh1 != sites.end())
        secMaxN = secClusters.at(isNeigh1 - sites.begin()).get();
      else if (isNeigh3 != sites.end())
        secMaxN = secClusters.at(isNeigh3 - sites.begin()).get();

      // S neighbor --------------------------------------------------
      if (isNeigh7 != sites.end())
        secMaxS = secClusters.at(isNeigh7 - sites.begin()).get();
      else if (isNeigh5 != sites.end())
        secMaxS = secClusters.at(isNeigh5 - sites.begin()).get();

      else if (isNeigh4 != sites.end() && isNeigh6 != sites.end()) {
        if ((secClusters.at(isNeigh4 - sites.begin()))->hwPt() == (secClusters.at(isNeigh6 - sites.begin()))->hwPt()) {
          // same E --> take 1
          if (mainCluster.hwEta() > 0)
            secMaxN = secClusters.at(isNeigh4 - sites.begin()).get();
          else
            secMaxN = secClusters.at(isNeigh6 - sites.begin()).get();
        }

        else {
          if ((secClusters.at(isNeigh4 - sites.begin()))->hwPt() > (secClusters.at(isNeigh6 - sites.begin()))->hwPt())
            secMaxS = secClusters.at(isNeigh4 - sites.begin()).get();
          else
            secMaxS = secClusters.at(isNeigh6 - sites.begin()).get();
        }

      }

      else if (isNeigh4 != sites.end())
        secMaxS = secClusters.at(isNeigh4 - sites.begin()).get();
      else if (isNeigh6 != sites.end())
        secMaxS = secClusters.at(isNeigh6 - sites.begin()).get();

      // N vs S neighbor --------------------------------------------------
      if (secMaxN != nullptr && secMaxS != nullptr) {
        if (secMaxN->hwPt() > secMaxS->hwPt())
          secondaryCluster = secMaxN;
        else
          secondaryCluster = secMaxS;
      } else {
        if (secMaxN != nullptr)
          secondaryCluster = secMaxN;
        else if (secMaxS != nullptr)
          secondaryCluster = secMaxS;
      }

      // trim primary cluster to remove overlap of TT
      int neigEta[8] = {0, -1, 0, 1, -1, 0, 1, 0};
      int neigPhi[8] = {3, 2, 2, 2, -2, -2, -2, -3};

      vector<pair<int, int>> TTPos(10);  // numbering of TT in cluster; <iEta, iPhi>
      TTPos.at(0) = make_pair(-1, 1);    // SW
      TTPos.at(1) = make_pair(0, 1);     // S
      TTPos.at(2) = make_pair(1, 1);     // SE
      TTPos.at(3) = make_pair(1, 0);     // E
      TTPos.at(4) = make_pair(1, -1);    // NE
      TTPos.at(5) = make_pair(0, -1);    // N
      TTPos.at(6) = make_pair(-1, -1);   // NW
      TTPos.at(7) = make_pair(-1, 0);    // W
      TTPos.at(8) = make_pair(0, 2);     // SS
      TTPos.at(9) = make_pair(0, -2);    // NN

      vector<CaloCluster::ClusterFlag> TTPosRemap(10);  // using geographical notation
      TTPosRemap.at(0) = CaloCluster::INCLUDE_SW;
      TTPosRemap.at(1) = CaloCluster::INCLUDE_S;
      TTPosRemap.at(2) = CaloCluster::INCLUDE_SE;
      TTPosRemap.at(3) = CaloCluster::INCLUDE_E;
      TTPosRemap.at(4) = CaloCluster::INCLUDE_NE;
      TTPosRemap.at(5) = CaloCluster::INCLUDE_N;
      TTPosRemap.at(6) = CaloCluster::INCLUDE_NW;
      TTPosRemap.at(7) = CaloCluster::INCLUDE_W;
      TTPosRemap.at(8) = CaloCluster::INCLUDE_SS;
      TTPosRemap.at(9) = CaloCluster::INCLUDE_NN;

      // loop over TT of secondary cluster, if there is overlap remove this towers from main cluster
      l1t::CaloCluster mainClusterTrim = mainCluster;
      int secondaryClusterHwPt = 0;

      if (!secClusters.empty()) {
        secondaryClusterHwPt = secondaryCluster->hwPt();

        int iSecIdxPosition = find_if(secClusters.begin(),
                                      secClusters.end(),
                                      [&](auto const& element) { return element.get() == secondaryCluster; }) -
                              secClusters.begin();
        int secondaryClusterSite = sites.at(iSecIdxPosition);

        for (unsigned int iTT = 0; iTT < TTPos.size(); iTT++) {
          //get this TT in the "frame" of the main
          int thisTTinMainEta = neigEta[secondaryClusterSite] + TTPos.at(iTT).first;
          int thisTTinMainPhi = neigPhi[secondaryClusterSite] + TTPos.at(iTT).second;
          pair<int, int> thisTT = make_pair(thisTTinMainEta, thisTTinMainPhi);
          // check if main cluster has this tower included; if true, switch it off
          auto thisTTItr = find(TTPos.begin(), TTPos.end(), thisTT);
          if (thisTTItr != TTPos.end()) {
            int idx = thisTTItr - TTPos.begin();
            if (secondaryCluster->checkClusterFlag(TTPosRemap.at(iTT)))
              mainClusterTrim.setClusterFlag(TTPosRemap.at(idx), false);
          }
        }
      }

      // re-compute main cluster energy + position
      const l1t::CaloTower& seed = l1t::CaloTools::getTower(towers, iEta, iPhi);
      const l1t::CaloTower& towerNW = l1t::CaloTools::getTower(towers, iEtaM, iPhiM);
      const l1t::CaloTower& towerN = l1t::CaloTools::getTower(towers, iEta, iPhiM);
      const l1t::CaloTower& towerNE = l1t::CaloTools::getTower(towers, iEtaP, iPhiM);
      const l1t::CaloTower& towerE = l1t::CaloTools::getTower(towers, iEtaP, iPhi);
      const l1t::CaloTower& towerSE = l1t::CaloTools::getTower(towers, iEtaP, iPhiP);
      const l1t::CaloTower& towerS = l1t::CaloTools::getTower(towers, iEta, iPhiP);
      const l1t::CaloTower& towerSW = l1t::CaloTools::getTower(towers, iEtaM, iPhiP);
      const l1t::CaloTower& towerW = l1t::CaloTools::getTower(towers, iEtaM, iPhi);
      const l1t::CaloTower& towerNN = l1t::CaloTools::getTower(towers, iEta, iPhiM2);
      const l1t::CaloTower& towerSS = l1t::CaloTools::getTower(towers, iEta, iPhiP2);

      int seedEt = seed.hwPt();
      int towerEtNW = towerNW.hwPt();
      int towerEtN = towerN.hwPt();
      int towerEtNE = towerNE.hwPt();
      int towerEtE = towerE.hwPt();
      int towerEtSE = towerSE.hwPt();
      int towerEtS = towerS.hwPt();
      int towerEtSW = towerSW.hwPt();
      int towerEtW = towerW.hwPt();
      int towerEtNN = towerNN.hwPt();
      int towerEtSS = towerSS.hwPt();

      // Recompute hw energy from towers
      tau.setHwPt(seedEt);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NW))
        tau.setHwPt(tau.hwPt() + towerEtNW);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_N))
        tau.setHwPt(tau.hwPt() + towerEtN);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NE))
        tau.setHwPt(tau.hwPt() + towerEtNE);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_E))
        tau.setHwPt(tau.hwPt() + towerEtE);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SE))
        tau.setHwPt(tau.hwPt() + towerEtSE);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_S))
        tau.setHwPt(tau.hwPt() + towerEtS);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SW))
        tau.setHwPt(tau.hwPt() + towerEtSW);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_W))
        tau.setHwPt(tau.hwPt() + towerEtW);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NN))
        tau.setHwPt(tau.hwPt() + towerEtNN);
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SS))
        tau.setHwPt(tau.hwPt() + towerEtSS);

      //Merging
      tau.setRawEt(tau.hwPt() + secondaryClusterHwPt);
      tau.setIsMerged(secondaryClusterHwPt > 0);

      // ==================================================================
      // Energy calibration
      // ==================================================================
      // Corrections function of ieta, ET, and cluster shape
      int calibPt = calibratedPt(mainCluster, towers, tau.rawEt(), tau.isMerged());
      tau.setHwPt(calibPt);

      // isolation
      int isoLeftExtension = params_->tauIsoAreaNrTowersEta();
      int isoRightExtension = params_->tauIsoAreaNrTowersEta();
      if (mainCluster.checkClusterFlag(CaloCluster::TRIM_LEFT))
        isoRightExtension++;
      else
        isoLeftExtension++;

      int isolBit = 0;
      int tauHwFootprint = tau.rawEt();
      unsigned int LUTaddress = isoLutIndex(tauHwFootprint, mainCluster.hwEta(), nrTowers);
      int hwEtSum = CaloTools::calHwEtSum(iEta,
                                          iPhi,
                                          towers,
                                          -isoLeftExtension,
                                          isoRightExtension,
                                          -1 * params_->tauIsoAreaNrTowersPhi(),
                                          params_->tauIsoAreaNrTowersPhi(),
                                          params_->tauPUSParam(2),
                                          CaloTools::CALO);
      int hwIsoEnergy = hwEtSum - tauHwFootprint;
      if (hwIsoEnergy < 0)
        hwIsoEnergy = 0;  // just in case the cluster is outside the window? should be very rare

      isolBit = (((hwIsoEnergy < (params_->tauIsolationLUT()->data(LUTaddress))) ||
                  (params_->tauIsolationLUT()->data(LUTaddress) > 255))
                     ? 1
                     : 0);
      //int isolBit2 = (((hwIsoEnergy < (params_->tauIsolationLUT2()->data(LUTaddress))) || (params_->tauIsolationLUT2()->data(LUTaddress)>255)) ? 1 : 0);
      //isolBit += (isolBit2 << 1);
      tau.setHwIso(isolBit);

      // development vars
      tau.setTowerIPhi(iPhi);
      tau.setTowerIEta(iEta);
      int qual = seed.hwQual();
      bool denomZeroFlag = ((qual & 0x1) > 0);
      bool eOverHFlag = ((qual & 0x2) > 0);
      int hasEM = (eOverHFlag || !denomZeroFlag);
      tau.setHasEM(hasEM > 0);
      tau.setIsoEt((short int)hwIsoEnergy);
      tau.setNTT((short int)nrTowers);

      // Physical eta/phi. Computed from ieta/iphi of the seed tower and the fine-grain position within the seed
      double eta = 0.;
      double phi = 0.;
      double seedEta = CaloTools::towerEta(iEta);
      double seedEtaSize = CaloTools::towerEtaSize(iEta);
      double seedPhi = CaloTools::towerPhi(iEta, iPhi);
      double seedPhiSize = CaloTools::towerPhiSize(iEta);
      if (mainCluster.fgEta() == 0)
        eta = seedEta;  // center
      else if (mainCluster.fgEta() == 2)
        eta = seedEta + seedEtaSize * 0.251;  // center + 1/4
      else if (mainCluster.fgEta() == 1)
        eta = seedEta - seedEtaSize * 0.251;  // center - 1/4

      //fgPhi is recomputed after trimming
      int fgPhi = 0;

      int EtUp = 0;
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NE))
        EtUp += towerEtNE;
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_N))
        EtUp += towerEtN;
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NW))
        EtUp += towerEtNW;
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_NN))
        EtUp += towerEtNN;
      int EtDown = 0;
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SE))
        EtDown += towerEtSE;
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_S))
        EtDown += towerEtS;
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SW))
        EtDown += towerEtSW;
      if (mainClusterTrim.checkClusterFlag(CaloCluster::INCLUDE_SS))
        EtDown += towerEtSS;
      //
      if (EtDown > EtUp)
        fgPhi = 2;
      else if (EtUp > EtDown)
        fgPhi = 1;

      if (fgPhi == 0)
        phi = seedPhi;  // center
      else if (fgPhi == 2)
        phi = seedPhi + seedPhiSize * 0.251;  // center + 1/4
      else if (fgPhi == 1)
        phi = seedPhi - seedPhiSize * 0.251;  // center - 1/4

      // Set 4-vector
      math::PtEtaPhiMLorentzVector calibP4((double)calibPt * params_->tauLsb(), eta, phi, 0.);
      tau.setP4(calibP4);

      // delete all sec clusters that were allocated with new
      secClusters.clear();
    }

  }  // end loop on clusters
}

// -----------------------------------------------------------------------------------
void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::dosorting(std::vector<l1t::Tau>& taus) {
  // prepare content to be sorted -- each phi ring contains 18 elements, with Et = 0 if no candidate exists
  math::PtEtaPhiMLorentzVector emptyP4;
  l1t::Tau tempTau(emptyP4, 0, 0, 0, 0);
  std::vector<std::vector<l1t::Tau>> tauEtaPos(params_->isoTauEtaMax(), std::vector<l1t::Tau>(18, tempTau));
  std::vector<std::vector<l1t::Tau>> tauEtaNeg(params_->isoTauEtaMax(), std::vector<l1t::Tau>(18, tempTau));

  for (unsigned int iTau = 0; iTau < taus.size(); iTau++) {
    if (taus.at(iTau).hwEta() > 0)
      tauEtaPos.at(taus.at(iTau).hwEta() - 1).at((72 - taus.at(iTau).hwPhi()) / 4) = taus.at(iTau);
    else
      tauEtaNeg.at(-(taus.at(iTau).hwEta() + 1)).at((72 - taus.at(iTau).hwPhi()) / 4) = taus.at(iTau);
  }

  AccumulatingSort<l1t::Tau> etaPosSorter(6);
  AccumulatingSort<l1t::Tau> etaNegSorter(6);
  std::vector<l1t::Tau> accumEtaPos;
  std::vector<l1t::Tau> accumEtaNeg;

  for (int ieta = 0; ieta < params_->isoTauEtaMax(); ++ieta) {
    // eta +
    std::vector<l1t::Tau>::iterator start_, end_;
    start_ = tauEtaPos.at(ieta).begin();
    end_ = tauEtaPos.at(ieta).end();
    BitonicSort<l1t::Tau>(down, start_, end_);
    etaPosSorter.Merge(tauEtaPos.at(ieta), accumEtaPos);

    // eta -
    start_ = tauEtaNeg.at(ieta).begin();
    end_ = tauEtaNeg.at(ieta).end();
    BitonicSort<l1t::Tau>(down, start_, end_);
    etaNegSorter.Merge(tauEtaNeg.at(ieta), accumEtaNeg);
  }

  // put all 12 candidates in the original tau vector, removing zero energy ones
  taus.clear();
  for (const l1t::Tau& acctau : accumEtaPos) {
    if (acctau.hwPt() > 0)
      taus.push_back(acctau);
  }
  for (const l1t::Tau& acctau : accumEtaNeg) {
    if (acctau.hwPt() > 0)
      taus.push_back(acctau);
  }
}

// -----------------------------------------------------------------------------------

void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::loadCalibrationLuts() {
  float minScale = 0.;
  float maxScale = 2.;
  float minScaleEta = 0.5;
  float maxScaleEta = 1.5;
  offsetBarrelEH_ = 0.5;
  offsetBarrelH_ = 1.5;
  offsetEndcapsEH_ = 0.;
  offsetEndcapsH_ = 1.5;

  // In the combined calibration LUT, upper 3-bits are used as LUT index:
  // (0=BarrelA, 1=BarrelB, 2=BarrelC, 3=EndCapA, 4=EndCapA, 5=EndCapA, 6=Eta)
  enum { LUT_UPPER = 3 };
  enum { LUT_OFFSET = 0x80 };
  l1t::LUT const* lut = params_->tauCalibrationLUT();
  unsigned int size = (1 << lut->nrBitsData());
  unsigned int nBins = (1 << (lut->nrBitsAddress() - LUT_UPPER));

  std::vector<float> emptyCoeff;
  emptyCoeff.resize(nBins, 0.);
  float binSize = (maxScale - minScale) / (float)size;
  for (unsigned iLut = 0; iLut < 6; ++iLut) {
    coefficients_.push_back(emptyCoeff);
    for (unsigned addr = 0; addr < nBins; addr++) {
      float y = (float)lut->data(iLut * LUT_OFFSET + addr);
      coefficients_[iLut][addr] = minScale + binSize * y;
    }
  }

  size = (1 << lut->nrBitsData());
  nBins = (1 << 6);  // can't auto-extract this now due to combined LUT.

  emptyCoeff.resize(nBins, 0.);
  binSize = (maxScaleEta - minScaleEta) / (float)size;
  coefficients_.push_back(emptyCoeff);
  for (unsigned addr = 0; addr < nBins; addr++) {
    float y = (float)lut->data(6 * LUT_OFFSET + addr);
    coefficients_.back()[addr] = minScaleEta + binSize * y;
  }
}

bool l1t::Stage2Layer2TauAlgorithmFirmwareImp1::compareTowers(l1t::CaloTower TT1, l1t::CaloTower TT2) {
  // 1. compare hwPt (for the moment no switch with E and H only, always use E+H)
  if (TT1.hwPt() < TT2.hwPt())
    return true;
  if (TT1.hwPt() > TT2.hwPt())
    return false;

  // 2. if equal pT, most central -- eta is in the range -32, 32 with ieta = 0 skipped
  if (abs(TT1.hwEta()) > abs(TT2.hwEta()))
    return true;
  if (abs(TT1.hwEta()) < abs(TT2.hwEta()))
    return false;

  // 3. if equal eta, compare phi (arbitrary)
  return (TT1.hwPhi() < TT2.hwPhi());  // towers towards S are favored (remember: N --> smaller phi, S --> larger phi)
}

bool l1t::Stage2Layer2TauAlgorithmFirmwareImp1::is3x3Maximum(const l1t::CaloTower& tower,
                                                             const std::vector<CaloTower>& towers,
                                                             l1t::CaloStage2Nav& caloNav) {
  int iEta = tower.hwEta();
  int iPhi = tower.hwPhi();

  // 1 : >
  // 2 : >=
  int mask[3][3] = {
      {1, 2, 2},
      {1, 0, 2},
      {1, 1, 2},
  };

  bool vetoTT = false;  // false if it is a local max i.e. no veto
  for (int deta = -1; deta < 2; deta++) {
    for (int dphi = -1; dphi < 2; dphi++) {
      int iEtaNeigh = caloNav.offsetIEta(iEta, deta);
      int iPhiNeigh = caloNav.offsetIPhi(iPhi, dphi);
      const l1t::CaloTower& towerNeigh = l1t::CaloTools::getTower(towers, iEtaNeigh, iPhiNeigh);
      if (mask[2 - (dphi + 1)][deta + 1] == 0)
        continue;
      if (mask[2 - (dphi + 1)][deta + 1] == 1)
        vetoTT = (tower.hwPt() < towerNeigh.hwPt());
      if (mask[2 - (dphi + 1)][deta + 1] == 2)
        vetoTT = (tower.hwPt() <= towerNeigh.hwPt());

      if (vetoTT)
        break;
    }
    if (vetoTT)
      break;
  }

  return (!vetoTT);  // negate because I ask if is a local maxima
}

std::vector<std::unique_ptr<l1t::CaloCluster>> l1t::Stage2Layer2TauAlgorithmFirmwareImp1::makeSecClusters(
    const std::vector<l1t::CaloTower>& towers,
    std::vector<int>& sites,
    const l1t::CaloCluster& mainCluster,
    l1t::CaloStage2Nav& caloNav) {
  constexpr int neigEta[8] = {0, -1, 0, 1, -1, 0, 1, 0};
  constexpr int neigPhi[8] = {3, 2, 2, 2, -2, -2, -2, -3};
  int clusterThreshold = floor(params_->egNeighbourThreshold() / params_->towerLsbSum());

  int iEtamain = mainCluster.hwEta();
  int iPhimain = mainCluster.hwPhi();

  std::vector<unique_ptr<CaloCluster>> secClusters;
  for (unsigned int isite = 0; isite < sites.size(); isite++) {
    // build full cluster at this site
    const int siteNumber = sites.at(isite);
    int iSecEta = caloNav.offsetIEta(iEtamain, neigEta[siteNumber]);
    int iSecPhi = caloNav.offsetIPhi(iPhimain, neigPhi[siteNumber]);

    const l1t::CaloTower& towerSec = l1t::CaloTools::getTower(towers, iSecEta, iSecPhi);

    math::XYZTLorentzVector emptyP4;
    auto secondaryCluster =
        std::make_unique<l1t::CaloCluster>(emptyP4, towerSec.hwPt(), towerSec.hwEta(), towerSec.hwPhi());

    secondaryCluster->setHwPtEm(towerSec.hwEtEm());
    secondaryCluster->setHwPtHad(towerSec.hwEtHad());
    secondaryCluster->setHwSeedPt(towerSec.hwPt());
    secondaryCluster->setHwPt(towerSec.hwPt());

    int iSecEtaP = caloNav.offsetIEta(iSecEta, 1);
    int iSecEtaM = caloNav.offsetIEta(iSecEta, -1);
    int iSecPhiP = caloNav.offsetIPhi(iSecPhi, 1);
    int iSecPhiP2 = caloNav.offsetIPhi(iSecPhi, 2);
    int iSecPhiM = caloNav.offsetIPhi(iSecPhi, -1);
    int iSecPhiM2 = caloNav.offsetIPhi(iSecPhi, -2);
    const l1t::CaloTower& towerNW = l1t::CaloTools::getTower(towers, iSecEtaM, iSecPhiM);
    const l1t::CaloTower& towerN = l1t::CaloTools::getTower(towers, iSecEta, iSecPhiM);
    const l1t::CaloTower& towerNE = l1t::CaloTools::getTower(towers, iSecEtaP, iSecPhiM);
    const l1t::CaloTower& towerE = l1t::CaloTools::getTower(towers, iSecEtaP, iSecPhi);
    const l1t::CaloTower& towerSE = l1t::CaloTools::getTower(towers, iSecEtaP, iSecPhiP);
    const l1t::CaloTower& towerS = l1t::CaloTools::getTower(towers, iSecEta, iSecPhiP);
    const l1t::CaloTower& towerSW = l1t::CaloTools::getTower(towers, iSecEtaM, iSecPhiP);
    const l1t::CaloTower& towerW = l1t::CaloTools::getTower(towers, iSecEtaM, iSecPhi);
    const l1t::CaloTower& towerNN = l1t::CaloTools::getTower(towers, iSecEta, iSecPhiM2);
    const l1t::CaloTower& towerSS = l1t::CaloTools::getTower(towers, iSecEta, iSecPhiP2);

    // just use E+H for clustering
    int towerEtNW = towerNW.hwPt();
    int towerEtN = towerN.hwPt();
    int towerEtNE = towerNE.hwPt();
    int towerEtE = towerE.hwPt();
    int towerEtSE = towerSE.hwPt();
    int towerEtS = towerS.hwPt();
    int towerEtSW = towerSW.hwPt();
    int towerEtW = towerW.hwPt();
    int towerEtNN = towerNN.hwPt();
    int towerEtSS = towerSS.hwPt();

    int towerEtEmNW = towerNW.hwEtEm();
    int towerEtEmN = towerN.hwEtEm();
    int towerEtEmNE = towerNE.hwEtEm();
    int towerEtEmE = towerE.hwEtEm();
    int towerEtEmSE = towerSE.hwEtEm();
    int towerEtEmS = towerS.hwEtEm();
    int towerEtEmSW = towerSW.hwEtEm();
    int towerEtEmW = towerW.hwEtEm();
    int towerEtEmNN = towerNN.hwEtEm();
    int towerEtEmSS = towerSS.hwEtEm();
    //
    int towerEtHadNW = towerNW.hwEtHad();
    int towerEtHadN = towerN.hwEtHad();
    int towerEtHadNE = towerNE.hwEtHad();
    int towerEtHadE = towerE.hwEtHad();
    int towerEtHadSE = towerSE.hwEtHad();
    int towerEtHadS = towerS.hwEtHad();
    int towerEtHadSW = towerSW.hwEtHad();
    int towerEtHadW = towerW.hwEtHad();
    int towerEtHadNN = towerNN.hwEtHad();
    int towerEtHadSS = towerSS.hwEtHad();

    if (towerEtNW < clusterThreshold)
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_NW, false);
    if (towerEtN < clusterThreshold) {
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_N, false);
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_NN, false);
    }
    if (towerEtNE < clusterThreshold)
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_NE, false);
    if (towerEtE < clusterThreshold)
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_E, false);
    if (towerEtSE < clusterThreshold)
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_SE, false);
    if (towerEtS < clusterThreshold) {
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_S, false);
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_SS, false);
    }
    if (towerEtSW < clusterThreshold)
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_SW, false);
    if (towerEtW < clusterThreshold)
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_W, false);
    if (towerEtNN < clusterThreshold)
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_NN, false);
    if (towerEtSS < clusterThreshold)
      secondaryCluster->setClusterFlag(CaloCluster::INCLUDE_SS, false);

    // compute cluster energy according to cluster flags
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_NW))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtNW);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_N))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtN);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_NE))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtNE);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_E))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtE);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_SE))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtSE);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_S))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtS);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_SW))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtSW);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_W))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtW);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_NN))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtNN);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_SS))
      secondaryCluster->setHwPt(secondaryCluster->hwPt() + towerEtSS);
    //
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_NW))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmNW);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_N))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmN);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_NE))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmNE);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_E))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmE);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_SE))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmSE);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_S))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmS);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_SW))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmSW);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_W))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmW);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_NN))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmNN);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_SS))
      secondaryCluster->setHwPtEm(secondaryCluster->hwPtEm() + towerEtEmSS);
    //
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_NW))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadNW);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_N))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadN);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_NE))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadNE);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_E))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadE);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_SE))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadSE);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_S))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadS);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_SW))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadSW);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_W))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadW);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_NN))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadNN);
    if (secondaryCluster->checkClusterFlag(CaloCluster::INCLUDE_SS))
      secondaryCluster->setHwPtHad(secondaryCluster->hwPtHad() + towerEtHadSS);

    // save this cluster in the vector
    secClusters.emplace_back(std::move(secondaryCluster));
  }
  return secClusters;
}

// isMerged=0,1 ; hasEM=0,1
unsigned int l1t::Stage2Layer2TauAlgorithmFirmwareImp1::calibLutIndex(int ieta, int Et, int hasEM, int isMerged) {
  int absieta = abs(ieta);
  if (absieta > 28)
    absieta = 28;
  if (Et > 255)
    Et = 255;

  unsigned int compressedEta = params_->tauCompressLUT()->data(absieta);
  unsigned int compressedEt = params_->tauCompressLUT()->data((0x1 << 5) + Et);

  unsigned int address = (compressedEta << 7) + (compressedEt << 2) + (hasEM << 1) +
                         isMerged;  //2 bits eta, 5 bits et, 1 bit hasEM, 1 bis isMerged
  // unsigned int address =  (compressedEt<<4)+(compressedEta<<2)+(hasEM<<1)+isMerged;
  return address;
}

int l1t::Stage2Layer2TauAlgorithmFirmwareImp1::calibratedPt(const l1t::CaloCluster& clus,
                                                            const std::vector<l1t::CaloTower>& towers,
                                                            int hwPt,
                                                            bool isMerged) {
  // get seed tower
  const l1t::CaloTower& seedTT = l1t::CaloTools::getTower(towers, CaloTools::caloEta(clus.hwEta()), clus.hwPhi());
  int qual = seedTT.hwQual();
  bool denomZeroFlag = ((qual & 0x1) > 0);
  bool eOverHFlag = ((qual & 0x2) > 0);
  int hasEM = (eOverHFlag || !denomZeroFlag);
  int isMergedI = (isMerged ? 1 : 0);

  unsigned int idx = calibLutIndex(clus.hwEta(), hwPt, hasEM, isMergedI);
  unsigned int corr = params_->tauCalibrationLUT()->data(idx);

  // now apply calibration factor: corrPt = rawPt * (corr[LUT] + 0.5)
  // where corr[LUT] is an integer mapped to the range [0, 2]
  int rawPt = hwPt;
  if (rawPt > 8191)
    rawPt = 8191;  // 13 bits for uncalibrated E

  int corrXrawPt = corr * rawPt;    // 17 bits
  int calibPt = (corrXrawPt >> 9);  // (10 bits) = (7 bits) + (9 bits)
  if (calibPt > 4095)
    calibPt = 4095;  // 12 bit in output

  return calibPt;
}

unsigned int l1t::Stage2Layer2TauAlgorithmFirmwareImp1::isoLutIndex(int Et, int hweta, unsigned int nrTowers) {
  // normalize to limits
  int aeta = abs(hweta);

  // input bits (NB: must be THE SAME in the input LUT for the compression)
  // int etaBits = 6  --> 64
  // int etBits  = 13 --> 8192
  // int nTTBits = 10 --> 1024
  if (Et >= 255)
    Et = 255;  // 8 bit for the input of compression LUT
  if (aeta >= 31)
    aeta = 31;
  if (nrTowers >= 1023)
    nrTowers = 1023;

  // get compressed value
  // NB: these also must MATCH the values in the LUT --> fix when new compression scheme is used
  // ultimately, the same compresison LUT as calibration will be used
  // etaCmprBits = 2;
  // EtCmprBits  = 4;//changed from 3, transparent to user
  // nTTCmprBits = 3;

  int etaCmpr = params_->tauCompressLUT()->data(aeta);
  int etCmpr = params_->tauCompressLUT()->data((0x1 << 5) + Et);  //offset: 5 bits from ieta
  int nTTCmpr = params_->tauCompressLUT()->data((0x1 << 5) + (0x1 << 8) +
                                                nrTowers);  //offset non-compressed: 5 bits from ieta, 8 bits from iEt

  // get the address -- NOTE: this also depends on the compression scheme!

  unsigned int address =
      ((etaCmpr << 10) | (etCmpr << 5) | nTTCmpr);  //ordering compressed: 2 bits iEta, 5 bits iEt, 5 bits nTT

  return address;
}
