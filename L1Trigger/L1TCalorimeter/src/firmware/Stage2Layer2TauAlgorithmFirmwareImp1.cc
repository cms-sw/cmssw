//
// ** class l1t::Stage2Layer2TauAlgorithmFirmwareImp1
// ** authors: L. Cadamuro, L. Mastrolorenzo, J. Brooke
// ** date:   2 Oct 2015
// ** Description: version of tau algorithm matching the jet-eg-tau merged implementation

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2TauAlgorithmFirmware.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"
#include "L1Trigger/L1TCalorimeter/interface/BitonicSort.h"

namespace l1t {
  bool operator > ( l1t::Tau& a, l1t::Tau& b )
  {
    if ( a.pt() == b.pt() ){
      if( a.hwPhi() == b.hwPhi() )
    return abs(a.hwEta()) > abs(b.hwEta());
      else
    return a.hwPhi() > b.hwPhi();
    }
    else
      return a.pt() > b.pt();
  }
}

l1t::Stage2Layer2TauAlgorithmFirmwareImp1::Stage2Layer2TauAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{

  loadCalibrationLuts();
}

l1t::Stage2Layer2TauAlgorithmFirmwareImp1::~Stage2Layer2TauAlgorithmFirmwareImp1() {
}

void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster> & clusters,
                                                             const std::vector<l1t::CaloTower>& towers,
                                                             std::vector<l1t::Tau> & taus) {
  
  // fill L1 candidates collections from clusters, merging neighbour clusters
  merging (clusters, towers, taus); 
  //FIXME: TO DO
  // isolation   (taus);
  dosorting(taus);
}

void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::merging(const std::vector<l1t::CaloCluster>& clusters,
                                                        const std::vector<l1t::CaloTower>& towers,
                                                        std::vector<l1t::Tau>& taus)
{
    // navigator
    l1t::CaloStage2Nav caloNav; 
  
    for (const auto& mainCluster : clusters)
    {
        // loop only on valid clusters
        // by construction of the clustering, they are local maxima in the 9x3 jet window
        if (mainCluster.isValid())
        {
            int iEta = mainCluster.hwEta();
            int iPhi = mainCluster.hwPhi();
            int iEtaP = caloNav.offsetIEta(iEta, 1);
            int iEtaM = caloNav.offsetIEta(iEta, -1);
            int iPhiP2 = caloNav.offsetIPhi(iPhi, 2);
            int iPhiP3 = caloNav.offsetIPhi(iPhi, 3);
            int iPhiM2 = caloNav.offsetIPhi(iPhi, -2);
            int iPhiM3 = caloNav.offsetIPhi(iPhi, -3);

            // get list of neighbor seeds and determine the highest E one
            std::vector<l1t::CaloTower> satellites;
            const l1t::CaloTower& towerN2  = l1t::CaloTools::getTower(towers, iEta,  iPhiM2);
            const l1t::CaloTower& towerN3  = l1t::CaloTools::getTower(towers, iEta,  iPhiM3);
            const l1t::CaloTower& towerN2W = l1t::CaloTools::getTower(towers, iEtaM, iPhiM2);
            const l1t::CaloTower& towerN2E = l1t::CaloTools::getTower(towers, iEtaP, iPhiM2);
            const l1t::CaloTower& towerS2  = l1t::CaloTools::getTower(towers, iEta,  iPhiP2);
            const l1t::CaloTower& towerS3  = l1t::CaloTools::getTower(towers, iEta,  iPhiP3);
            const l1t::CaloTower& towerS2W = l1t::CaloTools::getTower(towers, iEtaM, iPhiP2);
            const l1t::CaloTower& towerS2E = l1t::CaloTools::getTower(towers, iEtaP, iPhiP2);

            int seedThreshold    = floor(params_->egSeedThreshold()/params_->towerLsbSum()); 
            int clusterThreshold = floor(params_->egNeighbourThreshold()/params_->towerLsbSum());

            // get only local max, also ask that they are above seed threshold
            if (is3x3Maximum(towerN2, towers, caloNav)  && towerN2.hwPt()  >= seedThreshold && !mainCluster.checkClusterFlag(CaloCluster::INCLUDE_NN)) 
                satellites.push_back(towerN2);
            if (is3x3Maximum(towerN3, towers, caloNav)  && towerN3.hwPt()  >= seedThreshold)
                satellites.push_back(towerN3);
            if (is3x3Maximum(towerN2W, towers, caloNav) && towerN2W.hwPt() >= seedThreshold)
                satellites.push_back(towerN2W);
            if (is3x3Maximum(towerN2E, towers, caloNav) && towerN2E.hwPt() >= seedThreshold)
                satellites.push_back(towerN2E);
            if (is3x3Maximum(towerS2, towers, caloNav)  && towerS2.hwPt()  >= seedThreshold && !mainCluster.checkClusterFlag(CaloCluster::INCLUDE_SS)) 
                satellites.push_back(towerS2);
            if (is3x3Maximum(towerS3, towers, caloNav)  && towerS3.hwPt()  >= seedThreshold)
                satellites.push_back(towerS3);
            if (is3x3Maximum(towerS2W, towers, caloNav) && towerS2W.hwPt() >= seedThreshold)
                satellites.push_back(towerS2W);
            if (is3x3Maximum(towerS2E, towers, caloNav) && towerS2E.hwPt() >= seedThreshold)
                satellites.push_back(towerS2E);

            /*
            // get list of secondary clusters - candidates for merging in the double "T shape"
            std::vector<l1t::CaloCluster> satellites;
            const l1t::CaloCluster& clusterN2  = l1t::CaloTools::getCluster(clusters, iEta,  iPhiM2);
            const l1t::CaloCluster& clusterN3  = l1t::CaloTools::getCluster(clusters, iEta,  iPhiM3);
            const l1t::CaloCluster& clusterN2W = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiM2);
            const l1t::CaloCluster& clusterN2E = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiM2);
            const l1t::CaloCluster& clusterS2  = l1t::CaloTools::getCluster(clusters, iEta,  iPhiP2);
            const l1t::CaloCluster& clusterS3  = l1t::CaloTools::getCluster(clusters, iEta,  iPhiP3);
            const l1t::CaloCluster& clusterS2W = l1t::CaloTools::getCluster(clusters, iEtaM, iPhiP2);
            const l1t::CaloCluster& clusterS2E = l1t::CaloTools::getCluster(clusters, iEtaP, iPhiP2);

            if(clusterN2 .isValid()) satellites.push_back(clusterN2);
            if(clusterN3 .isValid()) satellites.push_back(clusterN3);
            if(clusterN2W.isValid()) satellites.push_back(clusterN2W);
            if(clusterN2E.isValid()) satellites.push_back(clusterN2E);
            if(clusterS2 .isValid()) satellites.push_back(clusterS2);
            if(clusterS3 .isValid()) satellites.push_back(clusterS3);
            if(clusterS2W.isValid()) satellites.push_back(clusterS2W);
            if(clusterS2E.isValid()) satellites.push_back(clusterS2E);
            
            

            // there are no neigbhours
            if (satellites.size() == 0)
            {
                //math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
                math::PtEtaPhiMLorentzVector p4(0., 0., 0., 0.);
                l1t::Tau tau (p4, mainCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
                taus.push_back (tau);
            }

            // there are neighbors
            else
            {
                // FIXME
                // will exploit ordering relation of clusters, based on E but also on eta and phi --> same as emulator??
                std::sort(satellites.begin(), satellites.end());
                const l1t::CaloCluster& secondaryCluster = satellites.back(); // by definition Esec < Emain
                // FIXME : the energy relation (clustering algo) is on seed or on cluster?
                // FIXME : what if energies are equal in the clustering algo?
                math::PtEtaPhiMLorentzVector p4(0., 0., 0., 0.);
                l1t::Tau tau (p4, mainCluster.hwPt()+secondaryCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
                taus.push_back (tau);
            }
            */

          
            if (satellites.size() == 0) // no merging candidate
            {
                //math::PtEtaPhiMLorentzVector p4(calibPt, eta, phi, 0.);
                math::PtEtaPhiMLorentzVector emptyP4;
                l1t::Tau tau (emptyP4, mainCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);

                // Corrections function of ieta, ET, and cluster shape
                //int calibPt = calibratedPt(cluster, egamma.hwPt()); // FIXME! for the moment no calibration
                int calibPt = mainCluster.hwPt();

                // Physical eta/phi. Computed from ieta/iphi of the seed tower and the fine-grain position within the seed
                // use fg positon of main cluster only
                double eta = 0.;
                double phi = 0.;
                double seedEta     = CaloTools::towerEta(mainCluster.hwEta());
                double seedEtaSize = CaloTools::towerEtaSize(mainCluster.hwEta());
                double seedPhi     = CaloTools::towerPhi(mainCluster.hwEta(), mainCluster.hwPhi());
                double seedPhiSize = CaloTools::towerPhiSize(mainCluster.hwEta());
                if(mainCluster.fgEta()==0)      eta = seedEta; // center
                else if(mainCluster.fgEta()==2) eta = seedEta + seedEtaSize*0.25; // center + 1/4
                else if(mainCluster.fgEta()==1) eta = seedEta - seedEtaSize*0.25; // center - 1/4
                if(mainCluster.fgPhi()==0)      phi = seedPhi; // center
                else if(mainCluster.fgPhi()==2) phi = seedPhi + seedPhiSize*0.25; // center + 1/4
                else if(mainCluster.fgPhi()==1) phi = seedPhi - seedPhiSize*0.25; // center - 1/4

                // Set 4-vector
                math::PtEtaPhiMLorentzVector calibP4((double)calibPt*params_->egLsb(), eta, phi, 0.);
                tau.setP4(calibP4);

                taus.push_back (tau);
            }

            else
            {
                std::sort(satellites.begin(), satellites.end(), compareTowers);
                l1t::CaloTower& secondaryTower = satellites.back();

                // make cluster around the selected neighbour
                math::XYZTLorentzVector emptyP4;
                l1t::CaloCluster secondaryCluster ( emptyP4, secondaryTower.hwPt(), secondaryTower.hwEta(), secondaryTower.hwPhi() ) ;
                secondaryCluster.setHwPtEm(secondaryTower.hwEtEm());
                secondaryCluster.setHwPtHad(secondaryTower.hwEtHad());
                secondaryCluster.setHwSeedPt(secondaryTower.hwPt());
                // H/E of the cluster is H/E of the seed, with possible threshold on H
                // H/E is currently encoded on 9 bits, from 0 to 1             
                // FIXME: I do not set H/E and FG of the secondary as it is not used for the moment
                // int hwEtHadTh = (tower.hwEtHad()>=hcalThreshold_ ? tower.hwEtHad() : 0);
                // int hOverE    = (tower.hwEtEm()>0 ? (hwEtHadTh<<9)/tower.hwEtEm() : 511);
                // if(hOverE>511) hOverE = 511; 
                // cluster.setHOverE(hOverE);
                // FG of the cluster is FG of the seed
                // bool fg = (secondaryTower.hwQual() & (0x1<<2));
                // secondaryCluster.setFgECAL((int)fg);
                
                // make cluster around the selected TT
                // look at the energies in neighbour towers
                int iSecEta   = secondaryCluster.hwEta();
                int iSecPhi   = secondaryCluster.hwPhi();
                int iSecEtaP  = caloNav.offsetIEta(iSecEta,  1);
                int iSecEtaM  = caloNav.offsetIEta(iSecEta, -1);
                int iSecPhiP  = caloNav.offsetIPhi(iSecPhi,  1);
                int iSecPhiP2 = caloNav.offsetIPhi(iSecPhi,  2);
                int iSecPhiM  = caloNav.offsetIPhi(iSecPhi, -1);
                int iSecPhiM2 = caloNav.offsetIPhi(iSecPhi, -2);
                const l1t::CaloTower& towerNW = l1t::CaloTools::getTower(towers, iSecEtaM, iSecPhiM);
                const l1t::CaloTower& towerN  = l1t::CaloTools::getTower(towers, iSecEta , iSecPhiM);
                const l1t::CaloTower& towerNE = l1t::CaloTools::getTower(towers, iSecEtaP, iSecPhiM);
                const l1t::CaloTower& towerE  = l1t::CaloTools::getTower(towers, iSecEtaP, iSecPhi );
                const l1t::CaloTower& towerSE = l1t::CaloTools::getTower(towers, iSecEtaP, iSecPhiP);
                const l1t::CaloTower& towerS  = l1t::CaloTools::getTower(towers, iSecEta , iSecPhiP);
                const l1t::CaloTower& towerSW = l1t::CaloTools::getTower(towers, iSecEtaM, iSecPhiP);
                const l1t::CaloTower& towerW  = l1t::CaloTools::getTower(towers, iSecEtaM, iSecPhi ); 
                const l1t::CaloTower& towerNN = l1t::CaloTools::getTower(towers, iSecEta , iSecPhiM2);
                const l1t::CaloTower& towerSS = l1t::CaloTools::getTower(towers, iSecEta , iSecPhiP2);

                // just use E+H for clustering
                int towerEtNW = towerNW.hwPt();
                int towerEtN  = towerN .hwPt();
                int towerEtNE = towerNE.hwPt();
                int towerEtE  = towerE .hwPt();
                int towerEtSE = towerSE.hwPt();
                int towerEtS  = towerS .hwPt();
                int towerEtSW = towerSW.hwPt();
                int towerEtW  = towerW .hwPt();
                int towerEtNN = towerNN.hwPt();
                int towerEtSS = towerSS.hwPt();
                
                int towerEtEmNW = towerNW.hwEtEm();
                int towerEtEmN  = towerN .hwEtEm();
                int towerEtEmNE = towerNE.hwEtEm();
                int towerEtEmE  = towerE .hwEtEm();
                int towerEtEmSE = towerSE.hwEtEm();
                int towerEtEmS  = towerS .hwEtEm();
                int towerEtEmSW = towerSW.hwEtEm();
                int towerEtEmW  = towerW .hwEtEm();
                int towerEtEmNN = towerNN.hwEtEm();
                int towerEtEmSS = towerSS.hwEtEm();
                //
                int towerEtHadNW = towerNW.hwEtHad();
                int towerEtHadN  = towerN .hwEtHad();
                int towerEtHadNE = towerNE.hwEtHad();
                int towerEtHadE  = towerE .hwEtHad();
                int towerEtHadSE = towerSE.hwEtHad();
                int towerEtHadS  = towerS .hwEtHad();
                int towerEtHadSW = towerSW.hwEtHad();
                int towerEtHadW  = towerW .hwEtHad();
                int towerEtHadNN = towerNN.hwEtHad();
                int towerEtHadSS = towerSS.hwEtHad();


                if(towerEtNW < clusterThreshold) secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
                if(towerEtN  < clusterThreshold)
                {
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_N , false);
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_NN, false);
                }
                if(towerEtNE < clusterThreshold) secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_NE, false);
                if(towerEtE  < clusterThreshold) secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_E , false);
                if(towerEtSE < clusterThreshold) secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_SE, false);
                if(towerEtS  < clusterThreshold)
                {
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_S , false);
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_SS, false);
                }
                if(towerEtSW < clusterThreshold) secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
                if(towerEtW  < clusterThreshold) secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_W , false);
                if(towerEtNN < clusterThreshold) secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_NN, false);
                if(towerEtSS < clusterThreshold) secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_SS, false);

                // now remove the overlapping clusters from the secondary
                // note: in firmware this is done vetoing TT on the MAIN cluster
                // so this simply recplicates the behaviour on the secondary
                // FIXME: in the future, this will be replaced by the tau trimming LUT

                vector <pair<int, int> > TTPos (10); // numbering of TT in cluster; iEta, iPhi
                TTPos.at(0) = make_pair (-1,  1); // SW
                TTPos.at(1) = make_pair (0,   1); // S
                TTPos.at(2) = make_pair (1,   1); // SE
                TTPos.at(3) = make_pair (1,   0); // E
                TTPos.at(4) = make_pair (1,  -1); // NE
                TTPos.at(5) = make_pair (0,  -1); // N
                TTPos.at(6) = make_pair (-1, -1); // NW
                TTPos.at(7) = make_pair (-1,  0); // W
                TTPos.at(8) = make_pair (0,   2); // SS
                TTPos.at(9) = make_pair (0,  -2); // NN

                vector <CaloCluster::ClusterFlag> TTPosRemap (10); // using geographical notation
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

                /*
                // neigbors
                vector <pair<int, int> > neigPos (8); // numbering of neighbors; iEta, iPhi
                neigPos.at(0) = make_pair (0,   3); // --
                neigPos.at(1) = make_pair (-1,  2); // --
                neigPos.at(2) = make_pair (0,   2); // --
                neigPos.at(3) = make_pair (1,   2); // S side   
                neigPos.at(4) = make_pair (-1, -2); // --
                neigPos.at(5) = make_pair (0,  -2); // --
                neigPos.at(6) = make_pair (1,  -2); // --  
                neigPos.at(7) = make_pair (0,  -3); // N side  
                */

                // relative position of secondary
                int deltaEta;
                if (iSecEta*iEta > 0) deltaEta = (iSecEta - iEta);
                else if (iSecEta > 0 ) deltaEta = (iSecEta - iEta -1); // must take into account the 0 that is skipped in TT numbering
                else deltaEta = (iSecEta - iEta +1); // in this case need to add 1 unit
                
                int deltaPhi = (iSecPhi - iPhi);
                while (deltaPhi >  36) deltaPhi -= 72; // phi is in 1 - 72 so dPhi (with sign) is limited to 36
                while (deltaPhi < -36) deltaPhi += 72; // phi is in 1 - 72 so dPhi (with sign) is limited to 36
                
                // loop on TT of secondary and decide if veto them
                for (unsigned int iTTsec = 0; iTTsec < TTPos.size(); iTTsec++)
                {
                    // is tower active in secondary?
                    //int thisTTeta = caloNav.offsetIEta(iSecEta,  TTPos.at(iTTsec).first);
                    //int thisTTphi = caloNav.offsetIPhi(iSecPhi,  TTPos.at(iTTsec).second);
                    bool isActive = secondaryCluster.checkClusterFlag (TTPosRemap.at(iTTsec));
                    if (isActive)
                    {
                        // find relative coordinates in the MAIN "reference frame"
                        int mainTTeta = caloNav.offsetIEta(iEta, deltaEta+TTPos.at(iTTsec).first);
                        int mainTTphi = caloNav.offsetIPhi(iPhi, deltaPhi+TTPos.at(iTTsec).second);
                        // now find to which geographical pos they correspond to
                        pair <int, int> coords = make_pair(mainTTeta, mainTTphi);
                        std::vector<pair<int,int>>::iterator res = std::find (TTPos.begin(), TTPos.end(), coords);
                        if (res != TTPos.end())
                        {
                            // check if overlapping TT of main is active
                            int idx = std::distance (TTPos.begin(), res);
                            if (mainCluster.checkClusterFlag(TTPosRemap.at(idx)))
                                secondaryCluster.setClusterFlag(TTPosRemap.at(iTTsec), false); 
                        }
                    }
                }

                // trim one eta-side
                // The side with largest energy will be kept
                int EtEtaRight = 0;
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) EtEtaRight += towerEtNE;
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_E))  EtEtaRight += towerEtE;
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) EtEtaRight += towerEtSE;
                int EtEtaLeft  = 0;
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) EtEtaLeft += towerEtNW;
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_W))  EtEtaLeft += towerEtW;
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) EtEtaLeft += towerEtSW;
                // favour most central part
                /*if(iEta>0) cluster.setClusterFlag(CaloCluster::TRIM_LEFT, (EtEtaRight> EtEtaLeft) );
                else       cluster.setClusterFlag(CaloCluster::TRIM_LEFT, (EtEtaRight>=EtEtaLeft) );*/
                //No iEta dependence in firmware
                secondaryCluster.setClusterFlag(CaloCluster::TRIM_LEFT, (EtEtaRight>= EtEtaLeft) );

                // finally compute secondary cluster energy
                if(secondaryCluster.checkClusterFlag(CaloCluster::TRIM_LEFT))
                {
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_NW, false);
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_W , false);
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_SW, false);
                }
                else
                {
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_NE, false);
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_E , false);
                    secondaryCluster.setClusterFlag(CaloCluster::INCLUDE_SE, false);
                }

                // compute cluster energy according to cluster flags
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtNW);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_N))  secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtN);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtNE);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_E))  secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtE);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtSE);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_S))  secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtS);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtSW);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_W))  secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtW);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NN)) secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtNN);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SS)) secondaryCluster.setHwPt(secondaryCluster.hwPt() + towerEtSS);
                //
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmNW);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_N))  secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmN);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmNE);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_E))  secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmE);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmSE);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_S))  secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmS);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmSW);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_W))  secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmW);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NN)) secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmNN);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SS)) secondaryCluster.setHwPtEm(secondaryCluster.hwPtEm() + towerEtEmSS);
                //
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NW)) secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadNW);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_N))  secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadN);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NE)) secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadNE);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_E))  secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadE);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SE)) secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadSE);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_S))  secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadS);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SW)) secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadSW);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_W))  secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadW);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_NN)) secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadNN);
                if(secondaryCluster.checkClusterFlag(CaloCluster::INCLUDE_SS)) secondaryCluster.setHwPtHad(secondaryCluster.hwPtHad() + towerEtHadSS);

                // it's the end! Perform the merging
                l1t::Tau tau (emptyP4, mainCluster.hwPt()+secondaryCluster.hwPt(), mainCluster.hwEta(), mainCluster.hwPhi(), 0);
                
                // ==================================================================
                // Energy calibration
                // ==================================================================

                // Corrections function of ieta, ET, and cluster shape
                //int calibPt = calibratedPt(cluster, egamma.hwPt()); // FIXME! for the moment no calibration
                int calibPt = mainCluster.hwPt()+secondaryCluster.hwPt();

                // Physical eta/phi. Computed from ieta/iphi of the seed tower and the fine-grain position within the seed
                // use fg positon of main cluster only
                double eta = 0.;
                double phi = 0.;
                double seedEta     = CaloTools::towerEta(mainCluster.hwEta());
                double seedEtaSize = CaloTools::towerEtaSize(mainCluster.hwEta());
                double seedPhi     = CaloTools::towerPhi(mainCluster.hwEta(), mainCluster.hwPhi());
                double seedPhiSize = CaloTools::towerPhiSize(mainCluster.hwEta());
                if(mainCluster.fgEta()==0)      eta = seedEta; // center
                else if(mainCluster.fgEta()==2) eta = seedEta + seedEtaSize*0.25; // center + 1/4
                else if(mainCluster.fgEta()==1) eta = seedEta - seedEtaSize*0.25; // center - 1/4
                if(mainCluster.fgPhi()==0)      phi = seedPhi; // center
                else if(mainCluster.fgPhi()==2) phi = seedPhi + seedPhiSize*0.25; // center + 1/4
                else if(mainCluster.fgPhi()==1) phi = seedPhi - seedPhiSize*0.25; // center - 1/4

                // Set 4-vector
                math::PtEtaPhiMLorentzVector calibP4((double)calibPt*params_->egLsb(), eta, phi, 0.);
                tau.setP4(calibP4);

                // save tau candidate which is now complete
                taus.push_back (tau);
            }
        }
    } // end loop on clusters
}


// -----------------------------------------------------------------------------------
void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::dosorting (std::vector<l1t::Tau>& taus)
{
    
    //Keep only 6 candidate with highest Pt in each eta-half
    std::vector<l1t::Tau> tauEtaPos;
    std::vector<l1t::Tau> tauEtaNeg;

    for (unsigned int iTau = 0; iTau < taus.size(); iTau++)
    {
        if (taus.at(iTau).hwEta() > 0) tauEtaPos.push_back (taus.at(iTau));
        else tauEtaNeg.push_back (taus.at(iTau));
    }

    std::vector<l1t::Tau>::iterator start_, end_;

    start_ = tauEtaPos.begin();  
    end_   = tauEtaPos.end();
    BitonicSort<l1t::Tau>(down, start_, end_);
    if (tauEtaPos.size()>6) tauEtaPos.resize(6);

    start_ = tauEtaNeg.begin();  
    end_   = tauEtaNeg.end();
    BitonicSort<l1t::Tau>(down, start_, end_);
    if (tauEtaNeg.size()>6) tauEtaNeg.resize(6);

    taus.clear();
    taus = tauEtaPos;
    taus.insert(taus.end(),tauEtaNeg.begin(),tauEtaNeg.end());

/*
    std::vector<pair<int,l1t::Tau>> tauEtaP;
    std::vector<pair<int,l1t::Tau>> tauEtaM;

    for (unsigned int iTau = 0; iTau < taus.size(); iTau++)
    {
        if (taus.at(iTau).hwEta() > 0) tauEtaP.push_back (make_pair (taus.at(iTau).hwPt(), taus.at(iTau)));
        else tauEtaM.push_back (make_pair (taus.at(iTau).hwPt(), taus.at(iTau)));
    }

    



    // select only 6 highest pT cands in eta+ and 6 highest pT cands in eta-
    taus.clear();
    
    sort(tauEtaP.begin(), tauEtaP.end());
    sort(tauEtaM.begin(), tauEtaM.end());
    reverse(tauEtaP.begin(), tauEtaP.end());
    reverse(tauEtaM.begin(), tauEtaM.end());

    for (unsigned int i = 0; i < tauEtaP.size() && i < 6; i++)
        taus.push_back(tauEtaP.at(i).second)

    for (unsigned int i = 0; i < tauEtaM.size() && i < 6; i++)
        taus.push_back(tauEtaM.at(i).second)

    return;
    */
}


// -----------------------------------------------------------------------------------

void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::loadCalibrationLuts()
{
  float minScale    = 0.;
  float maxScale    = 2.;
  float minScaleEta = 0.5;
  float maxScaleEta = 1.5;
  offsetBarrelEH_   = 0.5;
  offsetBarrelH_    = 1.5;
  offsetEndcapsEH_  = 0.;
  offsetEndcapsH_   = 1.5;

  // In the combined calibration LUT, upper 3-bits are used as LUT index:
  // (0=BarrelA, 1=BarrelB, 2=BarrelC, 3=EndCapA, 4=EndCapA, 5=EndCapA, 6=Eta)
  enum {LUT_UPPER = 3};
  enum {LUT_OFFSET = 0x80};
  l1t::LUT* lut = params_->tauCalibrationLUT();
  unsigned int size = (1 << lut->nrBitsData());
  unsigned int nBins = (1 << (lut->nrBitsAddress() - LUT_UPPER));


  std::vector<float> emptyCoeff;
  emptyCoeff.resize(nBins,0.);
  float binSize = (maxScale-minScale)/(float)size;
  for(unsigned iLut=0; iLut < 6; ++iLut ) {
    coefficients_.push_back(emptyCoeff);
    for(unsigned addr=0;addr<nBins;addr++) {
      float y = (float)lut->data(iLut*LUT_OFFSET + addr);
      coefficients_[iLut][addr] = minScale + binSize*y;
    }
  }

  size = (1 << lut->nrBitsData());
  nBins = (1 << 6); // can't auto-extract this now due to combined LUT.
  
  emptyCoeff.resize(nBins,0.);
  binSize = (maxScaleEta-minScaleEta)/(float)size;
  coefficients_.push_back(emptyCoeff);
  for(unsigned addr=0;addr<nBins;addr++) {
    float y = (float)lut->data(6*LUT_OFFSET + addr);
    coefficients_.back()[addr] = minScaleEta + binSize*y;
  }

}

bool l1t::Stage2Layer2TauAlgorithmFirmwareImp1::compareTowers (l1t::CaloTower TT1, l1t::CaloTower TT2)
{
    // 1. compare hwPt (for the moment no switch with E and H only, always use E+H)
    if (TT1.hwPt() < TT2.hwPt()) return true;
    if (TT1.hwPt() > TT2.hwPt()) return false;
    
    // 2. if equal pT, most central -- eta is in the range -32, 32 with ieta = 0 skipped
    if (abs(TT1.hwEta()) > abs(TT2.hwEta())) return true;
    if (abs(TT1.hwEta()) < abs(TT2.hwEta())) return false;

    // 3. if equal eta, compare phi (arbitrary)
    return (TT1.hwPhi() < TT2.hwPhi()); // towers towards S are favored (remember: N --> smaller phi, S --> larger phi)
}

bool l1t::Stage2Layer2TauAlgorithmFirmwareImp1::is3x3Maximum (const l1t::CaloTower& tower, const std::vector<CaloTower>& towers, l1t::CaloStage2Nav& caloNav)
{
    int iEta = tower.hwEta();
    int iPhi = tower.hwPhi();
    
    //int iEtaP = caloNav.offsetIEta(iEta,  1);
    //int iEtaM = caloNav.offsetIEta(iEta, -1);
    //int iPhiP = caloNav.offsetIPhi(iPhi,  1);
    //int iPhiM = caloNav.offsetIPhi(iPhi, -1);

    // 1 : >
    // 2 : >=
    int mask [3][3] = {
    { 1,2,2 },
    { 1,0,2 },
    { 1,1,2 },
    };

    bool vetoTT = false; // false if it is a local max i.e. no veto
    for (int deta = -1; deta < 2; deta++)
    {
        for (int dphi = -1; dphi < 2; dphi++)
        {
            int iEtaNeigh = caloNav.offsetIEta(iEta,  deta);
            int iPhiNeigh = caloNav.offsetIPhi(iPhi,  dphi);
            const l1t::CaloTower& towerNeigh = l1t::CaloTools::getTower(towers, iEtaNeigh, iPhiNeigh);
            if ( mask[2-(dphi+1)][deta +1] == 0 ) continue;
            if ( mask[2-(dphi+1)][deta +1] == 1 ) vetoTT = (tower.hwPt() <  towerNeigh.hwPt());
            if ( mask[2-(dphi+1)][deta +1] == 2 ) vetoTT = (tower.hwPt() <= towerNeigh.hwPt());
    
            if (vetoTT) break;
        }
    }

    return (!vetoTT); // negate because I ask if is a local maxima
}

