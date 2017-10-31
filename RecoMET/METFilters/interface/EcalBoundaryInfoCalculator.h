#ifndef ECALBOUNDARYINFOCALCULATOR_H_
#define ECALBOUNDARYINFOCALCULATOR_H_
#include <memory>
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalEndcapNavigator.h"
#include "RecoCaloTools/Navigation/interface/EcalPreshowerNavigator.h"
#include "RecoCaloTools/Navigation/interface/CaloTowerNavigator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/METReco/interface/BoundaryInformation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

enum CdOrientation {
    north, east, south, west, none
};

template<class EcalDetId> class EcalBoundaryInfoCalculator {

public:

    EcalBoundaryInfoCalculator();
    ~EcalBoundaryInfoCalculator();

    BoundaryInformation boundaryRecHits(const edm::Handle<EcalRecHitCollection>&, const EcalRecHit*,
                                        const edm::ESHandle<CaloTopology> theCaloTopology, const edm::ESHandle<EcalChannelStatus> ecalStatus,
                                        const edm::ESHandle<CaloGeometry> geometry) const;

    BoundaryInformation gapRecHits(const edm::Handle<EcalRecHitCollection>&, const EcalRecHit*, const edm::ESHandle<
                                   CaloTopology> theCaloTopology, const edm::ESHandle<EcalChannelStatus> ecalStatus, const edm::ESHandle<
                                   CaloGeometry> geometry) const;

    bool checkRecHitHasDeadNeighbour(const EcalRecHit& hit, const edm::ESHandle<EcalChannelStatus> ecalStatus, std::vector<
                                     int> &stati) const {

        stati.clear();
        EcalDetId hitdetid = EcalDetId(hit.id());

        if (hitdetid.subdet() == EcalBarrel) {

            EBDetId ebhitdetid = (EBDetId) hitdetid;

            int hitIeta = ebhitdetid.ieta();
            int hitIphi = ebhitdetid.iphi();

            for (int ieta = -1; ieta <= 1; ieta++) {
                for (int iphi = -1; iphi <= 1; iphi++) {
                    if ((iphi == 0 && ieta == 0) || iphi * ieta != 0)
                        //if (iphi == 0 && ieta == 0)
                        continue;
                    int neighbourIeta = hitIeta + ieta;
                    int neighbourIphi = hitIphi + iphi;
                    if (!EBDetId::validDetId(neighbourIeta, neighbourIphi)) {
                        if (neighbourIphi < 1)
                            neighbourIphi += 360;
                        if (neighbourIphi > 360)
                            neighbourIphi -= 360;
                        if (neighbourIeta == 0) {
                            neighbourIeta += ieta;
                        }
                    }

                    if (EBDetId::validDetId(neighbourIeta, neighbourIphi)) {

                        const EBDetId detid = EBDetId(neighbourIeta, neighbourIphi, EBDetId::ETAPHIMODE);
                        EcalChannelStatus::const_iterator chit = ecalStatus->find(detid);
                        int status = (chit != ecalStatus->end()) ? chit->getStatusCode() & 0x1F : -1;

                        if (status > 0) {
                            bool present = false;
                            for (std::vector<int>::const_iterator s = stati.begin(); s != stati.end(); ++s) {
                                if (*s == status) {
                                    present = true;
                                    break;
                                }
                            }
                            if (!present)
                                stati.push_back(status);
                        }
                    }
                }
            }

        } else if (hitdetid.subdet() == EcalEndcap) {

            EEDetId eehitdetid = (EEDetId) hitdetid;
            int hitIx = eehitdetid.ix();
            int hitIy = eehitdetid.iy();
            int hitIz = eehitdetid.zside();

            for (int ix = -1; ix <= 1; ix++) {
                for (int iy = -1; iy <= 1; iy++) {
                    if ((ix == 0 && iy == 0) || ix * iy != 0)
                        //if (ix == 0 && iy == 0)
                        continue;
                    int neighbourIx = hitIx + ix;
                    int neighbourIy = hitIy + iy;

                    if (EEDetId::validDetId(neighbourIx, neighbourIy, hitIz)) {

                        const EEDetId detid = EEDetId(neighbourIx, neighbourIy, hitIz, EEDetId::XYMODE);
                        EcalChannelStatus::const_iterator chit = ecalStatus->find(detid);
                        int status = (chit != ecalStatus->end()) ? chit->getStatusCode() & 0x1F : -1;

                        if (status > 0) {
                            bool present = false;
                            for (std::vector<int>::const_iterator s = stati.begin(); s != stati.end(); ++s) {
                                if (*s == status) {
                                    present = true;
                                    break;
                                }
                            }
                            if (!present)
                                stati.push_back(status);
                        }
                    }
                }
            }

        } else {
            edm::LogWarning("EcalBoundaryInfoCalculator") << "ERROR - RecHit belongs to wrong sub detector";
        }

        if (!stati.empty())
            return true;
        return false;

    }

    bool checkRecHitHasInvalidNeighbour(const EcalRecHit& hit, const edm::ESHandle<EcalChannelStatus> ecalStatus) const {
        //// return true, if *direct* neighbour is invalid

        EcalDetId hitdetid = EcalDetId(hit.id());

        if (hitdetid.subdet() == EcalBarrel) {

            EBDetId ebhitdetid = (EBDetId) hitdetid;

            int hitIeta = ebhitdetid.ieta();
            int hitIphi = ebhitdetid.iphi();

            for (int ieta = -1; ieta <= 1; ieta++) {
                for (int iphi = -1; iphi <= 1; iphi++) {
                    if ((iphi == 0 && ieta == 0) || iphi * ieta != 0)
                        //if (iphi == 0 && ieta == 0)
                        continue;
                    int neighbourIeta = hitIeta + ieta;
                    int neighbourIphi = hitIphi + iphi;
                    if (!EBDetId::validDetId(neighbourIeta, neighbourIphi)) {
                        if (neighbourIphi < 1)
                            neighbourIphi += 360;
                        if (neighbourIphi > 360)
                            neighbourIphi -= 360;
                        if (neighbourIeta == 0) {
                            neighbourIeta += ieta;
                        }
                    }

                    if (!EBDetId::validDetId(neighbourIeta, neighbourIphi)) {
                        return true;
                    }
                }
            }

        } else if (hitdetid.subdet() == EcalEndcap) {

            EEDetId eehitdetid = (EEDetId) hitdetid;
            int hitIx = eehitdetid.ix();
            int hitIy = eehitdetid.iy();
            int hitIz = eehitdetid.zside();

            for (int ix = -1; ix <= 1; ix++) {
                for (int iy = -1; iy <= 1; iy++) {
                    if ((ix == 0 && iy == 0) || ix * iy != 0)
                        //if (ix == 0 && iy == 0)
                        continue;
                    int neighbourIx = hitIx + ix;
                    int neighbourIy = hitIy + iy;

                    if (!EEDetId::validDetId(neighbourIx, neighbourIy, hitIz)) {
                        return true;
                    }
                }
            }

        } else {
            edm::LogWarning("EcalBoundaryInfoCalculator") << "ERROR - RecHit belongs to wrong sub detector";
        }

        return false;
    }

    void setDebugMode() {
        edm::LogInfo("EcalBoundaryInfoCalculator") << "set Debug Mode!";
        debug = true;
    }

private:

    EcalDetId makeStepInDirection(CdOrientation direction, const CaloNavigator<EcalDetId> * theNavi) const {
        EcalDetId next;
        switch (direction) {
        case north: {
                next = theNavi->north();
                break;
            }
        case east: {
                next = theNavi->east();
                break;
            }
        case south: {
                next = theNavi->south();
                break;
            }
        case west: {
                next = theNavi->west();
                break;
            }
        default:
	   	break;
        }
        return next;
    }

    CdOrientation goBackOneCell(CdOrientation currDirection, EcalDetId prev, CaloNavigator<EcalDetId> * theEcalNav) const {
        std::map<CdOrientation, CdOrientation>::iterator oIt = oppositeDirs.find(currDirection);
        CdOrientation oppDirection=none;
        if (oIt != oppositeDirs.end()) {
            oppDirection = oIt->second;
            theEcalNav->setHome(prev);
        }
        EcalDetId currDetId = theEcalNav->pos();

        return oppDirection;
    }

    CdOrientation turnRight(CdOrientation currDirection, bool reverseOrientation) const {
        //read nextDirection
        std::map<CdOrientation, CdOrientation> turnMap = nextDirs;
        if (reverseOrientation)
            turnMap = prevDirs;
        std::map<CdOrientation, CdOrientation>::iterator nIt = turnMap.find(currDirection);
        CdOrientation nextDirection=none;
        if (nIt != turnMap.end())
            nextDirection = (*nIt).second;
        else
            edm::LogWarning("EcalBoundaryInfoCalculator") << "ERROR - no Next Direction found!?!?";
        return nextDirection;
    }

    CdOrientation turnLeft(CdOrientation currDirection, bool reverseOrientation) const {
        //read nextDirection
        std::map<CdOrientation, CdOrientation> turnMap = prevDirs;
        if (reverseOrientation)
            turnMap = nextDirs;
        std::map<CdOrientation, CdOrientation>::iterator nIt = turnMap.find(currDirection);
        CdOrientation nextDirection=none;
        if (nIt != turnMap.end())
            nextDirection = (*nIt).second;
        else
            edm::LogWarning("EcalBoundaryInfoCalculator") << "ERROR - no Next Direction found!?!?";
        return nextDirection;
    }

    std::unique_ptr<CaloNavigator<EcalDetId>> initializeEcalNavigator(DetId startE, const edm::ESHandle<CaloTopology> theCaloTopology,
                                 EcalSubdetector ecalSubDet) const {
        std::unique_ptr<CaloNavigator<EcalDetId>> theEcalNav(nullptr);
        if (ecalSubDet == EcalBarrel) {
            theEcalNav.reset(new CaloNavigator<EcalDetId> ((EBDetId) startE, (theCaloTopology->getSubdetectorTopology(
                             DetId::Ecal, ecalSubDet))));
        } else if (ecalSubDet == EcalEndcap) {
            theEcalNav.reset(new CaloNavigator<EcalDetId> ((EEDetId) startE, (theCaloTopology->getSubdetectorTopology(
                             DetId::Ecal, ecalSubDet))));
        } else {
            edm::LogWarning("EcalBoundaryInfoCalculator") << "initializeEcalNavigator not implemented for subDet: " << ecalSubDet;
        }
        return theEcalNav;
    }

    std::map<CdOrientation, CdOrientation> nextDirs;
    std::map<CdOrientation, CdOrientation> prevDirs;
    std::map<CdOrientation, CdOrientation> oppositeDirs;
    bool debug;

};

template<class EcalDetId> EcalBoundaryInfoCalculator<EcalDetId>::EcalBoundaryInfoCalculator() {

    nextDirs.clear();
    nextDirs[north] = east;
    nextDirs[east] = south;
    nextDirs[south] = west;
    nextDirs[west] = north;

    prevDirs.clear();
    prevDirs[north] = west;
    prevDirs[east] = north;
    prevDirs[south] = east;
    prevDirs[west] = south;

    oppositeDirs.clear();
    oppositeDirs[north] = south;
    oppositeDirs[south] = north;
    oppositeDirs[east] = west;
    oppositeDirs[west] = east;

    debug = false;

}

template<class EcalDetId> EcalBoundaryInfoCalculator<EcalDetId>::~EcalBoundaryInfoCalculator() {
}

template<class EcalDetId> BoundaryInformation EcalBoundaryInfoCalculator<EcalDetId>::boundaryRecHits(const edm::Handle<
        EcalRecHitCollection>& RecHits, const EcalRecHit* hit, const edm::ESHandle<CaloTopology> theCaloTopology,
        edm::ESHandle<EcalChannelStatus> ecalStatus, edm::ESHandle<CaloGeometry> geometry) const {

    //initialize boundary information
    std::vector<EcalRecHit> boundaryRecHits;
    std::vector<DetId> boundaryDetIds;
    std::vector<int> stati;

    double boundaryEnergy = 0;
    double boundaryET = 0;
    int beCellCounter = 0;
    bool nextToBorder = false;

    boundaryRecHits.push_back(*hit);
    ++beCellCounter;
    boundaryEnergy += hit->energy();
    EcalDetId hitdetid = (EcalDetId) hit->id();
    boundaryDetIds.push_back(hitdetid);
    const CaloSubdetectorGeometry* subGeom = geometry->getSubdetectorGeometry(hitdetid);
    const CaloCellGeometry* cellGeom = subGeom->getGeometry(hitdetid);
    double eta = cellGeom->getPosition().eta();
    boundaryET += hit->energy() / cosh(eta);

    if (debug) {
        edm::LogInfo("EcalBoundaryInfoCalculator") << "Find Boundary RecHits...";

        if (hitdetid.subdet() == EcalBarrel) {
            edm::LogInfo("EcalBoundaryInfoCalculator") << "Starting at : (" << ((EBDetId) hitdetid).ieta() << "," << ((EBDetId) hitdetid).iphi() << ")";
        } else if (hitdetid.subdet() == EcalEndcap) {
            edm::LogInfo("EcalBoundaryInfoCalculator") << "Starting at : (" << ((EEDetId) hitdetid).ix() << "," << ((EEDetId) hitdetid).iy() << ")";
        }
    }

    //initialize navigator
    auto theEcalNav = initializeEcalNavigator(hitdetid, theCaloTopology, EcalDetId::subdet());
    CdOrientation currDirection = north;
    bool reverseOrientation = false;

    EcalDetId next(0);
    EcalDetId start = hitdetid;
    EcalDetId current = start;
    int current_status = 0;

    // Search until a dead cell is ahead
    bool startAlgo = false;
    int noDirs = 0;
    while (!startAlgo) {
        next = makeStepInDirection(currDirection, theEcalNav.get());
        theEcalNav->setHome(current);
        theEcalNav->home();
        EcalChannelStatus::const_iterator chit = ecalStatus->find(next);
        int status = (chit != ecalStatus->end()) ? chit->getStatusCode() & 0x1F : -1;
        if (status > 0) {
            stati.push_back(status);
            startAlgo = true;
            break;
        }
        currDirection = turnLeft(currDirection, reverseOrientation);
        ++noDirs;
        if (noDirs > 4) {
            throw cms::Exception("NoStartingDirection")
              << "EcalBoundaryInfoCalculator: No starting direction can be found: This should never happen if RecHit has a dead neighbour!";
        }
    }

    // go around dead clusters counter clock wise
    currDirection = turnRight(currDirection, reverseOrientation);

    // Search for next boundary element
    bool nextIsStart = false;
    bool atBorder = false;

    while (!nextIsStart) {

        bool nextStepFound = false;
        int status = 0;
        noDirs = 0;
        while (!nextStepFound) {
            next = makeStepInDirection(currDirection, theEcalNav.get());
            theEcalNav->setHome(current);
            theEcalNav->home();
            EcalChannelStatus::const_iterator chit = ecalStatus->find(next);
            status = (chit != ecalStatus->end()) ? chit->getStatusCode() & 0x1F : -1;
            if (status > 0) {
                // New dead cell found: update status std::vector of dead channels
                bool present = false;
                for (std::vector<int>::const_iterator s = stati.begin(); s != stati.end(); ++s) {
                    if (*s == status) {
                        present = true;
                        break;
                    }
                }
                if (!present)
                    stati.push_back(status);

                if (atBorder) {
                    nextStepFound = true;
                } else {
                    currDirection = turnRight(currDirection, reverseOrientation);
                }
            } else if (next == EcalDetId(0)) {
                // In case the Ecal border is reached -> go along dead cells
                currDirection = turnLeft(currDirection, reverseOrientation);
                atBorder = true;
            } else if (status == 0) {
                nextStepFound = true;
            }
            ++noDirs;
            if (noDirs > 4) {
              throw cms::Exception("NoNextDirection")
                << "EcalBoundaryInfoCalculator: No valid next direction can be found: This should never happen!";
            }
        }

        // make next step
        next = makeStepInDirection(currDirection, theEcalNav.get());

        if (next == start) {
            nextIsStart = true;
            if (debug)
                edm::LogInfo("EcalBoundaryInfoCalculator") << "Boundary path reached starting position!";
        }

        if (debug)
            edm::LogInfo("EcalBoundaryInfoCalculator") << "Next step: " << (EcalDetId) next << " Status: " << status << " Start: " << (EcalDetId) start;

        // save recHits and add energy if on the boundary (and not inside at border)
        if ((!atBorder || status == 0) && !nextIsStart) {
            boundaryDetIds.push_back(next);
            if (RecHits->find(next) != RecHits->end() && status == 0) {
                EcalRecHit nexthit = *RecHits->find(next);
                ++beCellCounter;
                boundaryRecHits.push_back(nexthit);
                boundaryEnergy += nexthit.energy();
                cellGeom = subGeom->getGeometry(hitdetid);
                eta = cellGeom->getPosition().eta();
                boundaryET += nexthit.energy() / cosh(eta);
            }
        }

        if (current_status == 0 && status == 0 && atBorder) {
            // this is for a special case, where dead cells are at border corner
            currDirection = turnRight(currDirection, reverseOrientation);
        } else {
            // if dead region along a border is left, turn left
            if (status == 0 && atBorder) {
                atBorder = false;
                currDirection = turnLeft(currDirection, reverseOrientation);
            }
            if (status == 0) {
                // if outside the cluster turn left to follow boundary
                currDirection = turnLeft(currDirection, reverseOrientation);
            } else {
                // else turn right to check if dead region can be left
                currDirection = turnRight(currDirection, reverseOrientation);
            }
        }

        // save currect position
        current = next;
        current_status = status;

    }

    if (debug) {
        edm::LogInfo("EcalBoundaryInfoCalculator") << "<<<<<<<<<<<<<<< Final Boundary object <<<<<<<<<<<<<<<";
        edm::LogInfo("EcalBoundaryInfoCalculator") << "no of neighbouring RecHits: " << boundaryRecHits.size();
        edm::LogInfo("EcalBoundaryInfoCalculator") << "no of neighbouring DetIds: " << boundaryDetIds.size();
        edm::LogInfo("EcalBoundaryInfoCalculator") << "boundary energy: " << boundaryEnergy;
        edm::LogInfo("EcalBoundaryInfoCalculator") << "boundary ET: " << boundaryET;
        edm::LogInfo("EcalBoundaryInfoCalculator") << "no of cells contributing to boundary energy: " << beCellCounter;
        edm::LogInfo("EcalBoundaryInfoCalculator") << "Channel stati: ";
        for (std::vector<int>::iterator it = stati.begin(); it != stati.end(); ++it) {
            edm::LogInfo("EcalBoundaryInfoCalculator") << *it << " ";
        }
        edm::LogInfo("EcalBoundaryInfoCalculator");
    }

    BoundaryInformation boundInfo;
    boundInfo.subdet = hitdetid.subdet();
    boundInfo.detIds = boundaryDetIds;
    boundInfo.recHits = boundaryRecHits;
    boundInfo.boundaryEnergy = boundaryEnergy;
    boundInfo.boundaryET = boundaryET;
    boundInfo.nextToBorder = nextToBorder;
    boundInfo.channelStatus = stati;

    return boundInfo;
}

template<class EcalDetId> BoundaryInformation EcalBoundaryInfoCalculator<EcalDetId>::gapRecHits(const edm::Handle<
        EcalRecHitCollection>& RecHits, const EcalRecHit* hit, const edm::ESHandle<CaloTopology> theCaloTopology,
        edm::ESHandle<EcalChannelStatus> ecalStatus, edm::ESHandle<CaloGeometry> geometry) const {

    //initialize boundary information
    std::vector<EcalRecHit> gapRecHits;
    std::vector<DetId> gapDetIds;

    double gapEnergy = 0;
    double gapET = 0;
    int gapCellCounter = 0;
    bool nextToBorder = false;

    gapRecHits.push_back(*hit);
    ++gapCellCounter;
    gapEnergy += hit->energy();
    EcalDetId hitdetid = (EcalDetId) hit->id();
    gapDetIds.push_back(hitdetid);
    const CaloSubdetectorGeometry* subGeom = geometry->getSubdetectorGeometry(hitdetid);
    const CaloCellGeometry* cellGeom = subGeom->getGeometry(hitdetid);
    double eta = cellGeom->getPosition().eta();
    gapET += hit->energy() / cosh(eta);

    if (debug) {
        edm::LogInfo("EcalBoundaryInfoCalculator") << "Find Border RecHits...";

        if (hitdetid.subdet() == EcalBarrel) {
            edm::LogInfo("EcalBoundaryInfoCalculator") << "Starting at : (" << ((EBDetId) hitdetid).ieta() << "," << ((EBDetId) hitdetid).iphi() << ")";
        } else if (hitdetid.subdet() == EcalEndcap) {
            edm::LogInfo("EcalBoundaryInfoCalculator") << "Starting at : (" << ((EEDetId) hitdetid).ix() << "," << ((EEDetId) hitdetid).iy() << ")";
        }
    }

    //initialize navigator
    auto theEcalNav = initializeEcalNavigator(hitdetid, theCaloTopology, EcalDetId::subdet());
    CdOrientation currDirection = north;
    bool reverseOrientation = false;

    EcalDetId next(0);
    EcalDetId start = hitdetid;
    EcalDetId current = start;

    // Search until a invalid cell is ahead
    bool startAlgo = false;
    int noDirs = 0;
    while (!startAlgo) {
        next = makeStepInDirection(currDirection, theEcalNav.get());
        theEcalNav->setHome(start);
        theEcalNav->home();
        if (next == EcalDetId(0)) {
            startAlgo = true;
            nextToBorder = true;
            break;
        }
        currDirection = turnLeft(currDirection, reverseOrientation);
        ++noDirs;
        if (noDirs > 4) {
            throw cms::Exception("NoStartingDirection")
              << "EcalBoundaryInfoCalculator: No starting direction can be found: This should never happen if RecHit is at border!";
        }
    }

    ////////// First: go along gap left
    CdOrientation startDirection = currDirection;
    currDirection = turnLeft(currDirection, reverseOrientation);

    // Search for next border element
    bool endIsFound = false;
    bool startIsEnd = false;

    while (!endIsFound) {

        bool nextStepFound = false;
        int status = 0;
        noDirs = 0;
        while (!nextStepFound) {
            next = makeStepInDirection(currDirection, theEcalNav.get());
            theEcalNav->setHome(current);
            theEcalNav->home();
            EcalChannelStatus::const_iterator chit = ecalStatus->find(next);
            status = (chit != ecalStatus->end()) ? chit->getStatusCode() & 0x1F : -1;
            if (status > 0) {
                // Find dead cell along border -> end of cluster
                endIsFound = true;
                break;
            } else if (next == EcalDetId(0)) {
                // In case the Ecal border -> go along gap
                currDirection = turnLeft(currDirection, reverseOrientation);
            } else if (status == 0) {
                if (RecHits->find(next) != RecHits->end()) {
                    nextStepFound = true;
                } else {
                    endIsFound = true;
                    break;
                }
            }
            ++noDirs;
            if (noDirs > 4) {
              throw cms::Exception("NoNextDirection")
                << "EcalBoundaryInfoCalculator: No valid next direction can be found: This should never happen!";
            }
        }

        // make next step
        next = makeStepInDirection(currDirection, theEcalNav.get());
        current = next;

        if (next == start) {
            startIsEnd = true;
            endIsFound = true;
            if (debug)
                edm::LogInfo("EcalBoundaryInfoCalculator") << "Path along gap reached starting position!";
        }

        if (debug) {
            edm::LogInfo("EcalBoundaryInfoCalculator") << "Next step: " << (EcalDetId) next << " Status: " << status << " Start: " << (EcalDetId) start;
            if (endIsFound)
                edm::LogInfo("EcalBoundaryInfoCalculator") << "End of gap cluster is found going left";
        }

        // save recHits and add energy
        if (!endIsFound) {
            gapDetIds.push_back(next);
            if (RecHits->find(next) != RecHits->end()) {
                EcalRecHit nexthit = *RecHits->find(next);
                ++gapCellCounter;
                gapRecHits.push_back(nexthit);
                gapEnergy += nexthit.energy();
                cellGeom = subGeom->getGeometry(next);
                eta = cellGeom->getPosition().eta();
                gapET += nexthit.energy() / cosh(eta);
            }
        }

        // turn right to follow gap
        currDirection = turnRight(currDirection, reverseOrientation);

    }

    ////////// Second: go along gap right
    theEcalNav->setHome(start);
    theEcalNav->home();
    current = start;
    currDirection = startDirection;
    currDirection = turnRight(currDirection, reverseOrientation);

    // Search for next border element
    endIsFound = false;

    if (!startIsEnd) {

        while (!endIsFound) {

            bool nextStepFound = false;
            int status = 0;
            noDirs = 0;
            while (!nextStepFound) {
                next = makeStepInDirection(currDirection, theEcalNav.get());
                theEcalNav->setHome(current);
                theEcalNav->home();
                EcalChannelStatus::const_iterator chit = ecalStatus->find(next);
                status = (chit != ecalStatus->end()) ? chit->getStatusCode() & 0x1F : -1;
                if (status > 0) {
                    // Find dead cell along border -> end of cluster
                    endIsFound = true;
                    break;
                } else if (next == EcalDetId(0)) {
                    // In case the Ecal border -> go along gap
                    currDirection = turnRight(currDirection, reverseOrientation);
                } else if (status == 0) {
                    if (RecHits->find(next) != RecHits->end()) {
                        nextStepFound = true;
                    } else {
                        endIsFound = true;
                        break;
                    }
                }
                ++noDirs;
                if (noDirs > 4) {
                  throw cms::Exception("NoStartingDirection")
                    << "EcalBoundaryInfoCalculator: No valid next direction can be found: This should never happen!";
                }
            }

            // make next step
            next = makeStepInDirection(currDirection, theEcalNav.get());
            current = next;

            if (debug) {
                edm::LogInfo("EcalBoundaryInfoCalculator")<< "Next step: " << (EcalDetId) next << " Status: " << status << " Start: " << (EcalDetId) start;
                if (endIsFound)
                    edm::LogInfo("EcalBoundaryInfoCalculator") << "End of gap cluster is found going right";
            }

            // save recHits and add energy
            if (!endIsFound) {
                gapDetIds.push_back(next);
                if (RecHits->find(next) != RecHits->end()) {
                    EcalRecHit nexthit = *RecHits->find(next);
                    ++gapCellCounter;
                    gapRecHits.push_back(nexthit);
                    gapEnergy += nexthit.energy();
                    cellGeom = subGeom->getGeometry(next);
                    eta = cellGeom->getPosition().eta();
                    gapET += nexthit.energy() / cosh(eta);
                }
            }

            // turn left to follow gap
            currDirection = turnLeft(currDirection, reverseOrientation);

        }
    }

    if (debug) {
        edm::LogInfo("EcalBoundaryInfoCalculator") << "<<<<<<<<<<<<<<< Final Gap object <<<<<<<<<<<<<<<";
        edm::LogInfo("EcalBoundaryInfoCalculator") << "No of RecHits along gap: " << gapRecHits.size();
        edm::LogInfo("EcalBoundaryInfoCalculator") << "No of DetIds along gap: " << gapDetIds.size();
        edm::LogInfo("EcalBoundaryInfoCalculator") << "Gap energy: " << gapEnergy;
        edm::LogInfo("EcalBoundaryInfoCalculator") << "Gap ET: " << gapET;
    }

    BoundaryInformation gapInfo;
    gapInfo.subdet = hitdetid.subdet();
    gapInfo.detIds = gapDetIds;
    gapInfo.recHits = gapRecHits;
    gapInfo.boundaryEnergy = gapEnergy;
    gapInfo.boundaryET = gapET;
    gapInfo.nextToBorder = nextToBorder;
    std::vector<int> stati;
    gapInfo.channelStatus = stati;

    return gapInfo;
}

#endif /*ECALBOUNDARYINFOCALCULATOR_H_*/
