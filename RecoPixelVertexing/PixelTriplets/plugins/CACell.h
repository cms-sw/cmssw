#ifndef RECOPIXELVERTEXING_PIXELTRIPLETS_CACELL_h
#define RECOPIXELVERTEXING_PIXELTRIPLETS_CACELL_h

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <cmath>
#include <array>

class CACell {
public:
    using Hit = RecHitsSortedInPhi::Hit;
    using CAntuplet = std::vector<CACell*>;

    CACell(const HitDoublets* doublets, int doubletId, const unsigned int cellId, const int innerHitId, const int outerHitId) :
    theCAState(0), theInnerHitId(innerHitId), theOuterHitId(outerHitId), theCellId(cellId), hasSameStateNeighbors(0), theDoublets(doublets), theDoubletId(doubletId),
    theInnerR(doublets->r(doubletId, HitDoublets::inner)), theOuterR(doublets->r(doubletId, HitDoublets::outer)),
    theInnerZ(doublets->z(doubletId, HitDoublets::inner)), theOuterZ(doublets->z(doubletId, HitDoublets::outer)) {
    }

    unsigned int getCellId() const {
        return theCellId;
    }

    Hit const & getInnerHit() const {
        return theDoublets->hit(theDoubletId, HitDoublets::inner);
    }

    Hit const & getOuterHit() const {
        return theDoublets->hit(theDoubletId, HitDoublets::outer);
    }

    float getInnerX() const {
        return theDoublets->x(theDoubletId, HitDoublets::inner);
    }

    float getOuterX() const {
        return theDoublets->x(theDoubletId, HitDoublets::outer);
    }

    float getInnerY() const {
        return theDoublets->y(theDoubletId, HitDoublets::inner);
    }

    float getOuterY() const {
        return theDoublets->y(theDoubletId, HitDoublets::outer);
    }

    float getInnerZ() const {
        return theInnerZ;
    }

    float getOuterZ() const {
        return theOuterZ;
    }

    float getInnerR() const {
        return theInnerR;
    }

    float getOuterR() const {
        return theOuterR;
    }

    float getInnerPhi() const {
        return theDoublets->phi(theDoubletId, HitDoublets::inner);
    }

    float getOuterPhi() const {
        return theDoublets->phi(theDoubletId, HitDoublets::outer);
    }

    unsigned int getInnerHitId() const {
        return theInnerHitId;
    }

    unsigned int getOuterHitId() const {
        return theOuterHitId;
    }

    void evolve() {

        hasSameStateNeighbors = 0;
        unsigned int numberOfNeighbors = theOuterNeighbors.size();

        for (unsigned int i = 0; i < numberOfNeighbors; ++i) {

            if (theOuterNeighbors[i]->getCAState() == theCAState) {

                hasSameStateNeighbors = 1;

                break;
            }
        }

    }

    void checkAlignmentAndTag(CACell* innerCell, const float ptmin, const float region_origin_x, const float region_origin_y, const float region_origin_radius, const float thetaCut, const float phiCut) {

        if (areAlignedRZ(innerCell, ptmin, thetaCut) && haveSimilarCurvature(innerCell, region_origin_x, region_origin_y, region_origin_radius, phiCut)) {
            tagAsInnerNeighbor(innerCell);
            innerCell->tagAsOuterNeighbor(this);
        }
    }

    bool areAlignedRZ(const CACell* otherCell, const float ptmin, const float thetaCut) const {

        float r1 = otherCell->getInnerR();
        float z1 = otherCell->getInnerZ();
        float distance_13_squared = (r1 - theOuterR)*(r1 - theOuterR) + (z1 - theOuterZ)*(z1 - theOuterZ);
        float tan_12_13_half_mul_distance_13_squared = fabs(z1 * (theInnerR - theOuterR) + theInnerZ * (theOuterR - r1) + theOuterZ * (r1 - theInnerR)) ;
        return tan_12_13_half_mul_distance_13_squared * ptmin <= thetaCut * distance_13_squared;
    }

    void tagAsOuterNeighbor(CACell* otherCell) {
        theOuterNeighbors.push_back(otherCell);
    }

    void tagAsInnerNeighbor(CACell* otherCell) {
        theInnerNeighbors.push_back(otherCell);
    }

    bool haveSimilarCurvature(const CACell* otherCell,
            const float region_origin_x, const float region_origin_y, const float region_origin_radius, const float phiCut) const {
        auto x1 = otherCell->getInnerX();
        auto y1 = otherCell->getInnerY();

        auto x2 = getInnerX();
        auto y2 = getInnerY();

        auto x3 = getOuterX();
        auto y3 = getOuterY();

        auto precision = 0.5f;
        auto offset = x2 * x2 + y2*y2;

        auto bc = (x1 * x1 + y1 * y1 - offset)*0.5f;

        auto cd = (offset - x3 * x3 - y3 * y3)*0.5f;

        auto det = (x1 - x2) * (y2 - y3) - (x2 - x3)* (y1 - y2);

        //points are aligned
        if (fabs(det) < precision)
            return true;

        auto idet = 1.f / det;

        auto x_center = (bc * (y2 - y3) - cd * (y1 - y2)) * idet;
        auto y_center = (cd * (x1 - x2) - bc * (x2 - x3)) * idet;

        auto radius = std::sqrt((x2 - x_center)*(x2 - x_center) + (y2 - y_center)*(y2 - y_center));
        auto centers_distance_squared = (x_center - region_origin_x)*(x_center - region_origin_x) + (y_center - region_origin_y)*(y_center - region_origin_y);

        auto minimumOfIntersectionRange = (radius - region_origin_radius)*(radius - region_origin_radius) - phiCut;

        if (centers_distance_squared >= minimumOfIntersectionRange) {
            auto minimumOfIntersectionRange = (radius + region_origin_radius)*(radius + region_origin_radius) + phiCut;
            return centers_distance_squared <= minimumOfIntersectionRange;
        } else {

            return false;
        }
        return true;


    }

    unsigned int getCAState() const {
        return theCAState;
    }
    // if there is at least one left neighbor with the same state (friend), the state has to be increased by 1.

    void updateState() {
        theCAState += hasSameStateNeighbors;
    }

    bool isRootCell(const unsigned int minimumCAState) const {
        return (theCAState >= minimumCAState);
    }

    // trying to free the track building process from hardcoded layers, leaving the visit of the graph
    // based on the neighborhood connections between cells.

    void findNtuplets(std::vector<CAntuplet>& foundNtuplets, CAntuplet& tmpNtuplet, const unsigned int minHitsPerNtuplet) const {

        // the building process for a track ends if:
        // it has no right neighbor
        // it has no compatible neighbor
        // the ntuplets is then saved if the number of hits it contains is greater than a threshold

        if (theOuterNeighbors.size() == 0) {
            if (tmpNtuplet.size() >= minHitsPerNtuplet - 1)
                foundNtuplets.push_back(tmpNtuplet);
            else
                return;
        } else {
            unsigned int numberOfOuterNeighbors = theOuterNeighbors.size();
            for (unsigned int i = 0; i < numberOfOuterNeighbors; ++i) {
                tmpNtuplet.push_back((theOuterNeighbors[i]));
                theOuterNeighbors[i]->findNtuplets(foundNtuplets, tmpNtuplet, minHitsPerNtuplet);
                tmpNtuplet.pop_back();
            }
        }
    }
    
    
private:
    std::vector<CACell*> theInnerNeighbors;
    std::vector<CACell*> theOuterNeighbors;

    unsigned int theCAState;

    const unsigned int theInnerHitId;
    const unsigned int theOuterHitId;
    const unsigned int theCellId;
    unsigned int hasSameStateNeighbors;

    const HitDoublets* theDoublets;
    const int theDoubletId;

    const float theInnerR;
    const float theOuterR;
    const float theInnerZ;
    const float theOuterZ;

};


#endif /*CACELL_H_ */
