#ifndef CACELL_H_
#define CACELL_H_

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

    unsigned int get_cell_id() const { return theCellId;}

    Hit const & get_inner_hit() const { return theDoublets->hit(theDoubletId, HitDoublets::inner);}

    Hit const & get_outer_hit() const { return theDoublets->hit(theDoubletId, HitDoublets::outer);}

    float get_inner_x() const { return theDoublets->x(theDoubletId, HitDoublets::inner);}

    float get_outer_x() const { return theDoublets->x(theDoubletId, HitDoublets::outer);}

    float get_inner_y() const { return theDoublets->y(theDoubletId, HitDoublets::inner);}

    float get_outer_y() const { return theDoublets->y(theDoubletId, HitDoublets::outer);}

    float get_inner_z() const { return theInnerZ;}

    float get_outer_z() const { return theOuterZ;}

    float get_inner_r() const { return theInnerR;}

    float get_outer_r() const { return theOuterR;}

    float get_inner_phi() const { return theDoublets->phi(theDoubletId, HitDoublets::inner);}

    float get_outer_phi() const { return theDoublets->phi(theDoubletId, HitDoublets::outer);}

    unsigned int get_inner_hit_id() const { return theInnerHitId;}

    unsigned int get_outer_hit_id() const { return theOuterHitId;}

    void evolve() {

        hasSameStateNeighbors = 0;
        unsigned int numberOfNeighbors = theOuterNeighbors.size();

        for (unsigned int i = 0; i < numberOfNeighbors; ++i) {

            if (theOuterNeighbors[i]->get_CA_state() == theCAState) {

                hasSameStateNeighbors = 1;

                break;
            }
        }

    }

    void check_alignment_and_tag(CACell* innerCell, const float ptmin) {

        if (are_aligned_RZ(innerCell, ptmin)) {
            tag_as_inner_neighbor(innerCell);
            innerCell->tag_as_outer_neighbor(this);
        }
    }

    bool are_aligned_RZ(const CACell* otherCell, const float ptmin) const {

        float r1 = otherCell->get_inner_r();
        float z1 = otherCell->get_inner_z();
        float distance_13_squared = (r1 - theOuterR)*(r1 - theOuterR) + (z1 - theOuterZ)*(z1 - theOuterZ);
        float tan_12_13_half = fabs(z1 * (theInnerR - theOuterR) + theInnerZ * (theOuterR - r1) + theOuterZ * (r1 - theInnerR)) / distance_13_squared;
        return tan_12_13_half * ptmin <= 0.00125f;
    }

    void tag_as_outer_neighbor(CACell* otherCell) {
        theOuterNeighbors.push_back(otherCell);
    }

    void tag_as_inner_neighbor(CACell* otherCell) {
        theInnerNeighbors.push_back(otherCell);
    }

    bool have_similar_curvature(const CACell*, const TrackingRegion&) const;

    unsigned int get_CA_state() const {
        return theCAState;
    }
    // if there is at least one left neighbor with the same state (friend), the state has to be increased by 1.
    void update_state() {
        theCAState += hasSameStateNeighbors;
    }

    bool is_root_cell(const unsigned int minimumCAState) const {
        return (theCAState >= minimumCAState);
    }

    // trying to free the track building process from hardcoded layers, leaving the visit of the graph
    // based on the neighborhood connections between cells.
    void find_ntuplets(std::vector<CAntuplet>&, CAntuplet&, const unsigned int) const;

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
