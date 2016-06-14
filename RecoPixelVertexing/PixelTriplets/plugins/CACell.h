/*
 * CACell.h
 *
 *  Created on: Jan 29, 2016
 *      Author: fpantale
 */

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
    theCAState(0), theInnerHitId(innerHitId), theOuterHitId(outerHitId), theCellId(cellId), hasSameStateNeighbors(0), theDoublets(doublets), theDoubletId(doubletId) {


        theInnerR = theDoublets->r(theDoubletId, HitDoublets::inner);
        theOuterR = theDoublets->r(theDoubletId, HitDoublets::outer);
        theInnerZ = theDoublets->z(theDoubletId, HitDoublets::inner);
        theOuterZ = theDoublets->z(theDoubletId, HitDoublets::outer);
    }

    unsigned int get_cell_id() const {
        return theCellId;

    }

    Hit const & get_inner_hit() const {
        return theDoublets->hit(theDoubletId, HitDoublets::inner);
    }

    Hit const & get_outer_hit() const {
        return theDoublets->hit(theDoubletId, HitDoublets::outer);
    }

    float get_inner_x() const {

        return theDoublets->x(theDoubletId, HitDoublets::inner);
    }

    float get_outer_x() const {
        return theDoublets->x(theDoubletId, HitDoublets::outer);
    }

    float get_inner_y() const {
        return theDoublets->y(theDoubletId, HitDoublets::inner);
        ;
    }

    float get_outer_y() const {
        return theDoublets->y(theDoubletId, HitDoublets::outer);
    }

    float get_inner_z() const {
        return theInnerZ;
    }

    float get_outer_z() const {
        return theOuterZ;
    }

    float get_inner_r() const {
        return theInnerR;
    }

    float get_outer_r() const {
        return theOuterR;

    }

    float get_inner_phi() const {
        return theDoublets->phi(theDoubletId, HitDoublets::inner);
    }

    float get_outer_phi() const {
        return theDoublets->phi(theDoubletId, HitDoublets::outer);

    }

    unsigned int get_inner_hit_id() const {
        return theInnerHitId;
    }

    unsigned int get_outer_hit_id() const {
        return theOuterHitId;
    }

    inline
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

    inline
    void check_alignment_and_tag(CACell* innerCell, const float ptmin) {

        if (are_aligned_RZ(innerCell, ptmin)) {
            tag_as_inner_neighbor(innerCell);
            innerCell->tag_as_outer_neighbor(this);
        }
    }

    inline
    bool are_aligned_RZ(const CACell* otherCell, const float ptmin) const {

        float r1 = otherCell->get_inner_r();
        float z1 = otherCell->get_inner_z();


        //  float inv_cos_theta = sqrt(theInnerZ*theInnerZ+theInnerR*theInnerR)/theInnerR;


        float distance_13_squared = (r1 - theOuterR)*(r1 - theOuterR) + (z1 - theOuterZ)*(z1 - theOuterZ);
        float tan_12_13 = 2 * fabs(z1 * (theInnerR - theOuterR) + theInnerZ * (theOuterR - r1) + theOuterZ * (r1 - theInnerR)) / distance_13_squared;

        //  return tan_12_13*ptmin*inv_cos_theta <= 0.005f;
        return tan_12_13 * ptmin <= 0.0025f;

    }

    // void check_alignment_and_tag(CACell*, const float ptmin );

    void tag_as_outer_neighbor(CACell* otherCell) {
        theOuterNeighbors.push_back(otherCell);
    }

    void tag_as_inner_neighbor(CACell* otherCell) {
        theInnerNeighbors.push_back(otherCell);
    }


    //    bool are_aligned_RZ(const CACell*, const float ptmin ) const;
    bool have_similar_curvature(const CACell*, const TrackingRegion&) const;

    void print_cell() const {
        std::cout << "\nprinting cell: " << theCellId << std::endl;
        std::cout << "CAState and hasSameStateNeighbors: " << theCAState << " " << hasSameStateNeighbors << std::endl;
        std::cout << "inner hit Id: " << theInnerHitId << " outer hit id: " << theOuterHitId << std::endl;

        std::cout << "it has inner and outer neighbors " << theInnerNeighbors.size() << " " << theOuterNeighbors.size() << std::endl;
        std::cout << "its inner neighbors are: " << std::endl;
        for (unsigned int i = 0; i < theInnerNeighbors.size(); ++i)
            std::cout << theInnerNeighbors.at(i)->get_cell_id() << std::endl;

        std::cout << "its outer neighbors are: " << std::endl;
        for (unsigned int i = 0; i < theOuterNeighbors.size(); ++i)
            std::cout << theOuterNeighbors.at(i)->get_cell_id() << std::endl;

    }


    // if there is at least one left neighbor with the same state (friend), the state has to be increased by 1.

    inline
    unsigned int get_CA_state() const {
        return theCAState;
    }


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

    float theInnerR;
    float theOuterR;
    float theInnerZ;
    float theOuterZ;


    const unsigned int theCellId;


    unsigned int hasSameStateNeighbors;
public:
    const HitDoublets* theDoublets;
    const int theDoubletId;



};




#endif /*CACELL_H_ */
