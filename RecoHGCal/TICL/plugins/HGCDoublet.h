// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 11/2018
// Copyright CERN

#ifndef __RecoHGCal_TICL_HGCDoublet_H__
#define __RecoHGCal_TICL_HGCDoublet_H__

#include <cmath>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

class HGCDoublet
{
  public:
    using HGCntuplet = std::vector<unsigned int>;

    HGCDoublet(const int innerClusterId, const int outerClusterId, const int doubletId, 
    const std::vector<reco::CaloCluster> *layerClusters) : 
        layerClusters_(layerClusters), theDoubletId_(doubletId), theInnerClusterId_(innerClusterId), theOuterClusterId_(outerClusterId),
        theInnerR_((*layerClusters)[innerClusterId].position().r()), theOuterR_((*layerClusters)[outerClusterId].position().r()),
        theInnerX_((*layerClusters)[innerClusterId].x()), theOuterX_((*layerClusters)[outerClusterId].x()),
        theInnerY_((*layerClusters)[innerClusterId].y()), theOuterY_((*layerClusters)[outerClusterId].y()),
        theInnerZ_((*layerClusters)[innerClusterId].z()), theOuterZ_((*layerClusters)[outerClusterId].z())
    {
    }

    double getInnerX() const
    {
        return theInnerX_;
    }

    double getOuterX() const
    {
        return theOuterX_;
    }

    double getInnerY() const
    {
        return theInnerY_;
    }

    double getOuterY() const
    {
        return theOuterY_;
    }

    double getInnerZ() const
    {
        return theInnerZ_;
    }

    double getOuterZ() const
    {
        return theOuterZ_;
    }

    double getInnerR() const
    {
        return theInnerR_;
    }

    double getOuterR() const
    {
        return theOuterZ_;
    }

    void tagAsOuterNeighbor(unsigned int otherDoublet)
    {
        theOuterNeighbors_.push_back(otherDoublet);
    }

    void tagAsInnerNeighbor(unsigned int otherDoublet)
    {
        theInnerNeighbors_.push_back(otherDoublet);
    }

    bool checkCompatibilityAndTag(std::vector<HGCDoublet> &allDoublets, const std::vector<int>& innerDoublets, float minCosTheta )
    {
        int nDoublets = innerDoublets.size();
        int constexpr VSIZE = 4;
        int ok[VSIZE];
        double xi[VSIZE];
        double yi[VSIZE];
        double zi[VSIZE];
        auto xo = getOuterX();
        auto yo = getOuterY();
        auto zo = getOuterZ();
        unsigned int doubletId = theDoubletId_;

        auto loop = [&](int i, int vs) {
            for (int j = 0; j < vs; ++j)
            {
                auto otherDoubletId = innerDoublets[i + j];
                auto &otherDoublet = allDoublets[otherDoubletId];
                xi[j] = otherDoublet.getInnerX();
                yi[j] = otherDoublet.getInnerY();
                zi[j] = otherDoublet.getInnerZ();
            }
            for (int j = 0; j < vs; ++j)
                ok[j] = areAligned(xi[j], yi[j], zi[j], xo, yo, zo, minCosTheta);

            for (int j = 0; j < vs; ++j)
            {
                auto otherDoubletId = innerDoublets[i + j];
                auto &otherDoublet = allDoublets[otherDoubletId];
                 if (ok[j])
                 {
                    otherDoublet.tagAsOuterNeighbor(doubletId);
                    allDoublets[doubletId].tagAsInnerNeighbor(otherDoubletId);

                 }
            }
        };
        auto lim = VSIZE * (nDoublets / VSIZE);
        for (int i = 0; i < lim; i += VSIZE)
            loop(i, VSIZE);
        loop(lim, nDoublets - lim);

        return theInnerNeighbors_.empty();
    }

    int areAligned(double xi, double yi, double zi, double xo, double yo, double zo, float minCosTheta)
    {
        return true;
    }


    // void checkAlignmentAndAct(std::vector<HGCDoublet> &allCells, CAntuple &innerCells, const double ptmin, const double region_origin_x,
    //                           const double region_origin_y, const double region_origin_radius, const double thetaCut,
    //                           const double phiCut, const double hardPtCut, std::vector<HGCDoublet::HGCntuplet> *foundTriplets)
    // {
    //     int ncells = innerCells.size();
    //     int constexpr VSIZE = 16;
    //     int ok[VSIZE];
    //     double r1[VSIZE];
    //     double z1[VSIZE];
    //     auto ro = getOuterR();
    //     auto zo = getOuterZ();
    //     unsigned int cellId = this - &allCells.front();
    //     auto loop = [&](int i, int vs) {
    //         for (int j = 0; j < vs; ++j)
    //         {
    //             auto koc = innerCells[i + j];
    //             auto &oc = allCells[koc];
    //             r1[j] = oc.getInnerR();
    //             z1[j] = oc.getInnerZ();
    //         }
    //         // this vectorize!
    //         for (int j = 0; j < vs; ++j)
    //             ok[j] = areAlignedRZ(r1[j], z1[j], ro, zo, ptmin, thetaCut);
    //         for (int j = 0; j < vs; ++j)
    //         {
    //             auto koc = innerCells[i + j];
    //             auto &oc = allCells[koc];
    //             if (ok[j] && haveSimilarCurvature(oc, ptmin, region_origin_x, region_origin_y,
    //                                               region_origin_radius, phiCut, hardPtCut))
    //             {
    //                 if (foundTriplets)
    //                     foundTriplets->emplace_back(HGCDoublet::HGCntuplet{koc, cellId});
    //                 else
    //                 {
    //                     oc.tagAsOuterNeighbor(cellId);
    //                 }
    //             }
    //         }
    //     };
    //     auto lim = VSIZE * (ncells / VSIZE);
    //     for (int i = 0; i < lim; i += VSIZE)
    //         loop(i, VSIZE);
    //     loop(lim, ncells - lim);
    // }

    // void checkAlignmentAndTag(std::vector<HGCDoublet> &allCells, CAntuple &innerCells, const double ptmin, const double region_origin_x,
    //                           const double region_origin_y, const double region_origin_radius, const double thetaCut,
    //                           const double phiCut, const double hardPtCut)
    // {
    //     checkAlignmentAndAct(allCells, innerCells, ptmin, region_origin_x, region_origin_y, region_origin_radius, thetaCut,
    //                          phiCut, hardPtCut, nullptr);
    // }
    // void checkAlignmentAndPushTriplet(std::vector<HGCDoublet> &allCells, CAntuple &innerCells, std::vector<HGCDoublet::HGCntuplet> &foundTriplets,
    //                                   const double ptmin, const double region_origin_x, const double region_origin_y,
    //                                   const double region_origin_radius, const double thetaCut, const double phiCut,
    //                                   const double hardPtCut)
    // {
    //     checkAlignmentAndAct(allCells, innerCells, ptmin, region_origin_x, region_origin_y, region_origin_radius, thetaCut,
    //                          phiCut, hardPtCut, &foundTriplets);
    // }

    // int areAlignedRZ(double r1, double z1, double ro, double zo, const double ptmin, const double thetaCut) const
    // {
    //     double radius_diff = std::abs(r1 - ro);
    //     double distance_13_squared = radius_diff * radius_diff + (z1 - zo) * (z1 - zo);

    //     double pMin = ptmin * std::sqrt(distance_13_squared); //this needs to be divided by radius_diff later

    //     double tan_12_13_half_mul_distance_13_squared = fabs(z1 * (getInnerR() - ro) + getInnerZ() * (ro - r1) + zo * (r1 - getInnerR()));
    //     return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
    // }

     // // trying to free the track building process from hardcoded layers, leaving the visit of the graph
    // // based on the neighborhood connections between cells.

    // void findNtuplets(std::vector<HGCDoublet> &allCells, std::vector<HGCntuplet> &foundNtuplets, HGCntuplet &tmpNtuplet, const unsigned int minClustersPerNtuplet) const
    // {

    //     // the building process for a track ends if:
    //     // it has no outer neighbor
    //     // it has no compatible neighbor
    //     // the ntuplets is then saved if the number of hits it contains is greater than a threshold

    //     if (tmpNtuplet.size() == minClustersPerNtuplet - 1)
    //     {
    //         foundNtuplets.push_back(tmpNtuplet);
    //     }
    //     else
    //     {
    //         unsigned int numberOfOuterNeighbors = theOuterNeighbors_.size();
    //         for (unsigned int i = 0; i < numberOfOuterNeighbors; ++i)
    //         {
    //             tmpNtuplet.push_back((theOuterNeighbors_[i]));
    //             allCells[theOuterNeighbors_[i]].findNtuplets(allCells, foundNtuplets, tmpNtuplet, minClustersPerNtuplet);
    //             tmpNtuplet.pop_back();
    //         }
    //     }
    // }

  private:
    const std::vector<reco::CaloCluster> *layerClusters_;
    std::vector<int> theOuterNeighbors_;
    std::vector<int> theInnerNeighbors_;

    const int theDoubletId_;
    const int theInnerClusterId_;
    const int theOuterClusterId_;

    const double theInnerR_;
    const double theOuterR_;
    const double theInnerX_;
    const double theOuterX_;
    const double theInnerY_;
    const double theOuterY_;
    const double theInnerZ_;
    const double theOuterZ_;
};

#endif /*HGCDoublet_H_ */
