
#include "RecoPPS/Local/interface/RPixRoadFinder.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "TMath.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#include <vector>
#include <memory>
#include <string>
#include <iostream>

//------------------------------------------------------------------------------------------------//

RPixRoadFinder::RPixRoadFinder(edm::ParameterSet const& parameterSet) : RPixDetPatternFinder(parameterSet) {
  verbosity_ = parameterSet.getUntrackedParameter<int>("verbosity");
  roadRadius_ = parameterSet.getParameter<double>("roadRadius");
  minRoadSize_ = parameterSet.getParameter<int>("minRoadSize");
  maxRoadSize_ = parameterSet.getParameter<int>("maxRoadSize");
  roadRadiusBadPot_ = parameterSet.getParameter<double>("roadRadiusBadPot");
}

//------------------------------------------------------------------------------------------------//

RPixRoadFinder::~RPixRoadFinder() {}

//------------------------------------------------------------------------------------------------//

void RPixRoadFinder::findPattern(bool* is2PlanePot) {
  Road temp_all_hits;
  Road temp_all_hits_2PlanePot[4];

  // convert local hit sto global and push them to a vector
  for (const auto& ds_rh2 : *hitVector_) {
    const auto myid = CTPPSPixelDetId(ds_rh2.id);
    for (const auto& it_rh : ds_rh2.data) {
      CTPPSGeometry::Vector localV(it_rh.point().x(), it_rh.point().y(), it_rh.point().z());
      const auto& globalV = geometry_->localToGlobal(ds_rh2.id, localV);
      math::Error<3>::type localError;
      localError[0][0] = it_rh.error().xx();
      localError[0][1] = it_rh.error().xy();
      localError[0][2] = 0.;
      localError[1][0] = it_rh.error().xy();
      localError[1][1] = it_rh.error().yy();
      localError[1][2] = 0.;
      localError[2][0] = 0.;
      localError[2][1] = 0.;
      localError[2][2] = 0.;
      if (verbosity_ > 2)
        edm::LogInfo("RPixRoadFinder") << "Hits = " << ds_rh2.data.size();

      DetGeomDesc::RotationMatrix theRotationMatrix = geometry_->sensor(myid)->rotation();
      AlgebraicMatrix33 theRotationTMatrix;
      theRotationMatrix.GetComponents(theRotationTMatrix(0, 0),
                                      theRotationTMatrix(0, 1),
                                      theRotationTMatrix(0, 2),
                                      theRotationTMatrix(1, 0),
                                      theRotationTMatrix(1, 1),
                                      theRotationTMatrix(1, 2),
                                      theRotationTMatrix(2, 0),
                                      theRotationTMatrix(2, 1),
                                      theRotationTMatrix(2, 2));

      math::Error<3>::type globalError = ROOT::Math::SimilarityT(theRotationTMatrix, localError);

      // create new collections for bad (2 planes) and good pots

      if (is2PlanePot[0] == true && myid.arm() == 0 && myid.station() == 2) {  // 45-220
        temp_all_hits_2PlanePot[0].emplace_back(PointInPlane{globalV, globalError, it_rh, myid});
      } else if (is2PlanePot[1] == true && myid.arm() == 0 && myid.station() == 0) {  // 45-210
        temp_all_hits_2PlanePot[1].emplace_back(PointInPlane{globalV, globalError, it_rh, myid});
      } else if (is2PlanePot[2] == true && myid.arm() == 1 && myid.station() == 0) {  // 56-210
        temp_all_hits_2PlanePot[2].emplace_back(PointInPlane{globalV, globalError, it_rh, myid});
      } else if (is2PlanePot[3] == true && myid.arm() == 1 && myid.station() == 2) {  // 56-220
        temp_all_hits_2PlanePot[3].emplace_back(PointInPlane{globalV, globalError, it_rh, myid});
      } else {
        temp_all_hits.emplace_back(PointInPlane{globalV, globalError, it_rh, myid});
      }
    }
  }

  Road::iterator it_gh1 = temp_all_hits.begin();
  Road::iterator it_gh2;

  patternVector_.clear();

  //look for points near wrt each other
  // starting algorithm
  while (it_gh1 != temp_all_hits.end() && temp_all_hits.size() >= minRoadSize_) {
    Road temp_road;

    it_gh2 = it_gh1;

    const auto currPoint = it_gh1->globalPoint;
    CTPPSPixelDetId currDet = CTPPSPixelDetId(it_gh1->detId);

    while (it_gh2 != temp_all_hits.end()) {
      bool same_pot = false;
      CTPPSPixelDetId tmpGh2Id = CTPPSPixelDetId(it_gh2->detId);
      if (currDet.rpId() == tmpGh2Id.rpId())
        same_pot = true;
      const auto subtraction = currPoint - it_gh2->globalPoint;

      if (subtraction.Rho() < roadRadius_ && same_pot) {  /// 1mm
        temp_road.push_back(*it_gh2);
        temp_all_hits.erase(it_gh2);
      } else {
        ++it_gh2;
      }
    }

    if (temp_road.size() >= minRoadSize_ && temp_road.size() < maxRoadSize_)
      patternVector_.push_back(temp_road);
  }
  // end of algorithm

  // 2PlanePot algorithm

  for (unsigned int i = 0; i < 4; i++) {
    if (is2PlanePot[i]) {
      Road::iterator it_gh1_bP = temp_all_hits_2PlanePot[i].begin();
      Road::iterator it_gh2_bP;

      while (it_gh1_bP != temp_all_hits_2PlanePot[i].end() && temp_all_hits_2PlanePot[i].size() >= 2) {
        Road temp_road;

        it_gh2_bP = it_gh1_bP;

        const auto currPoint = it_gh1_bP->globalPoint;

        while (it_gh2_bP != temp_all_hits_2PlanePot[i].end()) {
          const auto subtraction = currPoint - it_gh2_bP->globalPoint;

          if (subtraction.Rho() < roadRadiusBadPot_) {
            temp_road.push_back(*it_gh2_bP);
            temp_all_hits_2PlanePot[i].erase(it_gh2_bP);
          } else {
            ++it_gh2_bP;
          }
        }

        if (temp_road.size() == 2) {  // look for isolated tracks
          patternVector_.push_back(temp_road);
        }
      }
    }
  }
}
