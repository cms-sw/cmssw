#include "L1Trigger/L1TMuon/src/Phase2/L1TDisplacedMuonPtAssignment.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include "L1Trigger/L1TMuon/src/Phase2/EndcapTriggerPtAssignmentHelper.h"
#include "L1Trigger/L1TMuon/src/Phase2/BarrelTriggerPtAssignmentHelper.h"

#include "iostream"

using namespace L1TMuon;
using namespace EndcapTriggerPtAssignmentHelper;

L1TDisplacedMuonPtAssignment::L1TDisplacedMuonPtAssignment(const edm::ParameterSet& iConfig)
{
  // assignment_.reset(new L1TMuon::L1TDisplacedMuonPtAssignment(iConfig));

  // std::unique_ptr<EndcapTriggerPtAssignmentHelper> endcapHelper_;
  // std::unique_ptr<BarrelTriggerPtAssignmentHelper> barrelHelper_;
}

L1TDisplacedMuonPtAssignment::~L1TDisplacedMuonPtAssignment()
{
}

void L1TDisplacedMuonPtAssignment::calculatePositionPtBarrel(){}
void L1TDisplacedMuonPtAssignment::calculatePositionPtOverlap(){}
void L1TDisplacedMuonPtAssignment::calculatePositionPtEndcap()
{
  EvenOdd123 parity = EndcapTriggerPtAssignmentHelper::getParity(isEven[0], isEven[1],
                                                                 isEven[2], isEven[3]);

  // ignore invalid cases
  // put a warning message here!
  if (parity == EvenOdd123::Invalid)
    return;

  const float ddY123 = deltadeltaYcalculation(gp_st_layer3[0], gp_st_layer3[1], gp_st_layer3[2],
                                              gp_st_layer3[1].eta(), parity);
  // lowest pT assigned
  positionPt_ = 2.0;

  // walk through the DDY LUT and assign the pT that matches
  const int etaSector = EndcapTriggerPtAssignmentHelper::GetEtaPartition(gp_ME[1].eta());
  for (int i=0; i<EndcapTriggerPtAssignmentHelper::NPt2; i++){
    if (fabs(ddY123) <= EndcapTriggerPtAssignmentHelper::PositionPtDDYLUT[i][etaSector][int(parity)])
      positionPt_ = float(EndcapTriggerPtAssignmentHelper::PtBins2[i]);
    else
      break;
  }
}

void L1TDisplacedMuonPtAssignment::calculateDirectionPtBarrel()
{
  // check case
  int dt_stub_case = 0;//getBarrelStubCase(has_stub_mb1, has_stub_mb2, has_stub_mb3, has_stub_mb4);

  float barrel_direction_pt;

  switch(dt_stub_case){
  case 0:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt2Stubs(dPhi_barrel_dir_12, "DT1_DT2");
    break;
  case 1:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt2Stubs(dPhi_barrel_dir_13, "DT1_DT3");
    break;
  case 2:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt2Stubs(dPhi_barrel_dir_14, "DT1_DT4");
    break;
  case 3:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt2Stubs(dPhi_barrel_dir_23, "DT2_DT3");
    break;
  case 4:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt2Stubs(dPhi_barrel_dir_24, "DT2_DT4");
    break;
  case 5:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt2Stubs(dPhi_barrel_dir_34, "DT3_DT4");
    break;
  case 6:
    // first dphi is x value, second dphi is y value!!!!
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt3or4Stubs(dPhi_barrel_dir_12, dPhi_barrel_dir_13, "DT1_DT2__DT1_DT3");
    break;
  case 7:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt3or4Stubs(dPhi_barrel_dir_12, dPhi_barrel_dir_14, "DT1_DT2__DT1_DT4");
    break;
  case 8:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt3or4Stubs(dPhi_barrel_dir_13, dPhi_barrel_dir_14, "DT1_DT3__DT1_DT4");
    break;
  case 9:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt3or4Stubs(dPhi_barrel_dir_23, dPhi_barrel_dir_24, "DT2_DT3__DT2_DT4");
    break;
  case 10:
    barrel_direction_pt = BarrelTriggerPtAssignmentHelper::getDirectionPt3or4Stubs(dPhi_barrel_dir_14, dPhi_barrel_dir_23, "DT1_DT4__DT2_DT3");
    break;
  default:
    // all else fails; assign lowest possible pT
    barrel_direction_pt = 2;
    break;
  };
  std::cout << barrel_direction_pt << std::endl;
}

void L1TDisplacedMuonPtAssignment::calculateDirectionPtOverlap(){}
void L1TDisplacedMuonPtAssignment::calculateDirectionPtEndcapLow(){}
void L1TDisplacedMuonPtAssignment::calculateDirectionPtEndcapHigh(){}

void L1TDisplacedMuonPtAssignment::calculateHybridPtBarrel(){}
void L1TDisplacedMuonPtAssignment::calculateHybridPtOverlap(){}
void L1TDisplacedMuonPtAssignment::calculateHybridPtEndcapLow(){}
void L1TDisplacedMuonPtAssignment::calculateHybridPtEndcapHigh(){}

int L1TDisplacedMuonPtAssignment::getBarrelStubCase(bool MB1, bool MB2, bool MB3, bool MB4)
{
  if (    MB1 and     MB2 and not MB3 and not MB4) return 0;
  if (    MB1 and not MB2 and     MB3 and not MB4) return 1;
  if (    MB1 and not MB2 and not MB3 and     MB4) return 2;
  if (not MB1 and     MB2 and     MB3 and not MB4) return 3;
  if (not MB1 and     MB2 and not MB3 and     MB4) return 4;
  if (not MB1 and not MB2 and     MB3 and     MB4) return 5;

  if (    MB1 and     MB2 and     MB3 and not MB4) return 6;
  if (    MB1 and     MB2 and not MB3 and     MB4) return 7;
  if (    MB1 and not MB2 and     MB3 and     MB4) return 8;
  if (not MB1 and     MB2 and     MB3 and     MB4) return 9;

  if (MB1 and MB2 and MB3 and MB4) return 10;

  return -1;
}
