//FAMOS headers
#include "FastSimulation/CaloGeometryTools/interface/CaloSegment.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer2Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/ECALProperties.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

CaloSegment::CaloSegment(const CaloPoint& in,
                         const CaloPoint& out,
                         double si,
                         double siX0,
                         double siL0,
                         Material mat,
                         const CaloGeometryHelper* myCalorimeter)
    : entrance_(in),
      exit_(out),
      sentrance_(si),
      sX0entrance_(siX0),
      sL0entrance_(siL0),
      material_(mat)

{
  sexit_ = sentrance_ + std::sqrt((exit_ - entrance_).mag2());
  // Change this. CaloProperties from FamosShower should be used instead
  double radLenIncm = 999999;
  double intLenIncm = 999999;
  detector_ = in.whichDetector();
  if (detector_ != out.whichDetector() && mat != CRACK && mat != GAP && mat != ECALHCALGAP) {
    std::cout << " Problem in the segments " << detector_ << " " << out.whichDetector() << std::endl;
  }
  switch (mat) {
    case PbWO4: {
      int det = 0;
      if (in.whichSubDetector() == EcalBarrel)
        det = 1;
      if (in.whichSubDetector() == EcalEndcap)
        det = 2;

      radLenIncm = myCalorimeter->ecalProperties(det)->radLenIncm();
      intLenIncm = myCalorimeter->ecalProperties(det)->interactionLength();
    } break;
    case CRACK: {
      radLenIncm = 8.9;  //cracks : Al
      intLenIncm = 35.4;
    } break;
    case PS: {
      radLenIncm = myCalorimeter->layer1Properties(1)->radLenIncm();
      intLenIncm = myCalorimeter->layer1Properties(1)->interactionLength();
    } break;
    case HCAL: {
      radLenIncm = myCalorimeter->hcalProperties(1)->radLenIncm();
      intLenIncm = myCalorimeter->hcalProperties(1)->interactionLength();
    } break;
    case ECALHCALGAP: {
      // From Olga's & Patrick's talk PRS/JetMET 21 Sept 2004
      radLenIncm = 22.3;
      intLenIncm = 140;
    } break;
    case PSEEGAP: {
      // according to Sunanda 0.19 X0 (0.08X0 of polyethylene), support (0.06X0 of aluminium)  + other stuff
      // in the geometry 12 cm between layer and entrance of EE. Polyethylene is rather 48 and Al 8.9 (PDG)
      // for the inLen, just rescale according to PDG (85cm)
      radLenIncm = myCalorimeter->layer2Properties(1)->pseeRadLenIncm();
      intLenIncm = myCalorimeter->layer2Properties(1)->pseeIntLenIncm();
    } break;
    default:
      radLenIncm = 999999;
  }
  sX0exit_ = sX0entrance_ + (sexit_ - sentrance_) / radLenIncm;
  sL0exit_ = sL0entrance_ + (sexit_ - sentrance_) / intLenIncm;
  if (mat == GAP) {
    sX0exit_ = sX0entrance_;
    sL0exit_ = sL0entrance_;
  }
  length_ = sexit_ - sentrance_;
  X0length_ = sX0exit_ - sX0entrance_;
  L0length_ = sL0exit_ - sL0entrance_;
}

CaloSegment::XYZPoint CaloSegment::positionAtDepthincm(double depth) const {
  if (depth < sentrance_ || depth > sexit_)
    return XYZPoint();
  return XYZPoint(entrance_ + ((depth - sentrance_) / (sexit_ - sentrance_) * (exit_ - entrance_)));
}

CaloSegment::XYZPoint CaloSegment::positionAtDepthinX0(double depth) const {
  if (depth < sX0entrance_ || depth > sX0exit_)
    return XYZPoint();
  return XYZPoint(entrance_ + ((depth - sX0entrance_) / (sX0exit_ - sX0entrance_) * (exit_ - entrance_)));
}

CaloSegment::XYZPoint CaloSegment::positionAtDepthinL0(double depth) const {
  if (depth < sL0entrance_ || depth > sL0exit_)
    return XYZPoint();
  return XYZPoint(entrance_ + ((depth - sL0entrance_) / (sL0exit_ - sL0entrance_) * (exit_ - entrance_)));
}

double CaloSegment::x0FromCm(double cm) const { return sX0entrance_ + cm / length_ * X0length_; }

std::ostream& operator<<(std::ostream& ost, const CaloSegment& seg) {
  ost << " DetId ";
  if (!seg.entrance().getDetId().null())
    ost << seg.entrance().getDetId()();
  else {
    ost << seg.entrance().whichDetector();
    //  ost<< " Entrance side " << seg.entrance().getSide()
    ost << " Point " << (math::XYZVector)seg.entrance() << std::endl;
  }
  ost << "DetId ";
  if (!seg.exit().getDetId().null())
    ost << seg.exit().getDetId()();
  else
    ost << seg.exit().whichDetector();

  //  ost << " Exit side " << seg.exit().getSide()
  ost << " Point " << (math::XYZVector)seg.exit() << " " << seg.length() << " cm " << seg.X0length() << " X0 "
      << seg.L0length() << " Lambda0 ";
  switch (seg.material()) {
    case CaloSegment::PbWO4:
      ost << "PbWO4 ";
      break;
    case CaloSegment::CRACK:
      ost << "CRACK ";
      break;
    case CaloSegment::PS:
      ost << "PS ";
      break;
    case CaloSegment::HCAL:
      ost << "HCAL ";
      break;
    case CaloSegment::ECALHCALGAP:
      ost << "ECAL-HCAL GAP ";
      break;
    case CaloSegment::PSEEGAP:
      ost << "PS-ECAL GAP";
      break;
    default:
      ost << "GAP ";
  }
  return ost;
}
