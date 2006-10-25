//FAMOS headers
#include "FastSimulation/CaloGeometryTools/interface/CaloSegment.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer2Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"

CaloSegment::CaloSegment(const CaloPoint& in,
			 const CaloPoint& out,
			 double si,
			 double siX0, 
			 double siL0,
			 Material mat,
			 const CaloGeometryHelper* myCalorimeter):
  entrance_(in),
  exit_(out),
  sentrance_(si),
  sX0entrance_(siX0),
  sL0entrance_(siL0),
  material_(mat)

{

  sexit_= sentrance_+(exit_-entrance_).mag();
  // Change this. CaloProperties from FamosShower should be used instead 
  double radLenIncm=999999;
  double intLenIncm=999999;
  detector_=in.whichDetector();
  if(detector_!=out.whichDetector()&&mat!=CRACK&&mat!=GAP&&mat!=ECALHCALGAP) 
    {
      std::cout << " Problem in the segments " << detector_ << " " << out.whichDetector() <<std::endl;
    }
  switch (mat)
    {
    case PbWO4:
      {
	radLenIncm =
	  myCalorimeter->ecalProperties(1)->radLenIncm();
	intLenIncm =
	  myCalorimeter->ecalProperties(1)->interactionLength();
      }
      break;
    case CRACK:
      {
	radLenIncm=8.9;//cracks : Al
	intLenIncm=35.4;
      }
      break;
    case PS:
      {
	radLenIncm = 
	  myCalorimeter->layer1Properties(1)->radLenIncm();
	intLenIncm = 
	  myCalorimeter->layer1Properties(1)->interactionLength();
      }
      break;
    case HCAL:
      {
	radLenIncm = 
	  myCalorimeter->hcalProperties(1)->radLenIncm();
	intLenIncm = 
	  myCalorimeter->hcalProperties(1)->interactionLength();
      }
      break;
    case ECALHCALGAP:
      {
	// From Olga's & Patrick's talk PRS/JetMET 21 Sept 2004 
	radLenIncm = 22.3;
	intLenIncm = 140;
      }
      break;
    default:
      radLenIncm=999999;
    }
  sX0exit_ = sX0entrance_+(sexit_-sentrance_)/radLenIncm;
  sL0exit_ = sL0entrance_+(sexit_-sentrance_)/intLenIncm;
  if(mat==GAP) 
    {
      sX0exit_=sX0entrance_;
      sL0exit_=sL0entrance_;
    }
  length_ = sexit_-sentrance_;
  X0length_ = sX0exit_-sX0entrance_;
  L0length_ = sL0exit_-sL0entrance_;
}

HepPoint3D CaloSegment::positionAtDepthincm(double depth) const
{
  if (depth<sentrance_||depth>sexit_) return HepPoint3D();
  return HepPoint3D(entrance_+((depth-sentrance_)/(sexit_-sentrance_)*(exit_-entrance_)));
}

HepPoint3D CaloSegment::positionAtDepthinX0(double depth) const
{
  if (depth<sX0entrance_||depth>sX0exit_) return HepPoint3D();
  return HepPoint3D(entrance_+((depth-sX0entrance_)/(sX0exit_-sX0entrance_)*(exit_-entrance_)));
}

HepPoint3D CaloSegment::positionAtDepthinL0(double depth) const
{
  if (depth<sL0entrance_||depth>sL0exit_) return HepPoint3D();
  return HepPoint3D(entrance_+((depth-sL0entrance_)/(sL0exit_-sL0entrance_)*(exit_-entrance_)));
}

double CaloSegment::x0FromCm(double cm) const{
  return sX0entrance_+cm/length_*X0length_;
}

std::ostream & operator<<(std::ostream& ost ,const CaloSegment& seg)
{
  ost << " DetId " ;
  if(!seg.entrance().getDetId().null()) 
    ost << seg.entrance().getDetId()() ;
  else 
    {
      ost << seg.entrance().whichDetector() ;
      //  ost<< " Entrance side " << seg.entrance().getSide()
      ost << " Point " << (HepPoint3D)seg.entrance() << std::endl;
    }
  ost  << "DetId " ;
  if(!seg.exit().getDetId().null()) 
    ost << seg.exit().getDetId()() ;
  else
    ost << seg.exit().whichDetector() ;

  //  ost << " Exit side " << seg.exit().getSide() 
  ost << " Point " << (HepPoint3D)seg.exit() << " " << seg.length() << " cm " << seg.X0length() << " X0 " <<  seg.L0length() << " Lambda0 " ;
  switch (seg.material())
    {
    case CaloSegment::PbWO4:
      ost << "PbWO4 " ;
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
    default:
      ost << "GAP " ;
    }
  return ost;
};

