#ifndef CaloSegment_h
#define CaloSegment_h
/** \file FamosGeneric/FamosCalorimeters/interface/CrytalSegment.h 
 *
 * A segment between two CaloPoints. 
 * 
 */

//FAMOS headers
#include "FastSimulation/CaloGeometryTools/interface/CaloPoint.h"

#include <string>

class CaloGeometryHelper;
class CaloSegment 
{

 public: 

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;

  enum Material{PbWO4=0,CRACK=1,GAP=2,PS=3,HCAL=4,ECALHCALGAP=5,PSEEGAP=6};
  
  CaloSegment(const CaloPoint& in,const CaloPoint& out,double si,double siX0,double liX0,Material mat,
	      const CaloGeometryHelper * );
  ~CaloSegment(){;}
  /// absciss of the entrance (in cm)
  inline double sEntrance() const { return sentrance_;};
  /// absciss of the exit (in cm)
  inline double sExit() const { return sexit_;};
  /// absciss of the entrance (in X0)
  inline double sX0Entrance() const { return sX0entrance_;};
  /// absciss of the exit (in X0)
  inline double sX0Exit() const { return sX0exit_;};
  /// absciss of the entrance (in L0)
  inline double sL0Entrance() const { return sL0entrance_;};
  /// absciss of the exit (in L0)
  inline double sL0Exit() const { return sL0exit_;};  
  /// length of the segment (in cm)
  inline double length() const { return length_;};
  /// length of the segment (in X0)
  inline double X0length() const { return X0length_;};
  /// length of the segment (in L9)
  inline double L0length() const { return L0length_;};
  /// first point of the segment
  inline const CaloPoint& entrance() const { return entrance_;};
  /// last point of the segment (there are only two)
  inline const CaloPoint& exit() const { return exit_;};

  /// ordering operator wrt to the particle direction
  inline bool operator<(const CaloSegment & s) const
    { return sentrance_<s.sEntrance()&&sexit_<sExit(); }
  /// material
  inline Material material() const { return material_; }; 
  /// In which detector
  inline DetId::Detector whichDetector() const {return detector_;};
  /// space point corresponding to this depth (in cm)
  XYZPoint positionAtDepthincm(double depth) const ;
  /// space point corresponding to this depth (in X0)
  XYZPoint positionAtDepthinX0(double depth) const;
  /// space point corresponding to this depth (in L0)
  XYZPoint positionAtDepthinL0(double depth) const;

  /// cm to X0 conversion
  double x0FromCm(double cm) const;

  private:
  //  static ECALProperties myCaloProperties;
  CaloPoint  entrance_;
  CaloPoint  exit_; 
  double sentrance_;
  double sexit_;
  double sX0entrance_;
  double sX0exit_;
  double length_;
  double X0length_;
  double sL0entrance_;
  double sL0exit_;
  double L0length_;
  Material material_;
  DetId::Detector detector_;

 public:
  /// This class is used to determine if a point lies in the segment
  class inX0Segment
    {
    public:
      //      inSegment(const CaloSegment & ref):segment_(ref){;};
      inX0Segment(double depth):ref_(depth)
	{
	  //std::cout << "inSegment " << std::endl;
	};
      ~inX0Segment(){;};
      // in X0 !!!
//      bool operator() (double value) const
//	{
//	  return (value>segment_.sX0Entrance()&&value<segment_.sX0Exit());
//	}
      bool operator()(const CaloSegment & segment) const
	{
	  return (ref_>segment.sX0Entrance()&&ref_<segment.sX0Exit());
	}
    private:
      //      const CaloSegment & segment_;
      double ref_;
    };

  class inL0Segment
    {
    public:
      //      inSegment(const CaloSegment & ref):segment_(ref){;};
      inL0Segment(double depth):ref_(depth)
	{
	  //std::cout << "inSegment " << std::endl;
	};
      ~inL0Segment(){;};
      // in X0 !!!
//      bool operator() (double value) const
//	{
//	  return (value>segment_.sX0Entrance()&&value<segment_.sX0Exit());
//	}
      bool operator()(const CaloSegment & segment) const
	{
	  return (ref_>segment.sL0Entrance()&&ref_<segment.sL0Exit());
	}
    private:
      //      const CaloSegment & segment_;
      double ref_;
    };
  
 class inSegment
    {
    public:
      //      inSegment(const CaloSegment & ref):segment_(ref){;};
      inSegment(double depth):ref_(depth)
	{
	  //std::cout << "inSegment " << std::endl;
	};
      ~inSegment(){;};
      // in X0 !!!
//      bool operator() (double value) const
//	{
//	  return (value>segment_.sX0Entrance()&&value<segment_.sX0Exit());
//	}
      bool operator()(const CaloSegment & segment) const
	{
	  //	  std::cout << " Entrance " << segment.sEntrance() << " Exit " << segment.sExit() << " " << ref_ << " " << segment.whichDetector() << std::endl;
	  return (ref_>segment.sEntrance()&&ref_<segment.sExit());
	}
    private:
      //      const CaloSegment & segment_;
      double ref_;
    };
};
#include<iosfwd>
std::ostream& operator <<(std::ostream& o , const CaloSegment& cid);

#endif
