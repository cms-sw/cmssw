#ifndef ElementsInAnnulus_h
#define ElementsInAnnulus_h

#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Math/interface/angle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Point3D.h"

template <typename T, typename M, typename N, typename C>
  class ElementsInAnnulus {
   public:
    ElementsInAnnulus() {}
    ~ElementsInAnnulus() {}  
    const std::vector<edm::Ptr<C> > operator()(const T& coneAxis,const M& innerconeMetric,double innerconeSize,const N& outerconeMetric,double outerconeSize,const std::vector<edm::Ptr<C> >& elements)const{
      std::vector<edm::Ptr<C> > elementsInBand;
      for(typename std::vector<edm::Ptr<C> >::const_iterator element=elements.begin();element!=elements.end();++element) {
	double innerconeMetric_distance=innerconeMetric(coneAxis,(*element)->momentum());
	double outerconeMetric_distance=outerconeMetric(coneAxis,(*element)->momentum());
	if (innerconeMetric_distance>innerconeSize && outerconeMetric_distance<=outerconeSize)elementsInBand.push_back(*element);
      }
      return elementsInBand;
    }
};
template <typename T, typename M, typename N>
  class ElementsInAnnulus<T, M, N, std::pair<math::XYZPoint,float> > {
   public:
    ElementsInAnnulus() {}
    ~ElementsInAnnulus() {}  
    const std::vector<std::pair<math::XYZPoint,float> > operator()(const T& coneAxis,const M& innerconeMetric,double innerconeSize,const N& outerconeMetric,double outerconeSize,const std::vector<std::pair<math::XYZPoint,float> > & elements)const{
      std::vector<std::pair<math::XYZPoint,float> > elementsInBand;
      for(typename std::vector<std::pair<math::XYZPoint,float> >::const_iterator element=elements.begin();element!=elements.end();++element) {
	double innerconeMetric_distance=innerconeMetric(coneAxis,(*element).first);
	double outerconeMetric_distance=outerconeMetric(coneAxis,(*element).first);
	if (innerconeMetric_distance>innerconeSize && outerconeMetric_distance<=outerconeSize)elementsInBand.push_back(*element);
      }
      return elementsInBand;
    }    
};

#endif

