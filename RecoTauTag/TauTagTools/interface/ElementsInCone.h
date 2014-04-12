#ifndef ElementsInCone_h
#define ElementsInCone_h

#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Math/interface/angle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Point3D.h"

template <typename T, typename M, typename C>
  class ElementsInCone {
   public:
    ElementsInCone() {}
    ~ElementsInCone() {}
    const std::vector<edm::Ptr<C> > operator()(const T& coneAxis,const M& coneMetric,double coneSize, std::vector<edm::Ptr<C> > elements)const{
      std::vector<edm::Ptr<C> > elementsInCone;
      for(typename std::vector<edm::Ptr<C> >::const_iterator element=elements.begin();element!=elements.end();++element) {
	double distance = coneMetric(coneAxis,(*element)->momentum());
	if (distance<=coneSize)elementsInCone.push_back(*element);
      }
      return elementsInCone;
    }
  /*   const edm::RefVector<C> operator()(const T& coneAxis,const M& coneMetric,double coneSize,const edm::RefVector<C>& elements)const{ */
/*       edm::RefVector<C> elementsInCone; */
/*       for(typename edm::RefVector<C>::const_iterator element=elements.begin();element!=elements.end();++element) { */
/*         double distance = coneMetric(coneAxis,(*element)->momentum()); */
/*         if (distance<=coneSize)elementsInCone.push_back(*element); */
/*       } */
/*       return elementsInCone; */
/*     } */
};

template <typename T, typename M> 
class ElementsInCone<T, M, std::pair<math::XYZPoint,float> >{
   public:
    ElementsInCone() {}
    ~ElementsInCone() {}  
    const std::vector<std::pair<math::XYZPoint,float> > operator()(const T& coneAxis,const M& coneMetric,double coneSize,const std::vector<std::pair<math::XYZPoint,float> >& elements)const{
      std::vector<std::pair<math::XYZPoint,float> > elementsInCone;
      for(typename std::vector<std::pair<math::XYZPoint,float> >::const_iterator element=elements.begin();element!=elements.end();++element) {
	double distance = coneMetric(coneAxis,(*element).first);
	if (distance<=coneSize)elementsInCone.push_back(*element);
      }
      return elementsInCone;      
    }
};

#endif

