//
// Author:  Jan Heyninck
// Created: ?
//
// $Id: TopObject.h,v 1.4 2007/05/15 15:58:28 heyninck Exp $
//

#ifndef TopObject_h
#define TopObject_h

/**
  \class    TopObject TopObject.h "AnalysisDataFormats/TopObjects/interface/TopObject.h"
  \brief    High-level templated top object container

   TopObject is the templated base top object that wraps around reco objects

  \author   Jan Heyninck
  \version  $Id: TopObject.h,v 1.4 2007/05/15 15:58:28 heyninck Exp $
*/

#include <vector>


template <class ObjectType>
class TopObject : public ObjectType {

  public:

    TopObject() {}
    TopObject(ObjectType obj) : ObjectType(obj) {}
    virtual ~TopObject() {}

    void setResET(double);
    void setResEta(double);
    void setResPhi(double);
    void setResD(double);
    void setResPinv(double);
    void setResTheta(double);
    void setCovM(std::vector<double>);

    double getResET() const;
    double getResEta() const;
    double getResPhi() const;
    double getResD() const;
    double getResPinv() const;
    double getResTheta() const;
    std::vector<double> getCovM() const;

  protected:

    double resET_;
    double resEta_;
    double resPhi_;
    double resD_;
    double resPinv_;
    double resTheta_;
    int    parametrisation_;
    std::vector<double> covM_;

};


template <class ObjectType> void TopObject<ObjectType>::setResET(double et)       { resET_ = et; }
template <class ObjectType> void TopObject<ObjectType>::setResEta(double eta)     { resEta_ = eta; }
template <class ObjectType> void TopObject<ObjectType>::setResPhi(double phi)     { resPhi_ = phi; }
template <class ObjectType> void TopObject<ObjectType>::setResD(double d)         { resD_ = d; }
template <class ObjectType> void TopObject<ObjectType>::setResPinv(double pinv)   { resPinv_ = pinv; }
template <class ObjectType> void TopObject<ObjectType>::setResTheta(double theta) { resTheta_ = theta; }
template <class ObjectType> void TopObject<ObjectType>::setCovM(std::vector<double> c) { 
  covM_.clear();
  for(size_t i = 0; i < c.size(); i++) covM_.push_back(c[i]); 
}

template <class ObjectType> double TopObject<ObjectType>::getResET() const    	  	{ return resET_; }
template <class ObjectType> double TopObject<ObjectType>::getResEta() const    	  	{ return resEta_; }
template <class ObjectType> double TopObject<ObjectType>::getResPhi() const    	  	{ return resPhi_; }
template <class ObjectType> double TopObject<ObjectType>::getResD() const         	{ return resD_; }
template <class ObjectType> double TopObject<ObjectType>::getResPinv() const      	{ return resPinv_; }
template <class ObjectType> double TopObject<ObjectType>::getResTheta() const     	{ return resTheta_; }
template <class ObjectType> std::vector<double> TopObject<ObjectType>::getCovM() const 	{ return covM_; }


#endif
