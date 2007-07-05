//
// $Id: TopObject.h,v 1.6 2007/06/23 07:07:45 lowette Exp $
//

#ifndef TopObjects_TopObject_h
#define TopObjects_TopObject_h

/**
  \class    TopObject TopObject.h "AnalysisDataFormats/TopObjects/interface/TopObject.h"
  \brief    High-level templated top object container

   TopObject is the templated base top object that wraps around reco objects

  \author   Jan Heyninck
  \version  $Id: TopObject.h,v 1.6 2007/06/23 07:07:45 lowette Exp $
*/

#include <vector>


template <class ObjectType>
class TopObject : public ObjectType {

  public:

    TopObject();
    TopObject(ObjectType obj);
    virtual ~TopObject() {}

    double getResET() const;
    double getResEta() const;
    double getResPhi() const;
    double getResD() const;
    double getResPinv() const;
    double getResTheta() const;
    std::vector<double> getCovM() const;

    // FIXME: make these protected, once we have a base kinfit interface class
    void setResET(double et);
    void setResEta(double eta);
    void setResPhi(double phi);
    void setResD(double d);
    void setResPinv(double pinv);
    void setResTheta(double theta);
    void setCovM(std::vector<double>);

  protected:

    // resolution members
    double resET_;
    double resEta_;
    double resPhi_;
    double resD_;
    double resPinv_;
    double resTheta_;
    // covariance matrix (vector instead of matrix -> compact when not used)
    std::vector<double> covM_;

};


/// default constructor
template <class ObjectType> TopObject<ObjectType>::TopObject() :
  resET_(0), resEta_(0), resPhi_(0), resD_(0), resPinv_(0), resTheta_(0) {
}


/// constructor from a base object
template <class ObjectType> TopObject<ObjectType>::TopObject(ObjectType obj) :
  ObjectType(obj),
  resET_(0), resEta_(0), resPhi_(0), resD_(0), resPinv_(0), resTheta_(0) {
}


template <class ObjectType> double TopObject<ObjectType>::getResET() const    	  	{ return resET_; }
template <class ObjectType> double TopObject<ObjectType>::getResEta() const    	  	{ return resEta_; }
template <class ObjectType> double TopObject<ObjectType>::getResPhi() const    	  	{ return resPhi_; }
template <class ObjectType> double TopObject<ObjectType>::getResD() const         	{ return resD_; }
template <class ObjectType> double TopObject<ObjectType>::getResPinv() const      	{ return resPinv_; }
template <class ObjectType> double TopObject<ObjectType>::getResTheta() const     	{ return resTheta_; }
template <class ObjectType> std::vector<double> TopObject<ObjectType>::getCovM() const 	{ return covM_; }


template <class ObjectType> void TopObject<ObjectType>::setResET(double et)       { resET_ = et; }
template <class ObjectType> void TopObject<ObjectType>::setResEta(double eta)     { resEta_ = eta; }
template <class ObjectType> void TopObject<ObjectType>::setResPhi(double phi)     { resPhi_ = phi; }
template <class ObjectType> void TopObject<ObjectType>::setResD(double d)         { resD_ = d; }
template <class ObjectType> void TopObject<ObjectType>::setResPinv(double pinv)   { resPinv_ = pinv; }
template <class ObjectType> void TopObject<ObjectType>::setResTheta(double theta) { resTheta_ = theta; }
template <class ObjectType> void TopObject<ObjectType>::setCovM(std::vector<double> c) { 
  covM_.clear();
  for (size_t i = 0; i < c.size(); i++) covM_.push_back(c[i]); 
}


#endif
