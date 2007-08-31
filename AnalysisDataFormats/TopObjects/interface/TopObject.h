//
// $Id: TopObject.h,v 1.7.2.1 2007/08/30 16:53:23 heyninck Exp $
//

#ifndef TopObjects_TopObject_h
#define TopObjects_TopObject_h

/**
  \class    TopObject TopObject.h "AnalysisDataFormats/TopObjects/interface/TopObject.h"
  \brief    High-level templated top object container

   TopObject is the templated base top object that wraps around reco objects

  \author   Jan Heyninck
  \version  $Id: TopObject.h,v 1.7.2.1 2007/08/30 16:53:23 heyninck Exp $
*/

#include <vector>


template <class ObjectType>
class TopObject : public ObjectType {

  public:

    TopObject();
    TopObject(ObjectType obj);
    virtual ~TopObject() {}

    double getResA() const;
    double getResB() const;
    double getResC() const;
    double getResD() const;
    double getResET() const;
    double getResEta() const;
    double getResPhi() const;
    double getResTheta() const;
    std::vector<double> getCovM() const;

    // FIXME: make these protected, once we have a base kinfit interface class
    void setResA(double a);
    void setResB(double b);
    void setResC(double c);
    void setResD(double d);
    void setResET(double et);
    void setResEta(double eta);
    void setResPhi(double phi);
    void setResTheta(double theta);
    void setCovM(std::vector<double>);

  protected:

    // resolution members
    double resET_;
    double resEta_;
    double resPhi_;
    double resA_;
    double resB_;
    double resC_;
    double resD_;
    double resTheta_;
    // covariance matrix (vector instead of matrix -> compact when not used)
    std::vector<double> covM_;

};


/// default constructor
template <class ObjectType> TopObject<ObjectType>::TopObject() :
  resET_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0), resTheta_(0) {
}


/// constructor from a base object
template <class ObjectType> TopObject<ObjectType>::TopObject(ObjectType obj) :
  ObjectType(obj),
  resET_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0),  resTheta_(0) {
}



template <class ObjectType> double TopObject<ObjectType>::getResET() const    	  	{ return resET_; }
template <class ObjectType> double TopObject<ObjectType>::getResEta() const    	  	{ return resEta_; }
template <class ObjectType> double TopObject<ObjectType>::getResPhi() const    	  	{ return resPhi_; }
template <class ObjectType> double TopObject<ObjectType>::getResA() const         	{ return resA_; }
template <class ObjectType> double TopObject<ObjectType>::getResB() const         	{ return resB_; }
template <class ObjectType> double TopObject<ObjectType>::getResC() const         	{ return resC_; }
template <class ObjectType> double TopObject<ObjectType>::getResD() const         	{ return resD_; }
template <class ObjectType> double TopObject<ObjectType>::getResTheta() const     	{ return resTheta_; }
template <class ObjectType> std::vector<double> TopObject<ObjectType>::getCovM() const 	{ return covM_; }


template <class ObjectType> void TopObject<ObjectType>::setResET(double et)       { resET_ = et; }
template <class ObjectType> void TopObject<ObjectType>::setResEta(double eta)     { resEta_ = eta; }
template <class ObjectType> void TopObject<ObjectType>::setResPhi(double phi)     { resPhi_ = phi; }
template <class ObjectType> void TopObject<ObjectType>::setResA(double a)         { resA_ = a; }
template <class ObjectType> void TopObject<ObjectType>::setResB(double b)         { resB_ = b; }
template <class ObjectType> void TopObject<ObjectType>::setResC(double c)         { resC_ = c; }
template <class ObjectType> void TopObject<ObjectType>::setResD(double d)         { resD_ = d; }
template <class ObjectType> void TopObject<ObjectType>::setResTheta(double theta) { resTheta_ = theta; }
template <class ObjectType> void TopObject<ObjectType>::setCovM(std::vector<double> c) { 
  covM_.clear();
  for (size_t i = 0; i < c.size(); i++) covM_.push_back(c[i]); 
}


#endif
