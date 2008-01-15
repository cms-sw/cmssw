//
// $Id: PATObject.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
//

#ifndef DataFormats_PatCandidates_PATObject_h
#define DataFormats_PatCandidates_PATObject_h

/**
  \class    PATObject PATObject.h "DataFormats/PatCandidates/interface/PATObject.h"
  \brief    Templated PAT object container

   PATObject is the templated base PAT object that wraps around reco objects.

  \author   Steven Lowette
  \version  $Id: PATObject.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
*/

#include <vector>


namespace pat {


  template <class ObjectType>
  class PATObject : public ObjectType {

    public:

      PATObject();
      PATObject(ObjectType obj);
      virtual ~PATObject() {}

      float getResA() const;
      float getResB() const;
      float getResC() const;
      float getResD() const;
      float getResET() const;
      float getResEta() const;
      float getResPhi() const;
      float getResTheta() const;
      std::vector<float> getCovM() const;

      // FIXME: make these protected, once we have a base kinfit interface class
      void setResA(float a);
      void setResB(float b);
      void setResC(float c);
      void setResD(float d);
      void setResET(float et);
      void setResEta(float eta);
      void setResPhi(float phi);
      void setResTheta(float theta);
      void setCovM(std::vector<float>);

    protected:

      // resolution members
      float resET_;
      float resEta_;
      float resPhi_;
      float resA_;
      float resB_;
      float resC_;
      float resD_;
      float resTheta_;
      // covariance matrix (vector instead of matrix -> compact when not used)
      std::vector<float> covM_;

  };

  /// default constructor
  template <class ObjectType> PATObject<ObjectType>::PATObject() :
    resET_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0), resTheta_(0) {
  }

  /// constructor from a base object
  template <class ObjectType> PATObject<ObjectType>::PATObject(ObjectType obj) :
    ObjectType(obj),
    resET_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0),  resTheta_(0) {
  }

  template <class ObjectType> float PATObject<ObjectType>::getResET() const { return resET_; }
  template <class ObjectType> float PATObject<ObjectType>::getResEta() const { return resEta_; }
  template <class ObjectType> float PATObject<ObjectType>::getResPhi() const { return resPhi_; }
  template <class ObjectType> float PATObject<ObjectType>::getResA() const { return resA_; }
  template <class ObjectType> float PATObject<ObjectType>::getResB() const { return resB_; }
  template <class ObjectType> float PATObject<ObjectType>::getResC() const { return resC_; }
  template <class ObjectType> float PATObject<ObjectType>::getResD() const { return resD_; }
  template <class ObjectType> float PATObject<ObjectType>::getResTheta() const { return resTheta_; }
  template <class ObjectType> std::vector<float> PATObject<ObjectType>::getCovM() const { return covM_; }

  template <class ObjectType> void PATObject<ObjectType>::setResET(float et) { resET_ = et; }
  template <class ObjectType> void PATObject<ObjectType>::setResEta(float eta) { resEta_ = eta; }
  template <class ObjectType> void PATObject<ObjectType>::setResPhi(float phi) { resPhi_ = phi; }
  template <class ObjectType> void PATObject<ObjectType>::setResA(float a) { resA_ = a; }
  template <class ObjectType> void PATObject<ObjectType>::setResB(float b) { resB_ = b; }
  template <class ObjectType> void PATObject<ObjectType>::setResC(float c) { resC_ = c; }
  template <class ObjectType> void PATObject<ObjectType>::setResD(float d) { resD_ = d; }
  template <class ObjectType> void PATObject<ObjectType>::setResTheta(float theta) { resTheta_ = theta; }
  template <class ObjectType> void PATObject<ObjectType>::setCovM(std::vector<float> c) {
    covM_.clear();
    for (size_t i = 0; i < c.size(); i++) covM_.push_back(c[i]); 
  }


}

#endif
