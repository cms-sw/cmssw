//
// $Id: PATObject.h,v 1.1 2008/01/15 12:59:28 lowette Exp $
//

#ifndef DataFormats_PatCandidates_PATObject_h
#define DataFormats_PatCandidates_PATObject_h

/**
  \class    PATObject PATObject.h "DataFormats/PatCandidates/interface/PATObject.h"
  \brief    Templated PAT object container

   PATObject is the templated base PAT object that wraps around reco objects.

  \author   Steven Lowette
  \version  $Id: PATObject.h,v 1.1 2008/01/15 12:59:28 lowette Exp $
*/

#include <vector>


namespace pat {


  template <class ObjectType>
  class PATObject : public ObjectType {

    public:

      PATObject();
      PATObject(ObjectType obj);
      virtual ~PATObject() {}

      float resolutionA() const;
      float resolutionB() const;
      float resolutionC() const;
      float resolutionD() const;
      float resolutionET() const;
      float resolutionEta() const;
      float resolutionPhi() const;
      float resolutionTheta() const;
      const std::vector<float> & covMatrix() const;

      // FIXME: make these protected, once we have a base kinfit interface class
      void setResolutionA(float a);
      void setResolutionB(float b);
      void setResolutionC(float c);
      void setResolutionD(float d);
      void setResolutionET(float et);
      void setResolutionEta(float eta);
      void setResolutionPhi(float phi);
      void setResolutionTheta(float theta);
      void setCovMatrix(const std::vector<float> & c);

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
      // covariance matrix
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

  template <class ObjectType> float PATObject<ObjectType>::resolutionET() const { return resET_; }
  template <class ObjectType> float PATObject<ObjectType>::resolutionEta() const { return resEta_; }
  template <class ObjectType> float PATObject<ObjectType>::resolutionPhi() const { return resPhi_; }
  template <class ObjectType> float PATObject<ObjectType>::resolutionA() const { return resA_; }
  template <class ObjectType> float PATObject<ObjectType>::resolutionB() const { return resB_; }
  template <class ObjectType> float PATObject<ObjectType>::resolutionC() const { return resC_; }
  template <class ObjectType> float PATObject<ObjectType>::resolutionD() const { return resD_; }
  template <class ObjectType> float PATObject<ObjectType>::resolutionTheta() const { return resTheta_; }
  template <class ObjectType> const std::vector<float> & PATObject<ObjectType>::covMatrix() const { return covM_; }

  template <class ObjectType> void PATObject<ObjectType>::setResolutionET(float et) { resET_ = et; }
  template <class ObjectType> void PATObject<ObjectType>::setResolutionEta(float eta) { resEta_ = eta; }
  template <class ObjectType> void PATObject<ObjectType>::setResolutionPhi(float phi) { resPhi_ = phi; }
  template <class ObjectType> void PATObject<ObjectType>::setResolutionA(float a) { resA_ = a; }
  template <class ObjectType> void PATObject<ObjectType>::setResolutionB(float b) { resB_ = b; }
  template <class ObjectType> void PATObject<ObjectType>::setResolutionC(float c) { resC_ = c; }
  template <class ObjectType> void PATObject<ObjectType>::setResolutionD(float d) { resD_ = d; }
  template <class ObjectType> void PATObject<ObjectType>::setResolutionTheta(float theta) { resTheta_ = theta; }
  template <class ObjectType> void PATObject<ObjectType>::setCovMatrix(const std::vector<float> & c) {
    covM_.clear();
    for (size_t i = 0; i < c.size(); i++) covM_.push_back(c[i]); 
  }


}

#endif
