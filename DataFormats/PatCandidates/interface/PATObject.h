//
// $Id: PATObject.h,v 1.2 2008/01/16 20:33:23 lowette Exp $
//

#ifndef DataFormats_PatCandidates_PATObject_h
#define DataFormats_PatCandidates_PATObject_h

/**
  \class    PATObject PATObject.h "DataFormats/PatCandidates/interface/PATObject.h"
  \brief    Templated PAT object container

   PATObject is the templated base PAT object that wraps around reco objects.

  \author   Steven Lowette
  \version  $Id: PATObject.h,v 1.2 2008/01/16 20:33:23 lowette Exp $
*/

#include <DataFormats/Common/interface/Ref.h>

#include <vector>


namespace pat {


  template <class ObjectType>
  class PATObject : public ObjectType {

    public:

      PATObject();
      PATObject(const ObjectType & obj);
      PATObject(const edm::Ref<std::vector<ObjectType> > & ref);
      virtual ~PATObject() {}

      const ObjectType * originalObject() const;
      float resolutionA() const;
      float resolutionB() const;
      float resolutionC() const;
      float resolutionD() const;
      float resolutionET() const;
      float resolutionEta() const;
      float resolutionPhi() const;
      float resolutionTheta() const;
      const std::vector<float> & covMatrix() const;

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

      // reference back to the original object
      edm::Ref<std::vector<ObjectType> > refToOrig_;
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

  /// constructor from a base object (leaves invalid reference to original object!)
  template <class ObjectType> PATObject<ObjectType>::PATObject(const ObjectType & obj) :
    ObjectType(obj),
    refToOrig_(edm::ProductID(0)),
    resET_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0),  resTheta_(0) {
  }

  /// constructor from a ref to an object
  template <class ObjectType> PATObject<ObjectType>::PATObject(const edm::Ref<std::vector<ObjectType> > & ref) :
    ObjectType(*ref),
    refToOrig_(ref),
    resET_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0),  resTheta_(0) {
  }

  /// access to the original object; returns zero for null Ref and throws for unavailable collection
  template <class ObjectType> const ObjectType * PATObject<ObjectType>::originalObject() const {
    if (refToOrig_.isNull()) {
      // this object was not produced from a reference, so no link to the
      // original object exists -> return a 0-pointer
      return 0;
    } else if (!refToOrig_.isAvailable()) {
      throw edm::Exception(edm::errors::ProductNotFound) << "The original collection from which this PAT object was made is not present any more in the event, hence you cannot access the originating object anymore.";
      return 0;
    } else {
      return refToOrig_.get();
    }
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
