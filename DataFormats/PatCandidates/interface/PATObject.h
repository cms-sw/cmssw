#ifndef DataFormats_PatCandidates_PATObject_h
#define DataFormats_PatCandidates_PATObject_h

/** \class    pat::PATObject PATObject.h "DataFormats/PatCandidates/interface/PATObject.h"
 *
 *  \brief    Templated PAT object container
 *
 *  PATObject is the templated base PAT object that wraps around reco objects.
 *
 *  \author   Steven Lowette
 *
 *  \version  $Id: PATObject.h,v 1.7 2008/03/05 14:47:33 fronga Exp $
 *
 */

#include "DataFormats/Common/interface/RefToBase.h"
#include <vector>


namespace pat {

  template <class ObjectType>
  class PATObject : public ObjectType {
    public:

      /// default constructor
      PATObject();
      /// constructor from a base object (leaves invalid reference to original object!)
      PATObject(const ObjectType & obj);
      /// constructor from reference
      PATObject(const edm::RefToBase<ObjectType> & ref);
      /// destructor
      virtual ~PATObject() {}
    /// returns a clone
      virtual PATObject<ObjectType> * clone() const ; 
      /// access to the original object; returns zero for null Ref and throws for unavailable collection
      const ObjectType * originalObject() const;
      /// reference to original object. Returns a null reference if not available
      const edm::RefToBase<ObjectType> & originalObjectRef() const;
      /// standard deviation on A (see CMS Note 2006/023)
      float resolutionA() const;
      /// standard deviation on B (see CMS Note 2006/023)
      float resolutionB() const;
      /// standard deviation on C (see CMS Note 2006/023)
      float resolutionC() const;
      /// standard deviation on D (see CMS Note 2006/023)
      float resolutionD() const;
      /// standard deviation on transverse energy
      float resolutionEt() const;
      /// standard deviation on pseudorapidity
      float resolutionEta() const;
      /// standard deviation on azimuthal angle
      float resolutionPhi() const;
      /// standard deviation on polar angle
      float resolutionTheta() const;
      /// covariance matrix elements
      const std::vector<float> & covMatrix() const;
      /// set standard deviation on A (see CMS Note 2006/023)
      void setResolutionA(float a);
      /// set standard deviation on B (see CMS Note 2006/023)
      void setResolutionB(float b);
      /// set standard deviation on C (see CMS Note 2006/023)
      void setResolutionC(float c);
      /// set standard deviation on D (see CMS Note 2006/023)
      void setResolutionD(float d);
      /// set standard deviation on transverse energy
      void setResolutionEt(float et);
      /// set standard deviation on pseudorapidity
      void setResolutionEta(float eta);
      /// set standard deviation on azimuthal angle
      void setResolutionPhi(float phi);
      /// set standard deviation on polar angle
      void setResolutionTheta(float theta);
      /// set covariance matrix elements
      void setCovMatrix(const std::vector<float> & c);
      
    protected:
      // reference back to the original object
      edm::RefToBase<ObjectType> refToOrig_;
      /// standard deviation on transverse energy
      float resEt_;
      /// standard deviation on pseudorapidity
      float resEta_;
      /// standard deviation on azimuthal angle
      float resPhi_;
      /// standard deviation on A (see CMS Note 2006/023)
      float resA_;
      /// standard deviation on B (see CMS Note 2006/023)
      float resB_;
      /// standard deviation on C (see CMS Note 2006/023)
      float resC_;
      /// standard deviation on D (see CMS Note 2006/023)
      float resD_;
      /// standard deviation on polar angle
      float resTheta_;
      // covariance matrix elements
      std::vector<float> covM_;

  };


  template <class ObjectType> PATObject<ObjectType>::PATObject() :
    resEt_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0), resTheta_(0) {
  }

  template <class ObjectType> PATObject<ObjectType>::PATObject(const ObjectType & obj) :
    ObjectType(obj),
    refToOrig_(),
    resEt_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0),  resTheta_(0) {
  }

  template <class ObjectType> PATObject<ObjectType>::PATObject(const edm::RefToBase<ObjectType> & ref) :
    ObjectType(*ref),
    refToOrig_(ref),
    resEt_(0), resEta_(0), resPhi_(0), resA_(0), resB_(0), resC_(0), resD_(0),  resTheta_(0) {
  }

  template <class ObjectType> const ObjectType * PATObject<ObjectType>::originalObject() const {
    if (refToOrig_.isNull()) {
      // this object was not produced from a reference, so no link to the
      // original object exists -> return a 0-pointer
      return 0;
    // GIO: temporary disable the following: I have to find out why there is no "isAvailable()" in RefToBase...
    //     vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    //} else if (!refToOrig_.isAvailable()) {
    //  throw edm::Exception(edm::errors::ProductNotFound) << "The original collection from which this PAT object was made is not present any more in the event, hence you cannot access the originating object anymore.";
    //  return 0;
    //      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    } else {
      return refToOrig_.get();
    }
  }


  template <class ObjectType> 
  PATObject<ObjectType> * PATObject<ObjectType>::clone() const {
    return new PATObject<ObjectType> ( *this );
  }

  template <class ObjectType> 
  const edm::RefToBase<ObjectType> & PATObject<ObjectType>::originalObjectRef() const { return refToOrig_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionEt() const { return resEt_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionEta() const { return resEta_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionPhi() const { return resPhi_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionA() const { return resA_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionB() const { return resB_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionC() const { return resC_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionD() const { return resD_; }

  template <class ObjectType> 
  float PATObject<ObjectType>::resolutionTheta() const { return resTheta_; }

  template <class ObjectType> 
  const std::vector<float> & PATObject<ObjectType>::covMatrix() const { return covM_; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionEt(float et) { resEt_ = et; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionEta(float eta) { resEta_ = eta; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionPhi(float phi) { resPhi_ = phi; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionA(float a) { resA_ = a; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionB(float b) { resB_ = b; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionC(float c) { resC_ = c; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionD(float d) { resD_ = d; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setResolutionTheta(float theta) { resTheta_ = theta; }

  template <class ObjectType> 
  void PATObject<ObjectType>::setCovMatrix(const std::vector<float> & c) {
    //    covM_.clear();
    //    for (size_t i = 0; i < c.size(); i++) covM_.push_back(c[i]); 
    covM_ = c;
  }

}

#endif
