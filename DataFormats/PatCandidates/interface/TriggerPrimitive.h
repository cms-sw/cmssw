#ifndef DataFormats_PatCandidates_TriggerPrimitive_h
#define DataFormats_PatCandidates_TriggerPrimitive_h


//
// $Id: TriggerPrimitive.h,v 1.1.2.1 2008/04/08 09:40:53 vadler Exp $
//


/**
  \class    pat::TriggerPrimitive TriggerPrimitive.h "DataFormats/PatCandidates/interface/TriggerPrimitive.h"
  \brief    Analysis-level trigger primitive class

   TriggerPrimitive implements a container for trigger primitives' information within the 'pat' namespace.
   It inherits from LeafCandidate and adds trigger and filter names.

  \author   Volker Adler
  \version  $Id: TriggerPrimitive.h,v 1.1.2.1 2008/04/08 09:40:53 vadler Exp $
*/


#include <string>
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Association.h"


namespace pat {

  class TriggerPrimitive : public reco::LeafCandidate {

    public:

      TriggerPrimitive();
      TriggerPrimitive( const pat::TriggerPrimitive & aTrigPrim );
      TriggerPrimitive( const reco::Particle::LorentzVector & aVec, const std::string aTrig = "", const std::string aFilt = "" );
      TriggerPrimitive( const reco::Particle::PolarLorentzVector & aVec, const std::string aTrig = "", const std::string aFilt = "" );
      virtual ~TriggerPrimitive();
      
      virtual TriggerPrimitive * clone() const;
      
      const std::string & triggerName() const;
      const std::string & pathName() const;
      const std::string & filterName() const;
      const std::string & objectType() const;
      
      void setTriggerName( const std::string aTrig = "" );
      void setPathName( const std::string aPath = "" );
      void setFilterName( const std::string aFilt = "" );
      void setObjectType( const std::string anObj = "" );
      
    protected:
    
      std::string triggerName_;
      std::string filterName_;

  };
  

  /// collection of TriggerPrimitive
  typedef edm::OwnVector<TriggerPrimitive> TriggerPrimitiveCollection;
  /// persistent reference to a TriggerPrimitive
  typedef edm::Ref<TriggerPrimitiveCollection> TriggerPrimitiveRef;
  /// persistent reference to a TriggerPrimitiveCollection
  typedef edm::RefProd<TriggerPrimitiveCollection> TriggerPrimitiveRefProd;
  /// vector of reference to TriggerPrimitive in the same collection
  typedef edm::RefVector<TriggerPrimitiveCollection> TriggerPrimitiveRefVector;
  /// vector of reference to TriggerPrimitive in the same collection
  typedef edm::Association<TriggerPrimitiveCollection> TriggerPrimitiveMatch;

}


#endif
