#ifndef DataFormats_PatCandidates_TriggerPrimitive_h
#define DataFormats_PatCandidates_TriggerPrimitive_h


//
// $Id: TriggerPrimitive.h,v 1.1.4.2 2008/05/30 12:23:59 vadler Exp $
//


/**
  \class    pat::TriggerPrimitive TriggerPrimitive.h "DataFormats/PatCandidates/interface/TriggerPrimitive.h"
  \brief    Analysis-level trigger primitive class

   TriggerPrimitive implements a container for trigger primitives' information within the 'pat' namespace.
   It inherits from LeafCandidate and adds the following data members:
   - std::string filterName_        (name of the trigger filter the TriggerPrimitive was used in)
   - int         triggerObjectType_ (trigger object type as defined in DataFormats/HLTReco/interface/TriggerTypeDefs.h)
   In addition, the data member reco::Particle::pdgId_ (inherited via reco::LeafCandidate) is used
   to store the trigger object id from trigger::TriggerObject::id_.

  \author   Volker Adler
  \version  $Id: TriggerPrimitive.h,v 1.1.4.2 2008/05/30 12:23:59 vadler Exp $
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
      TriggerPrimitive( const reco::Particle::LorentzVector & aVec, const std::string aFilt = "", const int aType = 0, const int id = 0 );
      TriggerPrimitive( const reco::Particle::PolarLorentzVector & aVec, const std::string aFilt = "", const int aType = 0, const int id = 0 );
      virtual ~TriggerPrimitive();
      
      virtual TriggerPrimitive * clone() const;
      
      const std::string & filterName() const;
      const int           triggerObjectType() const;
      const int           triggerObjectId() const;
      
      void setFilterName( const std::string aFilt );
      void setTriggerObjectType( const int aType );
      void setTriggerObjectId( const int id );
      
    protected:
    
      std::string filterName_;
      int         triggerObjectType_;

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
