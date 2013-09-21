#ifndef DataFormats_PatCandidates_TriggerCondition_h
#define DataFormats_PatCandidates_TriggerCondition_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerCondition
//
// $Id: TriggerCondition.h,v 1.2 2013/06/11 13:24:49 vadler Exp $
//
/**
  \class    pat::TriggerCondition TriggerCondition.h "DataFormats/PatCandidates/interface/TriggerCondition.h"
  \brief    Analysis-level L1 trigger condition class

   TriggerCondition implements a container for trigger conditions' information within the 'pat' namespace.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerCondition

  \author   Volker Adler
  \version  $Id: TriggerCondition.h,v 1.2 2013/06/11 13:24:49 vadler Exp $
*/


#include <string>
#include <vector>

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

namespace pat {

  class TriggerCondition {

      /// Data Members

      /// Name of the condition
      std::string name_;
      /// Did condition succeed?
      bool accept_;
      /// L1 condition category as defined in CondFormats/L1TObjects/interface/L1GtFwd.h
      L1GtConditionCategory category_;
      /// L1 condition type as defined in CondFormats/L1TObjects/interface/L1GtFwd.h
      L1GtConditionType type_;
      /// Special identifiers for the used trigger object type as defined in
      /// DataFormats/HLTReco/interface/TriggerTypeDefs.h
      /// translated from L1GtObject type (DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h)
      std::vector< trigger::TriggerObjectType > triggerObjectTypes_;
      /// Indeces of trigger objects from succeeding combinations in pat::TriggerObjectCollection in event
      /// as produced together with the pat::TriggerAlgorithmCollection
      std::vector< unsigned > objectKeys_;

    public:

      /// Constructors and Desctructor

      /// Default constructor
      TriggerCondition();
      /// Constructor from condition name "only"
      TriggerCondition( const std::string & name );
      /// Constructor from values
      TriggerCondition( const std::string & name, bool accept );

      /// Destructor
      virtual ~TriggerCondition() {};

      /// Methods

      /// Set the condition name
      void setName( const std::string & name ) { name_ = name; };
      /// Set the success flag
      void setAccept( bool accept ) { accept_ = accept; };
      /// Set the condition category
      void setCategory( L1GtConditionCategory category ) { category_ = category; };
      void setCategory( int category )                   { category_ = L1GtConditionCategory( category ); };
      /// Set the condition type
      void setType( L1GtConditionType type ) { type_ = type; };
      void setType( int type )               { type_ = L1GtConditionType( type ); };
      /// Add a new trigger object type
      void addTriggerObjectType( trigger::TriggerObjectType triggerObjectType ) { triggerObjectTypes_.push_back( triggerObjectType ); }; // explicitely NOT checking for existence
      void addTriggerObjectType( int triggerObjectType )                        { addTriggerObjectType( trigger::TriggerObjectType( triggerObjectType ) ); };
      /// Add a new trigger object collection index
      void addObjectKey( unsigned objectKey ) { if ( ! hasObjectKey( objectKey ) ) objectKeys_.push_back( objectKey ); };
      /// Get the filter label
      const std::string & name() const { return name_; };
      /// Get the success flag
      bool wasAccept() const { return accept_; };
      /// Get the condition category
      int category() const { return int( category_ ); };
      /// Get the condition type
      int type() const { return int( type_ ); };
      /// Get the trigger object types
      std::vector< int > triggerObjectTypes() const;
      /// Checks, if a certain trigger object type is assigned
      bool hasTriggerObjectType( trigger::TriggerObjectType triggerObjectType ) const;
      bool hasTriggerObjectType( int triggerObjectType ) const { return hasTriggerObjectType( trigger::TriggerObjectType( triggerObjectType ) ); };
      /// Get all trigger object collection indeces
      const std::vector< unsigned > & objectKeys() const { return objectKeys_; };
      /// Checks, if a certain trigger object collection index is assigned
      bool hasObjectKey( unsigned objectKey ) const;

  };


  /// Collection of TriggerCondition
  typedef std::vector< TriggerCondition >                      TriggerConditionCollection;
  /// Persistent reference to an item in a TriggerConditionCollection
  typedef edm::Ref< TriggerConditionCollection >               TriggerConditionRef;
  /// Persistent reference to a TriggerConditionCollection product
  typedef edm::RefProd< TriggerConditionCollection >           TriggerConditionRefProd;
  /// Vector of persistent references to items in the same TriggerConditionCollection
  typedef edm::RefVector< TriggerConditionCollection >         TriggerConditionRefVector;
  /// Const iterator over vector of persistent references to items in the same TriggerConditionCollection
  typedef edm::RefVectorIterator< TriggerConditionCollection > TriggerConditionRefVectorIterator;

}


#endif
