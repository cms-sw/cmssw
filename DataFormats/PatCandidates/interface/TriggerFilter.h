#ifndef DataFormats_PatCandidates_TriggerFilter_h
#define DataFormats_PatCandidates_TriggerFilter_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerFilter
//
// $Id: TriggerFilter.h,v 1.10 2013/06/11 13:24:49 vadler Exp $
//
/**
  \class    pat::TriggerFilter TriggerFilter.h "DataFormats/PatCandidates/interface/TriggerFilter.h"
  \brief    Analysis-level HLTrigger filter class

   TriggerFilter implements a container for trigger filters' information within the 'pat' namespace.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerFilter

  \author   Volker Adler
  \version  $Id: TriggerFilter.h,v 1.10 2013/06/11 13:24:49 vadler Exp $
*/


#include <string>
#include <vector>

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

namespace pat {

  class TriggerFilter {

      /// Data Members

      /// Label of the filter
      std::string label_;
      /// CMSSW module type
      std::string type_;
      /// Indeces of trigger objects in pat::TriggerObjectCollection in event
      /// as produced together with the pat::TriggerFilterCollection
      std::vector< unsigned > objectKeys_;
      /// List of (unique) special identifiers for the used trigger object types as defined in
      /// trigger::TriggerObjectType (DataFormats/HLTReco/interface/TriggerTypeDefs.h),
      /// possibly empty or containing also zeroes
      std::vector< trigger::TriggerObjectType > triggerObjectTypes_;
      /// Indicator for filter status: -1: not run, 0: failed, 1: succeeded
      int status_;
      /// Indicator for being an L3 filter
      /// available starting from CMSSW_4_2_3
      bool saveTags_;

    public:

      /// Constructors and Desctructor

      /// Default constructor
      TriggerFilter();
      /// Constructor from std::string for filter label
      TriggerFilter( const std::string & label, int status = -1, bool saveTags = false );
      /// Constructor from edm::InputTag for filter label
      TriggerFilter( const edm::InputTag & tag, int status = -1, bool saveTags = false );

      /// Destructor
      virtual ~TriggerFilter() {};

      /// Methods

      /// Set the filter label
      void setLabel( const std::string & label ) { label_ = label; };
      /// Set the filter module type
      void setType( const std::string & type ) { type_  = type; };
      /// Add a new trigger object collection index
      void addObjectKey( unsigned objectKey ) { if ( ! hasObjectKey( objectKey ) ) objectKeys_.push_back( objectKey ); };
      /// Add a new trigger object type identifier
      void addTriggerObjectType( trigger::TriggerObjectType triggerObjectType ) { if ( ! hasTriggerObjectType( triggerObjectType ) ) triggerObjectTypes_.push_back( triggerObjectType ); };
      void addTriggerObjectType( int triggerObjectType )                        { addTriggerObjectType( trigger::TriggerObjectType( triggerObjectType ) ); };
      void addObjectId( trigger::TriggerObjectType triggerObjectType ) { addTriggerObjectType( triggerObjectType ); };                               // for backward compatibility
      void addObjectId( int triggerObjectType )                        { addTriggerObjectType( trigger::TriggerObjectType( triggerObjectType ) ); }; // for backward compatibility
      /// Set the filter status,
      /// only -1,0,1 accepted; returns 'false' (and does not modify the status) otherwise
      bool setStatus( int status );
      /// Set the L3 status
      void setSaveTags( bool saveTags ) { saveTags_ = saveTags; };
      /// Get the filter label
      const std::string & label() const { return label_; };
      /// Get the filter module type
      const std::string & type() const { return type_; };
      /// Get all trigger object collection indeces
      const std::vector< unsigned > & objectKeys() const { return objectKeys_; };
      /// Get all trigger object type identifiers
//       std::vector< trigger::TriggerObjectType > triggerObjectTypes() const { return triggerObjectTypes_; };
//       std::vector< trigger::TriggerObjectType > objectIds()          const { return triggerObjectTypes(); }; // for backward compatibility
      std::vector< int > triggerObjectTypes() const;  // for backward compatibilit
      std::vector< int > objectIds()          const { return triggerObjectTypes(); }; // for double backward compatibility
      /// Get the filter status
      int status() const { return status_; };
      /// Get the L3 status
      bool saveTags() const { return saveTags_; };
      bool isL3() const { return saveTags(); };
      bool isFiring() const { return ( saveTags() && status() == 1 ); };
      /// Checks, if a certain trigger object collection index is assigned
      bool hasObjectKey( unsigned objectKey ) const;
      /// Checks, if a certain trigger object type identifier is assigned
      bool hasTriggerObjectType( trigger::TriggerObjectType triggerObjectType ) const;
      bool hasTriggerObjectType( int triggerObjectType )                        const { return hasTriggerObjectType( trigger::TriggerObjectType( triggerObjectType ) ); };
      bool hasObjectId( trigger::TriggerObjectType triggerObjectType ) const { return hasTriggerObjectType( triggerObjectType ); };                               // for backward compatibility
      bool hasObjectId( int triggerObjectType )                        const { return hasTriggerObjectType( trigger::TriggerObjectType( triggerObjectType ) ); }; // for backward compatibility

  };


  /// Collection of TriggerFilter
  typedef std::vector< TriggerFilter >                      TriggerFilterCollection;
  /// Persistent reference to an item in a TriggerFilterCollection
  typedef edm::Ref< TriggerFilterCollection >               TriggerFilterRef;
  /// Persistent reference to a TriggerFilterCollection product
  typedef edm::RefProd< TriggerFilterCollection >           TriggerFilterRefProd;
  /// Vector of persistent references to items in the same TriggerFilterCollection
  typedef edm::RefVector< TriggerFilterCollection >         TriggerFilterRefVector;
  /// Const iterator over vector of persistent references to items in the same TriggerFilterCollection
  typedef edm::RefVectorIterator< TriggerFilterCollection > TriggerFilterRefVectorIterator;

}


#endif
