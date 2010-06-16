#ifndef DataFormats_PatCandidates_TriggerFilter_h
#define DataFormats_PatCandidates_TriggerFilter_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerFilter
//
// $Id: TriggerFilter.h,v 1.5 2010/04/20 21:39:46 vadler Exp $
//
/**
  \class    pat::TriggerFilter TriggerFilter.h "DataFormats/PatCandidates/interface/TriggerFilter.h"
  \brief    Analysis-level HLTrigger filter class

   TriggerFilter implements a container for trigger filters' information within the 'pat' namespace.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerFilter

  \author   Volker Adler
  \version  $Id: TriggerFilter.h,v 1.5 2010/04/20 21:39:46 vadler Exp $
*/


#include <string>
#include <vector>

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"

namespace pat {

  class TriggerFilter {

      /// data members
      std::string             label_;
      std::string             type_;
      std::vector< unsigned > objectKeys_;
      std::vector< int >      objectIds_; // special filter related object ID as defined in enum 'TriggerObjectType' in DataFormats/HLTReco/interface/TriggerTypeDefs.h
      int                     status_;    // -1: not run, 0: failed, 1: succeeded

    public:

      /// constructors and desctructor
      TriggerFilter();
      TriggerFilter( const std::string & label, int status = -1 );
      TriggerFilter( const edm::InputTag & tag, int status = -1 );
      virtual ~TriggerFilter() {};

      /// setters & getters
      void setLabel( const std::string & label ) { label_ = label; };
      void setType( const std::string & type )   { type_  = type; };
      void addObjectKey( unsigned objectKey )    { if ( ! hasObjectKey( objectKey ) ) objectKeys_.push_back( objectKey ); };
      void addObjectId( int objectId )           { if ( ! hasObjectId( objectId ) )   objectIds_.push_back( objectId ); };
      bool setStatus( int status ); // only -1,0,1 accepted; returns 'false' (and does not modify the status) otherwise
      std::string             label() const      { return label_; };
      std::string             type() const       { return type_; };
      std::vector< unsigned > objectKeys() const { return objectKeys_; };
      std::vector< int >      objectIds() const  { return objectIds_; };
      int                     status() const     { return status_; };
      bool                    hasObjectKey( unsigned objectKey ) const;
      bool                    hasObjectId( int objectId ) const;

  };


  /// collection of TriggerFilter
  typedef std::vector< TriggerFilter >                      TriggerFilterCollection;
  /// persistent reference to an item in a TriggerFilterCollection
  typedef edm::Ref< TriggerFilterCollection >               TriggerFilterRef;
  /// persistent reference to a TriggerFilterCollection product
  typedef edm::RefProd< TriggerFilterCollection >           TriggerFilterRefProd;
  /// vector of persistent references to items in the same TriggerFilterCollection
  typedef edm::RefVector< TriggerFilterCollection >         TriggerFilterRefVector;
  /// const iterator over vector of persistent references to items in the same TriggerFilterCollection
  typedef edm::RefVectorIterator< TriggerFilterCollection > TriggerFilterRefVectorIterator;

}


#endif
