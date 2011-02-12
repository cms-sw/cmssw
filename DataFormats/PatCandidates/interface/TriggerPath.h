#ifndef DataFormats_PatCandidates_TriggerPath_h
#define DataFormats_PatCandidates_TriggerPath_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerPath
//
// $Id: TriggerPath.h,v 1.6 2010/06/16 15:40:52 vadler Exp $
//
/**
  \class    pat::TriggerPath TriggerPath.h "DataFormats/PatCandidates/interface/TriggerPath.h"
  \brief    Analysis-level HLTrigger path class

   TriggerPath implements a container for trigger paths' information within the 'pat' namespace.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerPath

  \author   Volker Adler
  \version  $Id: TriggerPath.h,v 1.6 2010/06/16 15:40:52 vadler Exp $
*/


#include <string>
#include <vector>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"


namespace pat {

  typedef std::pair< bool, std::string > L1Seed;
  typedef std::vector< L1Seed >          L1SeedCollection;

  class TriggerPath {

      /// data members
      std::string                name_;
      unsigned                   index_;                // in trigger table
      unsigned                   prescale_;
      bool                       run_;
      bool                       accept_;
      bool                       error_;
      std::vector< std::string > modules_;              // filled in correct order by PATTriggerProducer; not necessarily in filter collection; consumes disc space
      std::vector< unsigned >    filterIndices_;        // also filled in correct order by PATTriggerProducer; indices of active filters in filter collection; needed, if 'modules_' kept empty
      unsigned                   lastActiveFilterSlot_; // index as from modules_
      L1SeedCollection           l1Seeds_;

    public:

      /// constructors and desctructor
      TriggerPath();
      TriggerPath( const std::string & name );
      TriggerPath( const std::string & name, unsigned index, unsigned prescale, bool run, bool accept, bool error, unsigned lastActiveFilterSlot );
      virtual ~TriggerPath() {};

      /// setters & getters
      void setName( const std::string & name )                        { name_                 = name; };
      void setIndex( unsigned index )                                 { index_                = index; };
      void setPrescale( unsigned prescale )                           { prescale_             = prescale; };
      void setRun( bool run )                                         { run_                  = run; };
      void setAccept( bool accept )                                   { accept_               = accept; };
      void setError( bool error )                                     { error_                = error; };
      void setLastActiveFilterSlot( unsigned lastActiveFilterSlot )   { lastActiveFilterSlot_ = lastActiveFilterSlot; };
      void addModule( const std::string & name )                      { modules_.push_back( name ); };
      void addFilterIndex( const unsigned index )                     { filterIndices_.push_back( index ); };
      void addL1Seed( const L1Seed & seed )                           { l1Seeds_.push_back( seed ); };
      void addL1Seed( bool decision, const std::string & expression ) { l1Seeds_.push_back( L1Seed( decision, expression ) ); };
      std::string                name() const                 { return name_; };
      unsigned                   index() const                { return index_; };
      unsigned                   prescale() const             { return prescale_; };
      bool                       wasRun() const               { return run_; };
      bool                       wasAccept() const            { return accept_; };
      bool                       wasError() const             { return error_; };
      unsigned                   lastActiveFilterSlot() const { return lastActiveFilterSlot_; };
      std::vector< std::string > modules() const              { return modules_; };
      std::vector< unsigned >    filterIndices() const        { return filterIndices_; };
      int                        indexModule( const std::string & name ) const;           // returns size of 'modules_' ( modules().size() ) if name unknown and -1 if 'modules_' not filled
      L1SeedCollection           l1Seeds() const              { return l1Seeds_; };
      std::vector< std::string > l1Seeds( const bool decision ) const;
      std::vector< std::string > acceptedL1Seeds() const      { return l1Seeds( true ); };
      std::vector< std::string > failedL1Seeds() const        { return l1Seeds( false ); };

  };


  /// collection of TriggerPath
  typedef std::vector< TriggerPath >                      TriggerPathCollection;
  /// persistent reference to an item in a TriggerPathCollection
  typedef edm::Ref< TriggerPathCollection >               TriggerPathRef;
  /// persistent reference to a TriggerPathCollection product
  typedef edm::RefProd< TriggerPathCollection >           TriggerPathRefProd;
  /// vector of persistent references to items in the same TriggerPathCollection
  typedef edm::RefVector< TriggerPathCollection >         TriggerPathRefVector;
  /// const iterator over vector of persistent references to items in the same TriggerPathCollection
  typedef edm::RefVectorIterator< TriggerPathCollection > TriggerPathRefVectorIterator;

}


#endif
