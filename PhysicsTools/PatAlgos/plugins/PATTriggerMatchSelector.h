#ifndef PhysicsTools_PatAlgos_PATTriggerMatchSelector_h
#define PhysicsTools_PatAlgos_PATTriggerMatchSelector_h


// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      PATTriggerMatchSelector
//
/**
  \class    pat::PATTriggerMatchSelector PATTriggerMatchSelector.h "PhysicsTools/PatAlgos/plugins/PATTriggerMatchSelector.h"
  \brief    

   .

  \author   Volker Adler
  \version  $Id: PATTriggerMatchSelector.h,v 1.1.2.2 2009/03/16 20:10:11 vadler Exp $
*/
//
// $Id: PATTriggerMatchSelector.h,v 1.1.2.2 2009/03/16 20:10:11 vadler Exp $
//


#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace pat {

  template< typename T1, typename T2 >
  class PATTriggerMatchSelector {
    
      bool                       andOr_;          // AND used if 'false', OR otherwise
      std::vector< unsigned >    filterIds_;      // special filter related ID as defined in enum 'TriggerObjectType' in DataFormats/HLTReco/interface/TriggerTypeDefs.h
      std::vector< std::string > filterLabels_;
      std::vector< std::string > pathNames_;
      std::vector< std::string > collectionTags_; // needs full tag strings (as from edm::InputTag::encode()), not only labels
  
    public:
    
      PATTriggerMatchSelector( const edm::ParameterSet & iConfig ) :
        andOr_( iConfig.getParameter< bool >( "andOr" ) ),
        filterIds_( iConfig.getParameter< std::vector< unsigned > >( "filterIds" ) ),
        filterLabels_( iConfig.getParameter< std::vector< std::string > >( "filterLabels" ) ),
        pathNames_( iConfig.getParameter< std::vector< std::string > >( "pathNames" ) ),
        collectionTags_( iConfig.getParameter< std::vector< std::string > >( "collectionTags" ) )
      {
      }
      
      bool operator()( const T1 & patObj, const T2 & trigObj ) const {
        if ( andOr_ ) { // OR
          for ( size_t i = 0; i < filterIds_.size(); ++i ) {
            if ( filterIds_.at( i ) == 0 || trigObj.hasFilterId( filterIds_.at( i ) ) ) return true;
          }
          for ( size_t j = 0; j < filterLabels_.size(); ++j ) {
            if ( filterLabels_.at( j ) == "*" || filterLabels_.at( j ) == "@" || trigObj.hasFilterLabel( filterLabels_.at( j ) ) ) return true;
          }
          for ( size_t k = 0; k < pathNames_.size(); ++k ) {
            if ( pathNames_.at( k ) == "*" || pathNames_.at( k ) == "@" || trigObj.hasPathName( pathNames_.at( k ) ) ) return true;
          }
          for ( size_t l = 0; l < collectionTags_.size(); ++l ) {
            if ( collectionTags_.at( l ) == "*" || collectionTags_.at( l ) == "@" || collectionTags_.at( l ) == trigObj.collection() ) return true;
          }
          return false;
        } else { // AND
          for ( size_t i = 0; i < filterIds_.size(); ++i ) {
            if ( filterIds_.at( i ) == 0 || trigObj.hasFilterId( filterIds_.at( i ) ) ) {
              for ( size_t j = 0; j < filterLabels_.size(); ++j ) {
                if ( filterLabels_.at( j ) == "*" || filterLabels_.at( j ) == "@" || trigObj.hasFilterLabel( filterLabels_.at( j ) ) ) {
                  for ( size_t k = 0; k < pathNames_.size(); ++k ) {
                    if ( pathNames_.at( k ) == "*" || pathNames_.at( k ) == "@" || trigObj.hasPathName( pathNames_.at( k ) ) ) {
                      for ( size_t l = 0; l < collectionTags_.size(); ++l ) {
                        if ( collectionTags_.at( l ) == "*" || collectionTags_.at( l ) == "@" || collectionTags_.at( l ) == trigObj.collection() ) {
                          return true;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          return false;
        }
        return false;
      }
      
  };
  
}


#endif
