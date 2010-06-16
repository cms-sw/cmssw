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
  \version  $Id: PATTriggerMatchSelector.h,v 1.4 2009/12/10 10:44:37 vadler Exp $
*/
//
// $Id: PATTriggerMatchSelector.h,v 1.4 2009/12/10 10:44:37 vadler Exp $
//


#include <string>
#include <vector>
#include <map>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/InputTag.h"


namespace pat {

  template< typename T1, typename T2 >
  class PATTriggerMatchSelector {

      bool                       andOr_;          // AND used if 'false', OR otherwise
      std::vector< int >         filterIds_;      // special filter related ID as defined in enum 'TriggerObjectType' in DataFormats/HLTReco/interface/TriggerTypeDefs.h
      std::vector< std::string > filterIdsEnum_;  // special filter related ID as defined in enum 'TriggerObjectType' in DataFormats/HLTReco/interface/TriggerTypeDefs.h
      std::vector< std::string > filterLabels_;
      std::vector< std::string > pathNames_;
      bool                       pathLastFilterAcceptedOnly_;
      std::vector< std::string > collectionTags_; // full tag strings (as from edm::InputTag::encode()) recommended, but also only labels allowed

    public:

      PATTriggerMatchSelector( const edm::ParameterSet & iConfig ) :
        andOr_( iConfig.getParameter< bool >( "andOr" ) ),
        filterIds_( iConfig.getParameter< std::vector< int > >( "filterIds" ) ),
        filterIdsEnum_( iConfig.getParameter< std::vector< std::string > >( "filterIdsEnum" ) ),
        filterLabels_( iConfig.getParameter< std::vector< std::string > >( "filterLabels" ) ),
        pathNames_( iConfig.getParameter< std::vector< std::string > >( "pathNames" ) ),
        pathLastFilterAcceptedOnly_( true ),
        collectionTags_( iConfig.getParameter< std::vector< std::string > >( "collectionTags" ) )
      {
        if ( iConfig.exists( "pathLastFilterAcceptedOnly" ) ) pathLastFilterAcceptedOnly_ = iConfig.getParameter< bool >( "pathLastFilterAcceptedOnly" );
      }

      bool operator()( const T1 & patObj, const T2 & trigObj ) const {
        std::map< std::string, trigger::TriggerObjectType > filterIdsEnumMap; // FIXME: Should be automated, but  h o w ?
        // L1
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1Mu"          , trigger::TriggerL1Mu ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1NoIsoEG"     , trigger::TriggerL1NoIsoEG ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1IsoEG"       , trigger::TriggerL1IsoEG ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1CenJet"      , trigger::TriggerL1CenJet ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1ForJet"      , trigger::TriggerL1ForJet ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1TauJet"      , trigger::TriggerL1TauJet ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1ETM"         , trigger::TriggerL1ETM ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1ETT"         , trigger::TriggerL1ETT ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1HTT"         , trigger::TriggerL1HTT ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1HTM"         , trigger::TriggerL1HTM ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1JetCounts"   , trigger::TriggerL1JetCounts ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1HfBitCounts" , trigger::TriggerL1HfBitCounts ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1HfRingEtSums", trigger::TriggerL1HfRingEtSums ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1TechTrig"    , trigger::TriggerL1TechTrig ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1Castor"      , trigger::TriggerL1Castor ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1BPTX"        , trigger::TriggerL1BPTX ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerL1GtExternal"  , trigger::TriggerL1GtExternal ) );
        // HLT
        filterIdsEnumMap.insert( std::make_pair( "TriggerPhoton"  , trigger::TriggerPhoton ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerElectron", trigger::TriggerElectron ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerMuon"    , trigger::TriggerMuon ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerTau"     , trigger::TriggerTau ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerJet"     , trigger::TriggerJet ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerBJet"    , trigger::TriggerBJet ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerMET"     , trigger::TriggerMET ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerTET"     , trigger::TriggerTET ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerTHT"     , trigger::TriggerTHT ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerMHT"     , trigger::TriggerMHT ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerTrack"   , trigger::TriggerTrack ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerCluster" , trigger::TriggerCluster ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerMETSig"  , trigger::TriggerMETSig ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerELongit" , trigger::TriggerELongit ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerMHTSig"  , trigger::TriggerMHTSig ) );
        filterIdsEnumMap.insert( std::make_pair( "TriggerHLongit" , trigger::TriggerHLongit ) );
        if ( andOr_ ) { // OR
          for ( size_t i = 0; i < filterIds_.size(); ++i ) {
            if ( filterIds_.at( i ) == 0 || trigObj.hasFilterId( filterIds_.at( i ) ) ) return true;
          }
          for ( size_t j = 0; j < filterLabels_.size(); ++j ) {
            if ( filterLabels_.at( j ) == "*" || filterLabels_.at( j ) == "@" || trigObj.hasFilterLabel( filterLabels_.at( j ) ) ) return true;
          }
          for ( size_t k = 0; k < pathNames_.size(); ++k ) {
            if ( pathNames_.at( k ) == "*" || pathNames_.at( k ) == "@" || trigObj.hasPathName( pathNames_.at( k ), pathLastFilterAcceptedOnly_ ) ) return true;
          }
          for ( size_t l = 0; l < collectionTags_.size(); ++l ) {
            if ( collectionTags_.at( l ) == "*" || collectionTags_.at( l ) == "@" || trigObj.hasCollection( collectionTags_.at( l ) ) ) return true;
          }
          for ( size_t m = 0; m < filterIdsEnum_.size(); ++m ) {
            if ( filterIdsEnum_.at( m ) == "*" || filterIdsEnum_.at( m ) == "@" ) return true;
            std::map< std::string, trigger::TriggerObjectType >::const_iterator iter( filterIdsEnumMap.find( filterIdsEnum_.at( m ) ) );
            if ( iter != filterIdsEnumMap.end() && trigObj.hasFilterId( iter->second ) ) return true;
          }
          return false;
        } else { // AND
          for ( size_t i = 0; i < filterIds_.size(); ++i ) {
            if ( filterIds_.at( i ) == 0 || trigObj.hasFilterId( filterIds_.at( i ) ) ) {
              for ( size_t j = 0; j < filterLabels_.size(); ++j ) {
                if ( filterLabels_.at( j ) == "*" || filterLabels_.at( j ) == "@" || trigObj.hasFilterLabel( filterLabels_.at( j ) ) ) {
                  for ( size_t k = 0; k < pathNames_.size(); ++k ) {
                    if ( pathNames_.at( k ) == "*" || pathNames_.at( k ) == "@" || trigObj.hasPathName( pathNames_.at( k ), pathLastFilterAcceptedOnly_ ) ) {
                      for ( size_t l = 0; l < collectionTags_.size(); ++l ) {
                        if ( collectionTags_.at( l ) == "*" || collectionTags_.at( l ) == "@" || trigObj.hasCollection( collectionTags_.at( l ) ) ) {
                          for ( size_t m = 0; m < filterIdsEnum_.size(); ++m ) {
                            if ( filterIdsEnum_.at( m ) == "*" || filterIdsEnum_.at( m ) == "@" ) return true;
                            std::map< std::string, trigger::TriggerObjectType >::const_iterator iter( filterIdsEnumMap.find( filterIdsEnum_.at( m ) ) );
                            if ( iter != filterIdsEnumMap.end() && trigObj.hasFilterId( iter->second ) ) return true;
                          }
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
