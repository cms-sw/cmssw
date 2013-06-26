#ifndef GenericTriggerEventFlag_H
#define GenericTriggerEventFlag_H


// -*- C++ -*-
//
// Package:    CommonTools/TriggerUtils
// Class:      GenericTriggerEventFlag
//
// $Id: GenericTriggerEventFlag.h,v 1.6 2012/01/20 18:18:11 vadler Exp $
//
/**
  \class    GenericTriggerEventFlag GenericTriggerEventFlag.h "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
  \brief    Provides a code based selection for trigger and DCS information in order to have no failing filters in the CMSSW path.

   [...]

  \author   Volker Adler
  \version  $Id: GenericTriggerEventFlag.h,v 1.6 2012/01/20 18:18:11 vadler Exp $
*/


#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <string>


class GenericTriggerEventFlag {

    // Utility classes
    edm::ESWatcher< AlCaRecoTriggerBitsRcd > * watchDB_;
    L1GtUtils                                  l1Gt_;
    HLTConfigProvider                          hltConfig_;
    bool                                       hltConfigInit_;
    // Configuration parameters
    bool        andOr_;
    std::string dbLabel_;
    unsigned    verbose_;
    bool               andOrDcs_;
    edm::InputTag      dcsInputTag_;
    std::vector< int > dcsPartitions_;
    bool               errorReplyDcs_;
    bool                       andOrGt_;
    edm::InputTag              gtInputTag_;
    edm::InputTag              gtEvmInputTag_;
    std::string                gtDBKey_;
    std::vector< std::string > gtLogicalExpressions_;
    bool                       errorReplyGt_;
    bool                       andOrL1_;
    bool                       l1BeforeMask_;
    std::string                l1DBKey_;
    std::vector< std::string > l1LogicalExpressionsCache_;
    std::vector< std::string > l1LogicalExpressions_;
    bool                       errorReplyL1_;
    bool                       andOrHlt_;
    edm::InputTag              hltInputTag_;
    std::string                hltDBKey_;
    std::vector< std::string > hltLogicalExpressionsCache_;
    std::vector< std::string > hltLogicalExpressions_;
    bool                       errorReplyHlt_;
    // Switches
    bool on_;
    bool onDcs_;
    bool onGt_;
    bool onL1_;
    bool onHlt_;
    // Member constants
    const std::string configError_;
    const std::string emptyKeyError_;

  public:

    // Constructors and destructor
    GenericTriggerEventFlag( const edm::ParameterSet & config ); // To be called from the ED module's c'tor
    ~GenericTriggerEventFlag();

    // Public methods
    bool on()  { return     on_  ; }
    bool off() { return ( ! on_ ); }
    void initRun( const edm::Run & run, const edm::EventSetup & setup );    // To be called from beginRun() methods
    bool accept( const edm::Event & event, const edm::EventSetup & setup ); // To be called from analyze/filter() methods

  private:

    // Private methods

    // DCS
    bool acceptDcs( const edm::Event & event );
    bool acceptDcsPartition( const edm::Handle< DcsStatusCollection > & dcsStatus, int dcsPartition ) const;

    // GT status bits
    bool acceptGt( const edm::Event & event );
    bool acceptGtLogicalExpression( const edm::Event & event, std::string gtLogicalExpression );

    // L1
    bool acceptL1( const edm::Event & event, const edm::EventSetup & setup );
    bool acceptL1LogicalExpression( const edm::Event & event, std::string l1LogicalExpression );

    // HLT
    bool acceptHlt( const edm::Event & event );
    bool acceptHltLogicalExpression( const edm::Handle< edm::TriggerResults > & hltTriggerResults, std::string hltLogicalExpression ) const;

    // Algos
    std::string expandLogicalExpression( const std::vector< std::string > & target, const std::string & expr, bool useAnd = false ) const;
    bool negate( std::string & word ) const;

  public:

    // Methods for expert analysis

    std::string gtDBKey()  { return gtDBKey_ ; } // can be empty
    std::string l1DBKey()  { return l1DBKey_ ; } // can be empty
    std::string hltDBKey() { return hltDBKey_; } // can be empty

    std::vector< std::string > expressionsFromDB( const std::string & key, const edm::EventSetup & setup );

};


#endif
