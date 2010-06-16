#ifndef DataFormats_PatCandidates_TriggerAlgorithm_h
#define DataFormats_PatCandidates_TriggerAlgorithm_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerAlgorithm
//
// $Id: TriggerAlgorithm.h,v 1.1 2010/04/20 21:41:01 vadler Exp $
//
/**
  \class    pat::TriggerAlgorithm TriggerAlgorithm.h "DataFormats/PatCandidates/interface/TriggerAlgorithm.h"
  \brief    Analysis-level L1 trigger algorithm class

   TriggerAlgorithm implements a container for L1 trigger algorithms' information within the 'pat' namespace.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerAlgorithm

  \author   Volker Adler
  \version  $Id: TriggerAlgorithm.h,v 1.1 2010/04/20 21:41:01 vadler Exp $
*/


#include <string>
#include <vector>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"


namespace pat {

  class TriggerAlgorithm {

      /// data members
      std::string name_;
      std::string alias_;
      bool        tech_;
      unsigned    bit_;
      unsigned    prescale_;
      bool        mask_;
      bool        decisionBeforeMask_;
      bool        decisionAfterMask_;

    public:

      /// constructors and desctructor
      TriggerAlgorithm();
      TriggerAlgorithm( const std::string & name );
      TriggerAlgorithm( const std::string & name, const std::string & alias, bool tech, unsigned bit, unsigned prescale, bool mask, bool decisionBeforeMask, bool decisionAfterMask );
      virtual ~TriggerAlgorithm() {};

      /// setters & getters
      void setName( const std::string & name )              { name_               = name; };
      void setAlias( const std::string & alias )            { alias_              = alias; };
      void setTechTrigger( bool tech )                      { tech_               = tech; };
      void setBit( unsigned bit )                           { bit_                = bit; };
      void setPrescale( unsigned prescale )                 { prescale_           = prescale; };
      void setMask( bool mask )                             { mask_               = mask; };
      void setDecisionBeforeMask( bool decisionBeforeMask ) { decisionBeforeMask_ = decisionBeforeMask; };
      void setDecisionAfterMas( bool decisionAfterMask )    { decisionAfterMask_  = decisionAfterMask; };
      std::string name() const               { return name_; };
      std::string alias() const              { return alias_; };
      bool        techTrigger() const        { return tech_; };
      unsigned    bit() const                { return bit_; };
      unsigned    prescale() const           { return prescale_; };
      bool        mask() const               { return mask_; };
      bool        decisionBeforeMask() const { return decisionBeforeMask_; };
      bool        decisionAfterMask() const  { return decisionAfterMask_; };
      bool        decision() const           { return decisionAfterMask(); };

  };


  /// collection of TriggerAlgorithm
  typedef std::vector< TriggerAlgorithm >                      TriggerAlgorithmCollection;
  /// persistent reference to an item in a TriggerAlgorithmCollection
  typedef edm::Ref< TriggerAlgorithmCollection >               TriggerAlgorithmRef;
  /// persistent reference to a TriggerAlgorithmCollection product
  typedef edm::RefProd< TriggerAlgorithmCollection >           TriggerAlgorithmRefProd;
  /// vector of persistent references to items in the same TriggerAlgorithmCollection
  typedef edm::RefVector< TriggerAlgorithmCollection >         TriggerAlgorithmRefVector;
  /// const iterator over vector of persistent references to items in the same TriggerAlgorithmCollection
  typedef edm::RefVectorIterator< TriggerAlgorithmCollection > TriggerAlgorithmRefVectorIterator;

}


#endif
