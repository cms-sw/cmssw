//
// $Id$
//


#include "DataFormats/PatCandidates/interface/TriggerAlgorithm.h"


using namespace pat;


/// default constructor

TriggerAlgorithm::TriggerAlgorithm() :
  name_(),
  alias_(),
  tech_(),
  bit_(),
  prescale_(),
  mask_(),
  decisionBeforeMask_(),
  decisionAfterMask_()
{}

/// constructor from values

TriggerAlgorithm::TriggerAlgorithm( const std::string & name ) :
  name_( name ),
  alias_(),
  tech_(),
  bit_(),
  prescale_(),
  mask_(),
  decisionBeforeMask_(),
  decisionAfterMask_()
{}

TriggerAlgorithm::TriggerAlgorithm( const std::string & name, const std::string & alias, bool tech, unsigned bit, unsigned prescale, bool mask, bool decisionBeforeMask, bool decisionAfterMask ) :
  name_( name ),
  alias_( alias),
  tech_( tech ),
  bit_( bit ),
  prescale_( prescale ),
  mask_( mask ),
  decisionBeforeMask_( decisionBeforeMask ),
  decisionAfterMask_( decisionAfterMask )
{}

/// getters
