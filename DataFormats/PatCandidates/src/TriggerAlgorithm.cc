//
// $Id: TriggerAlgorithm.cc,v 1.4 2011/11/30 13:41:14 vadler Exp $
//


#include "DataFormats/PatCandidates/interface/TriggerAlgorithm.h"


using namespace pat;


// Constructors and Destructor


// Default constructor
TriggerAlgorithm::TriggerAlgorithm() :
  name_(),
  alias_(),
  logic_(),
  tech_(),
  bit_(),
  gtlResult_(),
  prescale_(),
  mask_(),
  decisionBeforeMask_(),
  decisionAfterMask_()
{
  conditionKeys_.clear();
}


// Constructor from algorithm name only
TriggerAlgorithm::TriggerAlgorithm( const std::string & name ) :
  name_( name ),
  alias_(),
  logic_(),
  tech_(),
  bit_(),
  gtlResult_(),
  prescale_(),
  mask_(),
  decisionBeforeMask_(),
  decisionAfterMask_()
{
  conditionKeys_.clear();
}


// Constructors from values
TriggerAlgorithm::TriggerAlgorithm( const std::string & name, const std::string & alias, bool tech, unsigned bit, unsigned prescale, bool mask, bool decisionBeforeMask, bool decisionAfterMask ) :
  name_( name ),
  alias_( alias),
  logic_(),
  tech_( tech ),
  bit_( bit ),
  gtlResult_(),
  prescale_( prescale ),
  mask_( mask ),
  decisionBeforeMask_( decisionBeforeMask ),
  decisionAfterMask_( decisionAfterMask )
{
  conditionKeys_.clear();
}
TriggerAlgorithm::TriggerAlgorithm( const std::string & name, const std::string & alias, bool tech, unsigned bit, bool gtlResult, unsigned prescale, bool mask, bool decisionBeforeMask, bool decisionAfterMask ) :
  name_( name ),
  alias_( alias),
  logic_(),
  tech_( tech ),
  bit_( bit ),
  gtlResult_( gtlResult ),
  prescale_( prescale ),
  mask_( mask ),
  decisionBeforeMask_( decisionBeforeMask ),
  decisionAfterMask_( decisionAfterMask )
{
  conditionKeys_.clear();
}


// Methods


// Checks, if a certain trigger condition collection index is assigned
bool TriggerAlgorithm::hasConditionKey( unsigned conditionKey ) const
{
  for ( size_t iO = 0; iO < conditionKeys().size(); ++iO ) {
    if ( conditionKeys().at( iO ) == conditionKey ) {
      return true;
    }
  }
  return false;
}
