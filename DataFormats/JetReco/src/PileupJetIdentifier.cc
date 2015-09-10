#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"

#include <cstring>

// ------------------------------------------------------------------------------------------
StoredPileupJetIdentifier::StoredPileupJetIdentifier() 
{
  memset(this, 0, sizeof(StoredPileupJetIdentifier));
}

// ------------------------------------------------------------------------------------------
StoredPileupJetIdentifier::~StoredPileupJetIdentifier() 
{
}

// ------------------------------------------------------------------------------------------
PileupJetIdentifier::PileupJetIdentifier() 
{
  memset(this, 0, sizeof(PileupJetIdentifier));
}

// ------------------------------------------------------------------------------------------
PileupJetIdentifier::~PileupJetIdentifier() 
{
}
