#include "PhysicsTools/StarterKit/interface/HistoMET.h"


using pat::HistoMET;

// Constructor:


HistoMET::HistoMET( std::string dir ) : HistoGroup<MET>( dir, "MET", "met")
{

  // book relevant MET histograms
}

HistoMET::~HistoMET()
{
  // Root deletes histograms, not us
}


void HistoMET::fill( const MET * met, uint iPart)
{

  // First fill common 4-vector histograms
  HistoGroup<MET>::fill( met, iPart );

  // fill relevant MET histograms
}

// void HistoMet::clearVec()
// {
//   HistoGroup<MET>::clearVec();
// }
