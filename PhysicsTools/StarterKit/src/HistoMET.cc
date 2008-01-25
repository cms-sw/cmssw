#include "PhysicsTools/StarterKit/interface/HistoMET.h"


using pat::HistoMET;

// Constructor:


HistoMET::HistoMET( std::string dir, 
		    double pt1, double pt2, double m1, double m2 ) 
  : HistoGroup<MET>( dir, "MET", "met", pt1, pt2, m1, m2)
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
