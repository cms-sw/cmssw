
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentTrackSelector.h"

// constructor ----------------------------------------------------------------

AlignmentTrackSelector::AlignmentTrackSelector(const edm::ParameterSet & cfg) :
  ptMin( cfg.getParameter<double>( "ptMin" ) ) 
{
  std::cout <<"[AlignmentTrackSelector] constructed.\n";
  std::cout <<"[AlignmentTrackSelector] ptmin: " << ptMin << std::endl;
}

// destructor -----------------------------------------------------------------

AlignmentTrackSelector::~AlignmentTrackSelector()
{}

// select track ---------------------------------------------------------------

bool AlignmentTrackSelector::operator()( const reco::Track & trk ) const 
{
  // cout <<"[AlignmentTrackSelector] pt,ptmin: " << trk.pt() <<","<<ptMin<<endl;
  return trk.pt() > ptMin;
}

