#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"


//__________________________________________________________________________________________________
AlignmentAlgorithmBase::AlignmentAlgorithmBase( const edm::ParameterSet& cfg ) :
  debug( cfg.getParameter<bool>("debug") )
{

}

