/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "Alignment/RPTrackBased/interface/AlignmentAlgorithm.h"
#include "Alignment/RPTrackBased/interface/AlignmentTask.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//----------------------------------------------------------------------------------------------------

AlignmentAlgorithm::AlignmentAlgorithm(const edm::ParameterSet& ps, AlignmentTask *_t) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
  task(_t),
  singularLimit(ps.getParameter<double>("singularLimit")),
  useExternalFitter(ps.getParameter<bool>("useExternalFitter"))
{
}

