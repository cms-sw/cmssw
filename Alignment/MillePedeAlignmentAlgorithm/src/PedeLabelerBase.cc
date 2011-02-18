/** \file PedeLabelerBase.cc
 *
 * Baseclass for pede labelers
 *
 *  Original author: Andreas Mussgiller, January 2011
 *
 *  $Date: 2011/02/16 12:52:46 $
 *  $Revision: 1.1 $
 *  (last update by $Author: mussgill $)
 */

#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

// NOTE: Changing '+14' makes older binary files unreadable...
const unsigned int PedeLabelerBase::theMaxNumParam = RigidBodyAlignmentParameters::N_PARAM + 14;
const unsigned int PedeLabelerBase::theParamInstanceOffset = 300000;
const unsigned int PedeLabelerBase::theMinLabel = 1; // must be > 0

PedeLabelerBase::PedeLabelerBase(const TopLevelAlignables &alignables,
				 const edm::ParameterSet &config)
  :theOpenRunRange(std::make_pair<RunNumber,RunNumber>(cond::timeTypeSpecs[cond::runnumber].beginValue,
						       cond::timeTypeSpecs[cond::runnumber].endValue))
{
  
}
