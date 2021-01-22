/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include "CalibPPS/AlignmentRelative/interface/AlignmentAlgorithm.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentTask.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//----------------------------------------------------------------------------------------------------

AlignmentAlgorithm::AlignmentAlgorithm(const edm::ParameterSet& ps, AlignmentTask* _t)
    : verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
      task(_t),
      singularLimit(ps.getParameter<double>("singularLimit")) {}
