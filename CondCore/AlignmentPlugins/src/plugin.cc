/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Frederic Ronga on June 7, 2006
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"

#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(TrackerAlignmentRcd,Alignments);
REGISTER_PLUGIN(TrackerAlignmentErrorRcd,AlignmentErrors);
