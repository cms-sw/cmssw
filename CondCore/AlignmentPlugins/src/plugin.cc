/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Frederic Ronga on June 7, 2006
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorRcd.h"

#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentErrorRcd.h"

REGISTER_PLUGIN(GlobalPositionRcd,Alignments);
REGISTER_PLUGIN(TrackerAlignmentRcd,Alignments);
REGISTER_PLUGIN(TrackerAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(TrackerSurfaceDeformationRcd,AlignmentSurfaceDeformations);
REGISTER_PLUGIN(DTAlignmentRcd,Alignments);
REGISTER_PLUGIN(DTAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(CSCAlignmentRcd,Alignments);
REGISTER_PLUGIN(CSCAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(TrackerSurveyRcd,Alignments);
REGISTER_PLUGIN(TrackerSurveyErrorRcd,SurveyErrors);
REGISTER_PLUGIN(DTSurveyRcd,Alignments);
REGISTER_PLUGIN(DTSurveyErrorRcd,SurveyErrors);
REGISTER_PLUGIN(CSCSurveyRcd,Alignments);
REGISTER_PLUGIN(CSCSurveyErrorRcd,SurveyErrors);

REGISTER_PLUGIN(EBAlignmentRcd,Alignments);
REGISTER_PLUGIN(EBAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(EEAlignmentRcd,Alignments);
REGISTER_PLUGIN(EEAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(ESAlignmentRcd,Alignments);
REGISTER_PLUGIN(ESAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(HBAlignmentRcd,Alignments);
REGISTER_PLUGIN(HBAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(HEAlignmentRcd,Alignments);
REGISTER_PLUGIN(HEAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(HOAlignmentRcd,Alignments);
REGISTER_PLUGIN(HOAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(HFAlignmentRcd,Alignments);
REGISTER_PLUGIN(HFAlignmentErrorRcd,AlignmentErrors);
REGISTER_PLUGIN(ZDCAlignmentRcd,Alignments);
REGISTER_PLUGIN(ZDCAlignmentErrorRcd,AlignmentErrors);
