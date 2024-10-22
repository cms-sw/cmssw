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
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
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
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentErrorRcd.h"
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

#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentErrorExtendedRcd.h"

REGISTER_PLUGIN(GlobalPositionRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(TrackerAlignmentRcd, Alignments);
REGISTER_PLUGIN(TrackerSurfaceDeformationRcd, AlignmentSurfaceDeformations);
REGISTER_PLUGIN_NO_SERIAL(DTAlignmentRcd, Alignments);
REGISTER_PLUGIN(DTAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(CSCAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(CSCAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(GEMAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(GEMAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(TrackerSurveyRcd, Alignments);
REGISTER_PLUGIN(TrackerSurveyErrorRcd, SurveyErrors);
REGISTER_PLUGIN_NO_SERIAL(DTSurveyRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(DTSurveyErrorRcd, SurveyErrors);
REGISTER_PLUGIN_NO_SERIAL(CSCSurveyRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(CSCSurveyErrorRcd, SurveyErrors);

REGISTER_PLUGIN_NO_SERIAL(EBAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(EBAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(EEAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(EEAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(ESAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(ESAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(HBAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(HBAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(HEAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(HEAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(HOAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(HOAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(HFAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(HFAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(ZDCAlignmentRcd, Alignments);
REGISTER_PLUGIN_NO_SERIAL(ZDCAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN_NO_SERIAL(TrackerAlignmentErrorRcd, AlignmentErrors);
REGISTER_PLUGIN(TrackerAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(DTAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(CSCAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(GEMAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(EBAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(EEAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(ESAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(HBAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(HEAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(HOAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(HFAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
REGISTER_PLUGIN_NO_SERIAL(ZDCAlignmentErrorExtendedRcd, AlignmentErrorsExtended);
