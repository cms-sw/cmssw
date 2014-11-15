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
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorExtendedRcd.h"

#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentErrorExtendedRcd.h"

#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HBAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HEAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HOAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/HFAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/ZDCAlignmentErrorExtendedRcd.h"

REGISTER_PLUGIN(GlobalPositionRcd,Alignments);
REGISTER_PLUGIN(TrackerAlignmentRcd,Alignments);
REGISTER_PLUGIN(TrackerAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(TrackerSurfaceDeformationRcd,AlignmentSurfaceDeformations);
REGISTER_PLUGIN(DTAlignmentRcd,Alignments);
REGISTER_PLUGIN(DTAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(CSCAlignmentRcd,Alignments);
REGISTER_PLUGIN(CSCAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(TrackerSurveyRcd,Alignments);
REGISTER_PLUGIN(TrackerSurveyErrorExtendedRcd,SurveyErrors);
REGISTER_PLUGIN(DTSurveyRcd,Alignments);
REGISTER_PLUGIN(DTSurveyErrorExtendedRcd,SurveyErrors);
REGISTER_PLUGIN(CSCSurveyRcd,Alignments);
REGISTER_PLUGIN(CSCSurveyErrorExtendedRcd,SurveyErrors);

REGISTER_PLUGIN(EBAlignmentRcd,Alignments);
REGISTER_PLUGIN(EBAlignmentErrorExtendedRcd,AlignmentErrors);
REGISTER_PLUGIN(EEAlignmentRcd,Alignments);
REGISTER_PLUGIN(EEAlignmentErrorExtendedRcd,AlignmentErrors);
REGISTER_PLUGIN(ESAlignmentRcd,Alignments);
REGISTER_PLUGIN(ESAlignmentErrorExtendedRcd,AlignmentErrors);
REGISTER_PLUGIN(HBAlignmentRcd,Alignments);
REGISTER_PLUGIN(HBAlignmentErrorExtendedRcd,AlignmentErrors);
REGISTER_PLUGIN(HEAlignmentRcd,Alignments);
REGISTER_PLUGIN(HEAlignmentErrorExtendedRcd,AlignmentErrors);
REGISTER_PLUGIN(HOAlignmentRcd,Alignments);
REGISTER_PLUGIN(HOAlignmentErrorExtendedRcd,AlignmentErrors);
REGISTER_PLUGIN(HFAlignmentRcd,Alignments);
REGISTER_PLUGIN(HFAlignmentErrorExtendedRcd,AlignmentErrors);
REGISTER_PLUGIN(ZDCAlignmentRcd,Alignments);
REGISTER_PLUGIN(ZDCAlignmentErrorExtendedRcd,AlignmentErrors);
REGISTER_PLUGIN(TrackerAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(DTAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(CSCAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(EBAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(EEAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(ESAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(HBAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(HEAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(HOAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(HFAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
REGISTER_PLUGIN(ZDCAlignmentErrorExtendedRcd,AlignmentErrorsExtended);
