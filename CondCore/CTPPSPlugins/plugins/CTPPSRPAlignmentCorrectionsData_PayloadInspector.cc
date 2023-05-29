/****************************************************************************
 *
 * This is a part of PPS PI software.
 *
 ****************************************************************************/

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/PayloadReader.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "CondCore/CTPPSPlugins/interface/CTPPSRPAlignmentCorrectionsDataHelper.h"

namespace {
  typedef RPShift_History<CTPPSRPAlignment::RP::RP3, CTPPSRPAlignment::Shift::x, false, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP3_x;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP23, CTPPSRPAlignment::Shift::x, false, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP23_x;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP103, CTPPSRPAlignment::Shift::x, false, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP103_x;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP123, CTPPSRPAlignment::Shift::x, false, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP123_x;

  typedef RPShift_History<CTPPSRPAlignment::RP::RP3, CTPPSRPAlignment::Shift::x, true, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP3_x_uncertainty;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP23, CTPPSRPAlignment::Shift::x, true, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP23_x_uncertainty;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP103, CTPPSRPAlignment::Shift::x, true, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP103_x_uncertainty;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP123, CTPPSRPAlignment::Shift::x, true, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP123_x_uncertainty;

  typedef RPShift_History<CTPPSRPAlignment::RP::RP3, CTPPSRPAlignment::Shift::y, false, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP3_y;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP23, CTPPSRPAlignment::Shift::y, false, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP23_y;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP103, CTPPSRPAlignment::Shift::y, false, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP103_y;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP123, CTPPSRPAlignment::Shift::y, false, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP123_y;

  typedef RPShift_History<CTPPSRPAlignment::RP::RP3, CTPPSRPAlignment::Shift::y, true, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP3_y_uncertainty;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP23, CTPPSRPAlignment::Shift::y, true, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP23_y_uncertainty;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP103, CTPPSRPAlignment::Shift::y, true, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP103_y_uncertainty;
  typedef RPShift_History<CTPPSRPAlignment::RP::RP123, CTPPSRPAlignment::Shift::y, true, CTPPSRPAlignmentCorrectionsData>
      RPShift_History_RP123_y_uncertainty;
}  // namespace

PAYLOAD_INSPECTOR_MODULE(CTPPSRPAlignmentCorrectionsData) {
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP3_x);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP23_x);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP103_x);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP123_x);

  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP3_x_uncertainty);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP23_x_uncertainty);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP103_x_uncertainty);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP123_x_uncertainty);

  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP3_y);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP23_y);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP103_y);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP123_y);

  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP3_y_uncertainty);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP23_y_uncertainty);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP103_y_uncertainty);
  PAYLOAD_INSPECTOR_CLASS(RPShift_History_RP123_y_uncertainty);
}