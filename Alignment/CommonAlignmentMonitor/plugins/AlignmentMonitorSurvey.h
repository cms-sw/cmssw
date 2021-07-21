#ifndef Alignment_CommonAlignmentMonitor_AlignmentMonitorSurvey_H
#define Alignment_CommonAlignmentMonitor_AlignmentMonitorSurvey_H

// Package:     CommonAlignmentMonitor
// Class  :     AlignmentMonitorSurvey
//
// Store survey residuals in ROOT.
//
// Tree format is id:level:par[6].
// id: Alignable's ID (unsigned int).
// level: hierarchical level for which the survey residual is calculated (int).
// par[6]: survey residual (array of 6 doubles).
//
// Original Author:  Nhan Tran
//         Created:  10/8/07
// $Id: AlignmentMonitorSurvey.h,v 1.4 2008/02/22 01:21:45 cklae Exp $

#include <string>
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"

class AlignmentMonitorSurvey : public AlignmentMonitorBase {
public:
  AlignmentMonitorSurvey(const edm::ParameterSet&, edm::ConsumesCollector);

  void book() override;

  void event(const edm::Event&, const edm::EventSetup&, const ConstTrajTrackPairCollection&) override {}

private:
  std::vector<std::string> levelNames_;

  std::vector<align::StructureType> theLevels;
};

#endif
