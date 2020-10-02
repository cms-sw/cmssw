#include "CalibPPS/AlignmentRelative/interface/LocalTrackFit.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentTask.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentConstraint.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentGeometry.h"
#include "CalibPPS/AlignmentRelative/interface/JanAlignmentAlgorithm.h"
#include "CalibPPS/AlignmentRelative/interface/AlignmentResult.h"

#include <TVectorD.h>
#include <TGraph.h>
#include <TH1D.h>

namespace {
  namespace {
    LocalTrackFit ltf;

    AlignmentTask at;

    AlignmentConstraint ac;
	std::map<unsigned int, TVectorD> muitvd;

	std::vector<AlignmentConstraint> vac;

	DetGeometry dg;
    AlignmentGeometry ag;

	std::map<unsigned int, DetGeometry> muidg;
	std::set<unsigned int> sui;

	SingularMode sm;
	std::vector<SingularMode> vsm;

    JanAlignmentAlgorithm jaa;
	JanAlignmentAlgorithm::ScatterPlot jaasp;
	JanAlignmentAlgorithm::DetStat jaads;
	std::map<unsigned int, JanAlignmentAlgorithm::DetStat> muids;
	std::vector<TH1D*> vth;
    std::vector<TGraph*> vtg;  
    std::map< std::set<unsigned int>, JanAlignmentAlgorithm::ScatterPlot> msuisp;

    AlignmentResult ar;
    std::map<unsigned int, AlignmentResult> muiar;
  }
}
