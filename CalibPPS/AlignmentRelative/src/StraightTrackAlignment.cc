/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h"
#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "CalibPPS/AlignmentRelative/interface/StraightTrackAlignment.h"
#include "CalibPPS/AlignmentRelative/interface/Utilities.h"

#include "CalibPPS/AlignmentRelative/interface/IdealResult.h"
#include "CalibPPS/AlignmentRelative/interface/JanAlignmentAlgorithm.h"

#include <set>
#include <unordered_set>
#include <vector>
#include <string>

#include "TDecompLU.h"
#include "TH1D.h"
#include "TFile.h"
#include "TGraph.h"
#include "TCanvas.h"

//#define DEBUG

using namespace edm;
using namespace std;

//----------------------------------------------------------------------------------------------------

TH1D *StraightTrackAlignment::newResiduaHist(const char *name) {
  return new TH1D(name, ";residual   (mm)", 2000, -3., +3.);  // in mm
}

//----------------------------------------------------------------------------------------------------

TGraph *newGraph(const string &name, const string &title) {
  TGraph *g = new TGraph();
  g->SetName(name.c_str());
  g->SetTitle(title.c_str());
  return g;
}

//----------------------------------------------------------------------------------------------------

StraightTrackAlignment::RPSetPlots::RPSetPlots(const string &_name) : name(_name) {
  chisqn_lin_fitted = new TH1D("chi^2 norm, lin, fitted", ";#chi^{2}/ndf;", 5000, 0., 500.);
  chisqn_lin_selected = new TH1D("chi^2 norm, lin, selected", ";#chi^{2}/ndf;", 5000, 0., 500.);
  chisqn_log_fitted = new TH1D("chi^2 norm, log, fitted", ";log_{10}(#chi^{2}/ndf);", 700, -1., 6.);
  chisqn_log_selected = new TH1D("chi^2 norm, log, selected", ";log_{10}(#chi^{2}/ndf);", 700, -1., 6.);

  fitAxVsAyGraph_fitted = newGraph("ax vs. ay, fitted", ";a_{x}   (rad);a_{y}   (rad)");
  fitAxVsAyGraph_selected = newGraph("ax vs. ay, selected", ";a_{x}   (rad);a_{y}   (rad)");
  fitBxVsByGraph_fitted = newGraph("bx vs. by, fitted", ";b_{x}   (mm);b_{y}   (mm)");
  fitBxVsByGraph_selected = newGraph("bx vs. by, selected", ";b_{x}   (mm);b_{y}   (mm)");
}
//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::RPSetPlots::free() {
  delete chisqn_lin_fitted;
  delete chisqn_lin_selected;
  delete chisqn_log_fitted;
  delete chisqn_log_selected;

  delete fitAxVsAyGraph_fitted;
  delete fitAxVsAyGraph_selected;
  delete fitBxVsByGraph_fitted;
  delete fitBxVsByGraph_selected;
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::RPSetPlots::write() const {
  chisqn_lin_fitted->Write();
  chisqn_lin_selected->Write();
  chisqn_log_fitted->Write();
  chisqn_log_selected->Write();

  fitAxVsAyGraph_fitted->Write();
  fitAxVsAyGraph_selected->Write();
  fitBxVsByGraph_fitted->Write();
  fitBxVsByGraph_selected->Write();
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

StraightTrackAlignment::StraightTrackAlignment(const ParameterSet &ps)
    : verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),

      rpIds(ps.getParameter<vector<unsigned int> >("rpIds")),
      excludePlanes(ps.getParameter<vector<unsigned int> >("excludePlanes")),
      z0(ps.getParameter<double>("z0")),

      maxEvents(ps.getParameter<signed int>("maxEvents")),

      removeImpossible(ps.getParameter<bool>("removeImpossible")),
      requireNumberOfUnits(ps.getParameter<unsigned int>("requireNumberOfUnits")),
      requireAtLeast3PotsInOverlap(ps.getParameter<bool>("requireAtLeast3PotsInOverlap")),
      requireOverlap(ps.getParameter<bool>("requireOverlap")),
      cutOnChiSqPerNdf(ps.getParameter<bool>("cutOnChiSqPerNdf")),
      chiSqPerNdfCut(ps.getParameter<double>("chiSqPerNdfCut")),
      maxTrackAx(ps.getParameter<double>("maxTrackAx")),
      maxTrackAy(ps.getParameter<double>("maxTrackAy")),

      fileNamePrefix(ps.getParameter<string>("fileNamePrefix")),
      expandedFileNamePrefix(ps.getParameter<string>("expandedFileNamePrefix")),
      factoredFileNamePrefix(ps.getParameter<string>("factoredFileNamePrefix")),
      preciseXMLFormat(ps.getParameter<bool>("preciseXMLFormat")),
      saveXMLUncertainties(ps.getParameter<bool>("saveXMLUncertainties")),

      saveIntermediateResults(ps.getParameter<bool>("saveIntermediateResults")),
      taskDataFileName(ps.getParameter<string>("taskDataFileName")),
      taskDataFile(nullptr),

      task(ps),
      fitter(ps),

      buildDiagnosticPlots(ps.getParameter<bool>("buildDiagnosticPlots")),
      diagnosticsFile(ps.getParameter<string>("diagnosticsFile")),
      fitNdfHist_fitted(new TH1D("ndf_fitted", ";ndf;", 41, -4.5, 36.5)),
      fitNdfHist_selected(new TH1D("ndf_selected", ";ndf;", 41, -4.5, 36.5)),
      fitPHist_fitted(new TH1D("p_fitted", ";p value;", 100, 0., 1.)),
      fitPHist_selected(new TH1D("p_selected", ";p value;", 100, 0., 1.)),
      fitAxHist_fitted(new TH1D("ax_fitted", ";a_{x}   (rad);", 10000, -0.1, 0.1)),
      fitAxHist_selected(new TH1D("ax_selected", ";a_{x}   (rad);", 10000, -0.1, 0.1)),
      fitAyHist_fitted(new TH1D("ay_fitted", ";a_{y}   (rad);", 10000, -0.1, 0.1)),
      fitAyHist_selected(new TH1D("ay_selected", ";a_{y}   (rad);", 10000, -0.1, 0.1)),
      fitBxHist_fitted(new TH1D("bx_fitted", ";b_{x}   (mm);", 500, -30., 30.)),
      fitBxHist_selected(new TH1D("bx_selected", ";b_{x}   (mm);", 500, -30., 30.)),
      fitByHist_fitted(new TH1D("by_fitted", ";b_{y}   (mm);", 500, -30., 30.)),
      fitByHist_selected(new TH1D("by_selected", ";b_{y}   (mm);", 500, -30., 30.)),

      globalPlots("global") {
  // open task data file
  if (!taskDataFileName.empty())
    taskDataFile = new TFile(taskDataFileName.c_str(), "recreate");

  // instantiate algorithm objects
  // (and save them)
  vector<string> alNames(ps.getParameter<vector<string> >("algorithms"));
  for (unsigned int i = 0; i < alNames.size(); i++) {
    AlignmentAlgorithm *a = nullptr;

    if (alNames[i] == "Ideal") {
      IdealResult *ir = new IdealResult(ps, &task);
      a = ir;
    } else if (alNames[i] == "Jan") {
      JanAlignmentAlgorithm *jaa = new JanAlignmentAlgorithm(ps, &task);
      a = jaa;
    }

    if (a)
      algorithms.push_back(a);
    else
      throw cms::Exception("PPS") << "Unknown alignment algorithm `" << alNames[i] << "'.";
  }

  // get constraints type
  string ct = ps.getParameter<string>("constraintsType");
  if (ct.compare("fixedDetectors") == 0)
    constraintsType = ctFixedDetectors;
  else if (ct.compare("standard") == 0)
    constraintsType = ctStandard;
  else
    throw cms::Exception("PPS") << "Unknown constraints type `" << ct << "'.";

  // parse additional accepted RP sets
  string aars_str = ps.getParameter<string>("additionalAcceptedRPSets");

  size_t idx_b = 0, idx_e = string::npos;
  while (idx_b != string::npos) {
    // get one block - portion between successive ";"
    idx_e = aars_str.find(';', idx_b);
    size_t len = (idx_e == string::npos) ? string::npos : idx_e - idx_b;
    string block = aars_str.substr(idx_b, len);

    // process the block
    if (!block.empty()) {
      set<unsigned int> rpSet;

      // isolate bits (= RP ids)
      size_t bi_b = 0, bi_e = string::npos;
      while (bi_b != string::npos) {
        bi_e = block.find(',', bi_b);
        size_t bit_len = (bi_e == string::npos) ? string::npos : bi_e - bi_b;
        const string &bit = block.substr(bi_b, bit_len);

        unsigned int rp = atoi(bit.c_str());
        rpSet.insert(rp);

        bi_b = (bi_e == string::npos) ? string::npos : bi_e + 1;
      }

      additionalAcceptedRPSets.push_back(rpSet);
    }

    // move to next block
    idx_b = (idx_e == string::npos) ? string::npos : idx_e + 1;
  }
}

//----------------------------------------------------------------------------------------------------

StraightTrackAlignment::~StraightTrackAlignment() {
  if (taskDataFile)
    delete taskDataFile;

  for (vector<AlignmentAlgorithm *>::iterator it = algorithms.begin(); it != algorithms.end(); ++it)
    delete (*it);

  delete fitNdfHist_fitted;
  delete fitNdfHist_selected;
  delete fitPHist_fitted;
  delete fitPHist_selected;
  delete fitAxHist_fitted;
  delete fitAyHist_fitted;
  delete fitAxHist_selected;
  delete fitAyHist_selected;
  delete fitBxHist_fitted;
  delete fitByHist_fitted;
  delete fitBxHist_selected;
  delete fitByHist_selected;

  globalPlots.free();

  for (auto &p : rpSetPlots)
    p.second.free();

  for (auto it = residuaHistograms.begin(); it != residuaHistograms.end(); ++it) {
    delete it->second.total_fitted;
    delete it->second.total_selected;
    delete it->second.selected_vs_chiSq;
    for (auto sit = it->second.perRPSet_fitted.begin(); sit != it->second.perRPSet_fitted.end(); ++sit)
      delete sit->second;
    for (auto sit = it->second.perRPSet_selected.begin(); sit != it->second.perRPSet_selected.end(); ++sit)
      delete sit->second;
  }
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::begin(edm::ESHandle<CTPPSRPAlignmentCorrectionsData> hRealAlignment,
                                   edm::ESHandle<CTPPSGeometry> hRealGeometry,
                                   edm::ESHandle<CTPPSGeometry> hMisalignedGeometry) {
  // reset counters
  eventsTotal = 0;
  eventsFitted = 0;
  eventsSelected = 0;
  fittedTracksPerRPSet.clear();
  selectedTracksPerRPSet.clear();
  selectedHitsPerPlane.clear();

  // prepare geometry
  task.buildGeometry(rpIds, excludePlanes, hRealGeometry.product(), z0, task.geometry);

  // build matrix index mappings
  task.buildIndexMaps();

  // print geometry info
  if (verbosity > 1)
    task.geometry.print();

  // save task (including geometry) and fitter
  if (taskDataFile) {
    taskDataFile->WriteObject(&task, "task");
    taskDataFile->WriteObject(&fitter, "fitter");
  }

  // initiate the algorithms
  for (const auto &a : algorithms)
    a->begin(hRealGeometry.product(), hMisalignedGeometry.product());

  // get initial alignments
  initialAlignments = *hRealAlignment;
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::processEvent(const edm::EventID &eventId,
                                          const DetSetVector<TotemRPUVPattern> &uvPatternsStrip,
                                          const DetSetVector<CTPPSDiamondRecHit> &hitsDiamond,
                                          const edm::DetSetVector<CTPPSPixelRecHit> &hitsPixel,
                                          const DetSetVector<CTPPSPixelLocalTrack> &tracksPixel) {
  eventsTotal++;

  if (verbosity > 9)
    printf("\n---------- StraightTrackAlignment::ProcessEvent (%u:%llu)\n", eventId.run(), eventId.event());

  // -------------------- STEP 1: get hits from selected RPs

  HitCollection hitSelection;

  // strips
  for (const auto &pv : uvPatternsStrip) {
    const CTPPSDetId detId(pv.detId());
    const unsigned int rpDecId = detId.arm() * 100 + detId.station() * 10 + detId.rp();

    // skip if RP not selected
    if (find(rpIds.begin(), rpIds.end(), rpDecId) == rpIds.end())
      continue;

    // require exactly 1 U and 1 V pattern, i.e. unique U-V association
    unsigned int n_U = 0, n_V = 0;
    unsigned int idx_U = 0, idx_V = 0;
    for (unsigned int pi = 0; pi < pv.size(); pi++) {
      const TotemRPUVPattern &pattern = pv[pi];

      switch (pattern.projection()) {
        case TotemRPUVPattern::projU:
          n_U++;
          idx_U = pi;
          break;

        case TotemRPUVPattern::projV:
          n_V++;
          idx_V = pi;
          break;

        default:
          break;
      }
    }

    if (n_U != 1 || n_V != 1)
      continue;

    // skip if patterns not reasonable
    if (!pv[idx_U].fittable() || !pv[idx_V].fittable())
      continue;

    // add hits from the U and V pattern
    for (const auto &pattern : {pv[idx_U], pv[idx_V]}) {
      for (const auto &hitsDetSet : pattern.hits()) {
        // skip if sensor not in geometry
        if (!task.geometry.isValidSensorId(hitsDetSet.detId()))
          continue;

        const double &z = task.geometry.get(hitsDetSet.detId()).z;
        for (auto &hit : hitsDetSet)
          hitSelection.emplace_back(Hit(hitsDetSet.detId(), 2, hit.position(), hit.sigma(), z));
      }
    }
  }

  // diamonds
  for (const auto &pv : hitsDiamond) {
    // skip if RP not selected
    const CTPPSDetId detId(pv.detId());
    const unsigned int rpDecId = detId.arm() * 100 + detId.station() * 10 + detId.rp();

    // skip if RP not selected
    if (find(rpIds.begin(), rpIds.end(), rpDecId) == rpIds.end())
      continue;

    // skip if sensor not in geometry
    if (!task.geometry.isValidSensorId(detId))
      continue;

    const double &z = task.geometry.get(pv.detId()).z;

    for (const auto &h : pv)
      hitSelection.emplace_back(Hit(pv.detId(), 1, h.x(), h.xWidth() / sqrt(12.), z));
  }

  // pixels: data from rec hits
  map<unsigned int, unsigned int> pixelPlaneMultiplicity;
  for (const auto &pv : hitsPixel)
    pixelPlaneMultiplicity[pv.detId()] += pv.size();

  for (const auto &pv : hitsPixel) {
    // skip if RP not selected
    const CTPPSDetId detId(pv.detId());
    const unsigned int rpDecId = detId.arm() * 100 + detId.station() * 10 + detId.rp();

    // skip if RP not selected
    if (find(rpIds.begin(), rpIds.end(), rpDecId) == rpIds.end())
      continue;

    // skip if sensor not in geometry
    if (!task.geometry.isValidSensorId(detId))
      continue;

    // skip if plane multiplicity greater than 1
    if (pixelPlaneMultiplicity[pv.detId()] > 1)
      continue;

    for (const auto &h : pv) {
      const auto &dg = task.geometry.get(pv.detId());
      const double dz1 = dg.getDirectionData(1).dz;
      const double dz2 = dg.getDirectionData(2).dz;
      const double z = dg.z + h.point().x() * dz1 + h.point().y() * dz2;

      double x_unc = sqrt(h.error().xx());
      double y_unc = sqrt(h.error().yy());

      // TODO: Check this

      if (x_unc <= 0.)
        x_unc = 10E-3;  // mm
      if (y_unc <= 0.)
        y_unc = 10E-3;  // mm

      hitSelection.emplace_back(Hit(pv.detId(), 1, h.point().x(), x_unc, z));
      hitSelection.emplace_back(Hit(pv.detId(), 2, h.point().y(), y_unc, z));
    }
  }

  // pixels: data from tracks
  for (const auto &pv : tracksPixel) {
    const CTPPSDetId rpId(pv.detId());
    const unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    // skip if RP not selected
    if (find(rpIds.begin(), rpIds.end(), rpDecId) == rpIds.end())
      continue;

    // skip if more than 1 (valid) track in the RP
    unsigned int n_valid_tracks = 0;
    for (const auto &tr : pv) {
      if (tr.isValid())
        n_valid_tracks++;
    }

    if (n_valid_tracks > 1)
      continue;

    // go through all valid tracks
    for (const auto &tr : pv) {
      if (!tr.isValid())
        continue;

      // go through all associated rec hits
      for (const auto &hv : tr.hits()) {
        const CTPPSPixelDetId senId(hv.detId());

        // skip if sensor not in geometry
        if (!task.geometry.isValidSensorId(senId))
          continue;

        for (const auto &h : hv) {
          // skip hit if not used for fit
          if (!h.isUsedForFit())
            continue;

          const auto &dg = task.geometry.get(senId);
          const double dz1 = dg.getDirectionData(1).dz;
          const double dz2 = dg.getDirectionData(2).dz;
          const double z = dg.z + h.point().x() * dz1 + h.point().y() * dz2;

          double x_unc = sqrt(h.error().xx());
          double y_unc = sqrt(h.error().yy());

          if (x_unc <= 0.)
            x_unc = 10E-3;  // mm
          if (y_unc <= 0.)
            y_unc = 10E-3;  // mm

          hitSelection.emplace_back(Hit(senId, 1, h.point().x(), x_unc, z));
          hitSelection.emplace_back(Hit(senId, 2, h.point().y(), y_unc, z));
        }
      }
    }
  }

  if (hitSelection.empty())
    return;

  // -------------------- STEP 2: fit + outlier rejection

  LocalTrackFit trackFit;
  if (!fitter.fit(hitSelection, task.geometry, trackFit))
    return;

  set<unsigned int> selectedRPs;
  for (const auto &hit : hitSelection) {
    CTPPSDetId detId(hit.id);
    const unsigned int decRPId = detId.arm() * 100 + detId.station() * 10 + detId.rp();
    selectedRPs.insert(decRPId);
  }

  eventsFitted++;
  fittedTracksPerRPSet[selectedRPs]++;

  // -------------------- STEP 3: quality checks

  bool top = false, bottom = false, horizontal = false;
  unordered_set<unsigned int> units;
  for (const auto &rp : selectedRPs) {
    unsigned int rpIdx = rp % 10;
    unsigned int stId = rp / 10;
    unsigned int unitId = stId * 10;
    if (rpIdx > 2)
      unitId++;

    if (rpIdx == 0 || rpIdx == 4)
      top = true;
    if (rpIdx == 1 || rpIdx == 5)
      bottom = true;
    if (rpIdx == 2 || rpIdx == 3)
      horizontal = true;

    units.insert(unitId);
  }

  bool overlap = (top && horizontal) || (bottom && horizontal);

  bool rp_set_accepted = true;

  // impossible signature
  if (removeImpossible && top && bottom)
    rp_set_accepted = false;

  // cleanliness cuts
  if (units.size() < requireNumberOfUnits)
    rp_set_accepted = false;

  if (requireOverlap && !overlap)
    rp_set_accepted = false;

  if (requireAtLeast3PotsInOverlap && overlap && selectedRPs.size() < 3)
    rp_set_accepted = false;

  // is it an additional accepted RP set?
  if (find(additionalAcceptedRPSets.begin(), additionalAcceptedRPSets.end(), selectedRPs) !=
      additionalAcceptedRPSets.end())
    rp_set_accepted = true;

  if (verbosity > 5)
    printf("* rp set accepted: %u\n", rp_set_accepted);

  bool selected = rp_set_accepted;

  // too bad chisq
  if (cutOnChiSqPerNdf && trackFit.chiSqPerNdf() > chiSqPerNdfCut)
    selected = false;

  // parallelity cut
  if (fabs(trackFit.ax) > maxTrackAx || fabs(trackFit.ay) > maxTrackAy)
    selected = false;

  updateDiagnosticHistograms(hitSelection, selectedRPs, trackFit, selected);

  if (verbosity > 5)
    printf("* event selected: %u\n", selected);

  if (!selected)
    return;

  // update counters
  eventsSelected++;
  selectedTracksPerRPSet[selectedRPs]++;

  for (const auto &h : hitSelection)
    selectedHitsPerPlane[h.id]++;

  // -------------------- STEP 4: FEED ALGORITHMS

  for (auto &a : algorithms)
    a->feed(hitSelection, trackFit);

  // -------------------- STEP 5: ENOUGH TRACKS?

  if (eventsSelected == maxEvents)
    throw "StraightTrackAlignment: Number of tracks processed reached maximum";
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::updateDiagnosticHistograms(const HitCollection &selection,
                                                        const set<unsigned int> &selectedRPs,
                                                        const LocalTrackFit &trackFit,
                                                        bool trackSelected) {
  if (!buildDiagnosticPlots)
    return;

  fitNdfHist_fitted->Fill(trackFit.ndf);
  fitPHist_fitted->Fill(trackFit.pValue());
  fitAxHist_fitted->Fill(trackFit.ax);
  fitAyHist_fitted->Fill(trackFit.ay);
  fitBxHist_fitted->Fill(trackFit.bx);
  fitByHist_fitted->Fill(trackFit.by);

  globalPlots.chisqn_lin_fitted->Fill(trackFit.chiSqPerNdf());
  globalPlots.chisqn_log_fitted->Fill(log10(trackFit.chiSqPerNdf()));
  globalPlots.fitAxVsAyGraph_fitted->SetPoint(globalPlots.fitAxVsAyGraph_fitted->GetN(), trackFit.ax, trackFit.ay);
  globalPlots.fitBxVsByGraph_fitted->SetPoint(globalPlots.fitBxVsByGraph_fitted->GetN(), trackFit.bx, trackFit.by);

  if (trackSelected) {
    fitNdfHist_selected->Fill(trackFit.ndf);
    fitPHist_selected->Fill(trackFit.pValue());
    fitAxHist_selected->Fill(trackFit.ax);
    fitAyHist_selected->Fill(trackFit.ay);
    fitBxHist_selected->Fill(trackFit.bx);
    fitByHist_selected->Fill(trackFit.by);

    globalPlots.chisqn_lin_selected->Fill(trackFit.chiSqPerNdf());
    globalPlots.chisqn_log_selected->Fill(log10(trackFit.chiSqPerNdf()));
    globalPlots.fitAxVsAyGraph_selected->SetPoint(
        globalPlots.fitAxVsAyGraph_selected->GetN(), trackFit.ax, trackFit.ay);
    globalPlots.fitBxVsByGraph_selected->SetPoint(
        globalPlots.fitBxVsByGraph_selected->GetN(), trackFit.bx, trackFit.by);
  }

  auto it = rpSetPlots.find(selectedRPs);
  if (it == rpSetPlots.end())
    it = rpSetPlots.insert({selectedRPs, RPSetPlots(setToString(selectedRPs))}).first;

  it->second.chisqn_lin_fitted->Fill(trackFit.chiSqPerNdf());
  it->second.chisqn_log_fitted->Fill(log10(trackFit.chiSqPerNdf()));
  it->second.fitAxVsAyGraph_fitted->SetPoint(it->second.fitAxVsAyGraph_fitted->GetN(), trackFit.ax, trackFit.ay);
  it->second.fitBxVsByGraph_fitted->SetPoint(it->second.fitBxVsByGraph_fitted->GetN(), trackFit.bx, trackFit.by);

  if (trackSelected) {
    it->second.chisqn_lin_selected->Fill(trackFit.chiSqPerNdf());
    it->second.chisqn_log_selected->Fill(log10(trackFit.chiSqPerNdf()));
    it->second.fitAxVsAyGraph_selected->SetPoint(it->second.fitAxVsAyGraph_selected->GetN(), trackFit.ax, trackFit.ay);
    it->second.fitBxVsByGraph_selected->SetPoint(it->second.fitBxVsByGraph_selected->GetN(), trackFit.bx, trackFit.by);
  }

  for (const auto &hit : selection) {
    unsigned int id = hit.id;

    const DetGeometry &geom = task.geometry.get(id);
    const auto dirData = geom.getDirectionData(hit.dirIdx);

    double m = hit.position + dirData.s - (hit.z - geom.z) * dirData.dz;
    double x = trackFit.ax * hit.z + trackFit.bx;
    double y = trackFit.ay * hit.z + trackFit.by;
    double f = x * dirData.dx + y * dirData.dy;
    double R = m - f;

    auto it = residuaHistograms.find(id);
    if (it == residuaHistograms.end()) {
      it = residuaHistograms.insert(pair<unsigned int, ResiduaHistogramSet>(id, ResiduaHistogramSet())).first;
      char buf[30];
      sprintf(buf, "%u: total_fitted", id);
      it->second.total_fitted = newResiduaHist(buf);
      sprintf(buf, "%u: total_selected", id);
      it->second.total_selected = newResiduaHist(buf);
      it->second.selected_vs_chiSq = new TGraph();
      sprintf(buf, "%u: selected_vs_chiSq", id);
      it->second.selected_vs_chiSq->SetName(buf);
    }

    it->second.total_fitted->Fill(R);

    if (trackSelected) {
      it->second.total_selected->Fill(R);
      it->second.selected_vs_chiSq->SetPoint(it->second.selected_vs_chiSq->GetN(), trackFit.chiSqPerNdf(), R);
    }

    auto sit = it->second.perRPSet_fitted.find(selectedRPs);
    if (sit == it->second.perRPSet_fitted.end()) {
      char buf[10];
      sprintf(buf, "%u: ", id);
      string label = buf;
      label += setToString(selectedRPs);
      sit =
          it->second.perRPSet_fitted.insert(pair<set<unsigned int>, TH1D *>(selectedRPs, newResiduaHist(label.c_str())))
              .first;
    }

    sit->second->Fill(R);

    if (trackSelected) {
      sit = it->second.perRPSet_selected.find(selectedRPs);
      if (sit == it->second.perRPSet_selected.end()) {
        char buf[10];
        sprintf(buf, "%u: ", id);
        string label = buf;
        label += setToString(selectedRPs);
        sit = it->second.perRPSet_selected
                  .insert(pair<set<unsigned int>, TH1D *>(selectedRPs, newResiduaHist(label.c_str())))
                  .first;
      }

      sit->second->Fill(R);
    }
  }
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::buildConstraints(vector<AlignmentConstraint> &constraints) {
  constraints.clear();

  switch (constraintsType) {
    case ctFixedDetectors:
      task.buildFixedDetectorsConstraints(constraints);
      return;

    case ctStandard:
      task.buildStandardConstraints(constraints);
      return;
  }
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::finish() {
  // print statistics
  if (verbosity) {
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("\n>> StraightTrackAlignment::Finish\n");
    printf("\tevents total = %i\n", eventsTotal);
    printf("\tevents fitted = %i\n", eventsFitted);
    printf("\tevents selected = %i\n", eventsSelected);

    printf("\n* events per RP set:\n");
    printf("%30s  %10s%10s\n", "set of RPs", "fitted", "selected");

    for (auto it = fittedTracksPerRPSet.begin(); it != fittedTracksPerRPSet.end(); ++it) {
      const string &label = setToString(it->first);

      auto sit = selectedTracksPerRPSet.find(it->first);
      unsigned long sv = (sit == selectedTracksPerRPSet.end()) ? 0 : sit->second;

      printf("%30s :%10lu%10lu\n", label.c_str(), it->second, sv);
    }

    if (verbosity >= 2) {
      printf("\n* hits per plane:\n");
      for (const auto it : selectedHitsPerPlane) {
        printf("    ");
        printId(it.first);
        printf(" : %u\n", it.second);
      }
    }
  }

  // write diagnostics plots
  saveDiagnostics();

  // run analysis
  for (auto a : algorithms)
    a->analyze();

  // build constraints
  vector<AlignmentConstraint> constraints;
  buildConstraints(constraints);

  // save constraints
  if (taskDataFile)
    taskDataFile->WriteObject(&constraints, "constraints");

  if (verbosity) {
    printf("\n>> StraightTrackAlignment::Finish > %lu constraints built\n", constraints.size());
    for (unsigned int i = 0; i < constraints.size(); i++)
      printf("\t%s\n", constraints[i].name.c_str());
  }

  // solve
  vector<map<unsigned int, AlignmentResult> > results;
  for (auto algorithm : algorithms) {
    TDirectory *dir = nullptr;
    if (taskDataFile && saveIntermediateResults)
      dir = taskDataFile->mkdir((algorithm->getName() + "_data").c_str());

    results.resize(results.size() + 1);
    unsigned int rf = algorithm->solve(constraints, results.back(), dir);

    if (rf)
      throw cms::Exception("PPS") << "The Solve method of `" << algorithm->getName()
                                  << "' algorithm has failed (return value " << rf << ").";
  }

  // print results
  if (verbosity) {
    printf("\n>> StraightTrackAlignment::Finish > Print\n");

    printLineSeparator(results);
    printQuantitiesLine(results);
    printAlgorithmsLine(results);

    signed int prevRPId = -1;

    for (const auto &dit : task.geometry.getSensorMap()) {
      signed int rpId = CTPPSDetId(dit.first).rpId();
      if (rpId != prevRPId)
        printLineSeparator(results);
      prevRPId = rpId;

      printId(dit.first);
      printf(" ║");

      for (unsigned int q = 0; q < task.quantityClasses.size(); q++) {
        for (unsigned int a = 0; a < results.size(); a++) {
          const auto it = results[a].find(dit.first);
          if (it == results[a].end()) {
            if (algorithms[a]->hasErrorEstimate())
              printf("%19s", "----");
            else
              printf("%8s", "----");

            if (a + 1 == results.size())
              printf("║");
            else
              printf("│");

            continue;
          }

          const auto &ac = it->second;
          double v = 0., e = 0.;
          switch (task.quantityClasses[q]) {
            case AlignmentTask::qcShR1:
              v = ac.getShR1();
              e = ac.getShR1Unc();
              break;
            case AlignmentTask::qcShR2:
              v = ac.getShR2();
              e = ac.getShR2Unc();
              break;
            case AlignmentTask::qcShZ:
              v = ac.getShZ();
              e = ac.getShZUnc();
              break;
            case AlignmentTask::qcRotZ:
              v = ac.getRotZ();
              e = ac.getRotZUnc();
              break;
          }

          if (algorithms[a]->hasErrorEstimate())
            printf("%+8.1f ± %7.1f", v * 1E3, e * 1E3);
          else
            printf("%+8.1f", v * 1E3);

          if (a + 1 == results.size())
            printf("║");
          else
            printf("│");
        }
      }

      printf("\n");
    }

    printLineSeparator(results);
    printAlgorithmsLine(results);
    printQuantitiesLine(results);
    printLineSeparator(results);
  }

  // save results as alignment XML files
  for (unsigned int a = 0; a < results.size(); a++) {
    // convert readout-direction corrections to X and Y
    CTPPSRPAlignmentCorrectionsData convertedAlignments;
    for (const auto &p : results[a]) {
      const DetGeometry &d = task.geometry.get(p.first);
      const auto dir1 = d.getDirectionData(1);
      const auto dir2 = d.getDirectionData(2);

      const double det = dir1.dx * dir2.dy - dir1.dy * dir2.dx;
      const double sh_x = (dir2.dy * p.second.getShR1() - dir1.dy * p.second.getShR2()) / det;
      const double sh_y = (-dir2.dx * p.second.getShR1() + dir1.dx * p.second.getShR2()) / det;

      const double sh_x_e =
          sqrt(pow(dir2.dy / det * p.second.getShR1Unc(), 2) + pow(dir1.dy / det * p.second.getShR2Unc(), 2));
      const double sh_y_e =
          sqrt(pow(dir2.dx / det * p.second.getShR1Unc(), 2) + pow(dir1.dx / det * p.second.getShR2Unc(), 2));

      const CTPPSRPAlignmentCorrectionData corr(sh_x,
                                                sh_x_e,
                                                sh_y,
                                                sh_y_e,
                                                p.second.getShZ(),
                                                p.second.getShZUnc(),
                                                0.,
                                                0.,
                                                0.,
                                                0.,
                                                p.second.getRotZ(),
                                                p.second.getRotZUnc());
      convertedAlignments.setSensorCorrection(p.first, corr);
    }

    // write non-cumulative alignments
    if (!fileNamePrefix.empty()) {
      CTPPSRPAlignmentCorrectionsMethods::writeToXML(convertedAlignments,
                                                     fileNamePrefix + algorithms[a]->getName() + ".xml",
                                                     preciseXMLFormat,
                                                     saveXMLUncertainties,
                                                     true,
                                                     true,
                                                     true,
                                                     true);
    }

    // merge alignments
    CTPPSRPAlignmentCorrectionsData cumulativeAlignments;

    cumulativeAlignments.addCorrections(initialAlignments, false, true, true);
    cumulativeAlignments.addCorrections(convertedAlignments, false, true, true);

    // write expanded and factored results
    if (!expandedFileNamePrefix.empty() || !factoredFileNamePrefix.empty()) {
      CTPPSRPAlignmentCorrectionsData expandedAlignments;
      CTPPSRPAlignmentCorrectionsData factoredAlignments;

      if (verbosity)
        printf(">> Factorizing results of %s algorithm\n", algorithms[a]->getName().c_str());

      const bool equalWeights = false;
      factorRPFromSensorCorrections(
          cumulativeAlignments, expandedAlignments, factoredAlignments, task.geometry, equalWeights, verbosity);

      if (!expandedFileNamePrefix.empty()) {
        CTPPSRPAlignmentCorrectionsMethods::writeToXML(expandedAlignments,
                                                       expandedFileNamePrefix + algorithms[a]->getName() + ".xml",
                                                       preciseXMLFormat,
                                                       saveXMLUncertainties,
                                                       true,
                                                       true,
                                                       true,
                                                       true);
      }

      if (!factoredFileNamePrefix.empty()) {
        CTPPSRPAlignmentCorrectionsMethods::writeToXML(factoredAlignments,
                                                       factoredFileNamePrefix + algorithms[a]->getName() + ".xml",
                                                       preciseXMLFormat,
                                                       saveXMLUncertainties,
                                                       true,
                                                       true,
                                                       true,
                                                       true);
      }
    }
  }

  // prepare algorithms for destructions
  for (const auto &algorithm : algorithms)
    algorithm->end();
}

//----------------------------------------------------------------------------------------------------

string StraightTrackAlignment::setToString(const set<unsigned int> &s) {
  unsigned int N = s.size();
  if (N == 0)
    return "empty";

  string str;
  char buf[10];
  unsigned int i = 0;
  for (set<unsigned int>::iterator it = s.begin(); it != s.end(); ++it, ++i) {
    sprintf(buf, "%u", *it);
    str += buf;
    if (i < N - 1)
      str += ", ";
  }

  return str;
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::printN(const char *str, unsigned int N) {
  for (unsigned int i = 0; i < N; i++)
    printf("%s", str);
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::printLineSeparator(const std::vector<std::map<unsigned int, AlignmentResult> > &results) {
  printf("═════════════════════════╬");
  for (unsigned int q = 0; q < task.quantityClasses.size(); q++) {
    for (unsigned int a = 0; a < results.size(); a++) {
      printN("═", algorithms[a]->hasErrorEstimate() ? 18 : 8);
      if (a + 1 != results.size())
        printf("═");
    }
    printf("╬");
  }
  printf("\n");
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::printQuantitiesLine(const std::vector<std::map<unsigned int, AlignmentResult> > &results) {
  printf("                         ║");

  for (unsigned int q = 0; q < task.quantityClasses.size(); q++) {
    unsigned int size = 0;
    for (unsigned int a = 0; a < results.size(); a++)
      size += (algorithms[a]->hasErrorEstimate()) ? 18 : 8;
    size += algorithms.size() - 1;

    const string &tag = task.quantityClassTag(task.quantityClasses[q]);
    unsigned int space = (size - tag.size()) / 2;
    printN(" ", space);
    printf("%s", tag.c_str());
    printN(" ", size - space - tag.size());
    printf("║");
  }
  printf("\n");
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::printAlgorithmsLine(const std::vector<std::map<unsigned int, AlignmentResult> > &results) {
  printf("                         ║");

  for (unsigned int q = 0; q < task.quantityClasses.size(); q++) {
    for (unsigned int a = 0; a < results.size(); a++) {
      printf((algorithms[a]->hasErrorEstimate()) ? "%18s" : "%8s", algorithms[a]->getName().substr(0, 8).c_str());

      if (a + 1 == results.size())
        printf("║");
      else
        printf("│");
    }
  }
  printf("\n");
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::saveDiagnostics() const {
  if (diagnosticsFile.empty())
    return;

  TFile *df = new TFile(diagnosticsFile.c_str(), "recreate");
  if (df->IsZombie())
    throw cms::Exception("PPS") << "Cannot open file `" << diagnosticsFile << "' for writing.";

  if (buildDiagnosticPlots) {
    TDirectory *commonDir = df->mkdir("common");
    gDirectory = commonDir;

    fitNdfHist_fitted->Write();
    fitNdfHist_selected->Write();
    fitAxHist_fitted->Write();
    fitAyHist_fitted->Write();
    fitAxHist_selected->Write();
    fitAyHist_selected->Write();
    fitBxHist_fitted->Write();
    fitByHist_fitted->Write();
    fitBxHist_selected->Write();
    fitByHist_selected->Write();
    fitPHist_fitted->Write();
    fitPHist_selected->Write();

    gDirectory = commonDir->mkdir("plots global");
    globalPlots.write();

    TDirectory *ppsDir = commonDir->mkdir("plots per RP set");
    for (map<set<unsigned int>, RPSetPlots>::const_iterator it = rpSetPlots.begin(); it != rpSetPlots.end(); ++it) {
      gDirectory = ppsDir->mkdir(setToString(it->first).c_str());

      it->second.write();
    }

    TDirectory *resDir = commonDir->mkdir("residuals");
    for (map<unsigned int, ResiduaHistogramSet>::const_iterator it = residuaHistograms.begin();
         it != residuaHistograms.end();
         ++it) {
      char buf[10];
      sprintf(buf, "%u", it->first);
      gDirectory = resDir->mkdir(buf);
      it->second.total_fitted->Write();
      it->second.total_selected->Write();
      it->second.selected_vs_chiSq->Write();

      /*
      gDirectory = gDirectory->mkdir("fitted per RP set");
      for (map< set<unsigned int>, TH1D* >::iterator sit = it->second.perRPSet_fitted.begin();
          sit != it->second.perRPSet_fitted.end(); ++sit)
        sit->second->Write();
      gDirectory->cd("..");
*/

      gDirectory = gDirectory->mkdir("selected per RP set");
      TCanvas *c = new TCanvas;
      c->SetName("alltogether");
      unsigned int idx = 0;
      for (map<set<unsigned int>, TH1D *>::const_iterator sit = it->second.perRPSet_selected.begin();
           sit != it->second.perRPSet_selected.end();
           ++sit, ++idx) {
        sit->second->SetLineColor(idx + 1);
        sit->second->Draw((idx == 0) ? "" : "same");
        sit->second->Write();
      }
      c->Write();
    }
  }

  // save diagnostics of algorithms
  for (vector<AlignmentAlgorithm *>::const_iterator it = algorithms.begin(); it != algorithms.end(); ++it) {
    TDirectory *algDir = df->mkdir((*it)->getName().c_str());
    (*it)->saveDiagnostics(algDir);
  }

  delete df;
}
