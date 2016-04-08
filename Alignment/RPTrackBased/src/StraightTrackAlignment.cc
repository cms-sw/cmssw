/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTotemRP/RPRecoDataFormats/interface/RPTrackCandidate.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPTrackCandidateCollection.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "CondFormats/DataRecord/interface/VeryForwardRealGeometryRecord.h"
#include "Alignment/RPRecords/interface/RPRealAlignmentRecord.h"
#include "Alignment/RPTrackBased/interface/IdealResult.h"
#include "Alignment/RPTrackBased/interface/JanAlignmentAlgorithm.h"
#include "Alignment/RPTrackBased/interface/MillepedeAlgorithm.h"
#include "Alignment/RPTrackBased/interface/StraightTrackAlignment.h"

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

TH1D* StraightTrackAlignment::NewResiduaHist(const char *name)
{
  return new TH1D(name, ";residual   (mm)", 1000, -0.2, +0.2); // in mm
}

//----------------------------------------------------------------------------------------------------

StraightTrackAlignment::ChiSqHistograms::ChiSqHistograms(const string &name)
{
  lin_fitted = new TH1D((name + ", lin, all").c_str(), ";#chi^{2}/ndf;", 5000, 0., 500.);
  lin_selected = new TH1D((name + ", lin, selected").c_str(), ";#chi^{2}/ndf;", 5000, 0., 500.);
  log_fitted = new TH1D((name + ", log, all").c_str(), ";log_{10}(#chi^{2}/ndf);", 700, -1., 6.);
  log_selected = new TH1D((name + ", log, selected").c_str(), ";log_{10}(#chi^{2}/ndf);", 700, -1., 6.);
}

//----------------------------------------------------------------------------------------------------

TGraph* NewGraph(const string &name, const string &title)
{
  TGraph *g = new TGraph();
  g->SetName(name.c_str());
  g->SetTitle(title.c_str());
  return g;
}

//----------------------------------------------------------------------------------------------------

StraightTrackAlignment::StraightTrackAlignment(const ParameterSet& ps) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
  factorizationVerbosity(ps.getUntrackedParameter<unsigned int>("factorizationVerbosity", 0)),
  tagRecognizedPatterns(ps.getParameter<edm::InputTag>("tagRecognizedPatterns")),

  RPIds(ps.getParameter< vector<unsigned int> >("RPIds")),
  excludePlanes(ps.getParameter< vector<unsigned int> >("excludePlanes")),
  z0(ps.getParameter<double>("z0")),

  useExternalFitter(ps.getParameter<bool>("useExternalFitter")),
  tagExternalFit(ps.getParameter<edm::InputTag>("tagExternalFit")),

  maxEvents(ps.getParameter<unsigned int>("maxEvents")),

  removeImpossible(ps.getParameter<bool>("removeImpossible")),
  requireNumberOfUnits(ps.getParameter<unsigned int>("requireNumberOfUnits")),
  requireAtLeast3PotsInOverlap(ps.getParameter<bool>("requireAtLeast3PotsInOverlap")),
  requireOverlap(ps.getParameter<bool>("requireOverlap")),
  cutOnChiSqPerNdf(ps.getParameter<bool>("cutOnChiSqPerNdf")),
  chiSqPerNdfCut(ps.getParameter<double>("chiSqPerNdfCut")),
  runsWithoutHorizontalRPs(ps.getParameter< vector<unsigned int> >("runsWithoutHorizontalRPs")),

  fileNamePrefix(ps.getParameter<string>("fileNamePrefix")),
  cumulativeFileNamePrefix(ps.getParameter<string>("cumulativeFileNamePrefix")),
  expandedFileNamePrefix(ps.getParameter<string>("expandedFileNamePrefix")),
  factoredFileNamePrefix(ps.getParameter<string>("factoredFileNamePrefix")),
  preciseXMLFormat(ps.getParameter<bool>("preciseXMLFormat")),

  saveIntermediateResults(ps.getParameter<bool>("saveIntermediateResults")),
  taskDataFileName(ps.getParameter<string>("taskDataFileName")),
  taskDataFile(NULL),

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
  fitAxVsAyGraph_fitted(NewGraph("ax vs. ay_fitted", ";a_{x}   (rad);a_{y}   (rad)")), 
  fitAxVsAyGraph_selected(NewGraph("ax vs. ay_selected", ";a_{x}   (rad);a_{y}   (rad)")), 
  fitBxVsByGraph_fitted(NewGraph("bx vs. by_fitted", ";b_{x}   (mm);b_{y}   (mm)")), 
  fitBxVsByGraph_selected(NewGraph("bx vs. by_selected", ";b_{x}   (mm);b_{y}   (mm)")), 
  chiSqHists("chi^2/ndf global")
{
  // open task data file
  if (!taskDataFileName.empty())
    taskDataFile = new TFile(taskDataFileName.c_str(), "recreate");

  // instantiate algorithm objects
  // (and save them)
  vector<string> alNames(ps.getParameter< vector<string> >("algorithms"));
  for (unsigned int i = 0; i < alNames.size(); i++) {
    AlignmentAlgorithm *a = NULL;

    if (alNames[i].compare("Ideal") == 0)
      a = new IdealResult(ps, &task);

    if (alNames[i].compare("Jan") == 0) {
      JanAlignmentAlgorithm *jaa = new JanAlignmentAlgorithm(ps, &task);
      /*
      // TODO: this causes a crash
      if (taskDataFile && jaa)
        taskDataFile->WriteObject(jaa, "Jan");
      */
      a = jaa;
    }

    if (alNames[i].compare("Millepede") == 0)
      a = new MillepedeAlgorithm(ps, &task);

    if (a)
      algorithms.push_back(a);
    else
      throw cms::Exception("StraightTrackAlignment") << "Unknown alignment algorithm `" << alNames[i] << "'.";
  }

  // get constraints type
  string ct = ps.getParameter<string>("constraintsType");
  if (ct.compare("homogeneous") == 0) constraintsType = ctHomogeneous;
  else
    if (ct.compare("fixedDetectors") == 0) constraintsType = ctFixedDetectors;
    else
      if (ct.compare("dynamic") == 0) constraintsType = ctDynamic;
      else
        if (ct.compare("final") == 0) constraintsType = ctFinal;
        else
          throw cms::Exception("StraightTrackAlignment") << "Unknown constraints type `" << ct << "'.";
}

//----------------------------------------------------------------------------------------------------

StraightTrackAlignment::~StraightTrackAlignment()
{
  if (taskDataFile);
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
  delete fitAxVsAyGraph_fitted;
  delete fitAxVsAyGraph_selected;
  delete fitBxVsByGraph_fitted;
  delete fitBxVsByGraph_selected;
  
  delete chiSqHists.lin_fitted;
  delete chiSqHists.log_fitted;
  delete chiSqHists.lin_selected;
  delete chiSqHists.log_selected;

  for (map< set<unsigned int>, ChiSqHistograms >::iterator it = chiSqHists_perRP.begin(); it != chiSqHists_perRP.end(); ++it) {
    delete it->second.lin_fitted;
    delete it->second.log_fitted;
    delete it->second.lin_selected;
    delete it->second.log_selected;
  }

  for (map<unsigned int, ResiduaHistogramSet>::iterator it = residuaHistograms.begin(); it != residuaHistograms.end(); ++it) {
    delete it->second.total_fitted;
    delete it->second.total_selected;
    delete it->second.selected_vs_chiSq;
    for (map< set<unsigned int>, TH1D* >::iterator sit = it->second.perRPSet_fitted.begin();
          sit != it->second.perRPSet_fitted.end(); ++sit)
      delete sit->second;
    for (map< set<unsigned int>, TH1D* >::iterator sit = it->second.perRPSet_selected.begin();
          sit != it->second.perRPSet_selected.end(); ++sit)
      delete sit->second;
  }
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::Begin(const EventSetup &es)
{
  printf(">> StraightTrackAlignment::Begin\n");

  // reset counters
  eventsTotal = 0;
  eventsFitted = 0;
  eventsSelected = 0;
  fittedTracksPerRPSet.clear();
  selectedTracksPerRPSet.clear();
  
  // prepare geometry (in fact, this should be done whenever es gets changed)
  ESHandle<TotemRPGeometry> geomH;
  es.get<VeryForwardRealGeometryRecord>().get(geomH);
  task.BuildGeometry(RPIds, excludePlanes, geomH.product(), z0, task.geometry);

  // print geometry info
  if (verbosity > 1) {
    printf("> alignment geometry\n\t[matrix index/RP matrix index]\n");
    for (AlignmentGeometry::iterator it = task.geometry.begin(); it != task.geometry.end(); ++it)
      printf("%4u [%2u/%u] z = %+10.4f mm │ readout dir. %+.3f, %+.3f │ shift x = %+.3f mm, y = %+.3f mm, r = %+.3f mm │ %s-det\n",
          it->first, it->second.matrixIndex, it->second.rpMatrixIndex, it->second.z, it->second.dx,
          it->second.dy, it->second.sx, it->second.sy, it->second.s, (it->second.isU) ? "U" : "V");
  }
  
  // save task (including geometry) and fitter
  if (taskDataFile) {
    taskDataFile->WriteObject(&task, "task");
    taskDataFile->WriteObject(&fitter, "fitter");  
  }

  // initiate the algorithms
  for (vector<AlignmentAlgorithm *>::iterator it = algorithms.begin(); it != algorithms.end(); ++it)
    (*it)->Begin(es);
  
  // get initial alignments
  try {
    ESHandle<RPAlignmentCorrections> h;
    es.get<RPRealAlignmentRecord>().get(h);
    initialAlignments = *h;
  }
  catch (...) {}
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::ProcessEvent(const Event& event, const EventSetup&)
{
  if (verbosity > 9)
    printf("\n---------- StraightTrackAlignment::ProcessEvent > event %llu\n", event.id().event());
  
  // -------------------- STEP 1: get hits from selected RPs
  Handle< RPTrackCandidateCollection > trackColl;
  event.getByLabel(tagRecognizedPatterns, trackColl);

  bool skipHorRP = ( find(runsWithoutHorizontalRPs.begin(), runsWithoutHorizontalRPs.end(),
    event.id().run()/10000) != runsWithoutHorizontalRPs.end() );

  HitCollection selection;
  for (RPTrackCandidateCollection::const_iterator it = trackColl->begin(); it != trackColl->end(); ++it) {
    // skip non fittable candidates
    if (!it->second.Fittable())
      continue;

    // skip if RP not selected by user
    if (find(RPIds.begin(), RPIds.end(), it->first) == RPIds.end())
      continue;

    // skip horizontal RPs
    unsigned int rpNum = it->first % 10;
    if (skipHorRP && (rpNum == 2 || rpNum == 3))
      continue;
    
    const vector<TotemRPRecHit> &hits = it->second.TrackRecoHits();
    for (unsigned int i = 0; i < hits.size(); i++)
      selection.push_back(hits[i]);
  }

  eventsTotal++;

  if (selection.empty())
    return;

  // -------------------- STEP 2: fit + outlier rejection

  LocalTrackFit trackFit;
  if (! fitter.Fit(selection, task.geometry, trackFit))
     return;

  LocalTrackFit extTrackFit;
  if (useExternalFitter) {
    Handle< LocalTrackFit > hTrackFit;
    vector< Handle< LocalTrackFit> > hTrackFitList;
    event.getByLabel(tagExternalFit, hTrackFit);
    extTrackFit = *hTrackFit;
  }
  
  set<unsigned int> selectedRPs;
  for (const auto &hit : selection)
    selectedRPs.insert(hit.id/10);

  eventsFitted++;
  fittedTracksPerRPSet[selectedRPs]++;

  // -------------------- STEP 3: quality checks

  bool top = false, bottom = false, horizontal = false;
  unordered_set<unsigned int> units;
  for (const auto &rp : selectedRPs)
  {
    unsigned idx = rp % 10;
    unsigned unit = rp / 10;
    if (idx == 0 || idx == 4) top = true;
    if (idx == 1 || idx == 5) bottom = true;
    if (idx == 2 || idx == 3) horizontal = true;

    units.insert(unit);
  }
  bool overlap = (top && horizontal) || (bottom && horizontal);

  bool selected = true;

  // impossible signature
  if (removeImpossible && top && bottom)
    selected = false;

  // cleanliness cuts
  if (units.size() < requireNumberOfUnits)
    selected = false;
  
  if (requireOverlap && !overlap)
    selected = false;

  if (requireAtLeast3PotsInOverlap && overlap && selectedRPs.size() < 3)
    selected = false;

  // too bad chisq
  if (cutOnChiSqPerNdf && trackFit.ChiSqPerNdf() > chiSqPerNdfCut)
    selected = false;

  UpdateDiagnosticHistograms(selection, selectedRPs, trackFit, selected);

  if (verbosity > 5)
    printf("* SELECTED: %u\n", selected);

  if (!selected)
    return;
  
  eventsSelected++;
  selectedTracksPerRPSet[selectedRPs]++;
  
  // -------------------- STEP 4: FEED ALGORITHMS

  for (vector<AlignmentAlgorithm *>::iterator it = algorithms.begin(); it != algorithms.end(); ++it)
    (*it)->Feed(selection, trackFit, extTrackFit);

  // -------------------- STEP 5: ENOUGH TRACKS?

  if (eventsSelected == maxEvents)
      throw "Number of tracks processed reached maximum";
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::UpdateDiagnosticHistograms(const HitCollection &selection, 
      const set<unsigned int> &selectedRPs, const LocalTrackFit &trackFit, bool trackSelected)
{
  if (!buildDiagnosticPlots)
    return;

  fitNdfHist_fitted->Fill(trackFit.ndf);
  fitPHist_fitted->Fill(trackFit.PValue());
  fitAxHist_fitted->Fill(trackFit.ax);
  fitAyHist_fitted->Fill(trackFit.ay);
  fitBxHist_fitted->Fill(trackFit.bx);
  fitByHist_fitted->Fill(trackFit.by);
  fitAxVsAyGraph_fitted->SetPoint(fitAxVsAyGraph_fitted->GetN(), trackFit.ax, trackFit.ay);
  fitBxVsByGraph_fitted->SetPoint(fitBxVsByGraph_fitted->GetN(), trackFit.bx, trackFit.by);
  chiSqHists.lin_fitted->Fill(trackFit.ChiSqPerNdf());
  chiSqHists.log_fitted->Fill(log10(trackFit.ChiSqPerNdf()));

  if (trackSelected) {
    fitNdfHist_selected->Fill(trackFit.ndf);
    fitPHist_selected->Fill(trackFit.PValue());
    fitAxHist_selected->Fill(trackFit.ax);
    fitAyHist_selected->Fill(trackFit.ay);
    fitBxHist_selected->Fill(trackFit.bx);
    fitByHist_selected->Fill(trackFit.by);
    fitAxVsAyGraph_selected->SetPoint(fitAxVsAyGraph_selected->GetN(), trackFit.ax, trackFit.ay);
    fitBxVsByGraph_selected->SetPoint(fitBxVsByGraph_selected->GetN(), trackFit.bx, trackFit.by);
    chiSqHists.lin_selected->Fill(trackFit.ChiSqPerNdf());
    chiSqHists.log_selected->Fill(log10(trackFit.ChiSqPerNdf()));
  }

  map< set<unsigned int>, ChiSqHistograms >::iterator it = chiSqHists_perRP.find(selectedRPs);
  if (it == chiSqHists_perRP.end())
    it = chiSqHists_perRP.insert(pair< set<unsigned int>, ChiSqHistograms >(selectedRPs, ChiSqHistograms(SetToString(selectedRPs)))).first;
  it->second.lin_fitted->Fill(trackFit.ChiSqPerNdf());
  it->second.log_fitted->Fill(log10(trackFit.ChiSqPerNdf()));
  if (trackSelected) { 
    it->second.lin_selected->Fill(trackFit.ChiSqPerNdf());
    it->second.log_selected->Fill(log10(trackFit.ChiSqPerNdf()));
  }
  
  for (HitCollection::const_iterator hitCollectionIterator = selection.begin(); hitCollectionIterator != selection.end(); ++hitCollectionIterator) {
    unsigned int id = hitCollectionIterator->id;

    AlignmentGeometry::iterator dit = task.geometry.find(id);
    if (dit == task.geometry.end())
      continue;
    DetGeometry &d = dit->second;

    double m = hitCollectionIterator->position + d.s;
    double x = trackFit.ax * d.z + trackFit.bx;
    double y = trackFit.ay * d.z + trackFit.by;
    double f = x*d.dx + y*d.dy;
    double R = m - f;

    map<unsigned int, ResiduaHistogramSet>::iterator it = residuaHistograms.find(id);
    if (it == residuaHistograms.end()) {
      it = residuaHistograms.insert(pair<unsigned int, ResiduaHistogramSet>(id, ResiduaHistogramSet())).first;
      char buf[30];
      sprintf(buf, "%u: total_fitted", id); it->second.total_fitted = NewResiduaHist(buf);
      sprintf(buf, "%u: total_selected", id); it->second.total_selected = NewResiduaHist(buf);
      it->second.selected_vs_chiSq = new TGraph();
      sprintf(buf, "%u: selected_vs_chiSq", id);
      it->second.selected_vs_chiSq->SetName(buf);
    }
    it->second.total_fitted->Fill(R);
    if (trackSelected) {
      it->second.total_selected->Fill(R);
      it->second.selected_vs_chiSq->SetPoint(it->second.selected_vs_chiSq->GetN(), trackFit.ChiSqPerNdf(), R);
    }

    map< set<unsigned int>, TH1D* >::iterator sit = it->second.perRPSet_fitted.find(selectedRPs);
    if (sit == it->second.perRPSet_fitted.end()) {
      char buf[10];
      sprintf(buf, "%u: ", id);
      string label = buf;
      label += SetToString(selectedRPs);
      sit = it->second.perRPSet_fitted.insert(pair< set<unsigned int>, TH1D* >(selectedRPs, NewResiduaHist(label.c_str()))).first;
    }
    sit->second->Fill(R);
    
    if (trackSelected) {
      sit = it->second.perRPSet_selected.find(selectedRPs);
      if (sit == it->second.perRPSet_selected.end()) {
        char buf[10];
        sprintf(buf, "%u: ", id);
        string label = buf;
        label += SetToString(selectedRPs);
        sit = it->second.perRPSet_selected.insert(pair< set<unsigned int>, TH1D* >(selectedRPs, NewResiduaHist(label.c_str()))).first;
      }
      sit->second->Fill(R);
    }
  }
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::BuildStandardConstraints(vector<AlignmentConstraint> &constraints)
{
  constraints.clear();

  switch (constraintsType) {
    case ctHomogeneous:
      task.BuildHomogeneousConstraints(constraints);
      return;
    case ctFixedDetectors:
      task.BuildFixedDetectorsConstraints(constraints);
      return;
    case ctDynamic:
      return;
    case ctFinal:
      task.BuildOfficialConstraints(constraints);
      return;
  }
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::Finish()
{
  // print statistics
  if (verbosity) {
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("\n>> StraightTrackAlignment::Finish\n");
    printf("\tevents total = %lu\n", eventsTotal);
    printf("\tevents fitted = %lu\n", eventsFitted);
    printf("\tevents selected = %lu\n", eventsSelected);
    printf("\n%30s  %10s%10s\n", "set of RPs", "fitted", "selected");
    for (map< set<unsigned int>, unsigned long >::iterator it = fittedTracksPerRPSet.begin();
        it != fittedTracksPerRPSet.end(); ++it) {
      const string &label = SetToString(it->first);

      map< set<unsigned int>, unsigned long >::iterator sit = selectedTracksPerRPSet.find(it->first);
      unsigned long sv = (sit == selectedTracksPerRPSet.end()) ? 0 : sit->second;

      printf("%30s :%10lu%10lu\n", label.c_str(), it->second, sv);
    }
  }

  // write diagnostics plots
  SaveDiagnostics();

  // run analysis
  for (vector<AlignmentAlgorithm *>::iterator it = algorithms.begin(); it != algorithms.end(); ++it)
    (*it)->Analyze();

  // build constraints
  vector<AlignmentConstraint> constraints;
  if (constraintsType == ctDynamic) {
    // TODO
    printf(">> StraightTrackAlignment::Finish > Sorry, dynamic constraints not yet implemented.\n");
  } else {
    BuildStandardConstraints(constraints);
  }

  // save constraints
  if (taskDataFile)
    taskDataFile->WriteObject(&constraints, "constraints");  
  
  printf("\n>> StraightTrackAlignment::Finish > %lu constraints built\n", constraints.size());
  for (unsigned int i = 0; i < constraints.size(); i++) {
    printf("\t%25s, qc = %i, extended = %i\n", constraints[i].name.c_str(), constraints[i].forClass, constraints[i].extended);
  }

  // solve
  vector<RPAlignmentCorrections> results;
  for (vector<AlignmentAlgorithm *>::iterator it = algorithms.begin(); it != algorithms.end(); ++it) {
    TDirectory *dir = NULL;
    if (taskDataFile && saveIntermediateResults)
      dir = taskDataFile->mkdir(((*it)->GetName() + "_data").c_str());

    results.resize(results.size() + 1);
    unsigned int rf = (*it)->Solve(constraints, results.back(), dir);

    if (rf)
      throw cms::Exception("StraightTrackAlignment") << "The Solve method of `" << (*it)->GetName() 
        << "' algorithm has failed (return value " << rf << ").";
  }

  // print
  printf("\n>> StraightTrackAlignment::Finish > Print\n");

  PrintLineSeparator(results);
  PrintQuantitiesLine(results);
  PrintAlgorithmsLine(results);

  for (AlignmentGeometry::const_iterator dit = task.geometry.begin(); dit != task.geometry.end(); ++dit) {
    //  ═ ║ 
    if (dit->first % 10 == 0)
      PrintLineSeparator(results);

    printf("%4u ║", dit->first);

    for (unsigned int q = 0; q < task.quantityClasses.size(); q++) {
      for (unsigned int a = 0; a < results.size(); a++) {
        RPAlignmentCorrections::mapType::const_iterator it = results[a].sensors.find(dit->first);
        if (it == results[a].sensors.end()) {
          if (algorithms[a]->HasErrorEstimate())
            printf("%18s", "----│");
          else
            printf("%8s", "----│");
          continue;
        }

        const RPAlignmentCorrection &ac = it->second;
        double v = 0., e = 0.;
        switch (task.quantityClasses[q]) {
          case AlignmentTask::qcShR: v = ac.sh_r();   e = ac.sh_r_e(); break;
          case AlignmentTask::qcShZ:
          case AlignmentTask::qcRPShZ: v = ac.sh_z(); e = ac.sh_z_e(); break;
          case AlignmentTask::qcRotZ: v = ac.rot_z(); e = ac.rot_z_e(); break;
        }

        if (algorithms[a]->HasErrorEstimate())
          printf("%+8.1f ± %7.1f", v*1E3, e*1E3);
        else
          printf("%+8.1f", v*1E3);

        if (a + 1 == results.size())
          printf("║");
        else
          printf("│");
      }
    }

    printf("\n");
  }
  
  PrintLineSeparator(results);
  PrintAlgorithmsLine(results);
  PrintQuantitiesLine(results);
  PrintLineSeparator(results);

  // save results
  for (unsigned int a = 0; a < results.size(); a++) {
    // convert readout corrections to X and Y
    for (RPAlignmentCorrections::mapType::iterator it = results[a].sensors.begin();
        it != results[a].sensors.end(); ++it) {
      DetGeometry &d = task.geometry[it->first];
      double cos = d.dx, sin = d.dy;
      it->second.ReadoutTranslationToXY(cos, sin);
    }

    // write non-cumulative results
    if (!fileNamePrefix.empty())
      results[a].WriteXMLFile(fileNamePrefix + algorithms[a]->GetName() + ".xml",
        preciseXMLFormat, algorithms[a]->HasErrorEstimate());

    // merge alignments
    RPAlignmentCorrections cumulativeAlignments;
    cumulativeAlignments.AddCorrections(initialAlignments, false);
    cumulativeAlignments.AddCorrections(results[a], false, task.resolveShR,
      task.resolveShZ || task.resolveRPShZ, task.resolveRotZ);

    // synchronize XY and readout shifts, normalize z rotations
    for (RPAlignmentCorrections::mapType::iterator it = cumulativeAlignments.sensors.begin(); 
        it != cumulativeAlignments.sensors.end(); ++it) {
      DetGeometry &d = task.geometry[it->first];
      double cos = d.dx, sin = d.dy;
      it->second.XYTranslationToReadout(cos, sin);
      it->second.NormalizeRotationZ();
    }

    // write cumulative results
    if (!cumulativeFileNamePrefix.empty())
      cumulativeAlignments.WriteXMLFile(cumulativeFileNamePrefix + algorithms[a]->GetName() + ".xml",
        preciseXMLFormat, algorithms[a]->HasErrorEstimate());

    // write expanded and factored results
    if (!expandedFileNamePrefix.empty() || !factoredFileNamePrefix.empty()) {
      RPAlignmentCorrections expandedAlignments;
      RPAlignmentCorrections factoredAlignments;

      if (factorizationVerbosity)
        printf(">> Factorizing results of %s algorithm\n", algorithms[a]->GetName().c_str());
      
      cumulativeAlignments.FactorRPFromSensorCorrections(expandedAlignments, factoredAlignments,
        task.geometry, factorizationVerbosity);

      if (!expandedFileNamePrefix.empty())
        expandedAlignments.WriteXMLFile(expandedFileNamePrefix + algorithms[a]->GetName() + ".xml",
          preciseXMLFormat, algorithms[a]->HasErrorEstimate());

      if (!factoredFileNamePrefix.empty())
        factoredAlignments.WriteXMLFile(factoredFileNamePrefix + algorithms[a]->GetName() + ".xml",
          preciseXMLFormat, algorithms[a]->HasErrorEstimate());
    }
  }
  
  // prepare algorithms for destructions
  for (vector<AlignmentAlgorithm *>::iterator it = algorithms.begin(); it != algorithms.end(); ++it)
    (*it)->End();
}

//----------------------------------------------------------------------------------------------------

string StraightTrackAlignment::SetToString(const set<unsigned int> &s)
{
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

void StraightTrackAlignment::PrintN(const char *str, unsigned int N)
{
  for (unsigned int i = 0; i < N; i++)
    printf("%s", str);
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::PrintLineSeparator(const std::vector<RPAlignmentCorrections> &results)
{
  printf("═════╬");
  for (unsigned int q = 0; q < task.quantityClasses.size(); q++) {
    for (unsigned int a = 0; a < results.size(); a++) {
      PrintN("═", algorithms[a]->HasErrorEstimate() ? 18 : 8);
      if (a + 1 != results.size())
        printf("═");
    }
    printf("╬");
  }
  printf("\n");
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::PrintQuantitiesLine(const std::vector<RPAlignmentCorrections> &results)
{
  printf("     ║");

  for (unsigned int q = 0; q < task.quantityClasses.size(); q++) {

    unsigned int size = 0;
    for (unsigned int a = 0; a < results.size(); a++)
      size += (algorithms[a]->HasErrorEstimate()) ? 18 : 8;
    size += algorithms.size() - 1; 

    const string &tag = AlignmentTask::QuantityClassTag(task.quantityClasses[q]);
    unsigned int space = (size - tag.size())/2;
    PrintN(" ", space);
    printf("%s", tag.c_str());
    PrintN(" ", size - space - tag.size());
    printf("║");
  }
  printf("\n");
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::PrintAlgorithmsLine(const std::vector<RPAlignmentCorrections> &results)
{
  printf("     ║");

  for (unsigned int q = 0; q < task.quantityClasses.size(); q++)
    for (unsigned int a = 0; a < results.size(); a++) {
      printf((algorithms[a]->HasErrorEstimate()) ? "%18s" : "%8s", algorithms[a]->GetName().substr(0, 8).c_str());

      if (a + 1 == results.size())
        printf("║");
      else
        printf("│");
    }
  printf("\n");
}

//----------------------------------------------------------------------------------------------------

void StraightTrackAlignment::SaveDiagnostics() const
{
  if (diagnosticsFile.empty())
    return;

  TFile *df = new TFile(diagnosticsFile.c_str(), "recreate");
  if (df->IsZombie())
    throw cms::Exception("StraightTrackAlignment::SaveDiagnostics") << "Cannot open file `" << 
      diagnosticsFile << "' for writing.";

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
    fitAxVsAyGraph_fitted->Write();
    fitAxVsAyGraph_selected->Write();
    fitBxVsByGraph_fitted->Write();
    fitBxVsByGraph_selected->Write();
    fitPHist_fitted->Write();
    fitPHist_selected->Write();
    chiSqHists.lin_fitted->Write();
    chiSqHists.log_fitted->Write();
    chiSqHists.lin_selected->Write();
    chiSqHists.log_selected->Write();

    TDirectory *chiDir = commonDir->mkdir("chi^2 per RP set");
    for (map< set<unsigned int>, ChiSqHistograms >::const_iterator it = chiSqHists_perRP.begin(); it != chiSqHists_perRP.end(); ++it) {
      gDirectory = chiDir->mkdir(SetToString(it->first).c_str());
      it->second.lin_fitted->Write("lin_fitted");
      it->second.log_fitted->Write("log_fitted");
      it->second.lin_selected->Write("lin_selected");
      it->second.log_selected->Write("log_selected");
    } 

    TDirectory *resDir = commonDir->mkdir("residuals");
    for (map<unsigned int, ResiduaHistogramSet>::const_iterator it = residuaHistograms.begin(); it != residuaHistograms.end(); ++it) {
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
      TCanvas *c = new TCanvas; c->SetName("alltogether");
      unsigned int idx = 0;
      for (map< set<unsigned int>, TH1D* >::const_iterator sit = it->second.perRPSet_selected.begin();
          sit != it->second.perRPSet_selected.end(); ++sit, ++idx) {
        sit->second->SetLineColor(idx+1);
        sit->second->Draw((idx == 0) ? "" : "same");
        sit->second->Write();
      }
      c->Write();
    }
  }

  // save diagnostics of algorithms
  for (vector<AlignmentAlgorithm *>::const_iterator it = algorithms.begin(); it != algorithms.end(); ++it) {
    TDirectory *algDir = df->mkdir((*it)->GetName().c_str());
    (*it)->SaveDiagnostics(algDir);
  }
  
  delete df;
}

