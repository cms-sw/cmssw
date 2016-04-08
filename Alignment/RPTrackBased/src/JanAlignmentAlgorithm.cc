/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/RPTrackBased/interface/JanAlignmentAlgorithm.h"
#include "Alignment/RPTrackBased/interface/MatrixTools.h"
#include "Alignment/RPTrackBased/interface/AlignmentTask.h"

#include "TMatrixDSymEigen.h"
#include "TDecompSVD.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH2D.h"

#include <cmath>

//#define DEBUG 1

using namespace std;
using namespace edm;

JanAlignmentAlgorithm::JanAlignmentAlgorithm(const ParameterSet& ps, AlignmentTask *_t) :
  AlignmentAlgorithm(ps, _t),
  Sc(NULL), Mc(NULL)
{
  const ParameterSet& lps = ps.getParameterSet("JanAlignmentAlgorithm");
  weakLimit = lps.getParameter<double>("weakLimit");
  stopOnSingularModes = lps.getParameter<bool>("stopOnSingularModes");
  buildDiagnosticPlots = lps.getParameter<bool>("buildDiagnosticPlots");
}

//----------------------------------------------------------------------------------------------------

JanAlignmentAlgorithm::~JanAlignmentAlgorithm()
{
}

//----------------------------------------------------------------------------------------------------

void JanAlignmentAlgorithm::Begin(const edm::EventSetup&)
{
  // initialize M and S components
  Mc = new TVectorD[task->quantityClasses.size()];
  Sc = new TMatrixD* [task->quantityClasses.size()];
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    unsigned int rows = task->QuantitiesOfClass(task->quantityClasses[i]);

    Mc[i].ResizeTo(rows);
    Mc[i].Zero();

    Sc[i] = new TMatrixD[task->quantityClasses.size()];
    for (unsigned int j = 0; j < task->quantityClasses.size(); j++) {
      unsigned int cols = task->QuantitiesOfClass(task->quantityClasses[j]);
      Sc[i][j].ResizeTo(rows, cols);
      Sc[i][j].Zero();
    }
  }

  // prepare statistics plots
  if (buildDiagnosticPlots) {
    for(AlignmentGeometry::iterator it = task->geometry.begin(); it != task->geometry.end(); ++it) {
      unsigned int id = it->first;
      char buf[50];
      DetStat s;

      sprintf(buf, "%u: m distribution", id);
      s.m_dist = new TH1D(buf, ";u or v   (mm)", 100, -25, 25);

      for (unsigned int c = 0; c < task->quantityClasses.size(); c++) {
        sprintf(buf, "%u: coef, %s", id, task->QuantityClassTag(task->quantityClasses[c]).c_str());
        s.coefHist.push_back(new TH1D(buf, ";coefficient", 100, -2., +2.)); 
        
        sprintf(buf, "%u: R vs. coef, %s", id, task->QuantityClassTag(task->quantityClasses[c]).c_str());
        TGraph *g = new TGraph();
        g->SetName(buf);
        g->SetTitle(";coefficient;residual   (mm)");
        s.resVsCoef.push_back(g); 
      }

      statistics[id] = s;
    } 
  }

  events = 0;
}

//----------------------------------------------------------------------------------------------------

void JanAlignmentAlgorithm::Feed(const HitCollection &selection, const LocalTrackFit &trackFit,
  const LocalTrackFit &extTrackFit)
{
  if (verbosity > 9)
    printf("\n>> JanAlignmentAlgorithm::Feed\n");

  events++;

  // prepare fit - make z0 compatible
  double hax = trackFit.ax;
  double hay = trackFit.ay;
  double hbx = trackFit.bx + trackFit.ax * (task->geometry.z0 - trackFit.z0);
  double hby = trackFit.by + trackFit.ay * (task->geometry.z0 - trackFit.z0);

  // track parameters for Gamma coefficient calculations
  // either hat values or external fit
  double cax = hax, cbx=hbx, cay=hay, cby=hby; 
  if (useExternalFitter) {
    cax = extTrackFit.ax;
    cay = extTrackFit.ay;
    cbx = extTrackFit.bx + extTrackFit.ax * (task->geometry.z0 - extTrackFit.z0);
    cby = extTrackFit.by + extTrackFit.ay * (task->geometry.z0 - extTrackFit.z0);
  }

  // prepare Gamma matrices
  TMatrixD *Ga = new TMatrixD[task->quantityClasses.size()];
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    Ga[i].ResizeTo(selection.size(), Mc[i].GetNrows());
    Ga[i].Zero();
  }

  TMatrixD A(selection.size(), 4);
  TMatrixD Vi(selection.size(), selection.size());
  TVectorD m(selection.size());
  
  set<unsigned int> rpSet;
  if (buildDiagnosticPlots)
    for (HitCollection::const_iterator it = selection.begin(); it != selection.end(); ++it)
      rpSet.insert(it->id/10);

  // fill fit matrix and Gamma matrices
  unsigned int j = 0;
  
  for (HitCollection::const_iterator it = selection.begin(); it != selection.end(); ++it, ++j) {
    unsigned int id = it->id;

    // skip hits that don't have associated geometry record
    auto git = task->geometry.find(id);
    if (git == task->geometry.end())
      continue;

    DetGeometry &d = git->second;

    A(j, 0) = d.z * d.dx;
    A(j, 1) = d.dx;
    A(j, 2) = d.z * d.dy;
    A(j, 3) = d.dy;

    m(j) = it->position + d.s;  // in mm

    Vi(j, j) = 1./it->sigma/it->sigma;

    double C = d.dx, S = d.dy;

    //printf("%i\t%i\t%i\t%.3f\n", j, id, idx, measVec[j]);
    
    if (buildDiagnosticPlots) {
      statistics[id].m_dist->Fill(it->position);
    }

    for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
      switch (task->quantityClasses[i]) {
        case AlignmentTask::qcShR:
          Ga[i][j][d.matrixIndex] = -1.; break;
        case AlignmentTask::qcShZ:
          Ga[i][j][d.matrixIndex] = cax*C + cay*S; break;
        case AlignmentTask::qcRPShZ:
          Ga[i][j][d.rpMatrixIndex] = cax*C + cay*S; break;
        case AlignmentTask::qcRotZ:
          Ga[i][j][d.matrixIndex] = (cax*d.z + cbx - d.sx)*(-S) + (cay*d.z + cby - d.sy)*C; break;
      }

      if (buildDiagnosticPlots) {
        double hx = hax * d.z + hbx;  // in mm
        double hy = hay * d.z + hby;
        double R = m(j) - (hx*C + hy*S);    // (standard) residual
        double c = Ga[i][j][d.matrixIndex];
        DetStat &s = statistics[id];
        s.coefHist[i]->Fill(c);
        s.resVsCoef[i]->SetPoint(s.resVsCoef[i]->GetN(), c, R);

        if (task->quantityClasses[i] == AlignmentTask::qcRotZ) {
          map< set<unsigned int>, ScatterPlot>::iterator it = s.resVsCoefRot_perRPSet.find(rpSet);
          if (it == s.resVsCoefRot_perRPSet.end()) {
            ScatterPlot sp;
            sp.g = new TGraph();
            sp.h = new TH2D("", "", 40, -20., +20., 60, -0.15, +0.15);
            it = s.resVsCoefRot_perRPSet.insert(pair< set<unsigned int>, ScatterPlot>(rpSet, sp)).first;
          }
          it->second.g->SetPoint(it->second.g->GetN(), c, R);
          it->second.h->Fill(c, R);
        }
      }
    }
  }

  // sigma matrix
  TMatrixD AT(TMatrixD::kTransposed, A);
  TMatrixD ATViA(4, 4);
  ATViA = AT * Vi * A;
  TMatrixD ATViAI(ATViA);
  ATViAI = ATViA.Invert();  
  TMatrixD sigma(Vi);
  sigma -= Vi * A * ATViAI * AT * Vi;

  // traspose Gamma matrices
  TMatrixD *GaT = new TMatrixD[task->quantityClasses.size()];
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    GaT[i].ResizeTo(Mc[i].GetNrows(), selection.size());
    GaT[i].Transpose(Ga[i]);
  }

  // normalized residuals
  TVectorD r(selection.size());
  r = sigma * m;

  // increment M
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++)
    Mc[i] += GaT[i] * r;

  // increment S
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++)
    for (unsigned int j = 0; j < task->quantityClasses.size(); j++)
      Sc[i][j] += GaT[i] * sigma * Ga[j];

#ifdef DEBUG
  printf("* checking normalized residuals, selection.size = %u\n", selection.size());
  r.Print(); 

  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    printf("- class %u\n", i);
    TVectorD t(Mc[i].GetNrows());
    for (int j = 0; j < t.GetNrows(); j++)
      t[j] = 1.;
    t.Print();

    Ga[i].Print();

    TVectorD tt(selection.size());
    tt = sigma * Ga[i] * t;

    double ttn = sqrt(tt.Norm2Sqr());
    printf("|tt| = %E\n", ttn);
    if (ttn > 1E-8)
      tt.Print();
  }
#endif

  delete [] Ga;
  delete [] GaT;
}

//----------------------------------------------------------------------------------------------------

vector<SingularMode> JanAlignmentAlgorithm::Analyze()
{
  if (verbosity > 2)
    printf("\n>> JanAlignmentAlgorithm::Analyze\n");

  // calculate full dimension
  unsigned int dim = 0;
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++)
    dim += Mc[i].GetNrows();

  if (verbosity > 2) {
    printf("\tdetectors: %u\n", task->geometry.Detectors());
    printf("\tRPs: %u\n", task->geometry.RPs());
    printf("\tfull dimension: %u\n", dim);
    printf("\tquantity classes: %lu\n", task->quantityClasses.size());
  }

  // build full M
  M.ResizeTo(dim);
  unsigned int offset = 0;
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    M.SetSub(offset, Mc[i]);
    offset += Mc[i].GetNrows();
  }

  // build full S
  S.ResizeTo(dim, dim);
  unsigned int r_offset = 0, c_offset = 0;
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    c_offset = 0;
    unsigned int r_size = 0, c_size = 0;
    for (unsigned int j = 0; j < task->quantityClasses.size(); j++) {
      r_size = Sc[i][j].GetNrows();
      c_size = Sc[i][j].GetNcols();
      TMatrixDSub(S, r_offset, r_offset+r_size-1, c_offset, c_offset+c_size-1) = Sc[i][j];
      c_offset += c_size;
    }
    r_offset += r_size;
  }
 
  // analyze symmetricity
  if (verbosity > 2) {
    double maxDiff = 0., maxElem = 0.;
    for (unsigned int i = 0; i < dim; i++)
      for (unsigned int j = 0; j < dim; j++) {
        double diff = S[i][j] - S[j][i];
        if (fabs(diff) > maxDiff)
          maxDiff = diff;
        if (S[i][j] > maxElem)
          maxElem = S[i][j];
      }

    printf("\n* S matrix:\n\tdimension = %i\n\tmaximum asymmetry: %E\t(ratio to maximum element %E)\n", dim, maxDiff, maxDiff/maxElem);
  }

  // make a symmetric copy
  TMatrixDSym S_sym(dim);
  for (unsigned int j = 0; j < dim; j++)
    for (unsigned int i = 0; i < dim; i++) {
      S_sym[i][j] = S[i][j];
    }
  
  // eigen analysis of S
  TMatrixDSymEigen S_eig(S_sym);
  const TVectorD &S_eigVal_temp = S_eig.GetEigenValues();
  S_eigVal.ResizeTo(S_eigVal_temp.GetNrows());
  S_eigVal = S_eigVal_temp;
  const TMatrixD &S_eigVec_temp = S_eig.GetEigenVectors();
  S_eigVec.ResizeTo(S_eigVec_temp);
  S_eigVec = S_eigVec_temp;

  // identify singular modes
  for (int i = 0; i < S_eigVal.GetNrows(); i++) {
    double nev = S_eigVal[i] / events;
    if (fabs(nev) < singularLimit) {
      SingularMode sM;
      sM.val = S_eigVal[i];
      sM.vec.ResizeTo(dim);
      sM.vec = TMatrixDColumn(S_eigVec, i);
      sM.idx = i;
      singularModes.push_back(sM);
    }
  }
  
#if 0
  // print singular vectors
  if (singularModes.size() > 0) {
    printf("\n* S singular modes\n   | ");
    for (unsigned int i = 0; i < singularModes.size(); i++)
      printf("%+10.3E   ", singularModes[i].val);
    printf("\n-- | ");

    for (unsigned int i = 0; i < singularModes.size(); i++)
      printf("----------   ");
    printf("\n");

    for (unsigned int j = 0; j < dim; j++) {
      printf("%2u | ", j);
      for (unsigned int i = 0; i < singularModes.size(); i++) {
        printf("%+10.3E   ", singularModes[i].vec[j]);
      }
      printf("\n");
    }
  } else
    printf("\n* S has no singular modes\n");
#endif

  return singularModes;
}

//----------------------------------------------------------------------------------------------------

unsigned int JanAlignmentAlgorithm::Solve(const std::vector<AlignmentConstraint> &constraints,
  RPAlignmentCorrections &result, TDirectory *dir)
{
  printf(">> JanAlignmentAlgorithm::Solve\n");
  result.Clear();

  // build C matrix
  unsigned int dim = S.GetNrows();
  TMatrixD C(dim, constraints.size());
  TMatrixD C2(dim, constraints.size());
  for (unsigned int i = 0; i < constraints.size(); i++) {
    unsigned int offset = 0;
    for (unsigned int j = 0; j < task->quantityClasses.size(); j++) {
      const TVectorD &cv = constraints[i].coef.find(task->quantityClasses[j])->second;
      for (int k = 0; k < cv.GetNrows(); k++) {
        C[offset][i] = events * cv[k];
        C2[offset][i] = events * cv[k]*1E3;
        offset++;
      }
    }
  }

#ifdef DEBUG
  printf("\n* constraint matrix\n");
  Print(C);
#endif

  // build E matrix (singular vectors of S as its columns)
  TMatrixD E(S.GetNrows(), singularModes.size());
  for (unsigned int i = 0; i < singularModes.size(); i++)
    for (int j = 0; j < S.GetNrows(); j++)
      E(j, i) = singularModes[i].vec[j];

  // build CS matrix
  TMatrixDSym CS(dim + constraints.size());
  TMatrixDSym CS2(dim + constraints.size());
  CS.Zero();
  CS2.Zero();
  for (unsigned int j = 0; j < dim; j++)
    for (unsigned int i = 0; i < dim; i++) {
      CS[i][j] = S[i][j];
      CS2[i][j] = S[i][j];
    }
  for (unsigned int i = 0; i < constraints.size(); i++)
    for (unsigned int j = 0; j < dim; j++) {
      CS[j][dim + i] = CS[dim + i][j] =  C(j, i);
      CS2[j][dim + i] = CS2[dim + i][j] =  C2(j, i);
  }

  // eigen analysis of CS matrix
  TMatrixDSymEigen CS_eig(CS);
  TVectorD CS_eigVal = CS_eig.GetEigenValues();
  TMatrixD CS_eigVec = CS_eig.GetEigenVectors();

  // check regularity of CS matrix
  printf("\n* eigen values of CS and S matrices (events = %u)\n", events);
  printf("   #          CS    norm. CS               S     norm. S\n");
  unsigned int singularModeCount = 0;
  vector<unsigned int> weakModeIdx;
  for (int i = 0; i < CS_eigVal.GetNrows(); i++) {

    double CS_nev = CS_eigVal[i]/events;
    printf("%4i%+12.2E%+12.2E", i, CS_eigVal[i], CS_nev);
    if (fabs(CS_nev) < singularLimit) {
      singularModeCount++;
      printf(" (S)");
    } else
      if (fabs(CS_nev) < weakLimit) {
        weakModeIdx.push_back(i);
        printf(" (W)");
      } else {
        printf("    ");
      }

    if (i < S_eigVal.GetNrows()) {
      double S_nev = S_eigVal[i]/events;
      printf("%+12.2E%+12.2E", S_eigVal[i], S_nev);
      if (fabs(S_nev) < singularLimit)
        printf(" (S)");
      else
        if (fabs(S_nev) < weakLimit)
          printf(" (W)");
    }

    printf("\n");
  }

  // print weak vectors
  if (weakModeIdx.size() > 0) {
    unsigned int columns = 10;
    unsigned int first = 0;
    
    while (first < weakModeIdx.size()) {
      unsigned int last = first + columns;
      if (last >= weakModeIdx.size())
        last = weakModeIdx.size();

      printf("\n* CS weak modes\n    | ");
      for (unsigned int i = first; i < last; i++)
        printf("%+10.3E   ", CS_eigVal[weakModeIdx[i]]);
      printf("\n--- | ");
  
      for (unsigned int i = first; i < last; i++)
        printf("----------   ");
      printf("\n");
      
      // determine maximum elements
      vector<double> maxs;
      for (unsigned int i = first; i < last; i++) {
        double max = 0;
        for (unsigned int j = 0; j < dim + constraints.size(); j++) {
          double v = fabs(CS_eigVec(weakModeIdx[i], j));
          if (v > max)
            max = v;
        }
        maxs.push_back(max);
      }
  
      for (unsigned int j = 0; j < dim + constraints.size(); j++) {
        printf("%3u | ", j);
        for (unsigned int i = first; i < last; i++) {
          double v = CS_eigVec(weakModeIdx[i], j);
          if (fabs(v)/maxs[i-first] > 1E-3)
            printf("%+10.3E   ", v);
          else
            printf("         .   ");
        }
        printf("\n");
      }

      first = last;
    }
  } else
    printf("\n* CS has no weak modes\n");

  // check the regularity of C^T E
  if (E.GetNcols() == C.GetNcols()) {
    TMatrixD CTE(C, TMatrixD::kTransposeMult, E);
    Print(CTE, "* CTE matrix:");
    const double &det = CTE.Determinant();
    printf("\n* det(CTE) = %E, max(CTE) = %E, det(CTE)/max(CTE) = %E\n\tmax(C) = %E, max(E) = %E, det(CTE)/max(C)/max(E) = %E\n",
        det, CTE.Max(), det/CTE.Max(), C.Max(), E.Max(), det/C.Max()/E.Max());
  } else
    printf(">> JanAlignmentAlgorithm::Solve > WARNING: C matrix has %u, while E matrix %u columns.\n", C.GetNcols(), E.GetNcols());

  // stop if CS is singular
  if (singularModeCount > 0) {
    LogProblem("JanAlignmentAlgorithm") << "\n>> JanAlignmentAlgorithm::Solve > ERROR: There are "
      << singularModeCount << " singular modes in CS matrix.";
    if (stopOnSingularModes)
      return 1;
  }

  // build MV vector
  TVectorD MV(dim + constraints.size());
  for (unsigned int i = 0; i < dim; i++)
    MV[i] = M[i];
  for (unsigned int i = 0; i < constraints.size(); i++)
    MV[dim + i] = events*constraints[i].val;
  
  // perform inversion and solution
  TMatrixD CSI(TMatrixD::kInverted, CS);
  TMatrixD CS2I(TMatrixD::kInverted, CS2);
  TVectorD AL(MV);
  AL = CSI * MV;

  // evaluate error matrix
  TMatrixD S0(S); // new parts full of zeros
  S0.ResizeTo(dim + constraints.size(), dim + constraints.size());
  TMatrixD EM(CS);
  EM = CSI * S0 * CSI;
  
  TMatrixD EM2(CS2);
  EM2 = CS2I * S0 * CS2I;

  TMatrixD EMdiff(EM2 - EM);
  //Print(EMdiff, "EMdiff");
  
  double max1 = -1., max2 = -1., maxDiff = -1.;
  for (int i = 0; i < EMdiff.GetNrows(); i++)
    for (int j = 0; j < EMdiff.GetNcols(); j++) {
      if (maxDiff < fabs(EMdiff(i, j)))
        maxDiff = fabs(EMdiff(i, j));

      if (max1 < fabs(EM(i, j)))
        max1 = fabs(EM(i, j));
      
      if (max2 < fabs(EM2(i, j)))
        max2 = fabs(EM2(i, j));
  }

  printf("EM max = %E, EM2 max = %E, EM2 - EM max = %E\n", max1, max2, maxDiff);

  // tests
  TMatrixD &U = CS_eigVec;
  TMatrixD UT(TMatrixD::kTransposed, U);
  //TMatrixD CSEi(CS);
  //CSEi = UT * CS * U;
  //Print(CSEi, "CSEi");


  TMatrixD EMEi(EM);
  EMEi = UT * EM * U;
  //Print(EMEi, "*EMEi"); 

  double max = -1.;
  for (int i = 0; i < EMEi.GetNrows(); i++)
    for (int j = 0; j < EMEi.GetNcols(); j++)
      if (max < EMEi(i, j))
        max = EMEi(i, j);

  printf("max = %E\n", max);
  
  /*
  double th = max / 100;
  for (int i = 0; i < EMEi.GetNrows(); i++)
    for (int j = 0; j < EMEi.GetNcols(); j++)
      if (th < EMEi(i, j))
        printf("element %u, %u: %E\n", i, j, EMEi(i, j));
  */

  // print lambda values
  printf("\n* Lambda (from the contribution of singular modes to MV)\n");
  for (unsigned int i = 0; i < constraints.size(); i++)
    printf("\t%u (%25s)\t%+10.1E +- %10.1E\n", i, constraints[i].name.c_str(),
        AL[dim+i]*1E3,
        sqrt(EM[i+dim][i+dim])*1E3);
  
  
  // fill results
  unsigned int offset = 0;
  vector<unsigned int> offsets;
  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    offsets.push_back(offset);
    offset += Mc[i].GetNrows();
  }

  for (AlignmentGeometry::const_iterator dit = task->geometry.begin(); dit != task->geometry.end(); ++dit) {
    RPAlignmentCorrection r;

    for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
      unsigned idx = (task->quantityClasses[i] != AlignmentTask::qcRPShZ) ? dit->second.matrixIndex : dit->second.rpMatrixIndex;
      unsigned int fi = offsets[i] + idx;
      double v = AL[fi];
      double e = sqrt(EM[fi][fi]);
      switch (task->quantityClasses[i]) {
        case AlignmentTask::qcShR: r.SetTranslationR(v, e); break;
        case AlignmentTask::qcShZ: r.SetTranslationZ(v, e); break;
        case AlignmentTask::qcRPShZ: r.SetTranslationZ(v, e); break;
        case AlignmentTask::qcRotZ: r.SetRotationZ(v, e); break;
      }
    }

    result.SetSensorCorrection(dit->first, r);
  }

  // save matrices, eigen data, ...
  if (dir) {
    dir->cd();

    S.Write("S");
    S_eigVal.Write("S_eigen_values");
    S_eigVec.Write("S_eigen_vectors");
 
    E.Write("E");
    C.Write("C");

    CS.Write("CS");
    CS_eigVal.Write("CS_eigen_values");
    CS_eigVec.Write("CS_eigen_vectors");

    MV.Write("MV");
    AL.Write("AL");

    S0.Write("S0");
    EM.Write("EM");
  }

  // success
  return 0;
}

//----------------------------------------------------------------------------------------------------

void JanAlignmentAlgorithm::End()
{
  delete [] Mc;

  for (unsigned int i = 0; i < task->quantityClasses.size(); i++) {
    delete [] Sc[i];
  }
  delete [] Sc;
}

//----------------------------------------------------------------------------------------------------

void JanAlignmentAlgorithm::SaveDiagnostics(TDirectory *dir)
{
  if (!buildDiagnosticPlots)
    return;
  
  for (map<unsigned int, DetStat>::iterator it = statistics.begin(); it != statistics.end(); ++it) {
    char buf[50];
    sprintf(buf, "%u", it->first);
    gDirectory = dir->mkdir(buf);

    it->second.m_dist->Write();

    /*
    for (unsigned int c = 0; c < task->quantityClasses.size(); c++) {
      it->second.coefHist[c]->Write();
      it->second.resVsCoef[c]->Write();
    }
    */
    
    gDirectory = gDirectory->mkdir("R vs. rot. coef, per RP set");
    TCanvas *c = new TCanvas; c->SetName("R vs. rot. coef, overlapped");
    TH2D *frame =  new TH2D("frame", "frame", 100, -20., +20., 100, -0.15, +0.15);
    frame->Draw();
    unsigned int idx = 0;
    for (map< set<unsigned int>, ScatterPlot>::iterator iit = it->second.resVsCoefRot_perRPSet.begin(); 
      iit != it->second.resVsCoefRot_perRPSet.end(); ++iit, ++idx) {
      char buf[50];
      bool first = true;
      for (set<unsigned int>::iterator sit = iit->first.begin(); sit != iit->first.end(); ++sit)
        if (first) {
          sprintf(buf, "%u", *sit);
          first = false;
        } else
          sprintf(buf, "%s, %u", buf, *sit);
      iit->second.g->SetTitle(";rotation coefficient   (mm);residual   (mm)");
      iit->second.g->SetMarkerColor(idx+1);
      iit->second.g->SetName(buf);
      iit->second.g->Draw((idx == 0) ? "p" : "p");
      iit->second.g->Write();
      
      sprintf(buf, "%s (hist)", buf);
      iit->second.h->SetName(buf);
      iit->second.h->SetTitle(";rotation coefficient   (mm);residual   (mm)");
      iit->second.h->Write();
    }
    gDirectory->cd("..");
    c->Write();
  } 
}

