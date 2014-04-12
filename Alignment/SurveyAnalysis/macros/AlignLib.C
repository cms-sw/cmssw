#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TList.h"
#include "TMatrixD.h"
#include "TNtuple.h"
#include "TStyle.h"
#include "TText.h"
#include "TTree.h"
#include "TVectorD.h"

#include "NumbersAndNames.C"

#if !defined(__CINT__) && !defined(__MAKECINT__)
//#include "Alignment/CommonAlignment/src/AlignableObjectId.cc"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#endif

const int nPar = 6;

void setBranch(TTree* t, double p[], double r[])
{
  t->SetBranchStatus("*", 0); // disable all branches
  t->SetBranchStatus("Pos");  // enable Pos branch
  t->SetBranchStatus("Rot");  // enable Rot branch

  t->SetBranchAddress("Pos", p);
  t->SetBranchAddress("Rot", r);
}

void writeShifts(std::string path)
{
  TFile ft((path + "IOTruePositions.root"      ).c_str());
  TFile f0((path + "IOMisalignedPositions.root").c_str());
  TFile fp((path + "IOAlignedPositions.root"   ).c_str());

  TTree* tt = (TTree*)ft.Get("AlignablesOrgPos_1");

  const TList* keys = fp.GetListOfKeys();

  const std::string key0Name = keys->At(0)->GetName();

  const bool iter0Exist = ('0' == *(key0Name.end() - 1));

  const int nIteration = iter0Exist ? keys->GetSize() - 1 : keys->GetSize();
  const int nAlignable = tt->GetEntries();

  std::vector<TVectorD> post(nAlignable, TVectorD(3));
  std::vector<TMatrixD> rott(nAlignable, TMatrixD(3, 3));

  double pos[3];
  double rot[9];

  setBranch(tt, pos, rot);

  for (int n = 0; n < nAlignable; ++n)
  {
    tt->GetEntry(n);

    for (int p = 0; p < 3; ++p) post[n][p] = pos[p];
    for (int p = 0; p < 9; ++p) rott[n](p / 3, p % 3) = rot[p];
  }

  std::vector<TTree*> trees(nIteration + 1);

  trees[0] = (TTree*)f0.Get("AlignablesAbsPos_1");

  for (int i = 1; i <= nIteration; ++i)
  {
    trees[i] = (TTree*)fp.Get(keys->At(iter0Exist ? i : i - 1)->GetName());
  }

  TFile fout((path + "shifts.root").c_str(), "RECREATE");

  std::vector<TNtuple*> tuples(nIteration + 1);

  for (int i = 0; i <= nIteration; ++i)
  {
//     if (trees[i]->GetEntriesFast() != nAlignable)
//     {
//       std::cout << "Unmatched number of Alignables in " << trees[i]->GetName()
// 		<< std::endl;
//       return;
//     }

    std::ostringstream o; o << 't' << i;

    tuples[i] = new TNtuple(o.str().c_str(), "", "u:v:w:a:b:g");

    setBranch(trees[i], pos, rot);

    for (int n = 0; n < nAlignable; ++n)
    {
      trees[i]->GetEntry(n);

      TVectorD dr(3, pos);
      TMatrixD dR(3, 3, rot);

//       dr -= post[n];
//       dr *= 1e4;     // cm to microns
//       dR = TMatrixD(TMatrixD::kTransposed, rott[n]) * dR;
      dr -= post[n];     // in global frame
      dr *= 1e4;         // cm to microns
      dr = rott[n] * dr; // to local frame
      dR = dR * TMatrixD(TMatrixD::kTransposed, rott[n]);

      tuples[i]->Fill(dr[0], dr[1], dr[2],
		      1e3 * -std::atan2(dR(2, 1), dR(2, 2)),
		      1e3 *  std::asin(dR(2, 0)),
		      1e3 * -std::atan2(dR(1, 0), dR(0, 0)));
    }
  }

  fout.Write();

  for (int i = 0; i <= nIteration; ++i) delete tuples[i];
}

void writePars(std::string path)
{
  TFile ft((path + "IOTruePositions.root"      ).c_str());
  TFile f0((path + "IOMisalignedPositions.root").c_str());
  TFile fp((path + "IOAlignmentParameters.root").c_str());

  TTree* tt = (TTree*)ft.Get("AlignablesOrgPos_1");
  TTree* t0 = (TTree*)f0.Get("AlignablesAbsPos_1");

  const TList* keys = fp.GetListOfKeys();

  const std::string key0Name = keys->At(0)->GetName();

  const bool iter0Exist = ('0' == *(key0Name.end() - 1));

  const int nIteration = iter0Exist ? keys->GetSize() - 1 : keys->GetSize();
  const int nAlignable = tt->GetEntries();

  std::vector<TVectorD> post(nAlignable, TVectorD(3));
  std::vector<TMatrixD> rott(nAlignable, TMatrixD(3, 3));

  double pos[3];
  double rot[9];
  double par[nPar];

  setBranch(tt, pos, rot);

  for (int n = 0; n < nAlignable; ++n)
  {
    tt->GetEntry(n);

    for (int p = 0; p < 3; ++p) post[n][p] = pos[p];
    for (int p = 0; p < 9; ++p) rott[n](p / 3, p % 3) = rot[p];
  }

  TFile fout((path + "pars.root").c_str(), "RECREATE");

  std::vector<TNtuple*> tuples(nIteration + 1);

  tuples[0] = new TNtuple("t0", "", "u:v:w:a:b:g");

  setBranch(t0, pos, rot);

  for (int n = 0; n < nAlignable; ++n)
  {
    t0->GetEntry(n);

    TVectorD dr(3, pos);
    TMatrixD dR(3, 3, rot);

    dr -= post[n];     // in global frame
    dr *= 1e4;         // cm to microns
    dr = rott[n] * dr; // to local frame
    dR = dR * TMatrixD(TMatrixD::kTransposed, rott[n]);

    tuples[0]->Fill(dr[0], dr[1], dr[2],
		    1e3 * -std::atan2(dR(2, 1), dR(2, 2)),
		    1e3 *  std::asin(dR(2, 0)),
		    1e3 * -std::atan2(dR(1, 0), dR(0, 0)));
  }

  for (int i = 1; i <= nIteration; ++i)
  {
    TTree* tp = (TTree*)fp.Get(keys->At(iter0Exist ? i : i - 1)->GetName());

    tp->SetBranchStatus("*", 0); // disable all branches
    tp->SetBranchStatus("Par");  // enable Par branch
    tp->SetBranchAddress("Par", par);

    std::ostringstream o; o << 't' << i;

    tuples[i] = new TNtuple(o.str().c_str(), "", "u:v:w:a:b:g");

    const int nAlignable = tp->GetEntries();

    for (int n = 0; n < nAlignable; ++n)
    {
      tp->GetEntry(n);

      tuples[i]->Fill(par[0] * 1e4, par[1] * 1e4, par[2] * 1e4,
		      par[3] * 1e3, par[4] * 1e3, par[5] * 1e3);
    }
  }

  fout.Write();

  for (int i = 0; i <= nIteration; ++i) delete tuples[i];
}

void writeSurvey(std::string path, std::string type)
{
  static AlignableObjectId objId;

  TFile f0((path + "histograms.root").c_str());

  const TList* keys = f0.GetListOfKeys();

  const int nIteration = keys->GetSize();

  std::ostringstream o; o << path << "survey_" << type << ".root";

  TFile fout(o.str().c_str(), "RECREATE");

  std::vector<TNtuple*> tuples(nIteration);

  int level;
  double par[nPar];

  for (int i = 0; i < nIteration; ++i)
  {
    std::string name = keys->At(i)->GetName(); // due to iter 0
    name += "/survey";

    TTree* t0 = (TTree*)f0.Get(name.c_str());

    t0->SetBranchAddress("level", &level);
    t0->SetBranchAddress("par", par);

    std::ostringstream o; o << 't' << i + 1;

    tuples[i] = new TNtuple(o.str().c_str(), "", "u:v:w:a:b:g");

    const int nEntry = t0->GetEntries();

    for (int n = 0; n < nEntry; ++n)
    {
      t0->GetEntry(n);

      if (objId.nameToType(type) == level)
        tuples[i]->Fill(par[0] * 1e4, par[1] * 1e4, par[2] * 1e4,
			par[3] * 1e3, par[4] * 1e3, par[5] * 1e3);
    }
  }

  fout.Write();

  for (int i = 0; i < nIteration; ++i) delete tuples[i];
}

class AlignPlots
{
  typedef std::vector<float> AlignSet; // per iteration per parameter

  static const char* const titles_[nPar];

  public:

  AlignPlots(std::string file);
  AlignPlots(std::string file, std::vector<unsigned int> levels, int minHit = 0);

  void iters() const;

  void iter(int iter) const;

  void dump(int index, int iterN = 0) const;

  private:

  static float sum(const AlignSet&);
  static float sum2(const AlignSet&);
  static float width(const AlignSet&);

  std::string file_;

  std::vector<AlignSet> alignSets_[nPar];
};

const char* const AlignPlots::titles_[nPar] = 
  {"#Deltau (#mum)", "#Deltav (#mum)", "#Deltaw (#mum)",
   "#Delta#alpha (mrad)", "#Delta#beta (mrad)", "#Delta#gamma (mrad)"};

AlignPlots::AlignPlots(std::string file):
  file_(file)
{
  TFile fin((file_ + ".root").c_str());

  const TList* keys = fin.GetListOfKeys();

  TNtuple* t = (TNtuple*)fin.Get(keys->At(0)->GetName());

  const int nIteration = keys->GetSize();
  const int nAlignable = t->GetEntries();

  for (int p = 0; p < nPar; ++p)
    alignSets_[p].resize(nIteration, AlignSet(nAlignable));

  for (int i = 0; i < nIteration; ++i)
  {
    t = (TNtuple*)fin.Get(keys->At(i)->GetName());

    for (int n = 0; n < nAlignable; ++n)
    {
      t->GetEntry(n);

      const float* pars = t->GetArgs();

      for (int p = 0; p < nPar; ++p) alignSets_[p][i][n] = pars[p];
    }
  }
}

AlignPlots::AlignPlots(std::string file, std::vector<unsigned int> levels, int minHit):
  file_(file)
{
  const unsigned int nLevel = levels.size();

  if (nLevel == 0)
  {
    std::cout << "No input levels." << std::endl; return;
  }

  const CounterNames& cn = Helper::counterNames(levels[0]);

  if (cn.size() < nLevel)
  {
    std::cout << "Too many input levels for " << cn[0].second
	      << ". Max " << cn.size()
	      << std::endl;
    return;
  }

  std::string path = file.substr(0, file.find_last_of('/'));

  TFile fu((path + "/IOUserVariables.root").c_str());

  TTree* tu = (TTree*)fu.Get("T9_1");

  unsigned int id(0);
  int nHit(0);

  tu->SetBranchStatus("*", 0); // disable all branches
  tu->SetBranchStatus("Id");   // enable Id branch
  tu->SetBranchStatus("Nhit"); // enable Nhit branch
  tu->SetBranchAddress("Id"  , &id);
  tu->SetBranchAddress("Nhit", &nHit);

  TFile fin((file + ".root").c_str());

  const TList* keys = fin.GetListOfKeys();

  TNtuple* t = (TNtuple*)fin.Get(keys->At(0)->GetName());

  const int nIteration = keys->GetSize();
  const int nAlignable = t->GetEntries();

  for (int p = 0; p < nPar; ++p) alignSets_[p].resize(nIteration);

  for (int i = 0; i < nIteration; ++i)
  {
    for (int p = 0; p < nPar; ++p) alignSets_[p][i].reserve(nAlignable);

    t = (TNtuple*)fin.Get(keys->At(i)->GetName());

    for (int n = 0; n < nAlignable; ++n)
    {
      t->GetEntry(n);
      tu->GetEntry(n);

      const float* pars = t->GetArgs();

      bool selected = (nHit >= minHit);

      for (unsigned int l = 0; selected && l < nLevel; ++l)
	selected = (cn[l].first(id) == levels[l]);

      if (selected)
	for (int p = 0; p < nPar; ++p)
	  alignSets_[p][i].push_back(pars[p]);
    }
  }

  std::ostringstream o;

  for (unsigned int l = 0; l < nLevel; ++l)
    o << '_' << cn[l].second << levels[l];

  o << "_minHit" << minHit;
  file_ += o.str();
}

void AlignPlots::iters() const
{
  gStyle->SetOptTitle(0); // don't display title
  gStyle->SetOptStat(0);  // don't display stat box

  TGaxis::SetMaxDigits(3); // max digits for axis labels

  const int nIteration = alignSets_[0].size();
  const int nAlignable = alignSets_[0][0].size();

  if (0 == nAlignable)
  {
    std::cout << "0 Alignables selected." << std::endl; return;
  }

  TCanvas c("c", "c", 1200, 800);

  c.Divide(3, 2);

  std::vector<TGraph> graphs[nPar];

  for (int p = 0; p < nPar; ++p)
  {
    c.cd(p + 1);

  // Find min and max y-values over all iterations to set y-axis limits.

    std::vector<float> ylimits(nIteration);

    for (int i = 0; i < nIteration; ++i)
    {
      const AlignSet& aSet = alignSets_[p][i];

      float ymin = *std::min_element(aSet.begin(), aSet.end());
      float ymax = *std::max_element(aSet.begin(), aSet.end());

      ylimits[i] = std::max(std::abs(ymin), std::abs(ymax));
//     ylimits[i] = p < 3 ? 5e3 : .2;
    }

    float ylimit = *std::max_element(ylimits.begin(), ylimits.end());

    graphs[p].resize(nAlignable, nIteration);

    TGraph& g = graphs[p][0];

    g.SetMinimum(-ylimit);
    g.SetMaximum( ylimit);
    g.GetXaxis()->SetLimits(0., nIteration - 1.); // not SetRangeUser
    g.GetXaxis()->SetTitle("iteration");
    g.GetYaxis()->SetTitle(titles_[p]);
    g.GetYaxis()->SetTitleSize(.04);
    g.Draw("AP"); // need "P" to draw axes

    for (int n = 0; n < nAlignable; ++n)
    {
      TGraph& g = graphs[p][n];

      for (int i = 0; i < nIteration; ++i)
      {
        float y = alignSets_[p][i][n];

        g.SetPoint(i, i, isnan(y) ? 0.f : y);
      }

      g.Draw("L");
    }
  }

  std::ostringstream o;
  o << file_ << "_vs_iter";

  c.SaveAs((o.str() + ".png").c_str());
  c.SaveAs((o.str() + ".eps").c_str());
}

void AlignPlots::dump(int index, int iterN) const
{
  gStyle->SetOptTitle(0); // don't display title
  gStyle->SetOptStat(0);  // don't display stat box

  TGaxis::SetMaxDigits(3); // max digits for axis labels

  const int nIteration = alignSets_[0].size();
  const int nAlignable = alignSets_[0][0].size();

  if (0 >= iterN || nIteration <= iterN) iterN = nIteration;
  else ++iterN; // add 1 to include iteration 0

  if (index >= nAlignable)
  {
    std::cout << "Alignable index too big. "
              << "Number of Alignables is " << nAlignable
              << std::endl;
    return;
  }

  TCanvas c("c", "c", 1200, 800);

  c.Divide(3, 2);

  TGraph graphs[nPar];

  for (int p = 0; p < nPar; ++p)
  {
    c.cd(p + 1);

  // Find min and max y-values over all iterations to set y-axis limits.

    std::vector<float> ylimits(iterN);

    for (int i = 0; i < iterN; ++i)
    {
      ylimits[i] = std::abs(alignSets_[p][i][index]);
    }

    float ylimit = *std::max_element(ylimits.begin(), ylimits.end());

    TGraph& g = graphs[p];

    g.Set(iterN);
    g.SetMinimum(-ylimit);
    g.SetMaximum( ylimit);
    g.GetXaxis()->SetLimits(0., iterN - 1.); // not SetRangeUser
    g.GetXaxis()->SetTitle("iteration");
    g.GetYaxis()->SetTitle(titles_[p]);
    g.GetYaxis()->SetTitleSize(.04);
    g.Draw("AP"); // need "P" to draw axes

    for (int i = 0; i < iterN; ++i)
    {
      g.SetPoint(i, i, alignSets_[p][i][index]);
    }

    g.Draw("L");
  }

  std::ostringstream o;
  o << file_ << "_vs_iter_Alignable" << index;

  c.SaveAs((o.str() + ".png").c_str());
  c.SaveAs((o.str() + ".eps").c_str());
}

float AlignPlots::sum(const AlignSet& aSet)
{
  float sum(0.);

  for (unsigned int i = 0; i < aSet.size(); ++i)
    if (std::abs(aSet[i]) < 1e8) sum += aSet[i]; // avoid big numbers

  return sum;
}

float AlignPlots::sum2(const AlignSet& aSet)
{
  float sum(0.);

  for (unsigned int i = 0; i < aSet.size(); ++i)
    if (std::abs(aSet[i]) < 1e8) sum += aSet[i] * aSet[i]; // avoid big numbers

  return sum;
}

float AlignPlots::width(const AlignSet& aSet)
{
  float invN = 1. / aSet.size();
  float mean = sum(aSet) * invN;
  float rms2 = sum2(aSet) * invN - mean * mean;

  return rms2 > 0. ? 3. * std::sqrt(rms2) : 1e-3;
}

void AlignPlots::iter(int iter) const
{
  gStyle->SetOptTitle(0); // don't display title
  gStyle->SetOptStat(10); // show only entries
  gStyle->SetOptFit(0);   // show only entries
  gStyle->SetStatH(.2);   // set stat box height
  gStyle->SetStatW(.3);   // set stat box width

  const int nIteration = alignSets_[0].size();
  const int nAlignable = alignSets_[0][0].size();

  if (nIteration <= iter)
  {
    std::cout << "Iteration number too big. "
	      << "Setting to max " << nIteration - 1
	      << std::endl;

    iter = nIteration - 1;
  }

  if (0 == nAlignable)
  {
    std::cout << "0 Alignables selected." << std::endl; return;
  }

  TCanvas c("c", "c", 1200, 800);

  c.Divide(3, 2);

  TH1F hists[nPar];

  for (int p = 0; p < nPar; ++p)
  {
    const AlignSet& setI = alignSets_[p][iter];

    float mean = sum(setI) / setI.size();
    float rms3 = std::min(width(alignSets_[p][0]), width(setI));

    TH1F& h = hists[p];

    h.SetBins(50, mean - rms3, mean + rms3);

    for (int n = 0; n < nAlignable; ++n) h.Fill(setI[n]);

    c.cd(p + 1);
    h.SetXTitle(titles_[p]);
    h.SetTitleSize(.04);
    h.Fit("gaus", "LQ");

    TF1* f = h.GetFunction("gaus");

    f->SetLineWidth(1);
    f->SetLineColor(kRed);

    std::ostringstream o;

    o << std::fixed << std::setprecision(2)
      << "width = " << f->GetParameter(2);

    TText width; width.DrawTextNDC(.6, .8, o.str().c_str());
  }

  std::ostringstream o;
  o << file_ << "_iter" << iter;

  c.SaveAs((o.str() + ".png").c_str());
  c.SaveAs((o.str() + ".eps").c_str());
}

void compareShifts(std::string tree)
{
  gStyle->SetOptStat(0);

  TFile fm("merged/shifts.root");
  TFile ft("tracks/shifts.root");
  TFile fs("survey/shifts.root");

  TTree* tm = (TTree*)fm.Get(tree.c_str());
  TTree* tt = (TTree*)ft.Get(tree.c_str());
  TTree* ts = (TTree*)fs.Get(tree.c_str());

  tm->SetMarkerStyle(kCircle);
  tt->SetMarkerStyle(kPlus);

//   TH1F hx("hx", "#Deltax (#mum)", 50, -2e3, 2e3);
//   TH1F hy("hy", "#Deltay (#mum)", 50, -4e3, 4e3);
//   TH1F hz("hz", "#Deltaz (#mum)", 50, -4e3, 4e3);
//   TH1F ha("ha", "#Delta#omega_{x} (mrad)", 50, -0.2, 0.2);
//   TH1F hb("hb", "#Delta#omega_{y} (mrad)", 50, -0.2, 0.2);
//   TH1F hg("hg", "#Delta#omega_{z} (mrad)", 50, -0.2, 0.2);

  TCanvas c("c", "c", 1200, 800);

  c.Divide(3, 2);

  const char* const vars[6] = {"x", "y", "z", "a", "b", "g"};

  for (int i = 0; i < 6; ++i)
  {
    c.cd(i + 1);
    ts->Draw(vars[i]);
    tt->Draw(vars[i], "", "sameP");
    tm->Draw(vars[i], "", "sameP");
  }

  TLegend leg(.7, .7, .9, .9);

  leg.AddEntry(ts, "survey");
  leg.AddEntry(tt, "tracks");
  leg.AddEntry(tm, "merged");
  leg.Draw();

  c.SaveAs(("compareShifts_" + tree + ".png").c_str());
}
