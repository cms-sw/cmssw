#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <valarray>
#include <vector>

#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TList.h"
#include "TMatrixD.h"
#include "TNtuple.h"
#include "TStyle.h"
#include "TText.h"
#include "TTree.h"
#include "TVectorD.h"

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

  TTree* tt = (TTree*)ft.Get("AlignablesOrgPos:1");

  const TList* keys = fp.GetListOfKeys();

  const int nIteration = keys->GetSize();
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

  trees[0] = (TTree*)f0.Get("AlignablesAbsPos:1");

  for (int i = 0; i < nIteration; ++i)
  {
    trees[i + 1] = (TTree*)fp.Get(keys->At(i)->GetName());
  }

  TFile fout((path + "shifts.root").c_str(), "RECREATE");

  std::vector<TNtuple*> tuples(nIteration + 1);

  for (int i = 0; i <= nIteration; ++i)
  {
    std::ostringstream o; o << 't' << i;

    tuples[i] = new TNtuple(o.str().c_str(), "", "x:y:z:a:b:g");

    setBranch(trees[i], pos, rot);

    for (int n = 0; n < nAlignable; ++n)
    {
      trees[i]->GetEntry(n);

      TVectorD dr(3, pos);
      TMatrixD dR(3, 3, rot);

      dr -= post[n];
      dr *= 1e4;
      dR = rott[n].T() * dR;

      tuples[i]->Fill(dr[0], dr[1], dr[2],
		      -std::atan2(dR(2, 1), dR(2, 2)),
		       std::asin(dR(2, 0)),
		      -std::atan2(dR(1, 0), dR(0, 0)));
    }
  }

  fout.Write();

  for (int i = 0; i <= nIteration; ++i) delete tuples[i];
}

void writePars(std::string path)
{
  TFile f0((path + "IOAlignmentParameters.root").c_str());

  double pos[nPar];

  TFile fout((path + "pars.root").c_str(), "RECREATE");

  const TList* keys = f0.GetListOfKeys();

  const int nIteration = keys->GetSize();

  std::vector<TNtuple*> tuples(nIteration);

  for (int i = 0; i < nIteration; ++i)
  {
    TTree* t0 = (TTree*)f0.Get(keys->At(i)->GetName());

    t0->SetBranchStatus("*", 0); // disable all branches
    t0->SetBranchStatus("Par");  // enable Pos branch

    t0->SetBranchAddress("Par", pos);

    std::ostringstream o; o << 't' << i;

    tuples[i] = new TNtuple(o.str().c_str(), "", "u:v:w:a:b:g");

    const int nAlignable = t0->GetEntries();

    for (int n = 0; n < nAlignable; ++n)
    {
      t0->GetEntry(n);

      tuples[i]->Fill(pos[0] * 1e4, pos[1] * 1e4, pos[2] * 1e4,
		      pos[3], pos[4], pos[5]);
    }
  }

  fout.Write();

  for (int i = 0; i < nIteration; ++i) delete tuples[i];
}

class AlignPlots
{
  typedef std::valarray<float> AlignSet; // per iteration per parameter

  static const char* const titles_[nPar];

  public:

  AlignPlots(std::string file);

  void iters() const;

  void iter(int iter) const;

  private:

  std::string file_;

  std::vector<AlignSet> alignSets_[nPar];
};

const char* const AlignPlots::titles_[nPar] = 
  {"#Deltax (cm)", "#Deltay (cm)", "#Deltaz (cm)",
   "#Delta#omega_{x}", "#Delta#omega_{y}", "#Delta#omega_{z}"};

AlignPlots::AlignPlots(std::string file):
  file_(file)
{
  TFile fin = ((file_ + ".root").c_str());

  const TList* keys = fin.GetListOfKeys();

  TNtuple* t = (TNtuple*)fin.Get(keys->At(0)->GetName());

  const int nIteration = keys->GetSize();
  const int nAlignable = t->GetEntries();

  for (int p = 0; p < nPar; ++p) alignSets_[p].resize(nIteration, AlignSet(nAlignable));

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

void AlignPlots::iters() const
{
  gStyle->SetOptTitle(0); // don't display title
  gStyle->SetOptStat(0);  // don't display stat box

  TGaxis::SetMaxDigits(3); // max digits for axis labels

  const int nIteration = alignSets_[0].size();
  const int nAlignable = alignSets_[0][0].size();

  TCanvas c("c", "c", 1200, 800);

  c.Divide(3, 2);

  std::vector<TGraph> graphs[nPar];

  for (int p = 0; p < nPar; ++p)
  {
    graphs[p].resize(nAlignable, nIteration);

    c.cd(p + 1);

    TGraph& g = graphs[p][0];

    float ymin = alignSets_[p][0].min();
    float ymax = alignSets_[p][0].max();

    float ylimit = std::max(std::abs(ymin), std::abs(ymax));
//     float ylimit = p < 3 ? 5e3 : .2;

    g.SetMinimum(-ylimit);
    g.SetMaximum( ylimit);
    g.GetXaxis()->SetLimits(0., nIteration - 1.); // not SetRangeUser
    g.GetXaxis()->SetTitle("iteration");
    g.GetYaxis()->SetTitle(titles_[p]);
    g.Draw("AP"); // need "P" to draw axes

    for (int n = 0; n < nAlignable; ++n)
    {
      TGraph& g = graphs[p][n];

      for (int i = 0; i < nIteration; ++i)
      {
	g.SetPoint(i, i, alignSets_[p][i][n]);
      }

      g.Draw("L");
    }
  }

  c.SaveAs((file_ + "_vs_iter.png").c_str());
//   c.SaveAs((file_ + "_vs_iter.eps").c_str());
}

void AlignPlots::iter(int iter) const
{
  gStyle->SetOptTitle(0); // don't display title
  gStyle->SetOptStat(10); // show only entries
  gStyle->SetOptFit(0);   // show only entries

  const int nAlignable = alignSets_[0][0].size();

  TCanvas c("c", "c", 1200, 800);

  c.Divide(3, 2);

  TH1F hists[nPar];

  for (int p = 0; p < nPar; ++p)
  {
    const AlignSet& aSet = alignSets_[p][iter];

    TH1F& h = hists[p];

    h.SetBins(50, aSet.min(), aSet.max());
//     float xlimit = p < 3 ? 3000. : .1;
//     h.SetBins(50, -xlimit, xlimit);

    for (int n = 0; n < nAlignable; ++n)
    {
      h.Fill(aSet[n]);
    }

    c.cd(p + 1);
    h.SetXTitle(titles_[p]);
    h.Fit("gaus", "LQ");

    TF1* f = h.GetFunction("gaus");

    f->SetLineWidth(1);
    f->SetLineColor(kRed);

    std::ostringstream o;

    o << std::fixed << std::setprecision(2)
      << "width = " << f->GetParameter(2);

    TText width; width.DrawTextNDC(.6, .8, o.str().c_str());
  }

  std::ostringstream o; o << file_ << iter;

  c.SaveAs((o.str() + ".png").c_str());
//   c.SaveAs((o.str() + ".eps").c_str());
}
