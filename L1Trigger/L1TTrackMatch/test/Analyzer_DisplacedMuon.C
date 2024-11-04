
// // ----------------------------------------------------------------------------------------------------------------
// // Feasibility study of using L1 Tracks to identify Displaced Vertex
// //
// // By Bharadwaj Harikrishnan, May 2021
// // Edited by Ryan McCarthy, Sept 2021
// // ----------------------------------------------------------------------------------------------------------------

#include <TROOT.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TLatex.h>
#include <TVector3.h>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"
#include "TLeaf.h"
#include <TCanvas.h>
#include "TLegend.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TGraph.h"
#include "TMath.h"
#include <math.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <valarray>
#include <deque>
#include <THStack.h>
#include <TF2.h>
#include <TEllipse.h>
#include <TMarker.h>
#include <chrono>
#include <boost/variant.hpp>
using namespace std;

bool detailedPlots = false;
float d0_res = 0.02152; //cm

void SetPlotStyle();
void mySmallText(Double_t x, Double_t y, Color_t color, char *text);
void removeFlows(TH1F* h);
void removeFlows(TH2F* h);

class Track_Parameters
{
public:
  float pt;
  float d0;
  float dxy = -99999;
  float z0;
  float eta;
  float phi;
  float charge;
  float rho;
  int index;
  int pdgid = -99999;
  float vx;
  float vy;
  float vz;
  Track_Parameters* tp;
  float x0;
  float y0;
  int nstubs;
  float chi2rphi;
  float chi2rz;
  float bendchi2;
  float MVA1;

  float dist_calc(float x_dv, float y_dv, float x, float y){
    dxy = TMath::Sqrt((x_dv-x)*(x_dv-x) + (y_dv-y)*(y_dv-y));
    return dxy;
  }
  float x(float phi_T=0){
    return (-charge * rho * TMath::Sin(phi - charge*phi_T) + (d0 + charge * rho) * TMath::Sin(phi));
  }
  float y(float phi_T=0){
    return ( charge * rho * TMath::Cos(phi - charge*phi_T) - (d0 + charge * rho) * TMath::Cos(phi));
  }
  float z(float phi_T=0){
    float theta = 2 * TMath::ATan(TMath::Exp(-eta));
    return (z0 + rho*phi_T/TMath::Tan(theta));
  }
  float deltaPhi_T(Double_t phi1, Double_t phi2)
  {
    Double_t dPhi = phi1 - phi2;
    if (dPhi >= TMath::Pi())
      dPhi -= 2. * TMath::Pi();
    if (dPhi <= -TMath::Pi())
      dPhi += 2. * TMath::Pi();
    return dPhi;
  }
  float phi_T(float x, float y){
    float num = x - (d0 + charge * rho) * TMath::Sin(phi);
    float den = y + (d0 + charge * rho) * TMath::Cos(phi);
    return ((phi-TMath::ATan2(num,-den))/charge);
  }
  float z(float x, float y){
    float t = std::sinh(eta);
    float r = TMath::Sqrt(pow(x,2)+pow(y,2));
    return (z0+(t*r*(1+(pow(d0,2)/pow(r,2))+(1.0/6.0)*pow(r/(2*rho),2)))); // can do higher order terms if necessary from displaced math
  }
  Track_Parameters(float pt_in, float d0_in, float z0_in, float eta_in, float phi_in, int pdgid_in, float vx_in, float vy_in, float vz_in, float charge_in=0, int index_in=-1, Track_Parameters* tp_in=nullptr, int nstubs_in=0, float chi2rphi_in=0, float chi2rz_in=0, float bendchi2_in=0, float MVA1_in=0)
  {
    pt = pt_in;
    d0 = d0_in;
    z0 = z0_in;
    eta = eta_in;
    phi = phi_in;
    if(charge_in > 0){
      charge = 1;
    }
    else if (charge_in < 0){
      charge = -1;
    }
    else{
      charge = 0;
    }
    index = index_in;
    pdgid = pdgid_in;
    vx = vx_in;
    vy = vy_in;
    vz = vz_in;
    tp = tp_in;
    rho = fabs(1/charge_in);
    x0 = (rho+charge*d0)*TMath::Cos(phi-(charge*TMath::Pi()/2));
    y0 = (rho+charge*d0)*TMath::Sin(phi-(charge*TMath::Pi()/2));
    nstubs = nstubs_in;
    chi2rphi = chi2rphi_in;
    chi2rz = chi2rz_in;
    bendchi2 = bendchi2_in;
    MVA1 = MVA1_in;
  }
  Track_Parameters(){};
  ~Track_Parameters(){};
};

constexpr bool operator==(const Track_Parameters* lhs, const Track_Parameters& rhs)
{
  return (lhs->pt==rhs.pt && lhs->d0==rhs.d0 && lhs->z0==rhs.z0 && lhs->eta==rhs.eta && lhs->phi==rhs.phi);
}
constexpr bool operator==(const Track_Parameters& lhs, const Track_Parameters* rhs)
{
  return (lhs.pt==rhs->pt && lhs.d0==rhs->d0 && lhs.z0==rhs->z0 && lhs.eta==rhs->eta && lhs.phi==rhs->phi);
}
constexpr bool operator==(const Track_Parameters& lhs, const Track_Parameters& rhs)
{
  return (lhs.pt==rhs.pt && lhs.d0==rhs.d0 && lhs.z0==rhs.z0 && lhs.eta==rhs.eta && lhs.phi==rhs.phi);
}

std::valarray<float> calcPVec(Track_Parameters a, double_t v_x, double_t v_y)
{
  std::valarray<float> r_vec = {float(v_x)-a.x0,float(v_y)-a.y0};
  std::valarray<float> p_vec = {-r_vec[1],r_vec[0]};
  if(a.charge>0){
    p_vec *= -1;
  }
  p_vec /= TMath::Sqrt(pow(p_vec[0],2)+pow(p_vec[1],2));
  p_vec *= a.pt;
  return p_vec;
}

class Vertex_Parameters
{
public:
  Double_t x_dv;
  Double_t y_dv;
  Double_t z_dv;
  float score;
  Track_Parameters a;
  Track_Parameters b;
  int inTraj;
  bool matched = false;
  std::vector<Track_Parameters> tracks = {};
  float p_mag;
  float p2_mag;
  float openingAngle;
  float R_T;
  float cos_T;
  float alpha_T;
  float d_T;
  float chi2rphidofSum;
  float chi2rzdofSum;
  float bendchi2Sum;
  float MVA1Sum;
  int numStubsSum;
  float delta_z;
  float delta_eta;
  float phi;
  Vertex_Parameters(Double_t x_dv_in, Double_t y_dv_in, Double_t z_dv_in, Track_Parameters a_in, Track_Parameters b_in, float score_in=-1, int inTraj_in=4):   
    a(a_in),
    b(b_in)
  {
    x_dv = x_dv_in;
    y_dv = y_dv_in;
    z_dv = z_dv_in;
    score = score_in;
    tracks.push_back(a_in);
    tracks.push_back(b_in);
    inTraj = inTraj_in;
    std::valarray<float> p_trk_1 = calcPVec(a_in,x_dv_in,y_dv_in);
    std::valarray<float> p_trk_2 = calcPVec(b_in,x_dv_in,y_dv_in);
    std::valarray<float> p_tot = p_trk_1+p_trk_2;
    p_mag = TMath::Sqrt(pow(p_tot[0],2)+pow(p_tot[1],2));
    openingAngle = (p_trk_1[0]*p_trk_2[0]+p_trk_1[1]*p_trk_2[1]) / (TMath::Sqrt(pow(p_trk_1[0],2)+pow(p_trk_1[1],2))*TMath::Sqrt(pow(p_trk_2[0],2)+pow(p_trk_2[1],2)));
    R_T = TMath::Sqrt(pow(x_dv_in,2)+pow(y_dv_in,2));
    cos_T = (p_tot[0]*x_dv_in+p_tot[1]*y_dv_in)/(R_T*TMath::Sqrt(pow(p_tot[0],2)+pow(p_tot[1],2)));
    alpha_T = acos(cos_T);
    phi = atan2(p_tot[1],p_tot[0]);
    d_T = fabs(cos(phi)*y_dv_in-sin(phi)*x_dv_in);
    float chi2rphidof_1 = a_in.chi2rphi;
    float chi2rzdof_1 = a_in.chi2rz;
    float bendchi2_1 = a_in.bendchi2;
    float chi2rphidof_2 = b_in.chi2rphi;
    float chi2rzdof_2 = b_in.chi2rz;
    float bendchi2_2 = b_in.bendchi2;
    chi2rphidofSum = chi2rphidof_1 + chi2rphidof_2;
    chi2rzdofSum = chi2rzdof_1 + chi2rzdof_2;
    bendchi2Sum = bendchi2_1 + bendchi2_2;
    MVA1Sum = a_in.MVA1 + b_in.MVA1;
    numStubsSum = a_in.nstubs + b_in.nstubs;
    p2_mag = pow(a_in.pt,2)+pow(b_in.pt,2);
    delta_z = fabs(a_in.z(x_dv_in,y_dv_in)-b_in.z(x_dv_in,y_dv_in));
    delta_eta = fabs(a_in.eta-b_in.eta);
  }

  void addTrack(Track_Parameters trk){
    tracks.push_back(trk);
    std::valarray<float> p_tot = {0,0};
    for(auto track : tracks){
      p_tot+= calcPVec(track,x_dv,y_dv);
    }
    p_mag = TMath::Sqrt(pow(p_tot[0],2)+pow(p_tot[1],2));
    cos_T = (p_tot[0]*x_dv+p_tot[1]*y_dv)/(R_T*TMath::Sqrt(pow(p_tot[0],2)+pow(p_tot[1],2)));
    alpha_T = acos(cos_T);
    phi = atan2(p_tot[1],p_tot[0]);
    d_T = fabs(cos(phi)*y_dv-sin(phi)*x_dv);
    float chi2rphidof = trk.chi2rphi;
    float chi2rzdof = trk.chi2rz;
    float bendchi2 = trk.bendchi2;
    chi2rphidofSum+= chi2rphidof;
    chi2rzdofSum+= chi2rzdof;
    bendchi2Sum+= bendchi2;
    numStubsSum+= trk.nstubs;
    p2_mag+= pow(trk.pt,2);
    MVA1Sum+= trk.MVA1;
  }

  Vertex_Parameters(){};
  ~Vertex_Parameters(){};
};

constexpr bool operator==(const Vertex_Parameters& lhs, const Vertex_Parameters& rhs)
{
  return (lhs.x_dv==rhs.x_dv && lhs.y_dv==rhs.y_dv && lhs.z_dv==rhs.z_dv);
}

class Cut {
public:
  virtual ~Cut() = default;
  virtual TString getCutName() const = 0;
  virtual TString getCutLabel() const = 0;
  virtual float getParam(int it) const = 0;
  virtual float getCutValue() const = 0;
  virtual float getDoPlot() const = 0;
};

template <typename T>
class TypedCut : public Cut
{
public:
  TString cutName;
  TString cutLabel;
  std::vector<T>** params;
  T cutValue;
  bool doPlot;
  
  TypedCut(TString cutName_in, TString cutLabel_in, std::vector<T>** params_in, T cutValue_in, bool doPlot_in): cutName(cutName_in), cutLabel(cutLabel_in), params(params_in), cutValue(cutValue_in), doPlot(doPlot_in) {}
  TypedCut(){};
  ~TypedCut(){};
  TString getCutName() const
  {
    return cutName;
  }
  TString getCutLabel() const
  {
    return cutLabel;
  }
  float getParam(int it) const
  {
    T param = (*params)->at(it);
    return float(param);
  }
  float getCutValue() const
  {
    return float(cutValue);
  }
  float getDoPlot() const
  {
    return doPlot;
  }
};
  
class Plot {
public:
  virtual ~Plot() = default;
  virtual TString getVarName() const = 0;
  virtual TString getUnit() const = 0;
  virtual float getParam(int it) const = 0;
  virtual int getNumBins() const = 0;
  virtual float getMinBin() const = 0;
  virtual float getMaxBin() const = 0;
  virtual std::vector<float> getBins() const = 0;
  virtual bool getBool() const = 0;
};

template <typename T>
class TypedPlot : public Plot
{
public:
  TString varName;
  TString unit;
  std::vector<T>** params;
  int numBins;
  float minBin;
  float maxBin;
  std::vector<float> bins;
  bool variableBins;
  
  TypedPlot(TString varName_in, TString unit_in, std::vector<T>** params_in, int numBins_in, float minBin_in, float maxBin_in): varName(varName_in), unit(unit_in), params(params_in), numBins(numBins_in), minBin(minBin_in), maxBin(maxBin_in){
    variableBins = false;
  }
  TypedPlot(TString varName_in, TString unit_in, std::vector<T>** params_in, int numBins_in, std::vector<float> bins_in): varName(varName_in), unit(unit_in), params(params_in), numBins(numBins_in), bins(bins_in) {
    variableBins = true;
  }
  TypedPlot(){};
  ~TypedPlot(){};
  TString getVarName() const
  {
    return varName;
  }
  TString getUnit() const
  {
    return unit;
  }
  float getParam(int it) const
  {
    T param = (*params)->at(it);
    return float(param);
  }
  int getNumBins() const
  {
    return numBins;
  }
  float getMinBin() const
  {
    return minBin;
  }
  float getMaxBin() const
  {
    return maxBin;
  }
  std::vector<float> getBins() const
  {
    return bins;
  }
  bool getBool() const
  {
    return variableBins;
  }
};

void displayProgress(long current, long max)
{
  using std::cerr;
  if (max < 2500)
    return;
  if (current % (max / 2500) != 0 && current < max - 1)
    return;

  int width = 52; // Hope the terminal is at least that wide.
  int barWidth = width - 2;
  cerr << "\x1B[2K";    // Clear line
  cerr << "\x1B[2000D"; // Cursor left
  cerr << '[';
  for (int i = 0; i < barWidth; ++i)
    {
      if (i < barWidth * current / max)
	{
	  cerr << '=';
	}
      else
	{
	  cerr << ' ';
	}
    }
  cerr << ']';
  cerr << " " << Form("%8d/%8d (%5.2f%%)", (int)current, (int)max, 100.0 * current / max);
  cerr.flush();
}

template <typename T, typename S = TH1F>
void raiseMax(T *hist1, S *hist2=nullptr, T *hist3=nullptr, T *hist4=nullptr)
{ 
  Double_t max = hist1->GetBinContent(hist1->GetMaximumBin());
  if(hist2!=nullptr){
    Double_t max2 = hist2->GetBinContent(hist2->GetMaximumBin());
    if(max2>max) max = max2;
  }
  if(hist3!=nullptr){
    Double_t max3 = hist3->GetBinContent(hist3->GetMaximumBin());
    if(max3>max) max = max3;
  }
  if(hist4!=nullptr){
    Double_t max4 = hist4->GetBinContent(hist4->GetMaximumBin());
    if(max4>max) max = max4;
  }
  if(max>0.0){
    hist1->GetYaxis()->SetRangeUser(0.,1.2*max);
    if(hist2!=nullptr) hist2->GetYaxis()->SetRangeUser(0.,1.2*max);
    if(hist3!=nullptr) hist3->GetYaxis()->SetRangeUser(0.,1.2*max);
    if(hist4!=nullptr) hist4->GetYaxis()->SetRangeUser(0.,1.2*max);
  } 
}

void raiseMaxStack(TH1F* hist, THStack* stack)
{
  Double_t max = hist->GetMaximum();
  Double_t max2 = stack->GetMaximum();
  if(max2>max) max = max2;
 
  if(max>0.0){
    hist->GetYaxis()->SetRangeUser(0.,1.2*max);
    stack->GetYaxis()->SetRangeUser(0.,1.2*max);
  }
  
}

template <typename T, typename S>
void drawSame(T *hist1, S *hist2, T *hist3=nullptr, T *hist4=nullptr)
{
  if(hist1->GetBinContent(hist1->GetMaximumBin())!=0.0){
    hist1->Draw("HIST");
    hist2->Draw("HIST,SAME");
    if(hist3!=nullptr) hist3->Draw("HIST,SAME");
    if(hist4!=nullptr) hist4->Draw("HIST,SAME");
  }
  else if(hist2->GetBinContent(hist2->GetMaximumBin())!=0.0){
    hist2->Draw("HIST");
    if(hist3!=nullptr) hist3->Draw("HIST,SAME");
    if(hist4!=nullptr) hist4->Draw("HIST,SAME");
  }
  else if(hist3!=nullptr){
    if(hist3->GetBinContent(hist3->GetMaximumBin())!=0.0){
      hist3->Draw("HIST");
      if(hist4!=nullptr) hist4->Draw("HIST,SAME");
    }
  }
  else if(hist4!=nullptr){
    if(hist4->GetBinContent(hist4->GetMaximumBin())!=0.0){
      hist4->Draw("HIST");
    }
  }
  else{
    hist1->Draw("HIST");
  }
}

void drawSameStack(TH1F* hist, THStack* stack)
{
  if(hist->GetMaximum()!=0.0){
    hist->Draw("HIST");
    stack->Draw("HIST,SAME");
  }
  else if(stack->GetMaximum()!=0.0){
    stack->Draw("HIST");
  }
  else{
    hist->Draw("HIST");
  }
}

bool ComparePtTrack(Track_Parameters a, Track_Parameters b) { return a.pt > b.pt; }
bool CompareZ0Track(Track_Parameters a, Track_Parameters b) { return a.z0 > b.z0; }
bool CompareD0Track(Track_Parameters a, Track_Parameters b) { return a.d0 > b.d0; }
bool ComparePtVert(Vertex_Parameters v1, Vertex_Parameters v2) {return v1.a.pt > v2.a.pt; }
bool CompareDelzVert(Vertex_Parameters v1, Vertex_Parameters v2) {return v1.delta_z > v2.delta_z; }
bool CompareDtVert(Vertex_Parameters v1, Vertex_Parameters v2) {return v1.d_T > v2.d_T; }
bool CompareChi2rphidofSumVert(Vertex_Parameters v1, Vertex_Parameters v2) {return v1.chi2rphidofSum > v2.chi2rphidofSum; }
bool CompareRtVert(Vertex_Parameters v1, Vertex_Parameters v2) {return v1.R_T > v2.R_T; }

template<typename T>
std::vector<T> linspace(T start, T end, int num){
  std::vector<T> out;
  T delta = (end - start) / (num-1);
  for(int i=0; i<num-1; i++){
    out.push_back(start+delta*i);
  }
  out.push_back(end);
  return out;
}

std::vector<float> logspace(const float &a, const float &b, const int &k)
{
  std::vector<float> bins;
  float delta = (log10(b) - log10(a)) / k;
  for (int i = 0; i < (k+1); i++)
    {
      bins.push_back(pow(10, log10(a) + (i * delta)));
    }
  //std::cout<<"logspace bins: ";
  for(uint j=0; j<bins.size(); j++){
    //std::cout<<bins[j]<<" ";
  }
  //std::cout<<std::endl;
  return bins;
}

Double_t deltaPhi(Double_t phi1, Double_t phi2)
{
  Double_t dPhi = phi1 - phi2;
  if (dPhi > TMath::Pi())
    dPhi -= 2. * TMath::Pi();
  if (dPhi < -TMath::Pi())
    dPhi += 2. * TMath::Pi();
  return dPhi;
}

Double_t deltaR(Double_t eta1, Double_t phi1, Double_t eta2, Double_t phi2)
{
  Double_t dEta, dPhi;
  dEta = eta1 - eta2;
  dPhi = deltaPhi(phi1, phi2);
  return sqrt(dEta * dEta + dPhi * dPhi);
}

Double_t dist(Double_t x1, Double_t y1 , Double_t x2=0, Double_t y2=0){ // Distance between 2 points
  return (TMath::Sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)));
}

Double_t dist_Vertex(Double_t x_vtx, Double_t y_vtx, Track_Parameters a){ // Distance between track and displaced vertex
  float R = dist(x_vtx,y_vtx,a.x0,a.y0);
  return (fabs(R-(a.rho)));
}

Double_t dist_TPs(Track_Parameters* a, Track_Parameters* b); // Closest distance between 2 tracks
Double_t dist_TPs(Track_Parameters a, Track_Parameters b); // Closest distance between 2 tracks
bool CompareDeltaXY(Vertex_Parameters v1, Vertex_Parameters v2) {return dist_TPs(v1.a,v1.b) < dist_TPs(v2.a,v2.b); }

Int_t calcVertex(Track_Parameters a, Track_Parameters b, Double_t &x_vtx, Double_t &y_vtx, Double_t &z_vtx); 
// Identify the displaced vertex (x_vtx,y_vtx,z_vtx) and return the status 
//-2 = Circles with same center. No Intersection
//-1 = Circles don't Intersect. A point on the line connecting the centers is chosen.
// 0 = Only 1 Intersection not satisfying Z cutoff
// 1 = Only 1 Intersection satisfying Z cutoff
// 2 = Only 1 Intersection detectable dist(x,y)<20
// 3 = 2 Intersections 

void Analyzer_DisplacedMuon(TString inputFilePath,
			    TString outputDir,
			    float TP_maxD0 = 1.9,
			    float TP_minD0 = 0.0004196)
{
  TChain *tree = new TChain("L1TrackNtuple/eventTree"); 
  tree->Add(inputFilePath);
  //TChain *vertTree = new TChain("L1TrackNtuple/dispVertTree");
  //vertTree->Add(inputFilePath);
  std::string inputFileString(inputFilePath.Data());
  inputFileString = inputFileString.substr(inputFileString.find_last_of("/")+1);
  TString inputFile(inputFileString);
  std::cout<<"input: "<<inputFile<<std::endl;
  gROOT->SetBatch();
  gErrorIgnoreLevel = kWarning;

  SetPlotStyle();
  float barrelEta = 0.95;
  bool useEmulation = false;

  //Vertex parameter vectors
  vector<int> *trkVert_firstIndexTrk;
  vector<int> *trkVert_secondIndexTrk;
  vector<int> *trkVert_inTraj;
  vector<float> *trkVert_d_T;
  vector<float> *trkVert_R_T;
  vector<float> *trkVert_cos_T;
  vector<float> *trkVert_del_Z;
  vector<float> *trkVert_x;
  vector<float> *trkVert_y;
  vector<float> *trkVert_z;
  vector<float> *trkVert_openingAngle;
  vector<float> *trkVert_parentPt;
  vector<bool> *trkVert_isReal;
  vector<float> *trkVert_score;

  vector<float> *tpVert_d_T;
  vector<float> *tpVert_R_T;
  vector<float> *tpVert_cos_T;
  vector<float> *tpVert_x;
  vector<float> *tpVert_y;
  vector<float> *tpVert_z;
  vector<float> *tpVert_openingAngle;
  vector<float> *tpVert_parentPt;

  vector<float>   *trk_pt;
  vector<float>   *trk_eta;
  vector<float>   *trk_phi;
  vector<float>   *trk_d0;
  vector<float>   *trk_rinv;
  vector<float>   *trk_z0;
  vector<float>   *trk_chi2rphi;
  vector<float>   *trk_chi2rz;
  vector<float>   *trk_bendchi2;
  vector<int>     *trk_nstub;
  vector<int>     *trkExt_fake;
  vector<float>     *trk_MVA1;
  vector<int>     *trk_matchtp_pdgid;
  vector<int>     *trk_matchtp_isHToB;
  vector<bool>    *trk_matchtp_isHard;
  vector<float>   *trk_matchtp_pt;
  vector<float>   *trk_matchtp_eta;
  vector<float>   *trk_matchtp_phi;
  vector<float>   *trk_matchtp_z0;
  vector<float>   *trk_matchtp_d0;
  vector<float>   *trk_matchtp_x;
  vector<float>   *trk_matchtp_y;
  vector<float>   *trk_matchtp_z;
  vector<float>   *tp_pt;
  vector<float>   *tp_eta;
  vector<float>   *tp_phi;
  vector<float>   *tp_dxy;
  vector<float>   *tp_d0;
  vector<float>   *tp_z0;
  vector<float>   *tp_x;
  vector<float>   *tp_y;
  vector<float>   *tp_z;
  vector<int>     *tp_pdgid;
  vector<bool>    *tp_isHToB;
  vector<bool>    *tp_isHard;
  vector<int>     *tp_nmatch;
  vector<int>     *tp_nstub;
  vector<int>     *tp_eventid;
  vector<int>     *tp_charge;
  vector<float>   *matchtrk_pt;
  vector<float>   *matchtrk_eta;
  vector<float>   *matchtrk_phi;
  vector<float>   *matchtrk_z0;
  vector<float>   *matchtrk_d0;
  vector<float>   *matchtrk_rinv;
  vector<float>   *matchtrk_chi2rphi;
  vector<float>   *matchtrk_chi2rz;
  vector<float>   *matchtrk_bendchi2;
  vector<float>   *matchtrk_MVA1;
  vector<int>     *matchtrk_nstub;

  TBranch *b_trkVert_firstIndexTrk;
  TBranch *b_trkVert_secondIndexTrk;
  TBranch *b_trkVert_inTraj;
  TBranch *b_trkVert_d_T;
  TBranch *b_trkVert_R_T;
  TBranch *b_trkVert_cos_T;
  TBranch *b_trkVert_del_Z;
  TBranch *b_trkVert_x;
  TBranch *b_trkVert_y;
  TBranch *b_trkVert_z;
  TBranch *b_trkVert_openingAngle;
  TBranch *b_trkVert_parentPt;
  TBranch *b_trkVert_isReal;
  TBranch *b_trkVert_score;
  TBranch        *b_trk_pt;
  TBranch        *b_trk_eta;
  TBranch        *b_trk_phi;
  TBranch        *b_trk_d0;
  TBranch        *b_trk_rinv;
  TBranch        *b_trk_z0;
  TBranch        *b_trk_chi2rphi;
  TBranch        *b_trk_chi2rz;
  TBranch        *b_trk_bendchi2;
  TBranch        *b_trk_nstub;
  TBranch        *b_trkExt_fake;
  TBranch        *b_trk_MVA1;
  TBranch        *b_trk_matchtp_pdgid;
  TBranch        *b_trk_matchtp_isHToB;
  TBranch        *b_trk_matchtp_isHard;
  TBranch        *b_trk_matchtp_pt;
  TBranch        *b_trk_matchtp_eta;
  TBranch        *b_trk_matchtp_phi;
  TBranch        *b_trk_matchtp_z0;
  TBranch        *b_trk_matchtp_d0;
  TBranch        *b_trk_matchtp_x;
  TBranch        *b_trk_matchtp_y;
  TBranch        *b_trk_matchtp_z;
  TBranch        *b_tp_pt;
  TBranch        *b_tp_eta;
  TBranch        *b_tp_phi;
  TBranch        *b_tp_dxy;
  TBranch        *b_tp_d0;
  TBranch        *b_tp_z0;
  TBranch        *b_tp_x;
  TBranch        *b_tp_y;
  TBranch        *b_tp_z;
  TBranch        *b_tp_pdgid;
  TBranch        *b_tp_isHToB;
  TBranch        *b_tp_isHard;
  TBranch        *b_tp_nmatch;
  TBranch        *b_tp_nstub;
  TBranch        *b_tp_eventid;
  TBranch        *b_tp_charge;
  TBranch        *b_matchtrk_pt;
  TBranch        *b_matchtrk_eta;
  TBranch        *b_matchtrk_phi;
  TBranch        *b_matchtrk_z0;
  TBranch        *b_matchtrk_d0;
  TBranch        *b_matchtrk_rinv;
  TBranch        *b_matchtrk_chi2rphi;
  TBranch        *b_matchtrk_chi2rz;
  TBranch        *b_matchtrk_bendchi2;
  TBranch        *b_matchtrk_MVA1;
  TBranch        *b_matchtrk_nstub;

  trkVert_firstIndexTrk = 0;
  trkVert_secondIndexTrk = 0;
  trkVert_inTraj = 0;
  trkVert_d_T = 0;
  trkVert_R_T = 0;
  trkVert_cos_T = 0;
  trkVert_del_Z = 0;
  trkVert_x = 0;
  trkVert_y = 0;
  trkVert_z = 0;
  trkVert_openingAngle = 0;
  trkVert_parentPt = 0;
  trkVert_isReal = 0;
  trkVert_score = 0;
  trk_pt = 0;
  trk_eta = 0;
  trk_phi = 0;
  trk_d0 = 0;
  trk_rinv = 0;
  trk_z0 = 0;
  trk_chi2rphi = 0;
  trk_chi2rz = 0;
  trk_bendchi2 = 0;
  trk_nstub = 0;
  trkExt_fake = 0;
  trk_MVA1 = 0;
  trk_matchtp_pdgid = 0;
  trk_matchtp_isHToB = 0;
  trk_matchtp_isHard = 0;
  trk_matchtp_pt = 0;
  trk_matchtp_eta = 0;
  trk_matchtp_phi = 0;
  trk_matchtp_z0 = 0;
  trk_matchtp_d0 = 0;
  trk_matchtp_x = 0;
  trk_matchtp_y = 0;
  trk_matchtp_z = 0;
  tp_pt = 0;
  tp_eta = 0;
  tp_phi = 0;
  tp_dxy = 0;
  tp_d0 = 0;
  tp_z0 = 0;
  tp_x = 0;
  tp_y = 0;
  tp_z = 0;
  tp_pdgid = 0;
  tp_isHToB = 0;
  tp_isHard = 0;
  tp_nmatch = 0;
  tp_nstub = 0;
  tp_eventid = 0;
  tp_charge = 0;
  matchtrk_pt = 0;
  matchtrk_eta = 0;
  matchtrk_phi = 0;
  matchtrk_z0 = 0;
  matchtrk_d0 = 0;
  matchtrk_rinv = 0;
  matchtrk_chi2rphi = 0;
  matchtrk_chi2rz = 0;
  matchtrk_bendchi2 = 0;
  matchtrk_MVA1 = 0;
  matchtrk_nstub = 0;

  //tree->SetMakeClass(1);
  if(useEmulation){
    tree->SetBranchAddress("dvEmu_firstIndexTrk", &trkVert_firstIndexTrk, &b_trkVert_firstIndexTrk);
    tree->SetBranchAddress("dvEmu_secondIndexTrk", &trkVert_secondIndexTrk, &b_trkVert_secondIndexTrk);
    tree->SetBranchAddress("dvEmu_inTraj", &trkVert_inTraj, &b_trkVert_inTraj);
    tree->SetBranchAddress("dvEmu_d_T", &trkVert_d_T, &b_trkVert_d_T);
    tree->SetBranchAddress("dvEmu_R_T", &trkVert_R_T, &b_trkVert_R_T);
    tree->SetBranchAddress("dvEmu_cos_T", &trkVert_cos_T, &b_trkVert_cos_T);
    tree->SetBranchAddress("dvEmu_del_Z", &trkVert_del_Z, &b_trkVert_del_Z);
    tree->SetBranchAddress("dvEmu_x", &trkVert_x, &b_trkVert_x);
    tree->SetBranchAddress("dvEmu_y", &trkVert_y, &b_trkVert_y);
    tree->SetBranchAddress("dvEmu_z", &trkVert_z, &b_trkVert_z);
    tree->SetBranchAddress("dvEmu_openingAngle", &trkVert_openingAngle, &b_trkVert_openingAngle);
    tree->SetBranchAddress("dvEmu_parentPt", &trkVert_parentPt, &b_trkVert_parentPt);
    tree->SetBranchAddress("dvEmu_isReal", &trkVert_isReal, &b_trkVert_isReal);
    tree->SetBranchAddress("dvEmu_score", &trkVert_score, &b_trkVert_score);
    tree->SetBranchAddress("trkExtEmu_pt", &trk_pt, &b_trk_pt);
    tree->SetBranchAddress("trkExtEmu_eta", &trk_eta, &b_trk_eta);
    tree->SetBranchAddress("trkExtEmu_phi", &trk_phi, &b_trk_phi);
    tree->SetBranchAddress("trkExtEmu_d0", &trk_d0, &b_trk_d0);
    tree->SetBranchAddress("trkExtEmu_rho", &trk_rinv, &b_trk_rinv);
    tree->SetBranchAddress("trkExtEmu_z0", &trk_z0, &b_trk_z0);
    tree->SetBranchAddress("trkExtEmu_chi2rphi", &trk_chi2rphi, &b_trk_chi2rphi);
    tree->SetBranchAddress("trkExtEmu_chi2rz", &trk_chi2rz, &b_trk_chi2rz);
    tree->SetBranchAddress("trkExtEmu_bendchi2", &trk_bendchi2, &b_trk_bendchi2);
    tree->SetBranchAddress("trkExtEmu_nstub", &trk_nstub, &b_trk_nstub);
    tree->SetBranchAddress("trkExt_fake", &trkExt_fake, &b_trkExt_fake);
    tree->SetBranchAddress("trkExtEmu_MVA", &trk_MVA1, &b_trk_MVA1);
  }
  else{
    tree->SetBranchAddress("dv_firstIndexTrk", &trkVert_firstIndexTrk, &b_trkVert_firstIndexTrk);
    tree->SetBranchAddress("dv_secondIndexTrk", &trkVert_secondIndexTrk, &b_trkVert_secondIndexTrk);
    tree->SetBranchAddress("dv_inTraj", &trkVert_inTraj, &b_trkVert_inTraj);
    tree->SetBranchAddress("dv_d_T", &trkVert_d_T, &b_trkVert_d_T);
    tree->SetBranchAddress("dv_R_T", &trkVert_R_T, &b_trkVert_R_T);
    tree->SetBranchAddress("dv_cos_T", &trkVert_cos_T, &b_trkVert_cos_T);
    tree->SetBranchAddress("dv_del_Z", &trkVert_del_Z, &b_trkVert_del_Z);
    tree->SetBranchAddress("dv_x", &trkVert_x, &b_trkVert_x);
    tree->SetBranchAddress("dv_y", &trkVert_y, &b_trkVert_y);
    tree->SetBranchAddress("dv_z", &trkVert_z, &b_trkVert_z);
    tree->SetBranchAddress("dv_openingAngle", &trkVert_openingAngle, &b_trkVert_openingAngle);
    tree->SetBranchAddress("dv_parentPt", &trkVert_parentPt, &b_trkVert_parentPt);
    tree->SetBranchAddress("dv_isReal", &trkVert_isReal, &b_trkVert_isReal);
    tree->SetBranchAddress("dv_score", &trkVert_score, &b_trkVert_score);
    tree->SetBranchAddress("trkExt_pt", &trk_pt, &b_trk_pt);
    tree->SetBranchAddress("trkExt_eta", &trk_eta, &b_trk_eta);
    tree->SetBranchAddress("trkExt_phi", &trk_phi, &b_trk_phi);
    tree->SetBranchAddress("trkExt_d0", &trk_d0, &b_trk_d0);
    tree->SetBranchAddress("trkExt_rinv", &trk_rinv, &b_trk_rinv);
    tree->SetBranchAddress("trkExt_z0", &trk_z0, &b_trk_z0);
    tree->SetBranchAddress("trkExt_chi2rphi", &trk_chi2rphi, &b_trk_chi2rphi);
    tree->SetBranchAddress("trkExt_chi2rz", &trk_chi2rz, &b_trk_chi2rz);
    tree->SetBranchAddress("trkExt_bendchi2", &trk_bendchi2, &b_trk_bendchi2);
    tree->SetBranchAddress("trkExt_nstub", &trk_nstub, &b_trk_nstub);
    tree->SetBranchAddress("trkExt_fake", &trkExt_fake, &b_trkExt_fake);
    tree->SetBranchAddress("trkExt_MVA", &trk_MVA1, &b_trk_MVA1);
  }
  tree->SetBranchAddress("trkExt_matchtp_pdgid", &trk_matchtp_pdgid, &b_trk_matchtp_pdgid);
  tree->SetBranchAddress("trkExt_matchtp_isHToB", &trk_matchtp_isHToB, &b_trk_matchtp_isHToB);
  tree->SetBranchAddress("trkExt_matchtp_isHard", &trk_matchtp_isHard, &b_trk_matchtp_isHard);
  tree->SetBranchAddress("trkExt_matchtp_pt", &trk_matchtp_pt, &b_trk_matchtp_pt);
  tree->SetBranchAddress("trkExt_matchtp_eta", &trk_matchtp_eta, &b_trk_matchtp_eta);
  tree->SetBranchAddress("trkExt_matchtp_phi", &trk_matchtp_phi, &b_trk_matchtp_phi);
  tree->SetBranchAddress("trkExt_matchtp_z0", &trk_matchtp_z0, &b_trk_matchtp_z0);
  tree->SetBranchAddress("trkExt_matchtp_d0", &trk_matchtp_d0, &b_trk_matchtp_d0);
  tree->SetBranchAddress("trkExt_matchtp_x", &trk_matchtp_x, &b_trk_matchtp_x);
  tree->SetBranchAddress("trkExt_matchtp_y", &trk_matchtp_y, &b_trk_matchtp_y);
  tree->SetBranchAddress("trkExt_matchtp_z", &trk_matchtp_z, &b_trk_matchtp_z);
  tree->SetBranchAddress("tp_pt", &tp_pt, &b_tp_pt);
  tree->SetBranchAddress("tp_eta", &tp_eta, &b_tp_eta);
  tree->SetBranchAddress("tp_phi", &tp_phi, &b_tp_phi);
  tree->SetBranchAddress("tp_dxy", &tp_dxy, &b_tp_dxy);
  tree->SetBranchAddress("tp_d0", &tp_d0, &b_tp_d0);
  tree->SetBranchAddress("tp_z0", &tp_z0, &b_tp_z0);
  tree->SetBranchAddress("tp_x", &tp_x, &b_tp_x);
  tree->SetBranchAddress("tp_y", &tp_y, &b_tp_y);
  tree->SetBranchAddress("tp_z", &tp_z, &b_tp_z);
  tree->SetBranchAddress("tp_pdgid", &tp_pdgid, &b_tp_pdgid);
  tree->SetBranchAddress("tp_isHToB", &tp_isHToB, &b_tp_isHToB);
  tree->SetBranchAddress("tp_isHard", &tp_isHard, &b_tp_isHard);
  tree->SetBranchAddress("tp_nmatch", &tp_nmatch, &b_tp_nmatch);
  tree->SetBranchAddress("tp_nstub", &tp_nstub, &b_tp_nstub);
  tree->SetBranchAddress("tp_eventid", &tp_eventid, &b_tp_eventid);
  tree->SetBranchAddress("tp_charge", &tp_charge, &b_tp_charge);
  if(useEmulation){
    tree->SetBranchAddress("matchtrkExtEmu_pt", &matchtrk_pt, &b_matchtrk_pt);
    tree->SetBranchAddress("matchtrkExtEmu_eta", &matchtrk_eta, &b_matchtrk_eta);
    tree->SetBranchAddress("matchtrkExtEmu_phi", &matchtrk_phi, &b_matchtrk_phi);
    tree->SetBranchAddress("matchtrkExtEmu_z0", &matchtrk_z0, &b_matchtrk_z0);
    tree->SetBranchAddress("matchtrkExtEmu_d0", &matchtrk_d0, &b_matchtrk_d0);
    tree->SetBranchAddress("matchtrkExtEmu_rho", &matchtrk_rinv, &b_matchtrk_rinv);
    tree->SetBranchAddress("matchtrkExtEmu_chi2rphi", &matchtrk_chi2rphi, &b_matchtrk_chi2rphi);
    tree->SetBranchAddress("matchtrkExtEmu_chi2rz", &matchtrk_chi2rz, &b_matchtrk_chi2rz);
    tree->SetBranchAddress("matchtrkExtEmu_bendchi2", &matchtrk_bendchi2, &b_matchtrk_bendchi2);
    tree->SetBranchAddress("matchtrkExtEmu_MVA", &matchtrk_MVA1, &b_matchtrk_MVA1);
    tree->SetBranchAddress("matchtrkExtEmu_nstub", &matchtrk_nstub, &b_matchtrk_nstub);
  }
  else{
    tree->SetBranchAddress("matchtrkExt_pt", &matchtrk_pt, &b_matchtrk_pt);
    tree->SetBranchAddress("matchtrkExt_eta", &matchtrk_eta, &b_matchtrk_eta);
    tree->SetBranchAddress("matchtrkExt_phi", &matchtrk_phi, &b_matchtrk_phi);
    tree->SetBranchAddress("matchtrkExt_z0", &matchtrk_z0, &b_matchtrk_z0);
    tree->SetBranchAddress("matchtrkExt_d0", &matchtrk_d0, &b_matchtrk_d0);
    tree->SetBranchAddress("matchtrkExt_rinv", &matchtrk_rinv, &b_matchtrk_rinv);
    tree->SetBranchAddress("matchtrkExt_chi2rphi", &matchtrk_chi2rphi, &b_matchtrk_chi2rphi);
    tree->SetBranchAddress("matchtrkExt_chi2rz", &matchtrk_chi2rz, &b_matchtrk_chi2rz);
    tree->SetBranchAddress("matchtrkExt_bendchi2", &matchtrk_bendchi2, &b_matchtrk_bendchi2);
    tree->SetBranchAddress("matchtrkExt_MVA", &matchtrk_MVA1, &b_matchtrk_MVA1);
    tree->SetBranchAddress("matchtrkExt_nstub", &matchtrk_nstub, &b_matchtrk_nstub);
  }
  //preselection cuts and plots definitions
  // Cut assumptions: first cut is maxEta
  std::vector<std::unique_ptr<Cut>> preselCuts;
  std::unique_ptr<TypedCut<float>> cut0(new TypedCut<float>("maxEta","max #eta",&trk_eta,2.4,true));
  preselCuts.push_back(std::move(cut0));
  std::unique_ptr<TypedCut<float>> cut1(new TypedCut<float>("maxChi2rzdof","max #chi^{2}_{rz}",&trk_chi2rz,3.0,false));
  preselCuts.push_back(std::move(cut1));
  std::unique_ptr<TypedCut<float>> cut4(new TypedCut<float>("minMVA1","min MVA1",&trk_MVA1,0.2,false));
  preselCuts.push_back(std::move(cut4));
  std::unique_ptr<TypedCut<float>> cut5(new TypedCut<float>("minMVA1_D","min MVA1 D",&trk_MVA1,0.5,false));
  preselCuts.push_back(std::move(cut5));
  std::unique_ptr<TypedCut<int>> cut6(new TypedCut<int>("minNumStub_overlap","Quality",&trk_nstub,5,true));
  preselCuts.push_back(std::move(cut6));
  std::unique_ptr<TypedCut<float>> cut7(new TypedCut<float>("minPt","min p_{T}",&trk_pt,3.0,true));
  preselCuts.push_back(std::move(cut7));
  std::unique_ptr<TypedCut<float>> cut8(new TypedCut<float>("minD0_barrel","min d_{0} Bar",&trk_d0,0.06,false));
  preselCuts.push_back(std::move(cut8));
  std::unique_ptr<TypedCut<float>> cut9(new TypedCut<float>("minD0_disk","min d_{0}",&trk_d0,0.08,true));
  preselCuts.push_back(std::move(cut9));

  std::vector<std::unique_ptr<Cut>> preselCutsTP;
  std::unique_ptr<TypedCut<float>> tpCut0(new TypedCut<float>("maxEta","max #eta",&tp_eta,2.4,true));
  preselCutsTP.push_back(std::move(tpCut0));
  std::unique_ptr<TypedCut<float>> tpCut1(new TypedCut<float>("minPt","min p_{T}",&tp_pt,3.0,true));
  preselCutsTP.push_back(std::move(tpCut1));
  std::unique_ptr<TypedCut<float>> tpCut2(new TypedCut<float>("minD0_barrel","min d_{0} Barrel",&tp_d0,0.06,true));
  preselCutsTP.push_back(std::move(tpCut2));
  std::unique_ptr<TypedCut<float>> tpCut3(new TypedCut<float>("minD0_disk","min d_{0} Disk",&tp_d0,0.08,true));
  preselCutsTP.push_back(std::move(tpCut3));

  std::vector<std::unique_ptr<Plot>> varCutFlows;
  std::unique_ptr<TypedPlot<float>> plot0(new TypedPlot<float>("d0","cm",&trk_d0,20,logspace(0.01,10.0,20)));
  varCutFlows.push_back(std::move(plot0));
  std::unique_ptr<TypedPlot<float>> plot1(new TypedPlot<float>("pt","GeV",&trk_pt,20,logspace(2.0,100.0,20)));
  varCutFlows.push_back(std::move(plot1));
  std::unique_ptr<TypedPlot<float>> plot2(new TypedPlot<float>("eta","",&trk_eta,50,-2.5,2.5));
  varCutFlows.push_back(std::move(plot2));
  std::unique_ptr<TypedPlot<float>> plot3(new TypedPlot<float>("z0","cm",&trk_z0,100,-20.0,20.0));
  varCutFlows.push_back(std::move(plot3));
  std::unique_ptr<TypedPlot<float>> plot4(new TypedPlot<float>("phi","",&trk_phi,100,-2*TMath::Pi(),2*TMath::Pi()));
  varCutFlows.push_back(std::move(plot4));
  std::unique_ptr<TypedPlot<float>> plot6(new TypedPlot<float>("MVA1","",&trk_MVA1,100,0.0,1.0));
  varCutFlows.push_back(std::move(plot6));
  std::unique_ptr<TypedPlot<float>> plot8(new TypedPlot<float>("chi2rphidof","",&trk_chi2rphi,100,0.0,6.0));
  varCutFlows.push_back(std::move(plot8));
  std::unique_ptr<TypedPlot<float>> plot9(new TypedPlot<float>("chi2rzdof","",&trk_chi2rz,100,0.0,6.0));
  varCutFlows.push_back(std::move(plot9));
  std::unique_ptr<TypedPlot<float>> plot10(new TypedPlot<float>("bendchi2","",&trk_bendchi2,100,0.0,10.0));
  varCutFlows.push_back(std::move(plot10));

  std::vector<std::unique_ptr<Plot>> varCutFlowsTP;
  std::unique_ptr<TypedPlot<float>> tpPlot0(new TypedPlot<float>("d0","cm",&tp_d0,20,logspace(0.01,10.0,20)));
  varCutFlowsTP.push_back(std::move(tpPlot0));
  std::unique_ptr<TypedPlot<float>> tpPlot1(new TypedPlot<float>("pt","GeV",&tp_pt,20,logspace(2.0,100.0,20)));
  varCutFlowsTP.push_back(std::move(tpPlot1));
  std::unique_ptr<TypedPlot<float>> tpPlot2(new TypedPlot<float>("eta","",&tp_eta,50,-2.5,2.5));
  varCutFlowsTP.push_back(std::move(tpPlot2));
  std::unique_ptr<TypedPlot<float>> tpPlot3(new TypedPlot<float>("z0","cm",&tp_z0,100,-20.0,20.0));
  varCutFlowsTP.push_back(std::move(tpPlot3));
  std::unique_ptr<TypedPlot<float>> tpPlot4(new TypedPlot<float>("phi","",&tp_phi,100,-2*TMath::Pi(),2*TMath::Pi()));
  varCutFlowsTP.push_back(std::move(tpPlot4));
  std::unique_ptr<TypedPlot<float>> tpPlot6(new TypedPlot<float>("dxy","cm",&tp_dxy,50,-2.0,2.0));
  varCutFlowsTP.push_back(std::move(tpPlot6));
  
  std::vector<std::pair<std::unique_ptr<Plot>,std::unique_ptr<Plot> > > varCutFlows2D;
  std::unique_ptr<TypedPlot<float>> plot0X(new TypedPlot<float>("d0","cm",&trk_d0,200,-2.0,2.0));
  std::unique_ptr<TypedPlot<float>> plot0Y(new TypedPlot<float>("pt","GeV",&trk_pt,200,0.0,30.0));
  varCutFlows2D.push_back({std::move(plot0X),std::move(plot0Y)});
  std::unique_ptr<TypedPlot<float>> plot1X(new TypedPlot<float>("eta","",&trk_eta,200,-2.4,2.4));
  std::unique_ptr<TypedPlot<float>> plot1Y(new TypedPlot<float>("pt","GeV",&trk_pt,200,0.0,30.0));
  varCutFlows2D.push_back({std::move(plot1X),std::move(plot1Y)});
  std::unique_ptr<TypedPlot<float>> plot2X(new TypedPlot<float>("d0","cm",&trk_d0,200,-2.0,2.0));
  std::unique_ptr<TypedPlot<float>> plot2Y(new TypedPlot<float>("eta","",&trk_eta,200,-2.4,2.4));
  varCutFlows2D.push_back({std::move(plot2X),std::move(plot2Y)});
  std::unique_ptr<TypedPlot<float>> plot3X(new TypedPlot<float>("eta","",&trk_eta,200,-2.4,2.4));
  std::unique_ptr<TypedPlot<int>> plot3Y(new TypedPlot<int>("nstub","",&trk_nstub,7,0.0,7.0));
  varCutFlows2D.push_back({std::move(plot3X),std::move(plot3Y)});

  std::vector<std::pair<std::unique_ptr<Plot>,std::unique_ptr<Plot> > > varCutFlowsTP2D;
  std::unique_ptr<TypedPlot<float>> tpPlot0X(new TypedPlot<float>("d0","cm",&tp_d0,200,-2.0,2.0));
  std::unique_ptr<TypedPlot<float>> tpPlot0Y(new TypedPlot<float>("pt","GeV",&tp_pt,200,0.0,30.0));
  varCutFlowsTP2D.push_back({std::move(tpPlot0X),std::move(tpPlot0Y)});
  std::unique_ptr<TypedPlot<float>> tpPlot1X(new TypedPlot<float>("eta","",&tp_eta,200,-2.4,2.4));
  std::unique_ptr<TypedPlot<float>> tpPlot1Y(new TypedPlot<float>("pt","GeV",&tp_pt,200,0.0,30.0));
  varCutFlowsTP2D.push_back({std::move(tpPlot1X),std::move(tpPlot1Y)});
  std::unique_ptr<TypedPlot<float>> tpPlot2X(new TypedPlot<float>("d0","cm",&tp_d0,200,-2.0,2.0));
  std::unique_ptr<TypedPlot<float>> tpPlot2Y(new TypedPlot<float>("eta","",&tp_eta,200,-2.4,2.4));
  varCutFlowsTP2D.push_back({std::move(tpPlot2X),std::move(tpPlot2Y)});
  std::unique_ptr<TypedPlot<float>> tpPlot3X(new TypedPlot<float>("eta","",&tp_eta,200,-2.4,2.4));
  std::unique_ptr<TypedPlot<int>> tpPlot3Y(new TypedPlot<int>("nstub","",&tp_nstub,7,0.0,7.0));
  varCutFlowsTP2D.push_back({std::move(tpPlot3X),std::move(tpPlot3Y)});

  //std::vector<TString> trackType = {"primary","np","fake","PU","notHiggs"};
  std::vector<TString> trackType = {"primary","np"};
  //std::vector<TString> tpType = {"primary","np","PU","notHiggs","match",""};
  std::vector<TString> tpType = {"primary","np","match",""};
  std::vector<TString> plotModifiers = {"","_H","_L","_P","_D","_barrel","_disk"};
  if(!detailedPlots) plotModifiers = {""};
  uint preselCutsSize = 0;
  for(uint i=0; i<preselCuts.size(); i++){
    if(preselCuts[i]->getDoPlot()) preselCutsSize++;
  }
  TH1F* preselCutFlows[varCutFlows.size()][trackType.size()][preselCutsSize][plotModifiers.size()];
  TH2F* preselCutFlows2D[varCutFlows2D.size()][trackType.size()][preselCutsSize][plotModifiers.size()];
  TH1F* preselCutFlowsTP[varCutFlowsTP.size()][tpType.size()][preselCutsTP.size()][plotModifiers.size()];
  TH2F* preselCutFlowsTP2D[varCutFlowsTP2D.size()][tpType.size()][preselCutsTP.size()][plotModifiers.size()];
  //std::map<string,int> numPartCutFlows[trackType.size()][preselCuts.size()];
  //std::map<string,int> numPartCutFlowsTP[tpType.size()][preselCutsTP.size()];
  
  for(uint it=0; it<varCutFlows.size(); ++it){
    for(uint i=0; i<trackType.size(); ++i){
      uint i_plot = -1;
      for(uint jt=0; jt<preselCuts.size(); ++jt){
	if(preselCuts[jt]->getDoPlot()){
	  i_plot++;
	}
	else{
	  continue;
	}
	for(uint j=0; j<plotModifiers.size(); ++j){
	  TString name = "h_trk_"+varCutFlows[it]->getVarName()+"_"+trackType[i]+"_"+preselCuts[jt]->getCutName()+"Cut"+plotModifiers[j];
	  //std::cout<<"name: "<<name<<std::endl;
	  if(varCutFlows[it]->getBool()){
	    //std::cout<<"setting bins"<<std::endl;
	    TString labels = name+"; Track "+varCutFlows[it]->getVarName()+" ("+varCutFlows[it]->getUnit()+") ; Events ";
	    std::vector<float> bins = varCutFlows[it]->getBins();
	    TH1F* hist = new TH1F(name,labels,varCutFlows[it]->getNumBins(),bins.data());
	    preselCutFlows[it][i][i_plot][j] = hist;
	    TString varString = varCutFlows[it]->getVarName();
	    if(varString.Contains("d0") || varString.Contains("pt")){
	      //std::cout<<"labels: "<<labels<<std::endl;
	      std::string binValues = "[";
	      std::string binWidths = "[";

	      for(int ibin=1; ibin<(preselCutFlows[it][i][i_plot][j]->GetNbinsX()+1); ibin++){
		binValues+=to_string(preselCutFlows[it][i][i_plot][j]->GetBinContent(ibin)) + ", ";
		binWidths+=to_string(preselCutFlows[it][i][i_plot][j]->GetBinWidth(ibin)) + ", ";

	      }
	      binValues+="]";
	      binWidths+="]";

	      //std::cout<<"binValues: "<<binValues<<std::endl;
	      //std::cout<<"binWidths: "<<binWidths<<std::endl;

	    }
	  }
	  else{
	    //std::cout<<"else"<<std::endl;
	    float binWidth = (varCutFlows[it]->getMaxBin() - varCutFlows[it]->getMinBin()) / varCutFlows[it]->getNumBins();
	    TString binLabel = std::to_string(binWidth);
	    TString labels = name+"; Track "+varCutFlows[it]->getVarName()+" ("+varCutFlows[it]->getUnit()+") ; Events / "+binLabel+" "+varCutFlows[it]->getUnit();
	    TH1F* hist = new TH1F(name,labels,varCutFlows[it]->getNumBins(),varCutFlows[it]->getMinBin(),varCutFlows[it]->getMaxBin());
	    preselCutFlows[it][i][i_plot][j] = hist;
	  }
	}
      }
    }
  }
  
  for(uint it=0; it<varCutFlows2D.size(); ++it){
    for(uint i=0; i<trackType.size(); ++i){
      uint i_plot = -1;
      for(uint jt=0; jt<preselCuts.size(); ++jt){
	if(preselCuts[jt]->getDoPlot()){
	  i_plot++;
	}
	else{
	  continue;
	}
	for(uint j=0; j<plotModifiers.size(); ++j){
	  TString name = "h_trk_"+varCutFlows2D[it].second->getVarName()+"_vs_"+varCutFlows2D[it].first->getVarName()+"_"+trackType[i]+"_"+preselCuts[jt]->getCutName()+"Cut"+plotModifiers[j];
	  TString labels = name+"; Track "+varCutFlows2D[it].first->getVarName()+" ("+varCutFlows2D[it].first->getUnit()+") ; Track "+varCutFlows2D[it].second->getVarName()+" ("+varCutFlows2D[it].second->getUnit()+")";
	  TH2F* hist = new TH2F(name,labels,varCutFlows2D[it].first->getNumBins(),varCutFlows2D[it].first->getMinBin(),varCutFlows2D[it].first->getMaxBin(),varCutFlows2D[it].second->getNumBins(),varCutFlows2D[it].second->getMinBin(),varCutFlows2D[it].second->getMaxBin());
	  preselCutFlows2D[it][i][i_plot][j] = hist;
	}
      }
    }
  }

  for(uint it=0; it<varCutFlowsTP.size(); ++it){
    for(uint i=0; i<tpType.size(); ++i){
      for(uint jt=0; jt<preselCutsTP.size(); ++jt){
	for(uint j=0; j<plotModifiers.size(); ++j){
	  TString name = "h_tp_"+varCutFlowsTP[it]->getVarName()+"_"+tpType[i]+"_"+preselCutsTP[jt]->getCutName()+"Cut"+plotModifiers[j];
	  if(varCutFlowsTP[it]->getMaxBin()==varCutFlowsTP[it]->getMinBin()){
	    TString labels = name+"; Tp "+varCutFlowsTP[it]->getVarName()+" ("+varCutFlowsTP[it]->getUnit()+") ; Events ";
	    std::vector<float> bins = varCutFlowsTP[it]->getBins();
	    TH1F* hist = new TH1F(name,labels,varCutFlowsTP[it]->getNumBins(),bins.data());
	    preselCutFlowsTP[it][i][jt][j] = hist;
	  }
	  else{
	    float binWidth = (varCutFlowsTP[it]->getMaxBin() - varCutFlowsTP[it]->getMinBin()) / varCutFlowsTP[it]->getNumBins();
	    TString binLabel = std::to_string(binWidth);
	    TString labels = name+"; Tp "+varCutFlowsTP[it]->getVarName()+" ("+varCutFlowsTP[it]->getUnit()+") ; Events / "+binLabel+" "+varCutFlowsTP[it]->getUnit();
	    TH1F* hist = new TH1F(name,labels,varCutFlowsTP[it]->getNumBins(),varCutFlowsTP[it]->getMinBin(),varCutFlowsTP[it]->getMaxBin());
	    preselCutFlowsTP[it][i][jt][j] = hist;
	  }
	}
      }
    }
  }
    
  for(uint it=0; it<varCutFlowsTP2D.size(); ++it){
    for(uint i=0; i<tpType.size(); ++i){
      for(uint jt=0; jt<preselCutsTP.size(); ++jt){
	for(uint j=0; j<plotModifiers.size(); ++j){
	  TString name = "h_tp_"+varCutFlowsTP2D[it].second->getVarName()+"_vs_"+varCutFlowsTP2D[it].first->getVarName()+"_"+tpType[i]+"_"+preselCutsTP[jt]->getCutName()+"Cut"+plotModifiers[j];
	  TString labels = name+"; Tp "+varCutFlowsTP2D[it].first->getVarName()+" ("+varCutFlowsTP2D[it].first->getUnit()+") ; Tp "+varCutFlowsTP2D[it].second->getVarName()+" ("+varCutFlowsTP2D[it].second->getUnit()+")";
	  TH2F* hist = new TH2F(name,labels,varCutFlowsTP2D[it].first->getNumBins(),varCutFlowsTP2D[it].first->getMinBin(),varCutFlowsTP2D[it].first->getMaxBin(),varCutFlowsTP2D[it].second->getNumBins(),varCutFlowsTP2D[it].second->getMinBin(),varCutFlowsTP2D[it].second->getMaxBin());
	  preselCutFlowsTP2D[it][i][jt][j] = hist;
	}
      }
    }
  }

  //vertex cuts and plots definitions
  std::vector<std::unique_ptr<Cut>> vertCuts;
  std::unique_ptr<TypedCut<float>> vertCut0(new TypedCut<float>("minR_T_Res","min R_{T} #sigma_{d0}",&trkVert_R_T,d0_res,false));
  vertCuts.push_back(std::move(vertCut0));
  std::unique_ptr<TypedCut<float>> vertCut1(new TypedCut<float>("maxR_T","max R_{T}",&trkVert_R_T,20.0,false));
  vertCuts.push_back(std::move(vertCut1));
  std::unique_ptr<TypedCut<float>> vertCut2(new TypedCut<float>("max_trk_sumCharge","max #Sigma q",&trk_rinv,0,false));
  vertCuts.push_back(std::move(vertCut2));
  std::unique_ptr<TypedCut<float>> vertCut3(new TypedCut<float>("min_trk_sumCharge","min #Sigma q",&trk_rinv,0,true));
  vertCuts.push_back(std::move(vertCut3));
#if 0
  std::unique_ptr<TypedCut<float>> vertCut4(new TypedCut<float>("max_trk_sumBendChi2","max #Sigma #chi^{2}_{bend}",&trk_bendchi2,14.0,true));
  vertCuts.push_back(std::move(vertCut4));
  std::unique_ptr<TypedCut<float>> vertCut5(new TypedCut<float>("minCos_T","min cos_{T}",&trkVert_cos_T,0.96,true));
  vertCuts.push_back(std::move(vertCut5));
  std::unique_ptr<TypedCut<float>> vertCut6(new TypedCut<float>("max_trk_deltaEta","max #Delta #eta",&trk_eta,2.0,true));
  vertCuts.push_back(std::move(vertCut6));
  std::unique_ptr<TypedCut<float>> vertCut7(new TypedCut<float>("max_delZ","max #Delta z",&trkVert_del_Z,0.5,true));
  vertCuts.push_back(std::move(vertCut7));
  std::unique_ptr<TypedCut<float>> vertCut8(new TypedCut<float>("minR_T","min R_{T}",&trkVert_R_T,0.25,true));
  vertCuts.push_back(std::move(vertCut8));
  std::unique_ptr<TypedCut<float>> vertCut9(new TypedCut<float>("min_trk_highPt","min lead p_{T}",&trk_pt,13.0,true));
  vertCuts.push_back(std::move(vertCut9));
  std::unique_ptr<TypedCut<float>> vertCut10(new TypedCut<float>("max_trk_sumChi2rphidof","max #Sigma #chi^{2}_{r#phi}",&trk_chi2rphi,6.0,true));
  vertCuts.push_back(std::move(vertCut10));
  std::unique_ptr<TypedCut<int>> vertCut11(new TypedCut<int>("min_trk_sumNumStubs","min #Sigma n_{stub}",&trk_nstub,11,true));
  vertCuts.push_back(std::move(vertCut11));
  std::unique_ptr<TypedCut<float>> vertCut12(new TypedCut<float>("min_trk_highD0","min d_{0}",&trk_d0,0.3,true));
  vertCuts.push_back(std::move(vertCut12));

  //std::unique_ptr<TypedCut<float>> vertCut3p1(new TypedCut<float>("min_score0p68","score>0.68",&trkVert_score,0.68,true));
  //vertCuts.push_back(std::move(vertCut3p1));
  //std::unique_ptr<TypedCut<float>> vertCut3p2(new TypedCut<float>("min_score0p69","score>0.69",&trkVert_score,0.69,true));
  //vertCuts.push_back(std::move(vertCut3p2));
  std::unique_ptr<TypedCut<float>> vertCut3p3(new TypedCut<float>("min_score0p70","score>0.70",&trkVert_score,0.70,true));
  vertCuts.push_back(std::move(vertCut3p3));
  //std::unique_ptr<TypedCut<float>> vertCut3p4(new TypedCut<float>("min_score0p71","score>0.71",&trkVert_score,0.71,true));
  //vertCuts.push_back(std::move(vertCut3p4));
  //std::unique_ptr<TypedCut<float>> vertCut3p5(new TypedCut<float>("min_score0p72","score>0.72",&trkVert_score,0.72,true));
  //vertCuts.push_back(std::move(vertCut3p5));
  std::unique_ptr<TypedCut<float>> vertCut4(new TypedCut<float>("min_score0p8","score>0.8",&trkVert_score,0.8,true));
  vertCuts.push_back(std::move(vertCut4));
  std::unique_ptr<TypedCut<float>> vertCut5(new TypedCut<float>("min_score0p9","score>0.9",&trkVert_score,0.9,true));
  vertCuts.push_back(std::move(vertCut5));
  //std::unique_ptr<TypedCut<float>> vertCut51(new TypedCut<float>("min_score0p91","score>0.91",&trkVert_score,0.91,true));
  //vertCuts.push_back(std::move(vertCut51));
  //std::unique_ptr<TypedCut<float>> vertCut52(new TypedCut<float>("min_score0p92","score>0.92",&trkVert_score,0.92,true));
  //vertCuts.push_back(std::move(vertCut52));
  //std::unique_ptr<TypedCut<float>> vertCut53(new TypedCut<float>("min_score0p93","score>0.93",&trkVert_score,0.93,true));
  //vertCuts.push_back(std::move(vertCut53));
  std::unique_ptr<TypedCut<float>> vertCut54(new TypedCut<float>("min_score0p94","score>0.94",&trkVert_score,0.94,true));
  vertCuts.push_back(std::move(vertCut54));
  std::unique_ptr<TypedCut<float>> vertCut6(new TypedCut<float>("min_score0p95","score>0.95",&trkVert_score,0.95,true));
  vertCuts.push_back(std::move(vertCut6));
  std::unique_ptr<TypedCut<float>> vertCut7(new TypedCut<float>("min_score0p96","score>0.96",&trkVert_score,0.96,true));
  vertCuts.push_back(std::move(vertCut7));
  std::unique_ptr<TypedCut<float>> vertCut8(new TypedCut<float>("min_score0p97","score>0.97",&trkVert_score,0.97,true));
  vertCuts.push_back(std::move(vertCut8));
  std::unique_ptr<TypedCut<float>> vertCut9(new TypedCut<float>("min_score0p98","score>0.98",&trkVert_score,0.98,true));
  vertCuts.push_back(std::move(vertCut9));
  std::unique_ptr<TypedCut<float>> vertCut10(new TypedCut<float>("min_score0p99","score>0.99",&trkVert_score,0.99,true));
  vertCuts.push_back(std::move(vertCut10));
#endif

  std::unique_ptr<TypedCut<float>> vertCut10(new TypedCut<float>("min_score4p0","score>4.0",&trkVert_score,4.0,false));
  vertCuts.push_back(std::move(vertCut10));
  std::unique_ptr<TypedCut<float>> vertCut11(new TypedCut<float>("min_score4p5","score>4.5",&trkVert_score,4.5,false));
  vertCuts.push_back(std::move(vertCut11));
  std::unique_ptr<TypedCut<float>> vertCut12(new TypedCut<float>("min_score5p0","score>5.0",&trkVert_score,5.0,false));
  vertCuts.push_back(std::move(vertCut12));
  std::unique_ptr<TypedCut<float>> vertCut13(new TypedCut<float>("min_score5p1","score>5.1",&trkVert_score,5.1,false));
  vertCuts.push_back(std::move(vertCut13));
  std::unique_ptr<TypedCut<float>> vertCut14(new TypedCut<float>("min_score5p2","score>5.2",&trkVert_score,5.2,false));
  vertCuts.push_back(std::move(vertCut14));
  std::unique_ptr<TypedCut<float>> vertCut15(new TypedCut<float>("min_score5p3","score>5.3",&trkVert_score,5.3,false));
  vertCuts.push_back(std::move(vertCut15));
  std::unique_ptr<TypedCut<float>> vertCut16(new TypedCut<float>("min_score5p4","score>5.4",&trkVert_score,5.4,false));
  vertCuts.push_back(std::move(vertCut16));
  std::unique_ptr<TypedCut<float>> vertCut17(new TypedCut<float>("min_score5p5","score>5.5",&trkVert_score,5.5,true));
  vertCuts.push_back(std::move(vertCut17));
  std::unique_ptr<TypedCut<float>> vertCut18(new TypedCut<float>("min_score5p55","score>5.55",&trkVert_score,5.55,true));
  vertCuts.push_back(std::move(vertCut18));
  
  std::vector<std::unique_ptr<Plot>> vertCutFlows;
  std::unique_ptr<TypedPlot<float>> vertPlot0(new TypedPlot<float>("x","cm",&trkVert_x,100,-5.0,5.0));
  vertCutFlows.push_back(std::move(vertPlot0));
  std::unique_ptr<TypedPlot<float>> vertPlot1(new TypedPlot<float>("y","cm",&trkVert_y,100,-5.0,5.0));
  vertCutFlows.push_back(std::move(vertPlot1));
  std::unique_ptr<TypedPlot<float>> vertPlot2(new TypedPlot<float>("z","cm",&trkVert_z,100,-50.0,50.0));
  vertCutFlows.push_back(std::move(vertPlot2));
  std::unique_ptr<TypedPlot<float>> vertPlot5(new TypedPlot<float>("cos_T","",&trkVert_cos_T,40,-1.0,1.0));
  vertCutFlows.push_back(std::move(vertPlot5));
  std::unique_ptr<TypedPlot<float>> vertPlot6(new TypedPlot<float>("openingAngle","",&trkVert_openingAngle,40,-3.14,3.14));
  vertCutFlows.push_back(std::move(vertPlot6));
  std::unique_ptr<TypedPlot<float>> vertPlot7(new TypedPlot<float>("parentPt","GeV",&trkVert_parentPt,200,0.0,200.0));
  vertCutFlows.push_back(std::move(vertPlot7));
  std::unique_ptr<TypedPlot<float>> vertPlot8(new TypedPlot<float>("d_T","cm",&trkVert_d_T,40,0.0,1.0));
  vertCutFlows.push_back(std::move(vertPlot8));
  std::unique_ptr<TypedPlot<float>> vertPlot9(new TypedPlot<float>("R_T","cm",&trkVert_R_T,20,logspace(0.1,20.0,20)));
  vertCutFlows.push_back(std::move(vertPlot9));
  std::unique_ptr<TypedPlot<float>> vertPlot10(new TypedPlot<float>("highPt","GeV",&trk_pt,20,logspace(2.0,100.0,20)));
  vertCutFlows.push_back(std::move(vertPlot10));
  std::unique_ptr<TypedPlot<float>> vertPlot11(new TypedPlot<float>("lowPt","GeV",&trk_pt,100,0.0,100.0));
  vertCutFlows.push_back(std::move(vertPlot11));
  std::unique_ptr<TypedPlot<float>> vertPlot12(new TypedPlot<float>("highD0","cm",&trk_d0,80,-2.0,2.0));
  vertCutFlows.push_back(std::move(vertPlot12));
  std::unique_ptr<TypedPlot<float>> vertPlot13(new TypedPlot<float>("lowD0","cm",&trk_d0,80,-2.0,2.0));
  vertCutFlows.push_back(std::move(vertPlot13));
  std::unique_ptr<TypedPlot<float>> vertPlot14(new TypedPlot<float>("delZ","cm",&trkVert_del_Z,100,0.0,1.5));
  vertCutFlows.push_back(std::move(vertPlot14));
  std::unique_ptr<TypedPlot<float>> vertPlot15(new TypedPlot<float>("deltaZ0","cm",&trk_z0,100,0.0,10.0));
  vertCutFlows.push_back(std::move(vertPlot15));
  std::unique_ptr<TypedPlot<float>> vertPlot16(new TypedPlot<float>("deltaEta","",&trk_eta,100,0.0,2.4));
  vertCutFlows.push_back(std::move(vertPlot16));
  std::unique_ptr<TypedPlot<float>> vertPlot17(new TypedPlot<float>("deltaD0","cm",&trk_d0,100,0.0,10.0));
  vertCutFlows.push_back(std::move(vertPlot17));
  std::unique_ptr<TypedPlot<float>> vertPlot18(new TypedPlot<float>("deltaPhi","",&trk_phi,100,0.0,6.3));
  vertCutFlows.push_back(std::move(vertPlot18));
  std::unique_ptr<TypedPlot<int>> vertPlot20(new TypedPlot<int>("sumNumStubs","",&trk_nstub,12,0.0,12.0));
  vertCutFlows.push_back(std::move(vertPlot20));
  std::unique_ptr<TypedPlot<float>> vertPlot21(new TypedPlot<float>("sumChi2rphidof","",&trk_chi2rphi,200,0.0,8.0));
  vertCutFlows.push_back(std::move(vertPlot21));
  std::unique_ptr<TypedPlot<float>> vertPlot22(new TypedPlot<float>("sumChi2rzdof","",&trk_chi2rz,200,0.0,3.0));
  vertCutFlows.push_back(std::move(vertPlot22));
  std::unique_ptr<TypedPlot<float>> vertPlot23(new TypedPlot<float>("sumBendChi2","",&trk_bendchi2,200,0.0,14.0));
  vertCutFlows.push_back(std::move(vertPlot23));
  std::unique_ptr<TypedPlot<float>> vertPlot24(new TypedPlot<float>("sumMVA1","",&trk_MVA1,100,0.0,2.0));
  vertCutFlows.push_back(std::move(vertPlot24));
  std::unique_ptr<TypedPlot<float>> vertPlot26(new TypedPlot<float>("leadEta","",&trk_eta,50,-2.4,2.4));
  vertCutFlows.push_back(std::move(vertPlot26));
  std::unique_ptr<TypedPlot<float>> vertPlot27(new TypedPlot<float>("score","",&trkVert_score,50,0.0,1.0));
  vertCutFlows.push_back(std::move(vertPlot27));

  std::vector<std::unique_ptr<Plot>> vertCutFlowsTP;
  std::unique_ptr<TypedPlot<float>> vertPlotTP0(new TypedPlot<float>("x","cm",&tpVert_x,100,-5.0,5.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP0));
  std::unique_ptr<TypedPlot<float>> vertPlotTP1(new TypedPlot<float>("y","cm",&tpVert_y,100,-5.0,5.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP1));
  std::unique_ptr<TypedPlot<float>> vertPlotTP2(new TypedPlot<float>("z","cm",&tpVert_z,100,-50.0,50.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP2));
  std::unique_ptr<TypedPlot<float>> vertPlotTP4(new TypedPlot<float>("cos_T","",&tpVert_cos_T,40,-1.0,1.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP4));
  std::unique_ptr<TypedPlot<float>> vertPlotTP5(new TypedPlot<float>("openingAngle","",&tpVert_openingAngle,40,-3.14,3.14));
  vertCutFlowsTP.push_back(std::move(vertPlotTP5));
  std::unique_ptr<TypedPlot<float>> vertPlotTP6(new TypedPlot<float>("parentPt","GeV",&tpVert_parentPt,200,0.0,200.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP6));
  std::unique_ptr<TypedPlot<float>> vertPlotTP7(new TypedPlot<float>("d_T","cm",&tpVert_d_T,40,0.0,0.2));
  vertCutFlowsTP.push_back(std::move(vertPlotTP7));
  std::unique_ptr<TypedPlot<float>> vertPlotTP8(new TypedPlot<float>("R_T","cm",&tpVert_R_T,20,logspace(0.1,20.0,20)));
  vertCutFlowsTP.push_back(std::move(vertPlotTP8));
  std::unique_ptr<TypedPlot<float>> vertPlotTP9(new TypedPlot<float>("highPt","GeV",&tp_pt,20,logspace(2.0,100.0,20)));
  vertCutFlowsTP.push_back(std::move(vertPlotTP9));
  std::unique_ptr<TypedPlot<float>> vertPlotTP10(new TypedPlot<float>("lowPt","GeV",&tp_pt,100,0.0,100.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP10));
  std::unique_ptr<TypedPlot<float>> vertPlotTP11(new TypedPlot<float>("highD0","cm",&tp_d0,80,-2.0,2.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP11));
  std::unique_ptr<TypedPlot<float>> vertPlotTP12(new TypedPlot<float>("lowD0","cm",&tp_d0,80,-2.0,2.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP12));
  std::unique_ptr<TypedPlot<float>> vertPlotTP13(new TypedPlot<float>("deltaZ0","cm",&tp_z0,100,0.0,10.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP13));
  std::unique_ptr<TypedPlot<float>> vertPlotTP14(new TypedPlot<float>("deltaEta","",&tp_eta,100,0.0,2.4));
  vertCutFlowsTP.push_back(std::move(vertPlotTP14));
  std::unique_ptr<TypedPlot<float>> vertPlotTP15(new TypedPlot<float>("deltaD0","cm",&tp_d0,100,0.0,10.0));
  vertCutFlowsTP.push_back(std::move(vertPlotTP15));
  std::unique_ptr<TypedPlot<float>> vertPlotTP16(new TypedPlot<float>("deltaPhi","",&tp_phi,100,0.0,6.3));
  vertCutFlowsTP.push_back(std::move(vertPlotTP16));
  std::unique_ptr<TypedPlot<float>> vertPlotTP17(new TypedPlot<float>("leadEta","",&tp_eta,50,-2.4,2.4));
  vertCutFlowsTP.push_back(std::move(vertPlotTP17));

  std::vector<TString> vertType = {"matched","unmatched"};
  std::vector<TString> vertTypeTP = {"matched","all"};
  std::vector<TString> vertPlotTPModifiers = {"","_oneMatch"};

  TH1F* vertexCutFlows[vertCutFlows.size()][vertType.size()][vertCuts.size()];
  TH1F* vertexCutFlowsMatchTP[vertCutFlowsTP.size()][vertCuts.size()][vertPlotTPModifiers.size()];
  TH1F* vertexCutFlowsTP[vertCutFlowsTP.size()][vertPlotTPModifiers.size()];
  TH1F* vertexNumVertices[vertCuts.size()];
  TH1F* fiducialNumVertices[vertCuts.size()];

  for(uint i=0; i<vertCutFlows.size(); ++i){
    for(uint j=0; j<vertType.size(); ++j){
      for(uint k=0; k<vertCuts.size(); ++k){
	TString name = "h_trackVertex_"+vertCutFlows[i]->getVarName()+"_"+vertType[j]+"_"+vertCuts[k]->getCutName()+"Cut";
	if(vertCutFlows[i]->getMaxBin()==vertCutFlows[i]->getMinBin()){
	  TString labels = name+"; Track Vertex "+vertCutFlows[i]->getVarName()+" ("+vertCutFlows[i]->getUnit()+") ; Events ";
	  std::vector<float> bins = vertCutFlows[i]->getBins();
	  TH1F* hist = new TH1F(name,labels,vertCutFlows[i]->getNumBins(),bins.data());
	  vertexCutFlows[i][j][k] = hist;
	}
	else{
	  float binWidth = (vertCutFlows[i]->getMaxBin() - vertCutFlows[i]->getMinBin()) / vertCutFlows[i]->getNumBins();
	  TString binLabel = std::to_string(binWidth);
	  TString labels = name+"; Track Vertex "+vertCutFlows[i]->getVarName()+" ("+vertCutFlows[i]->getUnit()+") ; Events / "+binLabel+" "+vertCutFlows[i]->getUnit();
	  TH1F* hist = new TH1F(name,labels,vertCutFlows[i]->getNumBins(),vertCutFlows[i]->getMinBin(),vertCutFlows[i]->getMaxBin());
	  vertexCutFlows[i][j][k] = hist;
	}
      }
    }
  }

  for(uint i=0; i<vertCutFlowsTP.size(); ++i){
    for(uint k=0; k<vertCuts.size(); ++k){
      for(uint m=0; m<vertPlotTPModifiers.size(); ++m){
	TString name = "h_trueVertex_"+vertCutFlowsTP[i]->getVarName()+"_"+vertTypeTP[0]+"_"+vertCuts[k]->getCutName()+"Cut"+vertPlotTPModifiers[m];
	if(vertCutFlowsTP[i]->getMaxBin()==vertCutFlowsTP[i]->getMinBin()){
	  TString labels = name+"; True Vertex "+vertCutFlowsTP[i]->getVarName()+" ("+vertCutFlowsTP[i]->getUnit()+") ; Events ";
	  std::vector<float> bins = vertCutFlowsTP[i]->getBins();
	  TH1F* hist = new TH1F(name,labels,vertCutFlowsTP[i]->getNumBins(),bins.data());
	  vertexCutFlowsMatchTP[i][k][m] = hist;
	}
	else{
	  float binWidth = (vertCutFlowsTP[i]->getMaxBin() - vertCutFlowsTP[i]->getMinBin()) / vertCutFlowsTP[i]->getNumBins();
	  TString binLabel = std::to_string(binWidth);
	  TString labels = name+"; True Vertex "+vertCutFlowsTP[i]->getVarName()+" ("+vertCutFlowsTP[i]->getUnit()+") ; Events / "+binLabel+" "+vertCutFlowsTP[i]->getUnit();
	  TH1F* hist = new TH1F(name,labels,vertCutFlowsTP[i]->getNumBins(),vertCutFlowsTP[i]->getMinBin(),vertCutFlowsTP[i]->getMaxBin());
	  vertexCutFlowsMatchTP[i][k][m] = hist;
	}
      }
    } 
  }

  for(uint i=0; i<vertCutFlowsTP.size(); ++i){
    for(uint k=0; k<vertPlotTPModifiers.size(); ++k){
      TString name = "h_trueVertex_"+vertCutFlowsTP[i]->getVarName()+"_"+vertTypeTP[1]+vertPlotTPModifiers[k];
      if(vertCutFlowsTP[i]->getMaxBin()==vertCutFlowsTP[i]->getMinBin()){
	TString labels = name+"; True Vertex "+vertCutFlowsTP[i]->getVarName()+" ("+vertCutFlowsTP[i]->getUnit()+") ; Events ";
	std::vector<float> bins = vertCutFlowsTP[i]->getBins();
	TH1F* hist = new TH1F(name,labels,vertCutFlowsTP[i]->getNumBins(),bins.data());
	vertexCutFlowsTP[i][k] = hist;
      }
      else{
	float binWidth = (vertCutFlowsTP[i]->getMaxBin() - vertCutFlowsTP[i]->getMinBin()) / vertCutFlowsTP[i]->getNumBins();
	TString binLabel = std::to_string(binWidth);
	TString labels = name+"; True Vertex "+vertCutFlowsTP[i]->getVarName()+" ("+vertCutFlowsTP[i]->getUnit()+") ; Events / "+binLabel+" "+vertCutFlowsTP[i]->getUnit();
	TH1F* hist = new TH1F(name,labels,vertCutFlowsTP[i]->getNumBins(),vertCutFlowsTP[i]->getMinBin(),vertCutFlowsTP[i]->getMaxBin());
	vertexCutFlowsTP[i][k] = hist;
      }
    }
  }

  for(uint k=0; k<vertCuts.size(); ++k){
    TString name = "h_trackVertexNumVertices_"+vertCuts[k]->getCutName()+"Cut";
    TString labels = name+"; Number of Track Vertices"+" ; Events / 1.0";
    TH1F* hist = new TH1F(name,labels,40,0,40);
    vertexNumVertices[k] = hist;
  }

  for(uint k=0; k<vertCuts.size(); ++k){
    TString name = "h_fiducialNumVertices_"+vertCuts[k]->getCutName()+"Cut";
    TString labels = name+"; Number of Track Vertices"+" ; Events / 1.0";
    TH1F* hist = new TH1F(name,labels,40,0,40);
    fiducialNumVertices[k] = hist;
  }

  TH1F *h_numSelectedTrks = new TH1F("h_numSelectedTrks","h_numSelectedTrks; Number of Selected Tracks; Events / 1.0",100,0,100);
  TH1F *h_numSelectedTrks_zoomOut = new TH1F("h_numSelectedTrks_zoomOut","h_numSelectedTrks_zoomOut; Number of Selected Tracks; Events / 10.0",100,0,1000);
  TH1F *h_trk_H_T = new TH1F("h_trk_H_T","h_trk_H_T; Event Track Scalar p_{T} Sum [GeV]; Events / 10.0",100,0,1000);
  TH1F *h_trk_MET = new TH1F("h_trk_MET","h_trk_MET; Event Track Missing E_{T} [GeV]; Events / 4.0",100,0,400);
  TH1F *h_trk_oneMatch_H_T = new TH1F("h_trk_oneMatch_H_T","h_trk_oneMatch_H_T; Event Track Scalar p_{T} Sum [GeV]; Events / 10.0",100,0,1000);
  TH1F *h_trk_oneMatch_MET = new TH1F("h_trk_oneMatch_MET","h_trk_oneMatch_MET; Event Track Missing E_{T} [GeV]; Events / 4.0",100,0,400);
  TH1F *h_tp_H_T = new TH1F("h_tp_H_T","h_tp_H_T; Event TP Scalar p_{T} Sum [GeV]; Events / 10.0",100,0,1000);
  TH1F *h_tp_MET = new TH1F("h_tp_MET","h_tp_MET; Event TP Missing E_{T} [GeV]; Events / 4.0",100,0,400);
  TH1F *h_trueVertex_numAllCuts = new TH1F("h_trueVertex_numAllCuts","h_trueVertex_numAllCuts; TP Vertices; Events / 1.0",40,0,40);
  TH1F *h_trueVertex_numTPs = new TH1F("h_trueVertex_numTPs","h_trueVertex_numTPs; TPs Associated with Vertex; Events / 1.0",6,0,6);
  TH1F *h_trackVertexBranch_numAllCuts = new TH1F("h_trackVertexBranch_numAllCuts","h_trackVertexBranch_numAllCuts; Track Vertices; Events / 1.0",40,0,40);

  // Displaced Vertex Plots
  TH1F *h_res_tp_trk_x = new TH1F("h_res_tp_trk_x","h_res_tp_trk_x; x residual of vertex (cm) ; Events / 0.02 cm",100,-0.4,0.4);
  TH1F *h_res_tp_trk_y = new TH1F("h_res_tp_trk_y","h_res_tp_trk_y; y residual of vertex (cm) ; Events / 0.02 cm",100,-0.4,0.4);
  TH1F *h_res_tp_trk_x_zoomOut = new TH1F("h_res_tp_trk_x_zoomOut","h_res_tp_trk_x_zoomOut; x residual of vertex (cm) ; Events / 0.04 cm",500,-10,10);
  TH1F *h_res_tp_trk_y_zoomOut = new TH1F("h_res_tp_trk_y_zoomOut","h_res_tp_trk_y_zoomOut; y residual of vertex (cm) ; Events / 0.04 cm",500,-10,10);
  TH1F *h_res_tp_trk_z = new TH1F("h_res_tp_trk_z","h_res_tp_trk_z; z residual of vertex (cm) ; Events / 0.05 cm",200,-6,6);
  TH1F *h_res_tp_trk_r = new TH1F("h_res_tp_trk_r","h_res_tp_trk_r; r residual of vertex (cm) ; Events / 0.02 cm",100,-1,1);
  TH1F *h_res_tp_trk_phi = new TH1F("h_res_tp_trk_phi","h_res_tp_trk_phi; phi residual of vertex ; Events / 0.02",100,-1,1);

  TH2F *h_trueVertex_charge_vs_numTPs = new TH2F("h_trueVertex_charge_vs_numTPs","h_trueVertex_charge_vs_numTPs; TPs Associated with Vertex; Net Charge",6,0,6,12,-6,6);
  TH2F *h_correct_trackVertex_charge_vs_numTracks = new TH2F("h_correct_trackVertex_charge_vs_numTracks","h_correct_trackVertex_charge_vs_numTracks; Tracks Associated with Vertex; Net Charge",20,0,20,40,-20,20);
  TH2F *h_false_trackVertex_charge_vs_numTracks = new TH2F("h_false_trackVertex_charge_vs_numTracks","h_false_trackVertex_charge_vs_numTracks; Tracks Associated with Vertex; Net Charge",20,0,20,40,-20,20);
  
  std::string binVariable = "";
  std::vector<std::vector<double>> track_bins = {{-1,1}};
  std::vector<std::vector<double>> z0_bins;
  double z0_bin_min = -20.0;
  double z0_bin_width = 4.0;
  double z0_bin_max = z0_bin_min+z0_bin_width;
  while(z0_bin_max<20.0){
    z0_bins.push_back({z0_bin_min,z0_bin_max});
    z0_bin_min += (z0_bin_width/2);
    z0_bin_max += (z0_bin_width/2);
  }
  std::vector<std::vector<double>> phi_bins;
  double phi_bin_min = -TMath::Pi();
  double phi_bin_width = 0.6;
  double phi_bin_max = phi_bin_min+phi_bin_width;
  while(phi_bin_max<TMath::Pi()){
    phi_bins.push_back({phi_bin_min,phi_bin_max});
    phi_bin_min += (phi_bin_width/2);
    phi_bin_max += (phi_bin_width/2);
  }
  if(binVariable=="z0") track_bins = z0_bins;
  if(binVariable=="phi") track_bins = phi_bins;

  std::map<string,int> numPart_primary_noCuts{};
  std::map<string,int> numPart_primary_chi2rzdofCuts{};
  std::map<string,int> numPart_primary_bendchi2Cuts{};
  std::map<string,int> numPart_primary_chi2rphidofCuts{};
  std::map<string,int> numPart_primary_nstubCuts{};
  std::map<string,int> numPart_primary_ptCuts{};
  std::map<string,int> numPart_primary_d0Cuts{};
  std::map<string,int> numPart_primary_z0Cuts{};
  std::map<string,int> numPart_np_noCuts{};
  std::map<string,int> numPart_np_chi2rzdofCuts{};
  std::map<string,int> numPart_np_bendchi2Cuts{};
  std::map<string,int> numPart_np_chi2rphidofCuts{};
  std::map<string,int> numPart_np_nstubCuts{};
  std::map<string,int> numPart_np_ptCuts{};
  std::map<string,int> numPart_np_d0Cuts{};
  std::map<string,int> numPart_np_z0Cuts{};
  
  if (tree == 0) return;
  Long64_t nevt = tree->GetEntries();
  Long64_t n_findableEvent = 0;
  //nevt = 100;
  Vertex_Parameters geomTrackVertex;
  Vertex_Parameters geomTrueVertex;
  auto trackLoopTime = 0.;
  auto tpLoopTime = 0.;
  auto trueVertLoopTime = 0.;
  auto trueVertPlotLoopTime = 0.;
  auto trackVertLoopTime = 0.;
  auto matchLoopTime = 0.;
  auto trackVertPlotLoopTime = 0.;
  //std::cout<<"before event loop"<<std::endl;
  for (Long64_t i_evnt=0; i_evnt<nevt; i_evnt++) {
    //std::cout<<"event number: "<<i_evnt<<std::endl;
    tree->GetEntry(i_evnt);
    displayProgress(i_evnt, nevt);
    std::vector<std::deque<Track_Parameters>> binnedSelectedTracks;
    for(uint i=0; i<track_bins.size(); i++){
      binnedSelectedTracks.push_back({});
    }
    std::deque<Track_Parameters> selectedTracks;      // Tracks 
    std::deque<Track_Parameters> selectedTPs;         // Tracking particles
    std::vector<Vertex_Parameters> trueVertices;
    int maxPT_i = 0;
    bool oneMatch = false;
    bool findableEvent = false;
    std::valarray<float> trkMET = {0.0,0.0};
    float trkH_T = 0.0;

    vector<vector<int>> tpVert_indexTPs;
    vector<int> trkVert_indexMatch;
    tpVert_d_T = new vector<float> ();
    tpVert_R_T = new vector<float> ();
    tpVert_cos_T = new vector<float> ();
    tpVert_x = new vector<float> ();
    tpVert_y = new vector<float> ();
    tpVert_z = new vector<float> ();
    tpVert_openingAngle = new vector<float> ();
    tpVert_parentPt = new vector<float> ();
    
    // ----------------------------------------------------------------------------------------------------------------
    // track loop
    //std::cout<<"starting track loop"<<std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (uint it = 0; it < trk_pt->size(); ++it){
      bool isPrimary = true;
      if(inputFile.Contains("DarkPhoton")){
	isPrimary = trk_matchtp_isHard->at(it);
      }
      if(inputFile.Contains("DisplacedTrackJet")){
	isPrimary = trk_matchtp_isHToB->at(it);
      }
      //std::cout<<"track pt: "<<trk_pt->at(it)<<" eta: "<<trk_eta->at(it)<<" d0 : "<<trk_d0->at(it)<<" phi: "<<trk_phi->at(it)<<" z0: "<<trk_z0->at(it)<<" nstub: "<<trk_nstub<<std::endl;
      uint icut = 0;
      uint i_plot = -1;
      for(icut=0; icut<preselCuts.size(); ++icut){
	if(preselCuts[icut]->getDoPlot()){
	  i_plot++;
	}
	bool mods = true;
	TString cutName = preselCuts[icut]->getCutName();
	float cutValue = preselCuts[icut]->getCutValue();
	float param = preselCuts[icut]->getParam(it);
	if(cutName.Contains("D0") || cutName.Contains("Eta")) param = fabs(param);
	//std::cout<<"cutName: "<<cutName<<" cutValue: "<<cutValue<<" param: "<<param<<std::endl;      
	if(cutName.Contains("barrel") && fabs(trk_eta->at(it))>barrelEta) mods = false;
	if(cutName.Contains("disk") && fabs(trk_eta->at(it))<=barrelEta) mods = false;
	if(cutName.Contains("_H") && trk_pt->at(it)<=10) mods = false;
	if(cutName.Contains("_L") && trk_pt->at(it)>10) mods = false;
	if(cutName.Contains("_P") && fabs(trk_d0->at(it))>1) mods = false;
	if(cutName.Contains("_D") && fabs(trk_d0->at(it))<=1 ) mods = false;
	if(cutName.Contains("overlap") && (fabs(trk_eta->at(it))<=1.1 || fabs(trk_eta->at(it))>=1.7)) mods = false;
	if(mods){
	  if(cutName.Contains("max") && param>cutValue) break;
	  if(cutName.Contains("min") && param<cutValue) break;
	}
	//std::cout<<"passed cut"<<std::endl;
	if(!preselCuts[icut]->getDoPlot()) continue;
	for(uint i=0; i<trackType.size(); ++i){
	  bool primary = trkExt_fake->at(it)==1 && isPrimary;
	  if(trackType[i]=="primary" && !primary) continue;
	  if(trackType[i]=="np" && primary) continue;
	  if(trackType[i]=="fake" && trkExt_fake->at(it)!=0) continue;
	  if(trackType[i]=="PU" && trkExt_fake->at(it)!=2) continue;
	  if(trackType[i]=="notHiggs" && !(trkExt_fake->at(it)==1 && !isPrimary)) continue;
	  string partId = to_string(trk_matchtp_pdgid->at(it));
	  //numPartCutFlows[i][icut][partId]++;
	  for(uint j=0; j<plotModifiers.size(); ++j){
	    if(plotModifiers[j]=="_H" && trk_pt->at(it)<=10) continue;
	    if(plotModifiers[j]=="_L" && trk_pt->at(it)>10) continue;
	    if(plotModifiers[j]=="_P" && fabs(trk_d0->at(it))>1) continue;
	    if(plotModifiers[j]=="_D" && fabs(trk_d0->at(it))<=1) continue;
	    if(plotModifiers[j]=="_barrel" && fabs(trk_eta->at(it))>barrelEta) continue;
	    if(plotModifiers[j]=="_disk" && fabs(trk_eta->at(it))<=barrelEta) continue;
	    for(uint ivar=0; ivar<varCutFlows.size(); ++ivar){
	      param = varCutFlows[ivar]->getParam(it);
	      TString varName = varCutFlows[ivar]->getVarName();
	      if(varName.Contains("sector")){
		while (param < -TMath::Pi()/9 ) param += 2*TMath::Pi();
		while (param > TMath::Pi()*2 ) param -= 2*TMath::Pi();
		while (param > TMath::Pi()/9) param -= 2*TMath::Pi()/9;
	      }
	      if(varName.Contains("d0")){
		param = fabs(param);
	      }
	      preselCutFlows[ivar][i][i_plot][j]->Fill(param);
	    }
	    for(uint ivar2D=0; ivar2D<varCutFlows2D.size(); ++ivar2D){
	      float param1 = varCutFlows2D[ivar2D].first->getParam(it);
	      float param2 = varCutFlows2D[ivar2D].second->getParam(it);
	      TString varName1 = varCutFlows2D[ivar2D].first->getVarName();
	      TString varName2 = varCutFlows2D[ivar2D].second->getVarName();
	      if(varName1.Contains("sector")){
		while (param1 < -TMath::Pi()/9 ) param1 += 2*TMath::Pi();
		while (param1 > TMath::Pi()*2 ) param1 -= 2*TMath::Pi();
		while (param1 > TMath::Pi()/9) param1 -= 2*TMath::Pi()/9;
	      }
	      if(varName2.Contains("sector")){
		while (param2 < -TMath::Pi()/9 ) param2 += 2*TMath::Pi();
		while (param2 > TMath::Pi()*2 ) param2 -= 2*TMath::Pi();
		while (param2 > TMath::Pi()/9) param2 -= 2*TMath::Pi()/9;
	      }
	      preselCutFlows2D[ivar2D][i][i_plot][j]->Fill(param1,param2);
	    }
	  }
	}
      }
      if(icut==preselCuts.size()){
	Track_Parameters* tp_params = new Track_Parameters(trk_matchtp_pt->at(it), trk_matchtp_d0->at(it), trk_matchtp_z0->at(it), trk_matchtp_eta->at(it), trk_matchtp_phi->at(it), trk_matchtp_pdgid->at(it), trk_matchtp_x->at(it), trk_matchtp_y->at(it), trk_matchtp_z->at(it));
	for(uint i=0; i<track_bins.size(); i++){
	  float trkVariable = 0.0;
	  if(binVariable=="phi") trkVariable = trk_phi->at(it);
	  if (binVariable=="z0") trkVariable = fabs(trk_z0->at(it));
	  if(trkVariable<track_bins[i][1] && trkVariable>track_bins[i][0] ){
	    binnedSelectedTracks[i].push_back(Track_Parameters(trk_pt->at(it), -trk_d0->at(it), trk_z0->at(it), trk_eta->at(it), trk_phi->at(it), -99999, -999., -999., -999., trk_rinv->at(it), it, tp_params, trk_nstub->at(it), trk_chi2rphi->at(it), trk_chi2rz->at(it), trk_bendchi2->at(it), trk_MVA1->at(it)));
	  }
	}
	
	
	//std::cout<<"track params: "<<trk_pt->at(it)<<" "<<-trk_d0->at(it)<<" "<<trk_z0->at(it)<<" "<<trk_eta->at(it)<<" "<<trk_phi->at(it)<<" "<<trk_rinv->at(it)<<" "<<it<<" "<<trk_nstub->at(it)<<" "<<trk_chi2rphi->at(it)<<" "<<trk_chi2rz->at(it)<<" "<<trk_bendchi2->at(it)<<" "<<trk_MVA1->at(it)<<" "<<trk_MVA2->at(it)<<std::endl;
	selectedTracks.push_back(Track_Parameters(trk_pt->at(it), -trk_d0->at(it), trk_z0->at(it), trk_eta->at(it), trk_phi->at(it), -99999, -999., -999., -999., trk_rinv->at(it), it, tp_params, trk_nstub->at(it), trk_chi2rphi->at(it), trk_chi2rz->at(it), trk_bendchi2->at(it), trk_MVA1->at(it)));
	trkH_T += trk_pt->at(it);
	std::valarray<float> trackPtVec = {trk_pt->at(it)*cos(trk_phi->at(it)),trk_pt->at(it)*sin(trk_phi->at(it))};
	trkMET -= trackPtVec;
      }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    trackLoopTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    h_trk_H_T->Fill(trkH_T);
    h_trk_MET->Fill(TMath::Sqrt(pow(trkMET[0],2)+pow(trkMET[1],2)));
    h_numSelectedTrks->Fill(selectedTracks.size());
    //std::cout<<"num selected tracks: "<<selectedTracks.size()<<std::endl;
    h_numSelectedTrks_zoomOut->Fill(selectedTracks.size());
    
    // ----------------------------------------------------------------------------------------------------------------
    // tracking particle loop
    float tpH_T = 0.0;
    std::valarray<float> tpMET = {0.0,0.0};
    //std::cout<<"tp_pt size: "<<tp_pt->size()<<std::endl;
    //std::cout<<"starting tracking particle loop"<<std::endl;
    begin = std::chrono::steady_clock::now();
    for (int it = 0; it < (int)tp_pt->size(); it++){
      
      float tmp_d0 = tp_d0->at(it);	// Sign difference in the NTupleMaker
      float tmp_z0 = tp_z0->at(it);
	
      bool isPrimary = true;
      if(inputFile.Contains("DarkPhoton")) isPrimary = tp_isHard->at(it);
      if(inputFile.Contains("DisplacedTrackJet")) isPrimary = tp_isHToB->at(it);
      
      uint icut=0;
      for(icut=0; icut<preselCutsTP.size(); ++icut){
	bool mods = true;
	float param = preselCutsTP[icut]->getParam(it);
	TString cutName = preselCutsTP[icut]->getCutName();
	if(cutName.Contains("D0") || cutName.Contains("Eta")) param = fabs(param);
	float cutValue = preselCutsTP[icut]->getCutValue();
	//std::cout<<"cutName: "<<cutName<<" cutValue: "<<cutValue<<" param: "<<param<<std::endl;
	if(cutName.Contains("barrel") && fabs(tp_eta->at(it))>barrelEta) mods = false;
	if(cutName.Contains("disk") && fabs(tp_eta->at(it))<=barrelEta) mods = false;
	if(cutName.Contains("_H") && tp_pt->at(it)<=10) mods = false;
	if(cutName.Contains("_L") && tp_pt->at(it)>10) mods = false;
	if(cutName.Contains("_P") && fabs(tp_d0->at(it))>1) mods = false;
	if(cutName.Contains("_D") && fabs(tp_d0->at(it))<=1 ) mods = false;
	if(cutName.Contains("overlap") && (fabs(tp_eta->at(it))<=1.1 || fabs(tp_eta->at(it))>=1.7)) mods = false;
	if(mods){
	  if(cutName.Contains("max") && param>cutValue) break;
	  if(cutName.Contains("min") && param<cutValue) break;
	}
	//std::cout<<"passed cut"<<std::endl;
	for(uint i=0; i<tpType.size(); ++i){
	  bool primary = tp_eventid->at(it)==0 && isPrimary;
	  if(tpType[i]=="primary" && !primary) continue;
	  if(tpType[i]=="np" && primary) continue;
	  if(tpType[i]=="PU" && tp_eventid->at(it)==0) continue;
	  if(tpType[i]=="notHiggs" && !(tp_eventid->at(it)==0 && !isPrimary)) continue;
	  if(tpType[i]=="match" && tp_nmatch->at(it)==0) continue;
	  string partId = to_string(tp_pdgid->at(it));
	  //numPartCutFlowsTP[i][icut][partId]++;
	  for(uint j=0; j<plotModifiers.size(); ++j){
	    if(plotModifiers[j]=="_H" && tp_pt->at(it)<=10) continue;
	    if(plotModifiers[j]=="_L" && tp_pt->at(it)>10) continue;
	    if(plotModifiers[j]=="_P" && fabs(tp_d0->at(it))>1) continue;
	    if(plotModifiers[j]=="_D" && fabs(tp_d0->at(it))<=1) continue;
	    if(plotModifiers[j]=="_barrel" && fabs(tp_eta->at(it))>barrelEta) continue;
	    if(plotModifiers[j]=="_disk" && fabs(tp_eta->at(it))<=barrelEta) continue;
	    for(uint ivar=0; ivar<varCutFlowsTP.size(); ++ivar){
	      param = varCutFlowsTP[ivar]->getParam(it);
	      TString varName = varCutFlowsTP[ivar]->getVarName();
	      if(varName.Contains("sector")){
		while (param < -TMath::Pi()/9 ) param += 2*TMath::Pi();
		while (param > TMath::Pi()*2 ) param -= 2*TMath::Pi();
		while (param > TMath::Pi()/9) param -= 2*TMath::Pi()/9;
	      }
	      if(varName.Contains("d0")){
		param = fabs(param);
	      }
	      preselCutFlowsTP[ivar][i][icut][j]->Fill(param);
	    }
	    for(uint ivar2D=0; ivar2D<varCutFlowsTP2D.size(); ++ivar2D){
	      float param1 = varCutFlowsTP2D[ivar2D].first->getParam(it);
	      float param2 = varCutFlowsTP2D[ivar2D].second->getParam(it);
	      TString varName1 = varCutFlowsTP2D[ivar2D].first->getVarName();
	      TString varName2 = varCutFlowsTP2D[ivar2D].second->getVarName();
	      if(varName1.Contains("sector")){
		while (param1 < -TMath::Pi()/9 ) param1 += 2*TMath::Pi();
		while (param1 > TMath::Pi()*2 ) param1 -= 2*TMath::Pi();
		while (param1 > TMath::Pi()/9) param1 -= 2*TMath::Pi()/9;
	      }
	      if(varName2.Contains("sector")){
		while (param2 < -TMath::Pi()/9 ) param2 += 2*TMath::Pi();
		while (param2 > TMath::Pi()*2 ) param2 -= 2*TMath::Pi();
		while (param2 > TMath::Pi()/9) param2 -= 2*TMath::Pi()/9;
	      }
	      preselCutFlowsTP2D[ivar2D][i][icut][j]->Fill(param1,param2);
	    }
	  }
	}
      }
      if(icut==preselCutsTP.size() && tp_eventid->at(it)==0 && isPrimary==true){
	selectedTPs.push_back(Track_Parameters(tp_pt->at(it), tmp_d0, tmp_z0, tp_eta->at(it), tp_phi->at(it), tp_pdgid->at(it), tp_x->at(it), tp_y->at(it), tp_z->at(it), tp_charge->at(it), it));
	if (tp_eventid->at(it)>0){
	  tpH_T += tp_pt->at(it);
	  std::valarray<float> tpPtVec = {tp_pt->at(it)*cos(tp_phi->at(it)),tp_pt->at(it)*sin(tp_phi->at(it))};
	  tpMET -= tpPtVec;
	}
      }
    }
    end = std::chrono::steady_clock::now();
    tpLoopTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    h_tp_H_T->Fill(tpH_T);
    h_tp_MET->Fill(TMath::Sqrt(pow(tpMET[0],2)+pow(tpMET[1],2)));
    
    // --------------------------------------------------------------------------------------------
    //         Vertex finding in Tracking Particles
    // --------------------------------------------------------------------------------------------
    if (!(selectedTracks.size() >= 2)) continue;
    double_t x_dv = -9999.0;// (tp_x->at((*selectedTPs)[0]->index));//+tp_x->at((*selectedTPs)[1]->index))/2.0;
    double_t y_dv = -9999.0;// (tp_y->at((*selectedTPs)[0]->index));//+tp_y->at((*selectedTPs)[1]->index))/2.0;
    double_t z_dv = -9999.0;// (tp_z->at((*selectedTPs)[0]->index));//+tp_z->at((*selectedTPs)[1]->index))/2.0;
    
    if(selectedTPs.size()>=2){
      begin = std::chrono::steady_clock::now();
      //std::cout<<"vertex finding in TPs"<<std::endl;
      sort(selectedTPs.begin(), selectedTPs.end(), ComparePtTrack);
      while(selectedTPs.size()>1){
	bool foundTrueVertex = false;
	for( uint i=1; i<selectedTPs.size();){
	  int index0 = selectedTPs[0].index;
	  int index1 = selectedTPs[i].index;
	  if( fabs(tp_x->at(index0)-tp_x->at(index1))<0.0001 && fabs(tp_y->at(index0)-tp_y->at(index1))<0.0001 && fabs(tp_z->at(index0)-tp_z->at(index1))<0.0001 ){
	    x_dv = tp_x->at(index0);
	    y_dv = tp_y->at(index0);
	    z_dv = tp_z->at(index0);
	    if(dist(x_dv,y_dv)>d0_res && dist(x_dv,y_dv)<20){
	      //std::cout<<"true vertex: "<<x_dv<<" "<<y_dv<<" "<<z_dv<<" tp_pt: "<<selectedTPs[0].pt<<" "<<selectedTPs[i].pt<<" tp_d0: "<<selectedTPs[0].d0<<" "<<selectedTPs[1].d0<<" tp_z0: "<<selectedTPs[0].z0<<" "<<selectedTPs[1].z0<<" eventid's: "<<tp_eventid->at(selectedTPs[0].index)<<" "<<tp_eventid->at(selectedTPs[i].index)<<std::endl;
	      if(!foundTrueVertex){
		trueVertices.push_back(Vertex_Parameters(x_dv, y_dv, z_dv, selectedTPs[0], selectedTPs[i]) );
		foundTrueVertex = true;
	      }
	      else{
		trueVertices.back().addTrack(selectedTPs[i]);
	      }
	      selectedTPs.erase(selectedTPs.begin()+i);
	    }
	    else{
	      i++;
	    }
	  }
	  else{
	    i++;
	  }
	}
	selectedTPs.pop_front();
      }
      
      h_trueVertex_numAllCuts->Fill(trueVertices.size());
      float maxPT = 0.0;
      // loop through trueVertices and fill ntuple branches
      for(uint i=0; i<trueVertices.size(); i++){
	if(trueVertices[i].a.pt>maxPT){
	  maxPT = trueVertices[i].a.pt;
	  maxPT_i = i;
	}
	
	std::vector<int> itps;
	for(uint itrack=0; itrack<trueVertices[i].tracks.size(); itrack++){
	  itps.push_back(trueVertices[i].tracks[itrack].index);
	}
	tpVert_indexTPs.push_back(itps);
	tpVert_d_T->push_back(trueVertices[i].d_T);
	tpVert_cos_T->push_back(trueVertices[i].cos_T);
	tpVert_R_T->push_back(trueVertices[i].R_T);
	tpVert_x->push_back(trueVertices[i].x_dv);
	tpVert_y->push_back(trueVertices[i].y_dv);
	tpVert_z->push_back(trueVertices[i].z_dv);
	tpVert_openingAngle->push_back(trueVertices[i].openingAngle);
	tpVert_parentPt->push_back(trueVertices[i].p_mag);

	float netCharge = 0;
	for(uint itrack=0; itrack<trueVertices[i].tracks.size(); itrack++){
	  netCharge+=trueVertices[i].tracks[itrack].charge;
	}
	h_trueVertex_numTPs->Fill(trueVertices[i].tracks.size());
	h_trueVertex_charge_vs_numTPs->Fill(trueVertices[i].tracks.size(),netCharge);

	if(netCharge==0){
	  float chargeOfFirstTrack = 0.0;
	  for(uint itrack=0; itrack<trueVertices[i].tracks.size(); itrack++){
	    int itp = trueVertices[i].tracks[itrack].index;
	    float charge = trueVertices[i].tracks[itrack].charge;
	    bool hasTrack = (matchtrk_pt->at(itp)!=-999);
	    if(hasTrack){
	      if(chargeOfFirstTrack==0.0){
		chargeOfFirstTrack = charge;
	      }
	      else{
		if(charge == (-1*chargeOfFirstTrack)) findableEvent = true;
	      }
	    }
	  } 
	}
	
      }
      end = std::chrono::steady_clock::now();
      trueVertLoopTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
      begin = std::chrono::steady_clock::now();
      // fill true vertex plots using ntuple branches
      for (int it = 0; it < (int)tpVert_d_T->size(); it++){
	for(uint i=0; i<vertCutFlowsTP.size(); ++i){
	  float param;
	  for(uint k=0; k<vertPlotTPModifiers.size(); ++k){
	    if(vertPlotTPModifiers[k].Contains("oneMatch") && it!=maxPT_i) continue;
	    TString varName = vertCutFlowsTP[i]->getVarName();
	    if(varName.Contains("delta")){
	      float param1 = vertCutFlowsTP[i]->getParam(tpVert_indexTPs[it][0]);
	      for(uint iTP=1; iTP<tpVert_indexTPs[it].size(); ++iTP){
		float param2 = vertCutFlowsTP[i]->getParam(tpVert_indexTPs[it][iTP]);
		param = fabs(param1 - param2);
		vertexCutFlowsTP[i][k]->Fill(param);
	      }
	      continue;
	    }
	    else if(varName.Contains("high")){
	      param = vertCutFlowsTP[i]->getParam(tpVert_indexTPs[it][0]);
	      for(uint iTP=1; iTP<tpVert_indexTPs[it].size(); ++iTP){
		float param2 = vertCutFlowsTP[i]->getParam(tpVert_indexTPs[it][iTP]);
		if(fabs(param2)>fabs(param)) param = param2;
	      }
	    }
	    else if(varName.Contains("low")){
	      param = vertCutFlowsTP[i]->getParam(tpVert_indexTPs[it][0]);
	      for(uint iTP=1; iTP<tpVert_indexTPs[it].size(); ++iTP){
		float param2 = vertCutFlowsTP[i]->getParam(tpVert_indexTPs[it][iTP]);
		if(fabs(param2)<fabs(param)) param = param2;
	      }
	    }
	    else if(varName.Contains("lead")){
	      param = vertCutFlowsTP[i]->getParam(tpVert_indexTPs[it][0]);
	    }
	    else{
	      param = vertCutFlowsTP[i]->getParam(it);
	    }
	    vertexCutFlowsTP[i][k]->Fill(param);
	  }
	}
      }
      end = std::chrono::steady_clock::now();
      trueVertPlotLoopTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    }
    
    // --------------------------------------------------------------------------------------------
    //                Vertex finding in Tracks
    // --------------------------------------------------------------------------------------------
    //std::cout<<"vertex finding in tracks"<<std::endl;
    begin = std::chrono::steady_clock::now();
    sort(selectedTracks.begin(), selectedTracks.end(), ComparePtTrack);
    // find track vertices and fill ntuple branches
    std::vector<Vertex_Parameters> lastBinVertices;
    for(auto trackBin : binnedSelectedTracks){
      std::vector<Vertex_Parameters> binVertices;
      if(trackBin.size()<2) continue;
      sort(trackBin.begin(), trackBin.end(), ComparePtTrack);
      for(uint i=0; i<trackBin.size()-1; i++){
	for(uint j=i+1; j<trackBin.size(); j++){
	  if(dist_TPs(trackBin[i],trackBin[j])!=0) continue;
	  Double_t x_dv_trk = -9999.0;
	  Double_t y_dv_trk = -9999.0;
	  Double_t z_dv_trk = -9999.0;
	  int inTraj = calcVertex(trackBin[i],trackBin[j],x_dv_trk,y_dv_trk,z_dv_trk);
	  Vertex_Parameters vertex(x_dv_trk,y_dv_trk,z_dv_trk,trackBin[i],trackBin[j]);
	  bool isDupe = false;
	  for(auto oldVertex : lastBinVertices){
	    if(oldVertex==vertex) isDupe = true; 
	  }
	  if(!isDupe){
	    binVertices.push_back(vertex);
	    //trkVert_firstIndexTrk->push_back(trackBin[i].index);
	    //trkVert_secondIndexTrk->push_back(trackBin[j].index);
	    //trkVert_firstIndexPt->push_back(i);
	    //trkVert_secondIndexPt->push_back(j);
	    //trkVert_inTraj->push_back(inTraj);
	    //trkVert_d_T->push_back(vertex.d_T);
	    //trkVert_R_T->push_back(vertex.R_T);
	    //trkVert_cos_T->push_back(vertex.cos_T);
	    //trkVert_del_Z->push_back(vertex.delta_z);
	    //trkVert_x->push_back(vertex.x_dv);
	    //trkVert_y->push_back(vertex.y_dv);
	    //trkVert_z->push_back(vertex.z_dv);
	    //trkVert_openingAngle->push_back(vertex.openingAngle);
	    //trkVert_parentPt->push_back(vertex.p_mag);
	    //trkVert_delIndexPt->push_back(fabs(i-j));
	    //std::cout<<"found vertex"<<std::endl;
	    //trkVert_alt_x->push_back(x_trk_alt);
	    //trkVert_alt_y->push_back(y_trk_alt);
	    //trkVert_leadingCharge->push_back(vertex.a.charge);
	    //trkVert_subleadingCharge->push_back(vertex.b.charge);
	  }
	}
      }
      lastBinVertices = binVertices;
    }
    end = std::chrono::steady_clock::now();
    trackVertLoopTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    begin = std::chrono::steady_clock::now();
    //std::cout<<"match track and true vertices"<<std::endl;
    // match track vertices to true vertices
    for(int it = 0; it < (int)trkVert_x->size(); it++){
      int itp = trkVert_firstIndexTrk->at(it);
      //std::cout<<"match vertices first track index: "<<itp<<std::endl;
      Track_Parameters tp_params1(trk_matchtp_pt->at(itp), trk_matchtp_d0->at(itp), trk_matchtp_z0->at(itp), trk_matchtp_eta->at(itp), trk_matchtp_phi->at(itp), trk_matchtp_pdgid->at(itp), trk_matchtp_x->at(itp), trk_matchtp_y->at(itp), trk_matchtp_z->at(itp));
      //std::cout<<"trkVert xyz: "<<trkVert_x->at(it)<<" "<<trkVert_y->at(it)<<" "<<trkVert_z->at(it)<<" tp 1 params: "<<trk_matchtp_pt->at(itp)<<" "<<trk_matchtp_d0->at(itp)<<" "<<trk_matchtp_z0->at(itp)<<" "<<trk_matchtp_eta->at(itp)<<" "<<trk_matchtp_phi->at(itp)<<" "<<trk_matchtp_pdgid->at(itp)<<" "<<trk_matchtp_x->at(itp)<<" "<<trk_matchtp_y->at(itp)<<" "<<trk_matchtp_z->at(itp)<<std::endl;
      itp = trkVert_secondIndexTrk->at(it);
      //std::cout<<"match vertices second track index: "<<itp<<std::endl;
      Track_Parameters tp_params2(trk_matchtp_pt->at(itp), trk_matchtp_d0->at(itp), trk_matchtp_z0->at(itp), trk_matchtp_eta->at(itp), trk_matchtp_phi->at(itp), trk_matchtp_pdgid->at(itp), trk_matchtp_x->at(itp), trk_matchtp_y->at(itp), trk_matchtp_z->at(itp));
      //std::cout<<"trkVert xyz: "<<trkVert_x->at(it)<<" "<<trkVert_y->at(it)<<" "<<trkVert_z->at(it)<<" tp 2 params: "<<trk_matchtp_pt->at(itp)<<" "<<trk_matchtp_d0->at(itp)<<" "<<trk_matchtp_z0->at(itp)<<" "<<trk_matchtp_eta->at(itp)<<" "<<trk_matchtp_phi->at(itp)<<" "<<trk_matchtp_pdgid->at(itp)<<" "<<trk_matchtp_x->at(itp)<<" "<<trk_matchtp_y->at(itp)<<" "<<trk_matchtp_z->at(itp)<<std::endl;
      bool foundMatch = false;
      for(uint i=0; i<trueVertices.size(); i++){
	int numMatched = 0;
	for(uint j=0; j<trueVertices[i].tracks.size(); j++){
	  if(tp_params1==trueVertices[i].tracks[j] || tp_params2==trueVertices[i].tracks[j]) numMatched++;
	}
	if(numMatched>=2){
	  trkVert_indexMatch.push_back(i);
	  foundMatch = true;
	  break;
	}
      }
      if(!foundMatch) trkVert_indexMatch.push_back(-1);
    }
    end = std::chrono::steady_clock::now();
    matchLoopTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    begin = std::chrono::steady_clock::now();
    bool filledOneMatch[vertCuts.size()];
    bool isMatchedVec[trueVertices.size()][vertCuts.size()];
    uint numVertices[vertCuts.size()];
    for(uint i = 0; i<vertCuts.size(); i++){
      filledOneMatch[i] = false;
      numVertices[i] = 0;
      for(uint j = 0; j<trueVertices.size(); j++){
	isMatchedVec[j][i] = false;
      }
    }
    //std::cout<<"fill vertex plots"<<std::endl;
    // fill vertex plots
    for(int it = 0; it < (int)trkVert_x->size(); it++){
      for(uint i=0; i<vertCuts.size(); i++){
	TString cutName = vertCuts[i]->getCutName();
	float cutValue = vertCuts[i]->getCutValue();
	//std::cout<<"cutName: "<<cutName<<" vertex track indices: "<<trkVert_firstIndexTrk->at(it)<<" "<<trkVert_secondIndexTrk->at(it)<<std::endl;
	float param;
	if(cutName.Contains("delta")){
	  float param1 = vertCuts[i]->getParam(trkVert_firstIndexTrk->at(it));
	  float param2 = vertCuts[i]->getParam(trkVert_secondIndexTrk->at(it));
	  param = fabs(param1 - param2);
	}
	else if(cutName.Contains("sum")){
	  float param1 = vertCuts[i]->getParam(trkVert_firstIndexTrk->at(it));
	  float param2 = vertCuts[i]->getParam(trkVert_secondIndexTrk->at(it));
	  if(cutName.Contains("Charge")){
	    if(param1>0.0) param1 = 1.0;
	    if(param1<0.0) param1 = -1.0;
	    if(param2>0.0) param2 = 1.0;
	    if(param2<0.0) param2 = -1.0;
	  }
	  param = param1 + param2;
	}
	else if(cutName.Contains("high")){
	  param = vertCuts[i]->getParam(trkVert_firstIndexTrk->at(it));
	  float param2 = vertCuts[i]->getParam(trkVert_secondIndexTrk->at(it));
	  if(fabs(param2)>fabs(param)) param = param2;
	  if(cutName.Contains("D0") || cutName.Contains("Eta")) param = fabs(param);
	}
	else if(cutName.Contains("low")){
	  param = vertCuts[i]->getParam(trkVert_firstIndexTrk->at(it));
	  float param2 = vertCuts[i]->getParam(trkVert_secondIndexTrk->at(it));
	  if(fabs(param2)<fabs(param)) param = param2;
	  if(cutName.Contains("D0") || cutName.Contains("Eta")) param = fabs(param);
	}
	else if(cutName.Contains("lead")){
	  param = vertCuts[i]->getParam(trkVert_firstIndexTrk->at(it));
	  if(cutName.Contains("D0") || cutName.Contains("Eta")) param = fabs(param);
	}
	else{
	  param = vertCuts[i]->getParam(it);
	}
	//std::cout<<"trackVert cutName: "<<cutName<<" cutValue: "<<cutValue<<" param: "<<param<<std::endl;
	if(cutName.Contains("max") && param>cutValue) break;
	if(cutName.Contains("min") && param<cutValue) break;
	numVertices[i]++;
	for(uint j=0; j<vertType.size(); j++){
	  //if(vertType[j]=="matched" && (trkVert_indexMatch[it]==-1 || isMatchedVec[trkVert_indexMatch[it]][i])) continue;
	  if(vertType[j]=="matched" && trkVert_indexMatch[it]==-1) continue;
	  //if(vertType[j]=="unmatched" && (trkVert_indexMatch[it]!=-1 && !isMatchedVec[trkVert_indexMatch[it]][i])) continue;
	  if(vertType[j]=="unmatched" && trkVert_indexMatch[it]!=-1) continue;
	  for(uint k=0; k<vertCutFlows.size(); k++){
	    TString varName = vertCutFlows[k]->getVarName();
	    if(varName.Contains("delta")){
	      float param1 = vertCutFlows[k]->getParam(trkVert_firstIndexTrk->at(it));
	      float param2 = vertCutFlows[k]->getParam(trkVert_secondIndexTrk->at(it));
	      param = fabs(param1 - param2);
	    }
	    else if(varName.Contains("sum")){
	      float param1 = vertCutFlows[k]->getParam(trkVert_firstIndexTrk->at(it));
	      float param2 = vertCutFlows[k]->getParam(trkVert_secondIndexTrk->at(it));
	      if(varName.Contains("charge")){
		if(param1>0.0) param1 = 1.0;
		if(param1<0.0) param1 = -1.0;
		if(param2>0.0) param2 = 1.0;
		if(param2<0.0) param2 = -1.0;
	      }
	      param = param1 + param2;
	    }
	    else if(varName.Contains("high")){
	      param = vertCutFlows[k]->getParam(trkVert_firstIndexTrk->at(it));
	      float param2 = vertCutFlows[k]->getParam(trkVert_secondIndexTrk->at(it));
	      if(fabs(param2)>fabs(param)) param = param2;
	    }
	    else if(varName.Contains("low")){
	      param = vertCutFlows[k]->getParam(trkVert_firstIndexTrk->at(it));
	      float param2 = vertCutFlows[k]->getParam(trkVert_secondIndexTrk->at(it));
	      if(fabs(param2)<fabs(param)) param = param2;
	    }
	    else if(varName.Contains("lead")){
	      param = vertCutFlows[k]->getParam(trkVert_firstIndexTrk->at(it));
	    }
	    else{
	      param = vertCutFlows[k]->getParam(it);
	    }
	    vertexCutFlows[k][j][i]->Fill(param);
	  }
	  if(vertType[j]=="matched"){
	    for(uint k=0; k<vertPlotTPModifiers.size(); k++){
	      int jt = trkVert_indexMatch[it];
	      if(vertPlotTPModifiers[k].Contains("oneMatch") && filledOneMatch[i]) continue;
	      if(vertPlotTPModifiers[k].Contains("oneMatch")){
		jt = maxPT_i;
		filledOneMatch[i] = true;
	      }
	      else{
		if(isMatchedVec[jt][i]) continue;
		isMatchedVec[jt][i] = true;
	      }
	      
	      //resolution plot
	      if(!vertPlotTPModifiers[k].Contains("oneMatch") && i==(vertCuts.size()-1)){
		h_res_tp_trk_x->Fill(tpVert_x->at(jt)-trkVert_x->at(it));
		h_res_tp_trk_x_zoomOut->Fill(tpVert_x->at(jt)-trkVert_x->at(it));
		h_res_tp_trk_y->Fill(tpVert_y->at(jt)-trkVert_y->at(it));
		h_res_tp_trk_y_zoomOut->Fill(tpVert_y->at(jt)-trkVert_y->at(it));
		h_res_tp_trk_z->Fill(tpVert_z->at(jt)-trkVert_z->at(it));
	      }

	      for(uint m=0; m<vertCutFlowsTP.size(); m++){
		TString varName = vertCutFlowsTP[m]->getVarName();
		if(varName.Contains("delta")){
		  float param1 = vertCutFlowsTP[m]->getParam(tpVert_indexTPs[jt][0]);
		  for(uint iTP=1; iTP<tpVert_indexTPs[jt].size(); ++iTP){
		    float param2 = vertCutFlowsTP[m]->getParam(tpVert_indexTPs[jt][iTP]);
		    param = fabs(param1 - param2);
		    vertexCutFlowsMatchTP[m][i][k]->Fill(param);
		  }
		  continue;
		}
		else if(varName.Contains("high")){
		  param = vertCutFlowsTP[m]->getParam(tpVert_indexTPs[jt][0]);
		  for(uint iTP=1; iTP<tpVert_indexTPs[jt].size(); ++iTP){
		    float param2 = vertCutFlowsTP[m]->getParam(tpVert_indexTPs[jt][iTP]);
		    if(fabs(param2)>fabs(param)) param = param2;
		  }
		}
		else if(varName.Contains("low")){
		  param = vertCutFlowsTP[m]->getParam(tpVert_indexTPs[jt][0]);
		  for(uint iTP=1; iTP<tpVert_indexTPs[jt].size(); ++iTP){
		    float param2 = vertCutFlowsTP[m]->getParam(tpVert_indexTPs[jt][iTP]);
		    if(fabs(param2)<fabs(param)) param = param2;
		  }
		}
		else if(varName.Contains("lead")){
		  param = vertCutFlowsTP[m]->getParam(tpVert_indexTPs[jt][0]);
		}
		else{
		  param = vertCutFlowsTP[m]->getParam(jt);
		}
		vertexCutFlowsMatchTP[m][i][k]->Fill(param);
	      }
	    }
	    break;
	  }
	}	
      }
    }
    end = std::chrono::steady_clock::now();
    trackVertPlotLoopTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    h_trackVertexBranch_numAllCuts->Fill(numVertices[vertCuts.size()-1]);
    //std::cout<<"num vertices: "<<numVertices[vertCuts.size()-1]<<std::endl;
    for(uint i=0; i<vertCuts.size(); i++){
      vertexNumVertices[i]->Fill(numVertices[i]);
      if(findableEvent) fiducialNumVertices[i]->Fill(numVertices[i]);
    }
    if(findableEvent) n_findableEvent++;
      
    delete tpVert_d_T;
    delete tpVert_R_T;
    delete tpVert_cos_T;
    delete tpVert_x;
    delete tpVert_y;
    delete tpVert_z;
    delete tpVert_openingAngle;
    delete tpVert_parentPt;
#if 0
    delete trkVert_firstIndexTrk;
    delete trkVert_secondIndexTrk;
    delete trkVert_firstIndexPt;
    delete trkVert_secondIndexPt;
    delete trkVert_inTraj;
    delete trkVert_d_T;
    delete trkVert_R_T;
    delete trkVert_cos_T;
    delete trkVert_del_Z;
    delete trkVert_x;
    delete trkVert_y;
    delete trkVert_z;
    delete trkVert_openingAngle;
    delete trkVert_parentPt;
    delete trkVert_delIndexPt;
#endif
    

  } // End of Event Loop

  std::cout<<"nevt: "<<float(nevt)<<std::endl;
  std::cout<<"Time Report:"<<std::endl;
  std::cout<<"Avg track loop time       : "<<trackLoopTime / float(nevt)<<std::endl;
  std::cout<<"Avg tp    loop time       : "<<tpLoopTime / float(nevt)<<std::endl;
  std::cout<<"Avg true vertex time      : "<<trueVertLoopTime / float(nevt)<<std::endl;
  std::cout<<"Avg true vertex plot time : "<<trueVertPlotLoopTime / float(nevt)<<std::endl;
  std::cout<<"Avg track vertex time     : "<<trackVertLoopTime / float(nevt)<<std::endl;
  std::cout<<"Avg vertex match time     : "<<matchLoopTime / float(nevt)<<std::endl;
  std::cout<<"Avg track vertex plot time: "<<trackVertPlotLoopTime / float(nevt)<<std::endl;

  // ---------------------------------------------------------------------------------------------------------
  //some Histograms

  char ctxt[500];
  if(inputFile.Contains("DarkPhoton")){
    if(inputFile.Contains("cT0")){
      sprintf(ctxt, "Dark Photon, PU=0, #tau=0mm");
    }
    else if(inputFile.Contains("cT10000")){
      sprintf(ctxt, "Dark Photon, PU=0, #tau=10000mm");
    }
    else if(inputFile.Contains("cT5000")){
      sprintf(ctxt, "Dark Photon, PU=0, #tau=5000mm");
    }   
    else if(inputFile.Contains("cT100")){
      sprintf(ctxt, "Dark Photon, PU=200, #tau=100mm");
    }
    else if(inputFile.Contains("cT10")){
      if(inputFile.Contains("PU200")){
	sprintf(ctxt, "Dark Photon, PU=200, #tau=10mm");
      }
      else{
	sprintf(ctxt, "Dark Photon, PU=200, #tau=10mm");
      }
    }
  }
  else if(inputFile.Contains("DisplacedTrackJet")){
    if(inputFile.Contains("cT10")){
      if(inputFile.Contains("PU200")){
	sprintf(ctxt, "DispTrkJet, PU=200, #tau=10mm");
      }
      else{
	sprintf(ctxt, "DispTrkJet, PU=200, #tau=10mm");
      }
    }
  }
  else if(inputFile.Contains("NeutrinoGun")){
    sprintf(ctxt, "Neutrino Gun, PU=200");
  }
  else if(inputFile.Contains("DispMu")){
    if(inputFile.Contains("PU200")){
      sprintf(ctxt, "Displaced Mu, PU=200");
    }
    else{
      sprintf(ctxt, "Displaced Mu, PU=0");
    }
  }
  else if(inputFile.Contains("TTbar")){
    if(inputFile.Contains("PU200")){
      sprintf(ctxt, "TTbar, PU=200");
    }
    else{
      sprintf(ctxt, "TTbar, PU=0");
    }
  }
  else{
    sprintf(ctxt, " ");
  }
  TCanvas c;

  TString DIR = outputDir + "AnalyzerTrkPlots";
  TString makedir = "mkdir -p " + DIR;
  const char *mkDIR = makedir.Data();
  gSystem->Exec(mkDIR);
  TString PRESELDIR = DIR + "/PreselectionPlots";
  TString makedirPreSel = "mkdir -p " + PRESELDIR;
  const char *mkDIRPRESEL = makedirPreSel.Data();
  gSystem->Exec(mkDIRPRESEL);
  TString VERTDIR = DIR + "/VertexPlots";
  TString makedirVert = "mkdir -p " + VERTDIR;
  const char *mkDIRVERT = makedirVert.Data();
  gSystem->Exec(mkDIRVERT);

  TFile *fout;
  fout = new TFile(outputDir + "output_" + inputFile, "recreate");
  TLegend* l = new TLegend(0.82,0.3,0.98,0.7);
  l->SetFillColor(0);
  l->SetLineColor(0);
  l->SetTextSize(0.04);
  l->SetTextFont(42);
  
  std::cout<<"trkEffOverlay"<<std::endl;
  for(uint ivar=0; ivar<varCutFlows.size(); ++ivar){
    for(uint j=0; j<trackType.size(); ++j){
      for(uint k=0; k<plotModifiers.size(); ++k){
	l->Clear();
	TH1F* h_trkEff[preselCutsSize];
	uint i_plot = 0;
	for(uint mcut=1; mcut<preselCuts.size(); ++mcut){
	  if(preselCuts[mcut]->getDoPlot()){
	    i_plot++;
	  }
	  else{
	    continue;
	  }
	  //std::cout<<"trkEffOverlay i j m k: "<<i<<" "<<j<<" "<<m<<" "<<k<<std::endl;
	  h_trkEff[i_plot] = (TH1F*)preselCutFlows[ivar][j][0][k]->Clone();
	  h_trkEff[i_plot]->GetYaxis()->SetNoExponent(kTRUE);
	  removeFlows(h_trkEff[i_plot]);
	  h_trkEff[i_plot]->SetStats(0);
	  removeFlows(preselCutFlows[ivar][j][i_plot][k]);
	  TString cutLabel = preselCuts[mcut]->getCutLabel();
	  TString varString = varCutFlows[ivar]->getVarName();
	  h_trkEff[i_plot]->Divide(preselCutFlows[ivar][j][i_plot][k],h_trkEff[i_plot],1.0,1.0,"B");
	  if(i_plot!=10){
	    h_trkEff[i_plot]->SetLineColor(i_plot);
	    h_trkEff[i_plot]->SetMarkerColor(i_plot);
	  }
	  else{
	    h_trkEff[i_plot]->SetLineColor(40);
	    h_trkEff[i_plot]->SetMarkerColor(40);
	  }
	  //TString cutLabel = preselCuts[mcut]->getCutLabel();
	  //std::cout<<"cutName: "<<cutName<<std::endl;
	  l->AddEntry(h_trkEff[i_plot],cutLabel,"lp");
	  //TString varString = varCutFlows[ivar]->getVarName();
	  if(varString.Contains("d0") || varString.Contains("pt")){
	    std::cout<<"h_trkEffOverlay_"+varCutFlows[ivar]->getVarName()+"_"+trackType[j]+plotModifiers[k]+".pdf"<<std::endl;
	    std::cout<<"cut: "<<cutLabel<<std::endl;
	    std::string binValues = "[";
	    std::string binWidths = "[";
	    std::string binErrors = "[";
	    std::string binCenters = "[";
	    for(int ibin=1; ibin<(h_trkEff[i_plot]->GetNbinsX()+1); ibin++){
	      binValues+=to_string(h_trkEff[i_plot]->GetBinContent(ibin)) + ", ";
	      binWidths+=to_string(h_trkEff[i_plot]->GetBinWidth(ibin)) + ", ";
	      binErrors+=to_string(h_trkEff[i_plot]->GetBinError(ibin)) + ", ";
	      binCenters+=to_string(h_trkEff[i_plot]->GetBinCenter(ibin)) + ", ";
	    }
	    binValues+="]";
	    binWidths+="]";
	    binErrors+="]";
	    binCenters+="]";
	    std::cout<<"binValues: "<<binValues<<std::endl;
	    std::cout<<"binWidths: "<<binWidths<<std::endl;
	    std::cout<<"binErrors: "<<binErrors<<std::endl;
	    std::cout<<"binCenters: "<<binCenters<<std::endl;
	  }
	  if(i_plot==1){
	    raiseMax(h_trkEff[i_plot]);
	    h_trkEff[i_plot]->Draw();
	  }
	  else{
	    h_trkEff[i_plot]->Draw("SAME");
	  }
	}
	mySmallText(0.3, 0.9, 1, ctxt);
	l->Draw();
	/*
	TString label_cms="CMS";
	TLatex* Label_cms = new TLatex(0.15,0.92,label_cms);
	Label_cms->SetNDC();
	Label_cms->SetTextFont(61);
	Label_cms->SetTextSize(0.065);
	Label_cms->Draw();
	TString label_cms1="Simulation Phase-2 Preliminary";
	TLatex* Label_cms1 = new TLatex(0.232,0.92,label_cms1);
	Label_cms1->SetNDC();
	Label_cms1->SetTextSize(0.051);
	Label_cms1->SetTextFont(52);
	Label_cms1->Draw();*/
	c.SaveAs(PRESELDIR + "/h_trkEffOverlay_"+varCutFlows[ivar]->getVarName()+"_"+trackType[j]+plotModifiers[k]+".pdf");
	/*std::cout << "trkEffOverlay took "
		  << std::chrono::duration_cast<milli>(finish - start).count()
		  << " milliseconds\n";*/
      }
    }
  }

  std::cout<<"signalvsbg"<<std::endl;
  int m_primary = distance(trackType.begin(), find(trackType.begin(), trackType.end(), "primary"));
  int m_np = distance(trackType.begin(), find(trackType.begin(), trackType.end(), "np"));
  //int m_fake = distance(trackType.begin(), find(trackType.begin(), trackType.end(), "fake"));
  //int m_PU = distance(trackType.begin(), find(trackType.begin(), trackType.end(), "PU"));
  //int m_notHiggs = distance(trackType.begin(), find(trackType.begin(), trackType.end(), "notHiggs"));
  uint i_plot = -1;
  for(uint icut=0; icut<preselCuts.size(); ++icut){
    if(preselCuts[icut]->getDoPlot()){
      i_plot++;
    }
    else{
      continue;
    }
    for(uint j=0; j<plotModifiers.size(); ++j){
      for(uint kvar=0; kvar<varCutFlows.size(); ++kvar){
	TString cutName = preselCuts[icut]->getCutName();
	auto h_stack = new THStack("hs_"+varCutFlows[kvar]->getVarName()+"_"+cutName+"Cut"+plotModifiers[j],"Stacked BG histograms");
	float integralSum = 0;
	l->Clear();
	for(uint m=0; m<trackType.size(); ++m){
	  preselCutFlows[kvar][m][i_plot][j]->GetYaxis()->SetNoExponent(kTRUE);
	  removeFlows(preselCutFlows[kvar][m][i_plot][j]);
	  if(detailedPlots){
	    raiseMax(preselCutFlows[kvar][m][i_plot][j]);
	    preselCutFlows[kvar][m][i_plot][j]->Draw();
	    mySmallText(0.3, 0.9, 1, ctxt);
	    //preselCutFlows[kvar][m][icut][j]->Write("", TObject::kOverwrite);
	    c.SaveAs(PRESELDIR + "/"+ preselCutFlows[kvar][m][i_plot][j]->GetName() + ".pdf");
	  }
	  if(m!=9){
	    preselCutFlows[kvar][m][i_plot][j]->SetLineColor(m+1);
	    preselCutFlows[kvar][m][i_plot][j]->SetMarkerColor(m+1);
	  }
	  else{
	    preselCutFlows[kvar][m][i_plot][j]->SetLineColor(40);
	    preselCutFlows[kvar][m][i_plot][j]->SetMarkerColor(40);
	  }
	  if(trackType[m]=="fake" || trackType[m]=="PU" || trackType[m]=="notHiggs"){
	    integralSum+=preselCutFlows[kvar][m][i_plot][j]->Integral();
	  }
	  /*std::cout << "preselCutFlows took "
		    << std::chrono::duration_cast<milli>(finish - start).count()
		    << " milliseconds\n";*/
	}
	for(uint m=0; m<trackType.size(); ++m){
	  if(trackType[m]=="fake" || trackType[m]=="PU" || trackType[m]=="notHiggs"){
	    preselCutFlows[kvar][m][i_plot][j]->Scale(1./integralSum);
	    h_stack->Add(preselCutFlows[kvar][m][i_plot][j]);
	  }
	}
	
 	//h_stack->Draw("HIST");
	preselCutFlows[kvar][m_primary][i_plot][j]->Scale(1./preselCutFlows[kvar][m_primary][i_plot][j]->Integral());
	/*
	raiseMaxStack(preselCutFlows[kvar][m_primary][icut][j],h_stack);
	drawSameStack(preselCutFlows[kvar][m_primary][icut][j],h_stack);
	mySmallText(0.3, 0.9, 1, ctxt);
	l->Clear();
	l->AddEntry(preselCutFlows[kvar][m_primary][icut][j],"Primary","l");
	l->AddEntry(preselCutFlows[kvar][m_fake][icut][j],"Fake","l");
	l->AddEntry(preselCutFlows[kvar][m_PU][icut][j],"PU","l");
	l->AddEntry(preselCutFlows[kvar][m_notHiggs][icut][j],"notHiggs","l");
	l->Draw();
	c.SaveAs(PRESELDIR + "/h_signalVsBGStack_"+varCutFlows[kvar]->getVarName()+"_"+preselCuts[icut]->getCutName()+"Cut"+plotModifiers[j]+".pdf");
        */
	delete h_stack;
	for(uint m=0; m<trackType.size(); ++m){
	  preselCutFlows[kvar][m][i_plot][j]->Scale(1./preselCutFlows[kvar][m][i_plot][j]->Integral());
	  preselCutFlows[kvar][m][i_plot][j]->SetStats(0);
	}
	/*
	raiseMax(preselCutFlows[kvar][m_primary][icut][j],preselCutFlows[kvar][m_fake][icut][j],preselCutFlows[kvar][m_PU][icut][j],preselCutFlows[kvar][m_notHiggs][icut][j]);
	drawSame(preselCutFlows[kvar][m_primary][icut][j],preselCutFlows[kvar][m_fake][icut][j],preselCutFlows[kvar][m_PU][icut][j],preselCutFlows[kvar][m_notHiggs][icut][j]);
	mySmallText(0.3, 0.9, 1, ctxt);
	l->Draw();
	//std::cout<<"signalvsBGOverlay primary fake PU notHiggs: "<<m_primary<<" "<<m_fake<<" "<<m_PU<<" "<<m_notHiggs<<std::endl;
	//std::cout<<"signalvsBGOverlay k i j: "<<k<<" "<<i<<" "<<j<<std::endl;
	c.SaveAs(PRESELDIR + "/h_signalVsBGOverlay_"+varCutFlows[kvar]->getVarName()+"_"+preselCuts[icut]->getCutName()+"Cut"+plotModifiers[j]+".pdf");
	std::cout << "signalVsBGOverlay took "
		  << std::chrono::duration_cast<milli>(finish - start).count()
		  << " milliseconds\n";
        */
	raiseMax(preselCutFlows[kvar][m_primary][i_plot][j],preselCutFlows[kvar][m_np][i_plot][j]);
	drawSame(preselCutFlows[kvar][m_primary][i_plot][j],preselCutFlows[kvar][m_np][i_plot][j]);
	mySmallText(0.3, 0.9, 1, ctxt);
	l->Clear();
	l->AddEntry(preselCutFlows[kvar][m_primary][i_plot][j],"Primary","l");
	l->AddEntry(preselCutFlows[kvar][m_np][i_plot][j],"NP","l");
	l->Draw();
	preselCutFlows[kvar][m_primary][i_plot][j]->Write("", TObject::kOverwrite);
	preselCutFlows[kvar][m_np][i_plot][j]->Write("", TObject::kOverwrite);
	c.SaveAs(PRESELDIR + "/h_signalVsBG_"+varCutFlows[kvar]->getVarName()+"_"+preselCuts[icut]->getCutName()+"Cut"+plotModifiers[j]+".pdf");
      }
    }
  }
  std::cout<<"trackFindingEff"<<std::endl;
  int m_match = distance(tpType.begin(), find(tpType.begin(), tpType.end(), "match"));
  int m_tp = distance(tpType.begin(), find(tpType.begin(), tpType.end(), ""));
  for(uint icut=0; icut<preselCutsTP.size(); ++icut){
    TString cutName = preselCutsTP[icut]->getCutName();
    for(uint j=0; j<plotModifiers.size(); ++j){
      for(uint kvar=0; kvar<varCutFlowsTP.size(); ++kvar){
	for(uint m=0; m<tpType.size(); ++m){
	  preselCutFlowsTP[kvar][m][icut][j]->GetYaxis()->SetNoExponent(kTRUE);
	  removeFlows(preselCutFlowsTP[kvar][m][icut][j]);
	  if(detailedPlots){
	    raiseMax(preselCutFlowsTP[kvar][m][icut][j]);
	    preselCutFlowsTP[kvar][m][icut][j]->Draw();
	    mySmallText(0.3, 0.9, 1, ctxt);
	    //preselCutFlowsTP[kvar][m][icut][j]->Write("", TObject::kOverwrite);
	    c.SaveAs(PRESELDIR + "/"+ preselCutFlowsTP[kvar][m][icut][j]->GetName() + ".pdf");
	  }
	}
	preselCutFlowsTP[kvar][m_match][icut][j]->Divide(preselCutFlowsTP[kvar][m_match][icut][j],preselCutFlowsTP[kvar][m_tp][icut][j]);
	raiseMax(preselCutFlowsTP[kvar][m_match][icut][j]);
	preselCutFlowsTP[kvar][m_match][icut][j]->Draw();
	mySmallText(0.3, 0.9, 1, ctxt);
	c.SaveAs(PRESELDIR + "/h_trackFindingEff_"+varCutFlowsTP[kvar]->getVarName()+"_"+cutName+"Cut"+plotModifiers[j]+".pdf");
      }
    }
  }

  i_plot = -1;
  for(uint icut=0; icut<preselCuts.size(); ++icut){
    if(preselCuts[icut]->getDoPlot()){
      i_plot++;
    }
    else{
      continue;
    }
    TString cutName = preselCuts[icut]->getCutName();
    for(uint j=0; j<plotModifiers.size(); ++j){
      for(uint kvar=0; kvar<varCutFlows2D.size(); ++kvar){
	for(uint m=0; m<trackType.size(); ++m){
	  removeFlows(preselCutFlows2D[kvar][m][i_plot][j]);
	  preselCutFlows2D[kvar][m][i_plot][j]->Draw("COLZ");
	  mySmallText(0.3, 0.9, 1, ctxt);
	  c.SaveAs(PRESELDIR + "/"+ preselCutFlows2D[kvar][m][i_plot][j]->GetName() + ".pdf");
	}
      }
    }
  }
  std::cout<<"preselCutFlowsTP2D"<<std::endl;
  if(detailedPlots){
    for(uint icut=0; icut<preselCutsTP.size(); ++icut){
      TString cutName = preselCutsTP[icut]->getCutName();
      for(uint j=0; j<plotModifiers.size(); ++j){
	for(uint kvar=0; kvar<varCutFlowsTP2D.size(); ++kvar){
	  for(uint m=0; m<tpType.size(); ++m){
	    removeFlows(preselCutFlowsTP2D[kvar][m][icut][j]);
	    preselCutFlowsTP2D[kvar][m][icut][j]->Draw("COLZ");
	    mySmallText(0.3, 0.9, 1, ctxt);
	    c.SaveAs(PRESELDIR + "/"+ preselCutFlowsTP2D[kvar][m][icut][j]->GetName() + ".pdf");
	  }
	}
      }
    }
  }
  std::cout<<"eff_trueVertex"<<std::endl;
  for(uint i=0; i<vertPlotTPModifiers.size(); i++){
    for(uint j=0; j<vertCutFlowsTP.size(); j++){
      //if(vertPlotTPModifiers[i].Contains("oneMatch") && vertCutFlowsTP[j]->getVarName().Contains("highPt")) std::cout<<"onematch entries: "<<vertexCutFlowsTP[j][i]->GetEntries()<<" "<<vertexCutFlowsMatchTP[j][vertCuts.size()-1][i]->GetEntries()<<std::endl;
      removeFlows(vertexCutFlowsTP[j][i]);
      removeFlows(vertexCutFlowsMatchTP[j][vertCuts.size()-1][i]);
      TH1F* h_eff = (TH1F*)vertexCutFlowsMatchTP[j][vertCuts.size()-1][i]->Clone();
      h_eff->Divide(vertexCutFlowsMatchTP[j][vertCuts.size()-1][i],vertexCutFlowsTP[j][i],1.0,1.0,"B");
      raiseMax(h_eff);
      h_eff->SetStats(0);
      h_eff->SetAxisRange(0, 1.1, "Y");
      h_eff->Draw();
      mySmallText(0.3, 0.9, 1, ctxt);
      c.SaveAs(VERTDIR + "/h_eff_trueVertex_" + vertCutFlowsTP[j]->getVarName() + "_" + vertPlotTPModifiers[i] + ".pdf");
    }
  }
  std::cout<<"findEff_trueVertex"<<std::endl;
  for(uint i=0; i<vertCutFlowsTP.size(); i++){
    removeFlows(vertexCutFlowsTP[i][0]);
    l->Clear();
    for(uint j=0; j<vertCuts.size(); j++){
      removeFlows(vertexCutFlowsMatchTP[i][j][0]);
      TH1F* h_findEff = (TH1F*)vertexCutFlowsMatchTP[i][j][0]->Clone();
      h_findEff->Divide(vertexCutFlowsMatchTP[i][j][0],vertexCutFlowsTP[i][0],1.0,1.0,"B");
      h_findEff->SetStats(0);
      if(j!=9){
	h_findEff->SetLineColor(j+1);
	h_findEff->SetMarkerColor(j+1);
      }
      else{
	h_findEff->SetLineColor(40);
	h_findEff->SetMarkerColor(40);
      }
      l->AddEntry(h_findEff,vertCuts[j]->getCutLabel(),"lp");
      TString varString = vertCutFlowsTP[i]->getVarName();
      if(varString.Contains("R_T") || varString.Contains("highPt")){
	std::cout<<"/h_findEff_trueVertex_" + vertCutFlowsTP[i]->getVarName() + ".pdf"<<std::endl;
	std::cout<<"cut: "<<vertCuts[j]->getCutLabel()<<std::endl;
	std::string binValues = "[";
	std::string binWidths = "[";
	std::string binErrors = "[";
	std::string binCenters = "[";
	for(int ibin=1; ibin<(h_findEff->GetNbinsX()+1); ibin++){
	  binValues+=to_string(h_findEff->GetBinContent(ibin)) + ", ";
	  binWidths+=to_string(h_findEff->GetBinWidth(ibin)) + ", ";
	  binErrors+=to_string(h_findEff->GetBinError(ibin)) + ", ";
	  binCenters+=to_string(h_findEff->GetBinCenter(ibin)) + ", ";
	}
	binValues+="]";
	binWidths+="]";
	binErrors+="]";
	binCenters+="]";
	std::cout<<"binValues: "<<binValues<<std::endl;
	std::cout<<"binWidths: "<<binWidths<<std::endl;
	std::cout<<"binErrors: "<<binErrors<<std::endl;
	std::cout<<"binCenters: "<<binCenters<<std::endl;
      }
      if(j==0){
	raiseMax(h_findEff);
	h_findEff->SetAxisRange(0, 1.1, "Y");
	h_findEff->Draw();
      }
      else{
	h_findEff->Draw("SAME");
      }
    }
    mySmallText(0.3, 0.9, 1, ctxt);
    l->Draw();
    c.SaveAs(VERTDIR + "/h_findEff_trueVertex_" + vertCutFlowsTP[i]->getVarName() + ".pdf");
  }
  std::cout<<"fakeEff_trackVertex"<<std::endl;
  for(uint i=0; i<vertCutFlows.size(); i++){
    removeFlows(vertexCutFlows[i][1][0]);
    l->Clear();
    for(uint j=1; j<vertCuts.size(); j++){
      removeFlows(vertexCutFlows[i][1][j]);
      TH1F* h_fakeEff = (TH1F*)vertexCutFlows[i][1][j]->Clone();
      h_fakeEff->Divide(vertexCutFlows[i][1][j],vertexCutFlows[i][1][0],1.0,1.0,"B");
      h_fakeEff->SetStats(0);
      if(j!=10){
	h_fakeEff->SetLineColor(j);
	h_fakeEff->SetMarkerColor(j);
      }
      else{
	h_fakeEff->SetLineColor(40);
	h_fakeEff->SetMarkerColor(40);
      }
      l->AddEntry(h_fakeEff,vertCuts[j]->getCutLabel(),"lp");
      if(j==1){
	raiseMax(h_fakeEff);
	h_fakeEff->Draw();
      }
      else{
	h_fakeEff->Draw("SAME");
      }
    }
    mySmallText(0.3, 0.9, 1, ctxt);
    l->Draw();
    c.SaveAs(VERTDIR + "/h_fakeEff_trackVertex_" + vertCutFlows[i]->getVarName() + ".pdf");
  }
  std::cout<<"correctVsFalse"<<std::endl;
  for(uint i=0; i<vertCutFlows.size(); i++){
    for(uint j=0; j<vertCuts.size(); j++){
      l->Clear();
      for(uint k=0; k<vertType.size(); k++){
	removeFlows(vertexCutFlows[i][k][j]);
	vertexCutFlows[i][k][j]->SetStats(0);
	vertexCutFlows[i][k][j]->Scale(1./vertexCutFlows[i][k][j]->Integral());
	TString varString = vertCutFlows[i]->getVarName();
	if(varString.Contains("score") || varString.Contains("R_T")){
	  std::cout<<"/h_correctVsFalse_" + vertCutFlows[i]->getVarName() + "_" + vertCuts[j]->getCutName() + "Cut.pdf"<<std::endl;
	  std::cout<<"vertType: "<<vertType[k]<<std::endl;
	  std::string binValues = "[";
	  for(int ibin=1; ibin<(vertexCutFlows[i][k][j]->GetNbinsX()+1); ibin++){
	    binValues+=to_string(vertexCutFlows[i][k][j]->GetBinContent(ibin)) + ", ";
	  }
	  binValues+="]";
	  std::cout<<"binValues: "<<binValues<<std::endl;
	}
	if(k!=9){
	  vertexCutFlows[i][k][j]->SetLineColor(k+1);
	  vertexCutFlows[i][k][j]->SetMarkerColor(k+1);
	}
	else{
	  vertexCutFlows[i][k][j]->SetLineColor(40);
	  vertexCutFlows[i][k][j]->SetMarkerColor(40);
	}
	l->AddEntry(vertexCutFlows[i][k][j],vertType[k],"l");
	if(k==0){
	  removeFlows(vertexCutFlows[i][k+1][j]);
	  vertexCutFlows[i][k+1][j]->Scale(1./vertexCutFlows[i][k+1][j]->Integral());
	  raiseMax(vertexCutFlows[i][k][j],vertexCutFlows[i][k+1][j]);
	  vertexCutFlows[i][k][j]->Draw("HIST");
	}
	else{
	  vertexCutFlows[i][k][j]->Draw("HIST,SAME");
	}
      }
      mySmallText(0.3, 0.9, 1, ctxt);
      l->Draw();
      c.SaveAs(VERTDIR + "/h_correctVsFalse_" + vertCutFlows[i]->getVarName() + "_" + vertCuts[j]->getCutName() + "Cut.pdf");
    }
  }
  std::cout<<"trueVertex_charge_vs_numTPs"<<std::endl;
  h_trueVertex_charge_vs_numTPs->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_trueVertex_charge_vs_numTPs);
  h_trueVertex_charge_vs_numTPs->SetStats(0);
  c.SetLogz();
  h_trueVertex_charge_vs_numTPs->Draw("COLZ");
  mySmallText(0.4, 0.82, 1, ctxt);
  c.SaveAs(DIR + "/h_trueVertex_charge_vs_numTPs.pdf");
  delete h_trueVertex_charge_vs_numTPs;
  c.SetLogz(0);
  
  std::cout<<"numSelectedTrks"<<std::endl;
  h_numSelectedTrks->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numSelectedTrks);
  h_numSelectedTrks->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_numSelectedTrks->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_numSelectedTrks->GetName() + ".pdf");
  h_numSelectedTrks->SetStats(0);
  h_numSelectedTrks->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  c.SaveAs(DIR + "/"+ h_numSelectedTrks->GetName() + "_noStatBox.pdf");
  delete h_numSelectedTrks;

  h_numSelectedTrks_zoomOut->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numSelectedTrks_zoomOut);
  h_numSelectedTrks_zoomOut->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_numSelectedTrks_zoomOut->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_numSelectedTrks_zoomOut->GetName() + ".pdf");
  h_numSelectedTrks_zoomOut->SetStats(0);
  h_numSelectedTrks_zoomOut->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  c.SaveAs(DIR + "/"+ h_numSelectedTrks_zoomOut->GetName() + "_noStatBox.pdf");
  delete h_numSelectedTrks_zoomOut;

  std::cout<<"trk H_T"<<std::endl;
  h_trk_H_T->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_trk_H_T);
  h_trk_H_T->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_trk_H_T->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_trk_H_T->GetName() + ".pdf");
  delete h_trk_H_T;

  h_trk_oneMatch_H_T->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_trk_oneMatch_H_T);
  h_trk_oneMatch_H_T->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_trk_oneMatch_H_T->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_trk_oneMatch_H_T->GetName() + ".pdf");
  delete h_trk_oneMatch_H_T;

  std::cout<<"trkMET"<<std::endl;
  h_trk_MET->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_trk_MET);
  h_trk_MET->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_trk_MET->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_trk_MET->GetName() + ".pdf");
  delete h_trk_MET;

  h_trk_oneMatch_MET->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_trk_oneMatch_MET);
  h_trk_oneMatch_MET->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_trk_oneMatch_MET->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_trk_oneMatch_MET->GetName() + ".pdf");
  delete h_trk_oneMatch_MET;

  std::cout<<"tp HT"<<std::endl;
  h_tp_H_T->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_tp_H_T);
  h_tp_H_T->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_tp_H_T->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_tp_H_T->GetName() + ".pdf");
  delete h_tp_H_T;
  std::cout<<"tp MET"<<std::endl;
  h_tp_MET->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_tp_MET);
  h_tp_MET->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_tp_MET->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_tp_MET->GetName() + ".pdf");
  delete h_tp_MET;
#if 0
  int numPart = numPart_primary_noCuts.size();
  TH1F *h_numPart_primary_noCuts = new TH1F("h_numPart_primary_noCuts","h_numPart_primary_noCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_primary_chi2rzdofCuts = new TH1F("h_numPart_primary_chi2rzdofCuts","h_numPart_primary_chi2rzdofCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_primary_bendchi2Cuts = new TH1F("h_numPart_primary_bendchi2Cuts","h_numPart_primary_bendchi2Cuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_primary_chi2rphidofCuts = new TH1F("h_numPart_primary_chi2rphidofCuts","h_numPart_primary_chi2rphidofCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_primary_nstubCuts = new TH1F("h_numPart_primary_nstubCuts","h_numPart_primary_nstubCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_primary_ptCuts = new TH1F("h_numPart_primary_ptCuts","h_numPart_primary_ptCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_primary_d0Cuts = new TH1F("h_numPart_primary_d0Cuts","h_numPart_primary_d0Cuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_primary_z0Cuts = new TH1F("h_numPart_primary_z0Cuts","h_numPart_primary_z0Cuts; pdgid; Number of Particles",numPart,0,numPart);

  int binNum = 1;
  for(const auto & [key, value] : numPart_primary_noCuts){
    h_numPart_primary_noCuts->SetBinContent(binNum,value);
    h_numPart_primary_noCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_primary_noCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_primary_noCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_primary_ptCuts){
    h_numPart_primary_ptCuts->SetBinContent(binNum,value);
    h_numPart_primary_ptCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_primary_ptCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_primary_ptCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_primary_d0Cuts){
    h_numPart_primary_d0Cuts->SetBinContent(binNum,value);
    h_numPart_primary_d0Cuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_primary_d0Cuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_primary_d0Cuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_primary_chi2rzdofCuts){
    h_numPart_primary_chi2rzdofCuts->SetBinContent(binNum,value);
    h_numPart_primary_chi2rzdofCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_primary_chi2rzdofCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_primary_chi2rzdofCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_primary_bendchi2Cuts){
    h_numPart_primary_bendchi2Cuts->SetBinContent(binNum,value);
    h_numPart_primary_bendchi2Cuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_primary_bendchi2Cuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_primary_bendchi2Cuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_primary_chi2rphidofCuts){
    h_numPart_primary_chi2rphidofCuts->SetBinContent(binNum,value);
    h_numPart_primary_chi2rphidofCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_primary_chi2rphidofCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_primary_chi2rphidofCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_primary_nstubCuts){
    h_numPart_primary_nstubCuts->SetBinContent(binNum,value);
    h_numPart_primary_nstubCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_primary_nstubCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_primary_nstubCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_primary_z0Cuts){
    h_numPart_primary_z0Cuts->SetBinContent(binNum,value);
    h_numPart_primary_z0Cuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_primary_z0Cuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_primary_z0Cuts);
  h_numPart_primary_chi2rzdofCuts->SetName("partEff_pt_primary");
  h_numPart_primary_chi2rzdofCuts->GetXaxis()->SetTitle("pdgid");
  h_numPart_primary_chi2rzdofCuts->GetYaxis()->SetTitle("Cut Efficiency");
  h_numPart_primary_chi2rzdofCuts->Divide(h_numPart_primary_chi2rzdofCuts,h_numPart_primary_noCuts,1.0,1.0,"B");
  h_numPart_primary_chi2rzdofCuts->SetLineColor(1);
  h_numPart_primary_chi2rzdofCuts->SetMarkerColor(1);
  h_numPart_primary_chi2rzdofCuts->SetStats(0);
  h_numPart_primary_chi2rzdofCuts->GetYaxis()->SetRangeUser(0,1);
  h_numPart_primary_chi2rzdofCuts->GetXaxis()->SetRangeUser(0,numPart);
  h_numPart_primary_chi2rzdofCuts->Draw();
  h_numPart_primary_bendchi2Cuts->Divide(h_numPart_primary_bendchi2Cuts,h_numPart_primary_noCuts,1.0,1.0,"B");
  h_numPart_primary_bendchi2Cuts->SetLineColor(2);
  h_numPart_primary_bendchi2Cuts->SetMarkerColor(2);
  h_numPart_primary_bendchi2Cuts->SetStats(0);
  h_numPart_primary_bendchi2Cuts->Draw("SAME");
  h_numPart_primary_chi2rphidofCuts->Divide(h_numPart_primary_chi2rphidofCuts,h_numPart_primary_noCuts,1.0,1.0,"B");
  h_numPart_primary_chi2rphidofCuts->SetLineColor(3);
  h_numPart_primary_chi2rphidofCuts->SetMarkerColor(3);
  h_numPart_primary_chi2rphidofCuts->SetStats(0);
  h_numPart_primary_chi2rphidofCuts->Draw("SAME");
  h_numPart_primary_nstubCuts->Divide(h_numPart_primary_nstubCuts,h_numPart_primary_noCuts,1.0,1.0,"B");
  h_numPart_primary_nstubCuts->SetLineColor(4);
  h_numPart_primary_nstubCuts->SetMarkerColor(4);
  h_numPart_primary_nstubCuts->SetStats(0);
  h_numPart_primary_nstubCuts->Draw("SAME");
  h_numPart_primary_ptCuts->Divide(h_numPart_primary_ptCuts,h_numPart_primary_noCuts,1.0,1.0,"B");
  h_numPart_primary_ptCuts->SetLineColor(5);
  h_numPart_primary_ptCuts->SetMarkerColor(5);
  h_numPart_primary_ptCuts->SetStats(0);
  h_numPart_primary_ptCuts->Draw("SAME");
  h_numPart_primary_d0Cuts->Divide(h_numPart_primary_d0Cuts,h_numPart_primary_noCuts,1.0,1.0,"B");
  h_numPart_primary_d0Cuts->SetLineColor(6);
  h_numPart_primary_d0Cuts->SetMarkerColor(6);
  h_numPart_primary_d0Cuts->SetStats(0);
  h_numPart_primary_d0Cuts->Draw("SAME");
  h_numPart_primary_z0Cuts->Divide(h_numPart_primary_z0Cuts,h_numPart_primary_noCuts,1.0,1.0,"B");
  h_numPart_primary_z0Cuts->SetLineColor(7);
  h_numPart_primary_z0Cuts->SetMarkerColor(7);
  h_numPart_primary_z0Cuts->SetStats(0);
  h_numPart_primary_z0Cuts->Draw("SAME");
  mySmallText(0.4, 0.82, 1, ctxt);
  l->Clear();
  l->AddEntry(h_numPart_primary_chi2rzdofCuts,"#chi^{2}_{rz}/d.o.f Cut","lp");
  l->AddEntry(h_numPart_primary_bendchi2Cuts,"#chi^{2}_{bend} Cut","lp");
  l->AddEntry(h_numPart_primary_chi2rphidofCuts,"#chi^{2}_{r#phi}/d.o.f Cut","lp");
  l->AddEntry(h_numPart_primary_nstubCuts,"n_{stub} Cut","lp");
  l->AddEntry(h_numPart_primary_ptCuts,"p_{T} Cut","lp");
  l->AddEntry(h_numPart_primary_d0Cuts,"d_{0} Cut","lp");
  l->AddEntry(h_numPart_primary_z0Cuts,"z_{0} Cut","lp");
  l->Draw();
  c.SaveAs(DIR + "/h_partEffOverlay_pt_primary.pdf");
  delete h_numPart_primary_noCuts;
  delete h_numPart_primary_ptCuts;
  delete h_numPart_primary_d0Cuts;
  delete h_numPart_primary_chi2rzdofCuts;
  delete h_numPart_primary_bendchi2Cuts;
  delete h_numPart_primary_chi2rphidofCuts;
  delete h_numPart_primary_nstubCuts;
  delete h_numPart_primary_z0Cuts;
  
  numPart = numPart_np_noCuts.size();
  TH1F *h_numPart_np_noCuts = new TH1F("h_numPart_np_noCuts","h_numPart_np_noCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_np_chi2rzdofCuts = new TH1F("h_numPart_np_chi2rzdofCuts","h_numPart_np_chi2rzdofCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_np_bendchi2Cuts = new TH1F("h_numPart_np_bendchi2Cuts","h_numPart_np_bendchi2Cuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_np_chi2rphidofCuts = new TH1F("h_numPart_np_chi2rphidofCuts","h_numPart_np_chi2rphidofCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_np_nstubCuts = new TH1F("h_numPart_np_nstubCuts","h_numPart_np_nstubCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_np_ptCuts = new TH1F("h_numPart_np_ptCuts","h_numPart_np_ptCuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_np_d0Cuts = new TH1F("h_numPart_np_d0Cuts","h_numPart_np_d0Cuts; pdgid; Number of Particles",numPart,0,numPart);
  TH1F *h_numPart_np_z0Cuts = new TH1F("h_numPart_np_z0Cuts","h_numPart_np_z0Cuts; pdgid; Number of Particles",numPart,0,numPart);

  binNum = 1;
  for(const auto & [key, value] : numPart_np_noCuts){
    h_numPart_np_noCuts->SetBinContent(binNum,value);
    h_numPart_np_noCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_np_noCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_np_noCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_np_ptCuts){
    h_numPart_np_ptCuts->SetBinContent(binNum,value);
    h_numPart_np_ptCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_np_ptCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_np_ptCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_np_d0Cuts){
    h_numPart_np_d0Cuts->SetBinContent(binNum,value);
    h_numPart_np_d0Cuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_np_d0Cuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_np_d0Cuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_np_chi2rzdofCuts){
    h_numPart_np_chi2rzdofCuts->SetBinContent(binNum,value);
    h_numPart_np_chi2rzdofCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_np_chi2rzdofCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_np_chi2rzdofCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_np_bendchi2Cuts){
    h_numPart_np_bendchi2Cuts->SetBinContent(binNum,value);
    h_numPart_np_bendchi2Cuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_np_bendchi2Cuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_np_bendchi2Cuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_np_chi2rphidofCuts){
    h_numPart_np_chi2rphidofCuts->SetBinContent(binNum,value);
    h_numPart_np_chi2rphidofCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_np_chi2rphidofCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_np_chi2rphidofCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_np_nstubCuts){
    h_numPart_np_nstubCuts->SetBinContent(binNum,value);
    h_numPart_np_nstubCuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_np_nstubCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_np_nstubCuts);
  binNum = 1;
  for(const auto & [key, value] : numPart_np_z0Cuts){
    h_numPart_np_z0Cuts->SetBinContent(binNum,value);
    h_numPart_np_z0Cuts->GetXaxis()->SetBinLabel(binNum,key.c_str());
    binNum++;
  }
  h_numPart_np_z0Cuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_numPart_np_z0Cuts);
  h_numPart_np_chi2rzdofCuts->SetName("partEff_pt_np");
  h_numPart_np_chi2rzdofCuts->GetXaxis()->SetTitle("pdgid");
  h_numPart_np_chi2rzdofCuts->GetYaxis()->SetTitle("Cut Efficiency");
  h_numPart_np_chi2rzdofCuts->Divide(h_numPart_np_chi2rzdofCuts,h_numPart_np_noCuts,1.0,1.0,"B");
  h_numPart_np_chi2rzdofCuts->SetLineColor(1);
  h_numPart_np_chi2rzdofCuts->SetMarkerColor(1);
  h_numPart_np_chi2rzdofCuts->SetStats(0);
  h_numPart_np_chi2rzdofCuts->GetYaxis()->SetRangeUser(0,1);
  h_numPart_np_chi2rzdofCuts->GetXaxis()->SetRangeUser(0,numPart);
  h_numPart_np_chi2rzdofCuts->Draw();
  h_numPart_np_bendchi2Cuts->Divide(h_numPart_np_bendchi2Cuts,h_numPart_np_noCuts,1.0,1.0,"B");
  h_numPart_np_bendchi2Cuts->SetLineColor(2);
  h_numPart_np_bendchi2Cuts->SetMarkerColor(2);
  h_numPart_np_bendchi2Cuts->SetStats(0);
  h_numPart_np_bendchi2Cuts->Draw("SAME");
  h_numPart_np_chi2rphidofCuts->Divide(h_numPart_np_chi2rphidofCuts,h_numPart_np_noCuts,1.0,1.0,"B");
  h_numPart_np_chi2rphidofCuts->SetLineColor(3);
  h_numPart_np_chi2rphidofCuts->SetMarkerColor(3);
  h_numPart_np_chi2rphidofCuts->SetStats(0);
  h_numPart_np_chi2rphidofCuts->Draw("SAME");
  h_numPart_np_nstubCuts->Divide(h_numPart_np_nstubCuts,h_numPart_np_noCuts,1.0,1.0,"B");
  h_numPart_np_nstubCuts->SetLineColor(4);
  h_numPart_np_nstubCuts->SetMarkerColor(4);
  h_numPart_np_nstubCuts->SetStats(0);
  h_numPart_np_nstubCuts->Draw("SAME");
  h_numPart_np_ptCuts->Divide(h_numPart_np_ptCuts,h_numPart_np_noCuts,1.0,1.0,"B");
  h_numPart_np_ptCuts->SetLineColor(5);
  h_numPart_np_ptCuts->SetMarkerColor(5);
  h_numPart_np_ptCuts->SetStats(0);
  h_numPart_np_ptCuts->Draw("SAME");
  h_numPart_np_d0Cuts->Divide(h_numPart_np_d0Cuts,h_numPart_np_noCuts,1.0,1.0,"B");
  h_numPart_np_d0Cuts->SetLineColor(6);
  h_numPart_np_d0Cuts->SetMarkerColor(6);
  h_numPart_np_d0Cuts->SetStats(0);
  h_numPart_np_d0Cuts->Draw("SAME");
  h_numPart_np_z0Cuts->Divide(h_numPart_np_z0Cuts,h_numPart_np_noCuts,1.0,1.0,"B");
  h_numPart_np_z0Cuts->SetLineColor(7);
  h_numPart_np_z0Cuts->SetMarkerColor(7);
  h_numPart_np_z0Cuts->SetStats(0);
  h_numPart_np_z0Cuts->Draw("SAME");
  mySmallText(0.4, 0.82, 1, ctxt);
  l->Clear();
  l->AddEntry(h_numPart_np_chi2rzdofCuts,"#chi^{2}_{rz}/d.o.f Cut","lp");
  l->AddEntry(h_numPart_np_bendchi2Cuts,"#chi^{2}_{bend} Cut","lp");
  l->AddEntry(h_numPart_np_chi2rphidofCuts,"#chi^{2}_{r#phi}/d.o.f Cut","lp");
  l->AddEntry(h_numPart_np_nstubCuts,"n_{stub} Cut","lp");
  l->AddEntry(h_numPart_np_ptCuts,"p_{T} Cut","lp");
  l->AddEntry(h_numPart_np_d0Cuts,"d_{0} Cut","lp");
  l->AddEntry(h_numPart_np_z0Cuts,"z_{0} Cut","lp");
  l->Draw();
  c.SaveAs(DIR + "/h_partEffOverlay_pt_np.pdf");
  delete h_numPart_np_noCuts;
  delete h_numPart_np_ptCuts;
  delete h_numPart_np_d0Cuts;
  delete h_numPart_np_chi2rzdofCuts;
  delete h_numPart_np_bendchi2Cuts;
  delete h_numPart_np_chi2rphidofCuts;
  delete h_numPart_np_nstubCuts;
  delete h_numPart_np_z0Cuts;
#endif
  h_trueVertex_numAllCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_trueVertex_numAllCuts);
  h_trueVertex_numAllCuts->SetStats(0);
  h_trueVertex_numAllCuts->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_trueVertex_numAllCuts->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_trueVertex_numAllCuts->GetName() + ".pdf");
  delete h_trueVertex_numAllCuts;

  h_trackVertexBranch_numAllCuts->GetYaxis()->SetNoExponent(kTRUE);
  removeFlows(h_trackVertexBranch_numAllCuts);
  c.SetLogy();
  h_trackVertexBranch_numAllCuts->SetStats(0);
  h_trackVertexBranch_numAllCuts->Draw();
  mySmallText(0.4, 0.82, 1, ctxt);
  h_trackVertexBranch_numAllCuts->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_trackVertexBranch_numAllCuts->GetName() + ".pdf");
  delete h_trackVertexBranch_numAllCuts;
  c.SetLogy(0);
  
  std::cout<<"triggerEff"<<std::endl;
  TH1F *h_triggerEff = new TH1F("h_triggerEff","h_triggerEff; Cut Name; Percentage of Events Triggered",vertCuts.size(),0,vertCuts.size());
  for(uint i=0; i<vertCuts.size(); i++){
    float numTriggers = 0.0;
    removeFlows(vertexNumVertices[i]);
    for(int j=2; j<(vertexNumVertices[i]->GetNbinsX()+1); j++){
      numTriggers += vertexNumVertices[i]->GetBinContent(j);
    }
    std::cout<<"i cut: "<<i<<" numTriggers: "<<numTriggers<<std::endl;
    h_triggerEff->SetBinContent(i,numTriggers/nevt);
    h_triggerEff->GetXaxis()->SetBinLabel(i,vertCuts[i]->getCutName());
  }
  raiseMax(h_triggerEff);
  h_triggerEff->SetStats(0);
  h_triggerEff->Draw("HIST, TEXT");
  mySmallText(0.4, 0.82, 1, ctxt);
  c.SaveAs(DIR + "/"+ h_triggerEff->GetName() + ".pdf");
  delete h_triggerEff;
  for(uint i=0; i<vertCuts.size(); i++){
    delete vertexNumVertices[i];
  }

  std::cout<<"fiducial triggerEff"<<std::endl;
  std::cout<<"findable events: "<<n_findableEvent<<" fraction of total events: "<<(float)n_findableEvent/(float)nevt<<std::endl;
  TH1F *h_fiducialTriggerEff = new TH1F("h_fiducialTriggerEff","h_fiducialTriggerEff; Cut Name; Percentage of Findable Events Triggered",vertCuts.size(),0,vertCuts.size());
  for(uint i=0; i<vertCuts.size(); i++){
    float numTriggers = 0.0;
    removeFlows(fiducialNumVertices[i]);
    for(int j=2; j<(fiducialNumVertices[i]->GetNbinsX()+1); j++){
      numTriggers += fiducialNumVertices[i]->GetBinContent(j);
    }
    //std::cout<<"i cut: "<<i<<" numTriggers: "<<numTriggers<<std::endl;
    h_fiducialTriggerEff->SetBinContent(i,numTriggers/n_findableEvent);
    h_fiducialTriggerEff->GetXaxis()->SetBinLabel(i,vertCuts[i]->getCutName());
  }
  raiseMax(h_fiducialTriggerEff);
  h_fiducialTriggerEff->SetStats(0);
  h_fiducialTriggerEff->Draw("HIST, TEXT");
  mySmallText(0.4, 0.82, 1, ctxt);
  c.SaveAs(DIR + "/"+ h_fiducialTriggerEff->GetName() + ".pdf");
  delete h_fiducialTriggerEff;
  for(uint i=0; i<vertCuts.size(); i++){
    delete fiducialNumVertices[i];
  }
  
  char res[1000];
  float rms = 0;
  TF1* fit;
  fit = new TF1("fit", "gaus", -1, 1);
  removeFlows(h_res_tp_trk_x);
  h_res_tp_trk_x->Fit("fit","R");
  h_res_tp_trk_x->SetStats(0);
  h_res_tp_trk_x->Draw();
  rms = fit->GetParameter(2);
  sprintf(res, "RMS = %.4f", rms);
  mySmallText(0.22, 0.82, 1, res);
  mySmallText(0.4, 0.42, 1, ctxt);
  h_res_tp_trk_x->Write("", TObject::kOverwrite);
  
  std::cout<<"h_res_tp_trk_x"<<std::endl;
  std::string binValues = "[";
  std::string binCenters = "[";
  std::string fitValues = "[";
  for(int ibin=1; ibin<(h_res_tp_trk_x->GetNbinsX()+1); ibin++){
    binValues+=to_string(h_res_tp_trk_x->GetBinContent(ibin)) + ", ";
    binCenters+=to_string(h_res_tp_trk_x->GetBinCenter(ibin)) + ", ";
    fitValues+=to_string(fit->Eval(h_res_tp_trk_x->GetBinCenter(ibin))) + ", ";
  }
  binValues+="]";
  binCenters+="]";
  fitValues+="]";
  std::cout<<"binValues: "<<binValues<<std::endl;
  std::cout<<"binCenters: "<<binCenters<<std::endl;
  std::cout<<"fitValues: "<<fitValues<<std::endl;
  c.SaveAs(DIR + "/"+ h_res_tp_trk_x->GetName() + ".pdf");
  delete h_res_tp_trk_x;
  delete fit;

  fit = new TF1("fit", "gaus", -1, 1);
  removeFlows(h_res_tp_trk_x_zoomOut);
  h_res_tp_trk_x_zoomOut->Fit("fit","R");
  h_res_tp_trk_x_zoomOut->Draw();
  rms = fit->GetParameter(2);
  sprintf(res, "RMS = %.4f", rms);
  mySmallText(0.22, 0.82, 1, res);
  mySmallText(0.4, 0.42, 1, ctxt);
  h_res_tp_trk_x_zoomOut->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_res_tp_trk_x_zoomOut->GetName() + ".pdf");
  delete h_res_tp_trk_x_zoomOut;
  delete fit;

  fit = new TF1("fit", "gaus", -1, 1);
  removeFlows(h_res_tp_trk_y);
  h_res_tp_trk_y->Fit("fit","R");
  h_res_tp_trk_y->SetStats(0);
  h_res_tp_trk_y->Draw();
  rms = fit->GetParameter(2);
  sprintf(res, "RMS = %.4f", rms);
  mySmallText(0.22, 0.82, 1, res);
  mySmallText(0.4, 0.42, 1, ctxt);
  h_res_tp_trk_y->Write("", TObject::kOverwrite);
  std::cout<<"h_res_tp_trk_y"<<std::endl;
  binValues = "[";
  binCenters = "[";
  fitValues = "[";
  for(int ibin=1; ibin<(h_res_tp_trk_y->GetNbinsX()+1); ibin++){
    binValues+=to_string(h_res_tp_trk_y->GetBinContent(ibin)) + ", ";
    binCenters+=to_string(h_res_tp_trk_y->GetBinCenter(ibin)) + ", ";
    fitValues+=to_string(fit->Eval(h_res_tp_trk_y->GetBinCenter(ibin))) + ", ";
  }
  binValues+="]";
  binCenters+="]";
  fitValues+="]";
  std::cout<<"binValues: "<<binValues<<std::endl;
  std::cout<<"binCenters: "<<binCenters<<std::endl;
  std::cout<<"fitValues: "<<fitValues<<std::endl;
  c.SaveAs(DIR + "/"+ h_res_tp_trk_y->GetName() + ".pdf");
  delete h_res_tp_trk_y;
  delete fit;

  fit = new TF1("fit", "gaus", -1, 1);
  removeFlows(h_res_tp_trk_y_zoomOut);
  h_res_tp_trk_y_zoomOut->Fit("fit","R");
  h_res_tp_trk_y_zoomOut->Draw();
  rms = fit->GetParameter(2);
  sprintf(res, "RMS = %.4f", rms);
  mySmallText(0.22, 0.82, 1, res);
  mySmallText(0.4, 0.42, 1, ctxt);
  h_res_tp_trk_y_zoomOut->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_res_tp_trk_y_zoomOut->GetName() + ".pdf");
  delete h_res_tp_trk_y_zoomOut;
  delete fit;

  fit = new TF1("fit", "gaus", -1, 1);
  removeFlows(h_res_tp_trk_r);
  h_res_tp_trk_r->Fit("fit");
  h_res_tp_trk_r->Draw();
  rms = fit->GetParameter(2);
  sprintf(res, "RMS = %.4f", rms);
  mySmallText(0.22, 0.82, 1, res);
  mySmallText(0.4, 0.42, 1, ctxt);
  h_res_tp_trk_r->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_res_tp_trk_r->GetName() + ".pdf");
  delete h_res_tp_trk_r;
  delete fit;

  fit = new TF1("fit", "gaus", -1, 1);
  removeFlows(h_res_tp_trk_phi);
  h_res_tp_trk_phi->Fit("fit");
  h_res_tp_trk_phi->Draw();
  rms = fit->GetParameter(2);
  sprintf(res, "RMS = %.4f", rms);
  mySmallText(0.22, 0.82, 1, res);
  mySmallText(0.4, 0.42, 1, ctxt);
  h_res_tp_trk_phi->Write("", TObject::kOverwrite);
  c.SaveAs(DIR + "/"+ h_res_tp_trk_phi->GetName() + ".pdf");
  delete h_res_tp_trk_phi;
  delete fit;

  fit = new TF1("fit", "gaus", -10, 10);
  removeFlows(h_res_tp_trk_z);
  h_res_tp_trk_z->Fit("fit");
  h_res_tp_trk_z->SetStats(0);
  h_res_tp_trk_z->Draw();
  rms = fit->GetParameter(2);
  sprintf(res, "RMS = %.4f", rms);
  mySmallText(0.22, 0.82, 1, res);
  mySmallText(0.4, 0.42, 1, ctxt);
  h_res_tp_trk_z->Write("", TObject::kOverwrite);
  std::cout<<"h_res_tp_trk_z"<<std::endl;
  binValues = "[";
  binCenters = "[";
  fitValues = "[";
  for(int ibin=1; ibin<(h_res_tp_trk_z->GetNbinsX()+1); ibin++){
    binValues+=to_string(h_res_tp_trk_z->GetBinContent(ibin)) + ", ";
    binCenters+=to_string(h_res_tp_trk_z->GetBinCenter(ibin)) + ", ";
    fitValues+=to_string(fit->Eval(h_res_tp_trk_z->GetBinCenter(ibin))) + ", ";
  }
  binValues+="]";
  binCenters+="]";
  fitValues+="]";
  std::cout<<"binValues: "<<binValues<<std::endl;
  std::cout<<"binCenters: "<<binCenters<<std::endl;
  std::cout<<"fitValues: "<<fitValues<<std::endl;
  c.SaveAs(DIR + "/"+ h_res_tp_trk_z->GetName() + ".pdf");
  delete h_res_tp_trk_z;
  delete fit;

  //Geometric plot of circle projections and vertex locations
  /*Double_t x_min=0;
  Double_t x_max=0;
  Double_t y_min=0;
  Double_t y_max=0;
  Double_t x_values[6] = {geomTrackVertex.a.x0-geomTrackVertex.a.rho,geomTrackVertex.a.x0+geomTrackVertex.a.rho,geomTrackVertex.b.x0-geomTrackVertex.b.rho,geomTrackVertex.b.x0+geomTrackVertex.b.rho,geomTrackVertex.x_dv,geomTrueVertex.x_dv};
  Double_t y_values[6] = {geomTrackVertex.a.y0-geomTrackVertex.a.rho,geomTrackVertex.a.y0+geomTrackVertex.a.rho,geomTrackVertex.b.y0-geomTrackVertex.b.rho,geomTrackVertex.b.y0+geomTrackVertex.b.rho,geomTrackVertex.y_dv,geomTrueVertex.y_dv};
  for(uint i=0;i<6;i++){
    if(x_values[i]<x_min) x_min = x_values[i];
    if(x_values[i]>x_max) x_max = x_values[i];
    if(y_values[i]<y_min) y_min = y_values[i];
    if(y_values[i]>y_max) y_max = y_values[i];
  }
  x_min*=1.1;
  x_max*=1.1;
  y_min*=1.1;
  y_max*=1.1;
  c.DrawFrame(x_min,y_min,x_max,y_max);
  float trk1_POCA_x = geomTrackVertex.a.d0*sin(geomTrackVertex.a.phi);
  float trk1_POCA_y = -1*geomTrackVertex.a.d0*cos(geomTrackVertex.a.phi);
  float trk2_POCA_x = geomTrackVertex.b.d0*sin(geomTrackVertex.b.phi);
  float trk2_POCA_y = -1*geomTrackVertex.b.d0*cos(geomTrackVertex.b.phi);
  TEllipse *circleTrk1 = new TEllipse(geomTrackVertex.a.x0,geomTrackVertex.a.y0,geomTrackVertex.a.rho,geomTrackVertex.a.rho);
  circleTrk1->SetLineColor(kGreen);
  circleTrk1->SetFillStyle(0);
  TEllipse *circleTrk2 = new TEllipse(geomTrackVertex.b.x0,geomTrackVertex.b.y0,geomTrackVertex.b.rho,geomTrackVertex.b.rho);
  circleTrk2->SetLineColor(kBlack);
  circleTrk2->SetFillStyle(0);
  auto trackTraj1 = new TF1("trackTraj1","((sin([0])/cos([0]))*(x-[1]))+[2]",x_min,x_max);
  trackTraj1->SetParameters(geomTrackVertex.a.phi,trk1_POCA_x,trk1_POCA_y);
  trackTraj1->SetLineColor(kGreen);
  auto trackTraj2 = new TF1("trackTraj2","((sin([0])/cos([0]))*(x-[1]))+[2]",x_min,x_max);
  trackTraj2->SetParameters(geomTrackVertex.b.phi,trk2_POCA_x,trk2_POCA_y);
  trackTraj2->SetLineColor(kBlack);
  TMarker m1(geomTrackVertex.x_dv,geomTrackVertex.y_dv,8);
  TMarker m2(geomTrueVertex.x_dv,geomTrueVertex.y_dv,8);
  TMarker m3(trk1_POCA_x,trk1_POCA_y,5);
  TMarker m4(trk2_POCA_x,trk2_POCA_y,5);
  //std::cout<<"trk1 POCA: "<<trk1_POCA_x<<" "<<trk1_POCA_y<<" trk2 POCA: "<<trk2_POCA_x<<" "<<trk2_POCA_y<<std::endl;
  m1.SetMarkerColor(kRed);
  m2.SetMarkerColor(kBlue);
  m3.SetMarkerColor(kRed);
  m4.SetMarkerColor(kBlue);
  circleTrk1->Draw("SAME");
  circleTrk2->Draw("SAME");
  trackTraj1->Draw("SAME");
  trackTraj2->Draw("SAME");
  m1.Draw("SAME");
  m2.Draw("SAME");
  m3.Draw("SAME");
  m4.Draw("SAME");
  c.SaveAs(DIR + "/h_circleFitGeom.pdf");
  c.SaveAs(DIR + "/h_circleFitGeom.pdf");
  x_min = geomTrackVertex.x_dv;
  x_max = geomTrueVertex.x_dv;
  y_min = geomTrackVertex.y_dv;
  y_max = geomTrueVertex.y_dv;
  if(geomTrueVertex.x_dv<x_min){
    x_max = x_min;
    x_min = geomTrueVertex.x_dv; 
  }
  if(geomTrueVertex.y_dv<y_min){
    y_max = y_min;
    y_min = geomTrueVertex.y_dv; 
  }
  //std::cout<<"geom track vertex: "<<geomTrackVertex.x_dv<<" "<<geomTrackVertex.y_dv<<" geom true vertex: "<<geomTrueVertex.x_dv<<" "<<geomTrueVertex.y_dv<<std::endl;
  //std::cout<<"x_min: "<<x_min<<" x_max: "<<x_max<<" y_min: "<<y_min<<" y_max: "<<y_max<<std::endl;
  x_min-=2;
  x_max+=2;
  y_min-=2;
  y_max+=2;
  c.DrawFrame(x_min,y_min,x_max,y_max);
  circleTrk1->Draw("SAME");
  circleTrk2->Draw("SAME");
  //trackTraj1->Draw("SAME");
  //trackTraj2->Draw("SAME");
  m1.Draw("SAME");
  m2.Draw("SAME");
  m3.Draw("SAME");
  m4.Draw("SAME");
  c.SaveAs(DIR + "/h_circleFitGeomZoom.pdf");
  c.SaveAs(DIR + "/h_circleFitGeomZoom.pdf");
  x_min=geomTrackVertex.x_dv;
  x_max=geomTrackVertex.x_dv;
  y_min=geomTrackVertex.y_dv;
  y_max=geomTrackVertex.y_dv;
  Double_t x_values_POCA[4] = {geomTrackVertex.x_dv,geomTrueVertex.x_dv,trk1_POCA_x,trk2_POCA_x};
  Double_t y_values_POCA[4] = {geomTrackVertex.y_dv,geomTrueVertex.y_dv,trk1_POCA_y,trk2_POCA_y};
  for(uint i=0;i<4;i++){
    if(x_values_POCA[i]<x_min) x_min = x_values_POCA[i];
    if(x_values_POCA[i]>x_max) x_max = x_values_POCA[i];
    if(y_values_POCA[i]<y_min) y_min = y_values_POCA[i];
    if(y_values_POCA[i]>y_max) y_max = y_values_POCA[i];
  }
  x_min-=1;
  x_max+=1;
  y_min-=1;
  y_max+=1;
  c.DrawFrame(x_min,y_min,x_max,y_max);
  circleTrk1->Draw("SAME");
  circleTrk2->Draw("SAME");
  trackTraj1->Draw("SAME");
  trackTraj2->Draw("SAME");
  m1.Draw("SAME");
  m2.Draw("SAME");
  m3.Draw("SAME");
  m4.Draw("SAME");
  c.SaveAs(DIR + "/h_circleFitGeomPOCA.pdf");
  c.SaveAs(DIR + "/h_circleFitGeomPOCA.pdf");
  fout->Close()*/;

  for(uint i=0; i<varCutFlows.size(); i++){
    for(uint j=0; j<trackType.size(); j++){
      for(uint k=0; k<preselCutsSize; k++){
	for(uint l=0; l<plotModifiers.size(); l++){
	  delete preselCutFlows[i][j][k][l];
	}
      }
    }
  }
  for(uint i=0; i<varCutFlows2D.size(); i++){
    for(uint j=0; j<trackType.size(); j++){
      for(uint k=0; k<preselCutsSize; k++){
	for(uint l=0; l<plotModifiers.size(); l++){
	  delete preselCutFlows2D[i][j][k][l];
	}
      }
    }
  }
  for(uint i=0; i<varCutFlowsTP.size(); i++){
    for(uint j=0; j<tpType.size(); j++){
      for(uint k=0; k<preselCutsTP.size(); k++){
	for(uint l=0; l<plotModifiers.size(); l++){
	  delete preselCutFlowsTP[i][j][k][l];
	}
      }
    }
  }
  for(uint i=0; i<varCutFlowsTP2D.size(); i++){
    for(uint j=0; j<tpType.size(); j++){
      for(uint k=0; k<preselCutsTP.size(); k++){
	for(uint l=0; l<plotModifiers.size(); l++){
	  delete preselCutFlowsTP2D[i][j][k][l];
	}
      }
    }
  }
  for(uint i=0; i<vertCutFlows.size(); i++){
    for(uint j=0; j<vertType.size(); j++){
      for(uint k=0; k<vertCuts.size(); k++){
	delete vertexCutFlows[i][j][k];
      }
    }
  }
  for(uint i=0; i<vertCutFlowsTP.size(); i++){
    for(uint j=0; j<vertCuts.size(); j++){
      for(uint k=0; k<vertPlotTPModifiers.size(); k++){
	delete vertexCutFlowsMatchTP[i][j][k];
      }
    }
  }
  for(uint i=0; i<vertCutFlowsTP.size(); i++){
    for(uint j=0; j<vertPlotTPModifiers.size(); j++){
      delete vertexCutFlowsTP[i][j];
    }
  }

}
 

Double_t dist_TPs(Track_Parameters* a, Track_Parameters* b){  
  float x1 = a->x0; //   Centers of the circles
  float y1 = a->y0; // 
  float x2 = b->x0; // 
  float y2 = b->y0; // 
  float R1 = a->rho;   // Radii of the circles
  float R2 = b->rho;
  float R = dist(x1,y1,x2,y2); // Distance between centers
  if((R>=(R1-R2)) && (R<=(R1+R2))){
    return (0);
  }
  else if(R==0){
    return (-99999.0);
  }
  else{

    return(R-R1-R2);
  }
}

Double_t dist_TPs(Track_Parameters a, Track_Parameters b){  
  float x1 = a.x0; //   Centers of the circles
  float y1 = a.y0; // 
  float x2 = b.x0; // 
  float y2 = b.y0; // 
  float R1 = a.rho;   // Radii of the circles
  float R2 = b.rho;
  float R = dist(x1,y1,x2,y2); // Distance between centers
  if((R>=(R1-R2)) && (R<=(R1+R2))){
    return (0);
  }
  else if(R==0){
    return (-99999.0);
  }
  else{

    return(R-R1-R2);
  }
}

Int_t calcVertex(Track_Parameters a, Track_Parameters b, Double_t &x_vtx, Double_t &y_vtx, Double_t &z_vtx){
  float x1 = a.x0; //   Centers of the circles
  float y1 = a.y0; // 
  float x2 = b.x0; // 
  float y2 = b.y0; // 
  float R1 = a.rho;   // Radii of the circles
  float R2 = b.rho;
  float R = dist(x1,y1,x2,y2); // Distance between centers
  if(R==0) return -1;
  float co1 = (pow(R1,2)-pow(R2,2))/(2*pow(R,2));
  float radicand = (2/pow(R,2))*(pow(R1,2)+pow(R2,2))-(pow(pow(R1,2)-pow(R2,2),2)/pow(R,4))-1;
  float co2 = 0;
  if(radicand>0) co2 = 0.5*TMath::Sqrt(radicand);
  float ix1_x = 0.5*(x1+x2)+co1*(x2-x1)+co2*(y2-y1);
  float ix2_x = 0.5*(x1+x2)+co1*(x2-x1)-co2*(y2-y1);
  float ix1_y = 0.5*(y1+y2)+co1*(y2-y1)+co2*(x1-x2);
  float ix2_y = 0.5*(y1+y2)+co1*(y2-y1)-co2*(x1-x2);
  float ix1_z1 = a.z(ix1_x,ix1_y);
  float ix1_z2 = b.z(ix1_x,ix1_y);
  float ix1_delz = fabs(ix1_z1-ix1_z2); 
  float ix2_z1 = a.z(ix2_x,ix2_y);
  float ix2_z2 = b.z(ix2_x,ix2_y);
  float ix2_delz = fabs(ix2_z1-ix2_z2); 
  //std::cout<<"R: "<<R<<" co1: "<<co1<<" co2: "<<co2<<" ix1 delz: "<<ix1_delz<<" ix2 delz: "<<ix2_delz<<std::endl;
  //std::cout<<"ix1_x: "<<ix1_x<<" ix1_y: "<<ix1_y<<" ix2_x: "<<ix2_x<<" ix2_y: "<<ix2_y<<std::endl;
  //std::cout<<"ix1_z1: "<<ix1_z1<<" ix1_z2: "<<ix1_z2<<" ix2_z1: "<<ix2_z1<<" ix2_z2: "<<ix2_z2<<" trk 1 z0: "<<a.z0<<" trk 2 z0: "<<b.z0<<std::endl;
  float trk1_POCA[2] = {a.d0*sin(a.phi),-1*a.d0*cos(a.phi)};
  float trk2_POCA[2] = {b.d0*sin(b.phi),-1*b.d0*cos(b.phi)};
  float trk1_ix1_delxy[2] = {ix1_x-trk1_POCA[0],ix1_y-trk1_POCA[1]};
  float trk1_ix2_delxy[2] = {ix2_x-trk1_POCA[0],ix2_y-trk1_POCA[1]};
  float trk2_ix1_delxy[2] = {ix1_x-trk2_POCA[0],ix1_y-trk2_POCA[1]};
  float trk2_ix2_delxy[2] = {ix2_x-trk2_POCA[0],ix2_y-trk2_POCA[1]};
  float trk1_traj[2] = {cos(a.phi),sin(a.phi)};
  float trk2_traj[2] = {cos(b.phi),sin(b.phi)};
  bool trk1_ix1_inTraj = ((trk1_ix1_delxy[0]*trk1_traj[0]+trk1_ix1_delxy[1]*trk1_traj[1])>0) ? true : false;
  bool trk1_ix2_inTraj = ((trk1_ix2_delxy[0]*trk1_traj[0]+trk1_ix2_delxy[1]*trk1_traj[1])>0) ? true : false;
  bool trk2_ix1_inTraj = ((trk2_ix1_delxy[0]*trk2_traj[0]+trk2_ix1_delxy[1]*trk2_traj[1])>0) ? true : false;
  bool trk2_ix2_inTraj = ((trk2_ix2_delxy[0]*trk2_traj[0]+trk2_ix2_delxy[1]*trk2_traj[1])>0) ? true : false;
  //std::cout<<"ix1 inTraj: "<<trk1_ix1_inTraj<<" "<<trk2_ix1_inTraj<<" ix2 inTraj: "<<trk1_ix2_inTraj<<" "<<trk2_ix2_inTraj<<std::endl;
  if(trk1_ix1_inTraj&&trk2_ix1_inTraj&&trk1_ix2_inTraj&&trk2_ix2_inTraj){
    if(ix1_delz<ix2_delz){
      x_vtx = ix1_x;
      y_vtx = ix1_y;
      //x_alt = ix2_x;
      //y_alt = ix2_y;
      z_vtx = (ix1_z1+ix1_z2)/2;
      return 0;
    }
    else{
      x_vtx = ix2_x;
      y_vtx = ix2_y;
      //x_alt = ix1_x;
      //y_alt = ix1_y;
      z_vtx = (ix2_z1+ix2_z2)/2;
      return 0;
    }
  }
  if(trk1_ix1_inTraj&&trk2_ix1_inTraj){
    x_vtx = ix1_x;
    y_vtx = ix1_y;
    //x_alt = ix2_x;
    //y_alt = ix2_y;
    z_vtx = (ix1_z1+ix1_z2)/2;
    return 1;
  }
  if(trk1_ix2_inTraj&&trk2_ix2_inTraj){
    x_vtx = ix2_x;
    y_vtx = ix2_y;
    //x_alt = ix1_x;
    //y_alt = ix1_y;
    z_vtx = (ix2_z1+ix2_z2)/2;
    return 2;
  }
  else{
    if(ix1_delz<ix2_delz){
      x_vtx = ix1_x;
      y_vtx = ix1_y;
      //x_alt = ix2_x;
      //y_alt = ix2_y;
      z_vtx = (ix1_z1+ix1_z2)/2;
      return 3;
    }
    else{
      x_vtx = ix2_x;
      y_vtx = ix2_y;
      //x_alt = ix1_x;
      //y_alt = ix1_y;
      z_vtx = (ix2_z1+ix2_z2)/2;
      return 3;
    }
  }
  return 4;
}

void SetPlotStyle()
{
  // from ATLAS plot style macro

  // use plain black on white colors
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameFillColor(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetStatColor(0);
  gStyle->SetHistLineColor(1);

  gStyle->SetPalette(1);

  // set the paper & margin sizes
  gStyle->SetPaperSize(20, 26);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadRightMargin(0.19);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);

  // set title offsets (for axis label)
  gStyle->SetTitleXOffset(1.4);
  gStyle->SetTitleYOffset(1.0);

  // use large fonts
  gStyle->SetTextFont(42);
  gStyle->SetTextSize(0.05);
  gStyle->SetLabelFont(42, "x");
  gStyle->SetTitleFont(42, "x");
  gStyle->SetLabelFont(42, "y");
  gStyle->SetTitleFont(42, "y");
  gStyle->SetLabelFont(42, "z");
  gStyle->SetTitleFont(42, "z");
  gStyle->SetLabelSize(0.05, "x");
  gStyle->SetTitleSize(0.05, "x");
  gStyle->SetLabelSize(0.05, "y");
  gStyle->SetTitleSize(0.05, "y");
  gStyle->SetLabelSize(0.05, "z");
  gStyle->SetTitleSize(0.05, "z");

  // use bold lines and markers
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.2);
  gStyle->SetHistLineWidth(2.);
  gStyle->SetLineStyleString(2, "[12 12]");

  // get rid of error bar caps
  gStyle->SetEndErrorSize(0.);

  // do not display any of the standard histogram decorations
  gStyle->SetOptTitle(0);
  //gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  // put tick marks on top and RHS of plots
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
}

void mySmallText(Double_t x, Double_t y, Color_t color, char *text)
{
  Double_t tsize = 0.044;
  TLatex l;
  l.SetTextSize(tsize);
  l.SetNDC();
  l.SetTextColor(color);
  l.DrawLatex(x, y, text);
}

void removeFlows(TH1F* h)
{
  int nbins = h->GetNbinsX();
  double underflow = h->GetBinContent(0);
  double overflow = h->GetBinContent(nbins+1);
  h->AddBinContent(1,underflow);
  h->AddBinContent(nbins,overflow);
  h->SetBinContent(0,0);
  h->SetBinContent(nbins+1,0);
}

void removeFlows(TH2F* h)
{
}
