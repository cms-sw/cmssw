#include "iostream"
#include "fstream"
#include "sstream"
#include "vector"
#include "map"
#include "string"
#include "TCanvas.h"
#include "TTree.h"
#include "TH1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TObjArray.h"
#include "TString.h"
#include "TKey.h"
#include "TLegend.h"
#include "THStack.h"

struct Params{
  string   _varexp;
  string   _selection;
  string   _option;
  string   _legend;
  Long64_t _nentries;
  Long64_t _firstentry;  
  string   _Hname;
  TH1     *_TH;
  bool     _1D;
};

class DrawTree{

public:
  static int fN;

  DrawTree(TTree *tree);
  ~DrawTree(){};

  void setTree(TTree *tree){_tree=tree;}
  TTree* getTree(){return _tree;}

  void setLegend(float xmin=.8, float ymin=.6, float xsize=.2, float ysize=.2,char* header="");
  void setMarkerStyle(int i){_MarkerStyle=i;_MarkerCount0=0;}
  void setMarkerColor(int i){_MarkerColor=i;_MarkerCount1=0;}
  void setXmin(float val){xmin=val;}
  void setXmax(float val){xmax=val;}
  void setYmin(float val){ymin=val;}
  void setYmax(float val){ymax=val;}
  void setTitle(char* xopt="",char* yopt="");

  void add(char* varexp,char* selection="",Option_t *option="",char* legend="",Long64_t nentries=100000000, Long64_t firstentry=0);
  
  void Draw();

  void setHRef(TH1* h){_TH=h;}
private:

  void Project(Params& A);

  TTree* _tree;
  
  std::vector<Params>  _VPar;

  int _MarkerCount0,_MarkerCount1;
  int _MarkerStyle;
  int _MarkerColor;

  int _countAdd;
  double xmin, xmax, ymin, ymax;

  THStack _stack;
  TLegend *_legend;

  TH1 * _TH;

  std::string xtitle;
  std::string ytitle;
};

int DrawTree::fN=0;

DrawTree::DrawTree(TTree *tree):_tree(tree),_MarkerCount0(0),_MarkerCount1(0),_MarkerStyle(20),_MarkerColor(0),_countAdd(0),_TH(0){

  fN++;

  xmin=100000;
  xmax=-100000;
  ymin=100000;
  ymax=-100000;

  setLegend();
}

void DrawTree::setTitle(char* xopt,char* yopt){
  xtitle.append(xopt);
  ytitle.append(yopt);
}


void DrawTree::setLegend(float xmin, float ymin, float xsize, float ysize,char* header){
    _legend = new TLegend(xmin,ymin,xmin+xsize,ymin+ysize);
    _legend->SetTextAlign(12);
    _legend->SetTextColor(1);
    _legend->SetTextSize(0.03);
    _legend->SetFillStyle(0);
    _legend->SetFillColor(0);
    _legend->SetBorderSize(0);

    _legend->SetHeader(header);

}

void DrawTree::add(char* varexp,char* selection,Option_t *option,char* legend,Long64_t nentries, Long64_t firstentry){

  Params A;
  A._varexp=varexp;
  A._selection=selection;
  A._option=option;
  A._legend=legend;
  A._nentries=nentries;
  A._firstentry=firstentry;

  Project(A);
  _VPar.push_back(A);
  
}

void DrawTree::Draw(){
  bool first=true;

  for (size_t i=0;i<_VPar.size();i++){
    if (_VPar[i]._1D){
      _stack.Draw("nostack");
      break;
    }else{
      if (_TH == NULL ){
	_VPar[i]._TH->GetXaxis()->SetRangeUser(xmin,xmax);
	_VPar[i]._TH->GetXaxis()->SetRangeUser(xmin,xmax);
	_VPar[i]._TH->GetYaxis()->SetRangeUser(ymin,ymax);
	_VPar[i]._TH->GetYaxis()->SetRangeUser(ymin,ymax);
      }
	
      if (!first)
	_VPar[i]._legend.append("same");
      
      if (_TH != NULL && first){
	_VPar[i]._legend.append("same");
	_TH->Draw();
      }

      if (first){
	_VPar[i]._TH->GetXaxis()->SetTitle(xtitle.c_str());
	_VPar[i]._TH->GetYaxis()->SetTitle(ytitle.c_str());
      }
 
      _VPar[i]._TH->SetTitle("");
      _VPar[i]._TH->SetStats(0);
      if (_VPar[i]._TH->GetEntries()){
	_VPar[i]._TH->Draw(_VPar[i]._legend.c_str());
	first=false;
      }
    }
  }
  _legend->Draw();
}

void DrawTree::Project(Params& A){

  char name[128];
  sprintf(name,"drawTree%d_%d",fN,_countAdd++);
  A._Hname=name;
  _tree->Project(name,A._varexp.c_str(),A._selection.c_str(),A._option.c_str(),A._nentries,A._firstentry); 

  TH1* h = (TH1*)gDirectory->Get(A._Hname.c_str());

  h->SetMarkerStyle(_MarkerStyle+(_MarkerCount0++));
  h->SetMarkerColor(_MarkerColor+(_MarkerCount1++));
  A._TH=h;
  _stack.Add(h);  
   cout << A._legend.c_str() << endl;
  _legend->AddEntry(h,A._legend.c_str(),"P");

  float axmin=h->GetXaxis()->GetXmin();
  float axmax=h->GetXaxis()->GetXmax();

  if (xmin>axmin)
    xmin=axmin;
  if (xmax<axmax)
    xmax=axmax;

  if (A._varexp.find(":") != string::npos){
    A._1D=false;
    float aymin=h->GetYaxis()->GetXmin();
    float aymax=h->GetYaxis()->GetXmax();

    if (ymin>aymin)
      ymin=aymin;
    if (ymax<aymax)
      ymax=aymax;

  }
}
