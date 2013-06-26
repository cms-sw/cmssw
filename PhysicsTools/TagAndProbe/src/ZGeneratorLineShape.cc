#include "PhysicsTools/TagAndProbe/interface/ZGeneratorLineShape.h"

ClassImp(ZGeneratorLineShape)

ZGeneratorLineShape::ZGeneratorLineShape(const char *name, const char *title,
					 RooAbsReal& _m, 
					 const char* genfile, const char* histoName
					 ): 
  RooAbsPdf(name,title),
  m("m","m", this,_m),  
  dataHist(0)
{
  TFile *f_gen= TFile::Open(genfile);
  TH1F* mass_th1f = (TH1F*)  f_gen->Get(histoName);
  dataHist = new RooDataHist("Mass_gen", "Mass_gen", _m, mass_th1f );
  f_gen->Close();
}


ZGeneratorLineShape::ZGeneratorLineShape(const ZGeneratorLineShape& other, const char* name):
  RooAbsPdf(other,name),
  m("m", this,other.m),
  dataHist(other.dataHist)
{
}


Double_t ZGeneratorLineShape::evaluate() const{

  // std::cout<<"gen shape: m, evaluate= "<<m<<", "<<dataHist->weight(m.arg())<<std::endl;
  return dataHist->weight(m.arg()) ;
}
