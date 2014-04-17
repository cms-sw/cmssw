#include <iostream>
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TObject.h"
#include "TH1F.h"
#include "TNamed.h"
#include "TList.h"
#include "TKey.h"
#include "TClass.h"
#include <string>
#include <vector>

CommonAnalyzer::CommonAnalyzer(TFile* file, const char* run, const char* mod, const char* path, const char* prefix):
  _file(file), _runnumber(run), _module(mod), _path(path), _prefix(prefix) { }

CommonAnalyzer::CommonAnalyzer(const CommonAnalyzer& dtca):
  _file(dtca._file), _runnumber(dtca._runnumber), _module(dtca._module), _path(dtca._path), 
  _prefix(dtca._prefix){ }

CommonAnalyzer& CommonAnalyzer::operator=(const CommonAnalyzer& dtca) {
  
  if(this != &dtca) {
    _file = dtca._file;
    _runnumber = dtca._runnumber;
    _module = dtca._module;
    _path = dtca._path;
    _prefix = dtca._prefix;
  }
  return *this;
}

void CommonAnalyzer::setRunNumber(const char* run) { _runnumber = run; }
void CommonAnalyzer::setFile(TFile* file) { _file = file; }
void CommonAnalyzer::setModule(const char* mod) { _module = mod; }
void CommonAnalyzer::setPath(const char* path) { _path = path; }
void CommonAnalyzer::setPrefix(const char* prefix) { _prefix = prefix; }

const std::string& CommonAnalyzer::getRunNumber() const {return _runnumber;}
const std::string& CommonAnalyzer::getModule() const {return _module;}
const std::string& CommonAnalyzer::getPath() const {return _path;}
const std::string& CommonAnalyzer::getPrefix() const {return _prefix;}

TObject* CommonAnalyzer::getObject(const char* name) const {

  TObject* obj = 0;

  std::string fullpath = _module + "/" + _path;
  if(_file) {
    bool ok = _file->cd(fullpath.c_str());
    if(ok && gDirectory) {
      obj = gDirectory->Get(name);
    }
  }
  return obj;

}

TNamed* CommonAnalyzer::getObjectWithSuffix(const char* name, const char* suffix) const {

  TNamed* obj = (TNamed*)getObject(name);

  if(obj) {
    if(!strstr(obj->GetTitle(),"run")) {
      char htitle[300];
      sprintf(htitle,"%s %s run %s",obj->GetTitle(),suffix,_runnumber.c_str());
      obj->SetTitle(htitle);
    }
  }
  return obj;

}

const std::vector<unsigned int> CommonAnalyzer::getRunList() const {

  std::vector<unsigned int> runlist;

  std::string fullpath = _module + "/" + _path;
  if(_file) {
    bool ok = _file->cd(fullpath.c_str());
    if(ok && gDirectory) {
      TList* keys = gDirectory->GetListOfKeys();
      TListIter it(keys);
      TKey* key=0;
      while((key=(TKey*)it.Next())) {
	std::cout << key->GetName() << std::endl;
	TClass cl(key->GetClassName());
	if (cl.InheritsFrom("TDirectory") && strstr(key->GetName(),"run_") != 0 ) {
	  unsigned int run;
	  sscanf(key->GetName(),"run_%u",&run);
	  runlist.push_back(run);
	} 
      }

    }
  }
  //  sort(runlist);
  return runlist;
  
}

TH1F* CommonAnalyzer::getBinomialRatio(const CommonAnalyzer& denom, const char* name, const int rebin) const {
  
  TH1F* den = (TH1F*)denom.getObject(name);
  TH1F* num = (TH1F*)getObject(name);
  TH1F* ratio =0;
  
  if(den!=0 && num!=0) {
    
    TH1F* denreb=den;
    TH1F* numreb=num;
    if(rebin>0) {
      denreb = (TH1F*)den->Rebin(rebin,"denrebinned");
      numreb = (TH1F*)num->Rebin(rebin,"numrebinned");
    }
    
    ratio = new TH1F(*numreb);
    ratio->SetDirectory(0);
    ratio->Reset();
    ratio->Sumw2();
    ratio->Divide(numreb,denreb,1,1,"B");
    delete denreb;
    delete numreb;
  }
  
  return ratio;
}
