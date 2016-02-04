#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"

int PerformancePayloadFromTFormula::InvalidPos=-1;

#include <iostream>
using namespace std;

float PerformancePayloadFromTFormula::getResult(PerformanceResult::ResultType r ,BinningPointByMap p) const {
  check();
  //
  // which formula to use?
  //
  if (! isInPayload(r,p)) return PerformancePayload::InvalidResult;

  // nice, what to do here???
  TFormula * formula = compiledFormulas_[resultPos(r)];
  //
  // prepare the vector to pass, order counts!!!
  //
  std::vector<BinningVariables::BinningVariablesType> t = myBinning();
  
  // sorry, TFormulas just work up to dimension==4
  Double_t values[4];
  int i=0;
  for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end();++it, ++i){
    values[i] = p.value(*it);    
  }
  //
  // i need a non const version #$%^
  return formula->EvalPar(values);
}

bool PerformancePayloadFromTFormula::isOk(BinningPointByMap p) const {
  
  std::vector<BinningVariables::BinningVariablesType> t = myBinning();
  
  for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end();++it){
    if (!   p.isKeyAvailable(*it)) return false;
    float v = p.value(*it);
    int pos = limitPos(*it);
    std::pair<float, float> limits = (pl.limits())[pos];
    if (v<limits.first || v>limits.second) return false;
  }
  return true;
}

bool PerformancePayloadFromTFormula::isInPayload(PerformanceResult::ResultType res,BinningPointByMap point) const {
  check();
  // first, let's see if it is available at all
  if (resultPos(res) == PerformancePayloadFromTFormula::InvalidPos) return false;
  
  if ( ! isOk(point)) return false;
  return true;
}


void PerformancePayloadFromTFormula::check() const {
  if (pl.formulas().size() == compiledFormulas_.size()) return;
  //
  // otherwise, compile!
  //
  for (unsigned int i=0; i< pl.formulas().size(); ++i){
    TFormula* t = new TFormula("rr",(pl.formulas()[i]).c_str()); //FIXME: "rr" should be unique!
    t->Compile();
    compiledFormulas_.push_back(t);
  }
}

void PerformancePayloadFromTFormula::printFormula(PerformanceResult::ResultType res) const {
  check();
  //
  // which formula to use?
  //
  if (resultPos(res) == PerformancePayloadFromTFormula::InvalidPos)  {
    cout << "Warning: result not available!" << endl;
  }
  
  // nice, what to do here???
  TFormula * formula = compiledFormulas_[resultPos(res)];
  cout << "-- Formula: " << formula->GetExpFormula("p") << endl;
  // prepare the vector to pass, order counts!!!
  //
  std::vector<BinningVariables::BinningVariablesType> t = myBinning();
  
  for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end();++it){
    int pos = limitPos(*it);
    std::pair<float, float> limits = (pl.limits())[pos];
    cout << "      Variable: " << *it << " with limits: " << "from: " << limits.first  << " to: " << limits.second << endl;
  }

}



#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(PerformancePayloadFromTFormula);
