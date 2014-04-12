#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromBinnedTFormula.h"

#include "FWCore/Utilities/interface/Exception.h"

const int PerformancePayloadFromBinnedTFormula::InvalidPos=-1;

#include <iostream>
using namespace std;

void PerformancePayloadFromBinnedTFormula::initialize() {
  for (unsigned int t=0; t< pls.size(); ++t){
    std::vector <boost::shared_ptr<TFormula> > temp;
      for (unsigned int i=0; i< (pls[t].formulas()).size(); ++i){
	PhysicsTFormulaPayload  tmp = pls[t];
	boost::shared_ptr<TFormula> tt(new TFormula("rr",tmp.formulas()[i].c_str()));
	tt->Compile();
	temp.push_back(tt);
      }
      compiledFormulas_.push_back(temp);
    }
}

const boost::shared_ptr<TFormula>& PerformancePayloadFromBinnedTFormula::getFormula(PerformanceResult::ResultType r ,const BinningPointByMap& p ) const {
  //
  // chooses the correct rectangular region
  //
  if (! isInPayload(r,p)) { throw cms::Exception("MalformedPerfPayload") << "Requested performance data not available!"; }
  unsigned int region;
  bool ok =  isOk(p,region);
  if (ok == false) { throw cms::Exception("MalformedPerfPayload") << "Requested variable does not match internal structure!"; }

  return compiledFormulas_[region][resultPos(r)];

}

float PerformancePayloadFromBinnedTFormula::getResult(PerformanceResult::ResultType r ,const BinningPointByMap& _p) const {
  BinningPointByMap p = _p;
  //
  // which formula to use?
  //
  if (! isInPayload(r,p)) return PerformancePayload::InvalidResult;

  // nice, what to do here???
  //  TFormula * formula = compiledFormulas_[resultPos(r)];
  //

  const boost::shared_ptr<TFormula>& formula = getFormula(r,p);

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
  // i need a non const version 
  return formula->EvalPar(values);
}

bool PerformancePayloadFromBinnedTFormula::isOk(const BinningPointByMap& _p,unsigned int& whichone) const {
  
  BinningPointByMap p = _p;
  //
  // change: look on whether a single rectangularr region matches
  //
  for (unsigned int ti=0; ti< pls.size(); ++ti){
    bool result = true;
    std::vector<BinningVariables::BinningVariablesType>  t = myBinning();
    for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end();++it){
      //
      // now looking into a single payload
      //
      if (!   p.isKeyAvailable(*it)) return false;
      float v = p.value(*it);
      int pos = limitPos(*it);
      std::pair<float, float> limits = (pls[ti].limits())[pos];
      if (v<limits.first || v>limits.second) result= false;
    }
    if (result == true)  {
      whichone = ti;
      return true;
    }
  }
  whichone = 9999;
  return false;
}

bool PerformancePayloadFromBinnedTFormula::isInPayload(PerformanceResult::ResultType res,const BinningPointByMap& point) const {
  // first, let's see if it is available at all
  if (resultPos(res) == PerformancePayloadFromBinnedTFormula::InvalidPos) return false;
  unsigned int whocares;
  if ( ! isOk(point,whocares)) return false;
  return true;
}

void PerformancePayloadFromBinnedTFormula::printFormula(PerformanceResult::ResultType res,const BinningPointByMap& point) const {
  //
  // which formula to use?
  //
  if (resultPos(res) == PerformancePayloadFromBinnedTFormula::InvalidPos)  {
    cout << "Warning: result not available!" << endl;
  }
  
  // nice, what to do here???
  const boost::shared_ptr<TFormula>& formula = getFormula(res, point);
  unsigned int whichone;
  isOk(point,whichone);
  cout << "-- Formula: " << formula->GetExpFormula("p") << endl;
  // prepare the vector to pass, order counts!!!
  //
  std::vector<BinningVariables::BinningVariablesType> t = myBinning();
  
  for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end();++it){
    int pos = limitPos(*it);
    std::pair<float, float> limits = (pls[whichone].limits())[pos];
    cout << "      Variable: " << *it << " with limits: " << "from: " << limits.first  << " to: " << limits.second << endl;
  }

}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(PerformancePayloadFromBinnedTFormula);
