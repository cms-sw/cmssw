#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"

const int PerformancePayloadFromTFormula::InvalidPos = -1;

#include <iostream>
using namespace std;

void PerformancePayloadFromTFormula::initialize() {
  for (std::vector<std::string>::const_iterator formula = pl.formulas().begin(); formula != pl.formulas().end();
       ++formula) {
    const auto formulaUniqueName = edm::createGlobalIdentifier();
    //be sure not to add TFormula to ROOT's global list
    auto temp = std::make_shared<TFormula>(formulaUniqueName.c_str(), formula->c_str(), false);
    temp->Compile();
    compiledFormulas_.emplace_back(std::move(temp));
  }
}

float PerformancePayloadFromTFormula::getResult(PerformanceResult::ResultType r, const BinningPointByMap& _p) const {
  BinningPointByMap p = _p;
  //
  // which formula to use?
  //
  if (!isInPayload(r, p)) {
    edm::LogError("PerformancePayloadFromTFormula")
        << "Missing formula in conditions. Maybe code/conditions are inconsistent" << std::endl;
    assert(false);
  }

  const TFormula* formula = compiledFormulas_[resultPos(r)].get();
  //
  // prepare the vector to pass, order counts!!!
  //
  std::vector<BinningVariables::BinningVariablesType> t = myBinning();

  // sorry, TFormulas just work up to dimension==4
  Double_t values[4];
  int i = 0;
  for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end(); ++it, ++i) {
    values[i] = p.value(*it);
  }
  //
  return formula->EvalPar(values);
}

bool PerformancePayloadFromTFormula::isOk(const BinningPointByMap& _p) const {
  BinningPointByMap p = _p;
  std::vector<BinningVariables::BinningVariablesType> t = myBinning();

  for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end(); ++it) {
    if (!p.isKeyAvailable(*it))
      return false;
    float v = p.value(*it);
    int pos = limitPos(*it);
    std::pair<float, float> limits = (pl.limits())[pos];
    if (v < limits.first || v > limits.second)
      return false;
  }
  return true;
}

bool PerformancePayloadFromTFormula::isInPayload(PerformanceResult::ResultType res,
                                                 const BinningPointByMap& point) const {
  // first, let's see if it is available at all
  if (resultPos(res) == PerformancePayloadFromTFormula::InvalidPos)
    return false;

  if (!isOk(point))
    return false;
  return true;
}

void PerformancePayloadFromTFormula::printFormula(PerformanceResult::ResultType res) const {
  //
  // which formula to use?
  //
  if (resultPos(res) == PerformancePayloadFromTFormula::InvalidPos) {
    cout << "Warning: result not available!" << endl;
    return;
  }

  const TFormula* formula = compiledFormulas_[resultPos(res)].get();
  cout << "-- Formula: " << formula->GetExpFormula("p") << endl;
  // prepare the vector to pass, order counts!!!
  //
  std::vector<BinningVariables::BinningVariablesType> t = myBinning();

  for (std::vector<BinningVariables::BinningVariablesType>::const_iterator it = t.begin(); it != t.end(); ++it) {
    int pos = limitPos(*it);
    std::pair<float, float> limits = (pl.limits())[pos];
    cout << "      Variable: " << *it << " with limits: "
         << "from: " << limits.first << " to: " << limits.second << endl;
  }
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(PerformancePayloadFromTFormula);
