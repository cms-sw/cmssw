#ifndef PerformancePayloadFromBinnedTFormula_h
#define PerformancePayloadFromBinnedTFormula_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTFormulaPayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"

#include <algorithm>
#include <string>
#include <vector>

#include "TFormula.h"

#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"

class PerformancePayloadFromBinnedTFormula : public PerformancePayload {
  //  class PerformancePayloadFromBinnedTFormula : public PerformancePayload, public PhysicsPerformancePayload {
public:
  static const int InvalidPos;

  PerformancePayloadFromBinnedTFormula(const std::vector<PerformanceResult::ResultType>& r,
                                       const std::vector<BinningVariables::BinningVariablesType>& b,
                                       const std::vector<PhysicsTFormulaPayload>& in)
      : pls(in), results_(r), variables_(b) {
    initialize();
  }

  void initialize() override;

  PerformancePayloadFromBinnedTFormula() {}
  ~PerformancePayloadFromBinnedTFormula() override { compiledFormulas_.clear(); }

  float getResult(PerformanceResult::ResultType,
                  const BinningPointByMap&) const override;  // gets from the full payload

  virtual bool isParametrizedInVariable(const BinningVariables::BinningVariablesType p) const {
    return (limitPos(p) != PerformancePayloadFromBinnedTFormula::InvalidPos);
  }

  bool isInPayload(PerformanceResult::ResultType, const BinningPointByMap&) const override;

  const std::vector<PhysicsTFormulaPayload>& formulaPayloads() const { return pls; }

  void printFormula(PerformanceResult::ResultType res, const BinningPointByMap&) const;

protected:
  virtual std::vector<BinningVariables::BinningVariablesType> myBinning() const { return variables_; }

  virtual int limitPos(const BinningVariables::BinningVariablesType b) const {
    std::vector<BinningVariables::BinningVariablesType>::const_iterator p;
    p = find(variables_.begin(), variables_.end(), b);
    if (p == variables_.end())
      return PerformancePayloadFromBinnedTFormula::InvalidPos;
    return ((p - variables_.begin()));
  }

  virtual int resultPos(PerformanceResult::ResultType r) const {
    std::vector<PerformanceResult::ResultType>::const_iterator p;
    p = find(results_.begin(), results_.end(), r);
    if (p == results_.end())
      return PerformancePayloadFromBinnedTFormula::InvalidPos;
    return ((p - results_.begin()));
  }

  bool isOk(const BinningPointByMap& p, unsigned int&) const;

  const std::shared_ptr<TFormula>& getFormula(PerformanceResult::ResultType, const BinningPointByMap&) const;

  //
  // now this is a vector, since we can have different rectangular regions in the same object
  //
  std::vector<PhysicsTFormulaPayload> pls;
  //
  // the variable mapping
  //
  std::vector<PerformanceResult::ResultType> results_;
  std::vector<BinningVariables::BinningVariablesType> variables_;

  //
  // the transient part; now a vector of vector; CHANGE CHECK!!!!!
  //

  // the compiled functions
  std::vector<std::vector<std::shared_ptr<TFormula> > > compiledFormulas_ COND_TRANSIENT;
  ;

  COND_SERIALIZABLE;
};

#endif
