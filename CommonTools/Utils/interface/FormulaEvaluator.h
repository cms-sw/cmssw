#ifndef CommonTools_Utils_FormulaEvaluator_h
#define CommonTools_Utils_FormulaEvaluator_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     FormulaEvaluator
// 
/**\class FormulaEvaluator FormulaEvaluator.h "CommonTools/Utils/interface/FormulaEvaluator.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 23 Sep 2015 21:12:11 GMT
//

// system include files
#include <array>
#include <vector>
#include <memory>

// user include files

// forward declarations
namespace reco {
  namespace formula {
    struct ArrayAdaptor {
    ArrayAdaptor(double const* iStart, size_t iSize):
      m_start(iStart), m_size(iSize) {}
      size_t size() const { return m_size; }
      bool empty() const {return m_size == 0; }
      double const* start() const { return m_start; }
    private:
      double const* m_start;
      size_t m_size;
    };
    inline double const* startingAddress(ArrayAdaptor const& iV) {
      if(iV.empty()) {
        return nullptr;
      }
      return iV.start();
    }

    class EvaluatorBase;
    inline double const* startingAddress(std::vector<double> const& iV) {
      if(iV.empty()) {
        return nullptr;
      }
      return &iV[0];
    }

    template<size_t t>
      inline double const* startingAddress(std::array<double,t> const& iV) {
      if(iV.empty()) {
        return nullptr;
      }
      return &iV[0];
    }

  }

  class FormulaEvaluator
  {
    
  public:
    explicit FormulaEvaluator(std::string const& iFormula);

    // ---------- const member functions ---------------------
    template<typename V, typename P>
      double evaluate( V const& iVariables, P const& iParameters) const {
      if (m_nVariables > iVariables.size()) {
        throwWrongNumberOfVariables(iVariables.size());
      }
      if (m_nParameters > iParameters.size()) {
        throwWrongNumberOfParameters(iParameters.size());
      }
      return evaluate( formula::startingAddress(iVariables),
                       formula::startingAddress(iParameters));
    }
    
    unsigned int numberOfParameters() const { return m_nParameters; }
    unsigned int numberOfVariables() const { return m_nVariables; }
    
  private:
    double evaluate(double const* iVariables, double const* iParameters) const;

    void throwWrongNumberOfVariables(size_t) const ;
    void throwWrongNumberOfParameters(size_t) const;

    std::shared_ptr<formula::EvaluatorBase const> m_evaluator; 
    unsigned int m_nVariables = 0;
    unsigned int m_nParameters = 0;

};
}

#endif
