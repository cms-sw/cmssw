#ifndef CommonTools_Utils_formulaVariableEvaluator_h
#define CommonTools_Utils_formulaVariableEvaluator_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     reco::formula::VariableEvaluator
// 
/**\class reco::formula::VariableEvaluator formulaVariableEvaluator.h "formulaVariableEvaluator.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 23 Sep 2015 18:06:27 GMT
//

// system include files

// user include files
#include "formulaEvaluatorBase.h"

// forward declarations

namespace reco {
  namespace formula {
    class VariableEvaluator : public EvaluatorBase
    {
      
    public:
      explicit VariableEvaluator(unsigned int iIndex) : m_index(iIndex) {}
      

      // ---------- const member functions ---------------------
      double evaluate(double const* iVariables, double const* iParameters) const override final;
      
    private:
      VariableEvaluator(const VariableEvaluator&) = delete;
      
      const VariableEvaluator& operator=(const VariableEvaluator&) = delete;
      
      // ---------- member data --------------------------------
      unsigned int m_index;
    };
  }
}


#endif
