#ifndef CommonTools_Utils_formulaEvaluatorBase_h
#define CommonTools_Utils_formulaEvaluatorBase_h
// -*- C++ -*-
//
// Package:     CommonTools/Utils
// Class  :     reco::formula::EvaluatorBase
// 
/**\class reco::formula::EvaluatorBase formulaEvaluatorBase.h "formulaEvaluatorBase.h"

 Description: Base class for formula evaluators

 Usage:
    Used as an internal detail on the reco::FormulaEvalutor class. 
    Base class for all objects used in the abstract evaluation tree where one node
    corresponds to one syntax element of the formula.

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 23 Sep 2015 16:26:00 GMT
//

// system include files

// user include files

// forward declarations

namespace reco {
  namespace formula {
    class EvaluatorBase
    {
      
    public:
      enum class Precidence { 
        kPlusMinus = 1,
          kMultDiv = 2,
          kPower = 3,
          kFunction = 4, //default
          kParenthesis = 5,
          kUnaryMinusOperator = 6
          };

      EvaluatorBase();
      EvaluatorBase(Precidence);
      virtual ~EvaluatorBase();
      
      // ---------- const member functions ---------------------
      //inputs are considered to be 'arrays' which have already been validated to 
      // be of the appropriate length
      virtual double evaluate(double const* iVariables, double const* iParameters) const = 0;

      unsigned int precidence() const { return m_precidence; }
      void setPrecidenceToParenthesis() { m_precidence = static_cast<unsigned int>(Precidence::kParenthesis); }

    private:
      EvaluatorBase(const EvaluatorBase&) = delete; 
      
      const EvaluatorBase& operator=(const EvaluatorBase&) = delete;
      
      // ---------- member data --------------------------------
      unsigned int m_precidence;
    };
  }
}


#endif
