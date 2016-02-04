#ifndef ExprEvalSingleton_h
#define ExprEvalSingleton_h

#include "DetectorDescription/Base/interface/Singleton.h"

//////////////////////////////////////////////////////////////////////////////
// Choose the Evaluator here:
//  The concrete evaluator must be a subclass of class ExprEvalInterface, which
//  defines the interface (which can be used to write Evaluator-implementation-
//  independent code !
   #include "DetectorDescription/ExprAlgo/interface/ClhepEvaluator.h"
   typedef ClhepEvaluator UseThisEvaluator;
//////////////////////////////////////////////////////////////////////////////


// full name compatible with the header file name
typedef DDI::Singleton<UseThisEvaluator> ExprEvalSingleton;

// short name
typedef DDI::Singleton<UseThisEvaluator> ExprEval;


//////////////////////////////////////////////////////////////////////////////
// usage:
//
// DDExprEvaluator & eval_sgtn = DDExpreEvalSingleton::instance();
// eval_sgtn.set(...) ; and so on ...
// 
// the same singleton can be addressed using:
// DDExprEval::instance()
//
////////////////////////////////////////////////////////////////////////////// 
#endif
