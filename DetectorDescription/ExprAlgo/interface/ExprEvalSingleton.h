#ifndef ExprEvalSingleton_h
#define ExprEvalSingleton_h

#include "DetectorDescription/Base/interface/Singleton.h"
#include "DetectorDescription/ExprAlgo/interface/ClhepEvaluator.h"
// full name compatible with the header file name
typedef DDI::Singleton<ClhepEvaluator> ExprEvalSingleton;

// short name
typedef DDI::Singleton<ClhepEvaluator> ExprEval;


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
