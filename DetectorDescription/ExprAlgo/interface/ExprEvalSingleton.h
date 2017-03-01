#ifndef ExprEvalSingleton_h
#define ExprEvalSingleton_h

#include "DetectorDescription/Base/interface/Singleton.h"
#include "DetectorDescription/ExprAlgo/interface/ClhepEvaluator.h"

class ClhepEvaluator;

typedef DDI::Singleton<ClhepEvaluator> ExprEvalSingleton;

#endif
