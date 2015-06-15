#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "ExpressionEvaluatorCutT.h"

typedef ExpressionEvaluatorCutT<CutApplicatorBase> ExpressionEvaluatorCut;

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  ExpressionEvaluatorCut,
                  "ExpressionEvaluatorCut");
