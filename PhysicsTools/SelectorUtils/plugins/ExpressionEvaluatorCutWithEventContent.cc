#include "CommonTools/Utils/interface/ExpressionEvaluator.h"
#include "CommonTools/Utils/interface/ExpressionEvaluatorTemplates.h"

#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "ExpressionEvaluatorCutT.h"

typedef ExpressionEvaluatorCutT<CutApplicatorWithEventContentBase> ExpressionEvaluatorCutWithEventContent;

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  ExpressionEvaluatorCutWithEventContent,
                  "ExpressionEvaluatorCutWithEventContent");
