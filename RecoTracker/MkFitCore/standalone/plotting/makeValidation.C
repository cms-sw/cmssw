#include "plotting/StackValidation.cpp+"

void makeValidation(const TString& label = "",
                    const TString& extra = "",
                    const Bool_t cmsswComp = false,
                    const TString& suite = "forPR") {
  StackValidation Stacks(label, extra, cmsswComp, suite);
  Stacks.MakeValidationStacks();
}
