#include "TClass.h"

#include <string>
#include <vector>

// Begin AUTO code, some classes commented out.

std::vector<std::string> classes = {
    // "mkfit::IterationConfig",
    "mkfit::IterationLayerConfig",
    "mkfit::IterationParams",
    // "mkfit::IterationSeedPartition",
    "mkfit::IterationConfig",
    "mkfit::IterationsInfo"};

// End AUTO code.

/*
    1. When running for the first time, after changing of classes:
       Review extracto.pl
       Run: ./extracto.pl ../SteeringParams.h
       Cut-n-paste code fragments above and into Config.LinkDef.h

    2. To run:
         # setup root environment
         make
         root.exe dump_vars.C
       Then cut-n-paste NLOHMANN defines into SteeringParams.cc 
*/

void dump_vars() {
  gSystem->Load("libConfigDict.so");

  for (auto &cls : classes) {
    printf("NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(%s,\n", cls.c_str());

    TClass *tc = TClass::GetClass(cls.c_str());
    TList *ml = tc->GetListOfDataMembers();
    TIter it(ml);
    TDataMember *dm = (TDataMember *)it.Next();
    while (dm) {
      // dm->GetTypeName(), dm->GetFullTypeName(), dm->GetTrueTypeName(),
      printf("  /* %s */   %s", dm->GetTypeName(), dm->GetName());
      dm = (TDataMember *)it.Next();
      if (dm)
        printf(",");
      printf("\n");
    }
    printf(")\n\n");
  }
}
