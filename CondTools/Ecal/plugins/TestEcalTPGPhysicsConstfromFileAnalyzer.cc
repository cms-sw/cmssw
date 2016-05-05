#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGPhysicsConstfromFile.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalTPGPhysicsConstfromFile> ExTestEcalTPGPhysicsConstfromFile;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGPhysicsConstfromFile);
