#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGPedfromFile.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalTPGPedfromFile> ExTestEcalTPGPedfromFile;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGPedfromFile);
