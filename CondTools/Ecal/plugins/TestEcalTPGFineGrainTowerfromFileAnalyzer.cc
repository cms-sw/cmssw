#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGFineGrainTowerfromFile.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalTPGFineGrainTowerfromFile> ExTestEcalTPGFineGrainTowerfromFile;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGFineGrainTowerfromFile);
