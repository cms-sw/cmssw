#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGFineGrainStripfromFile.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalTPGFineGrainStripfromFile> ExTestEcalTPGFineGrainStripfromFile;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGFineGrainStripfromFile);
