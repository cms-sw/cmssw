#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGSpikeThresholdfromFile.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalTPGSpikeThresholdfromFile> ExTestEcalTPGSpikeThresholdfromFile;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGSpikeThresholdfromFile);
