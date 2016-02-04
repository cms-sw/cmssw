#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CSCDBChipSpeedCorrectionHandler.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBChipSpeedCorrectionImpl> CSCDBChipSpeedCorrectionPopConAnalyzer;

DEFINE_FWK_MODULE(CSCDBChipSpeedCorrectionPopConAnalyzer);
