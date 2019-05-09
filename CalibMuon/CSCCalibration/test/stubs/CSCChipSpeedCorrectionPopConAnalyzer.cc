#include "CSCDBChipSpeedCorrectionHandler.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::CSCDBChipSpeedCorrectionImpl> CSCDBChipSpeedCorrectionPopConAnalyzer;

DEFINE_FWK_MODULE(CSCDBChipSpeedCorrectionPopConAnalyzer);
