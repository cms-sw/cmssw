#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondTools/RunInfo/interface/PopConBTransitionSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::PopConBTransitionSourceHandler<Alignments> >
    EcalAlignmentPopConBTransitionAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(EcalAlignmentPopConBTransitionAnalyzer);
