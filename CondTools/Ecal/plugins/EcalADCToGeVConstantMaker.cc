#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalADCToGeVTagWriter.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalADCToGeVTagWriter> EcalADCToGeVTagWriter;

//define this as a plug-in
DEFINE_FWK_MODULE(EcalADCToGeVTagWriter);
