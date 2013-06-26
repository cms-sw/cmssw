#include "FWCore/Framework/interface/MakerMacros.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutGenerator.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalOmdsCalibrations.h"

#include "FWCore/Framework/interface/SourceFactory.h"

DEFINE_FWK_MODULE(HcalLutGenerator);

DEFINE_FWK_EVENTSETUP_SOURCE(HcalOmdsCalibrations);
