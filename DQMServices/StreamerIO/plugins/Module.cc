#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/StreamerIO/plugins/DQMStreamerReader.h"

typedef edm::DQMStreamerReader DQMStreamerReader;
DEFINE_FWK_INPUT_SOURCE(DQMStreamerReader);
