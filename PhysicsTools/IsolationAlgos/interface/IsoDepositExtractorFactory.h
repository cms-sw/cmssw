#ifndef IsolationAlgos_IsoDepositExtractorFactory_H
#define IsolationAlgos_IsoDepositExtractorFactory_H


namespace edm {class ParameterSet;}
namespace reco { namespace isodeposit { class IsoDepositExtractor; }}
#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory< reco::isodeposit::IsoDepositExtractor* (const edm::ParameterSet&) > IsoDepositExtractorFactory;
#endif
