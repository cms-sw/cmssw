#ifndef DaqSource_DaqReaderPluginFactory_h
#define DaqSource_DaqReaderPluginFactory_h

/** \class DaqReaderPluginFactory
 *  Plugin factory for actual data sources for DaqSource.
 *
 *  \author N. Amapane - CERN
 */

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include <IORawData/DaqSource/interface/DaqBaseReader.h>

namespace edm {class ParameterSet;}

typedef edmplugin::PluginFactory<DaqBaseReader *(const edm::ParameterSet &)> DaqReaderPluginFactory;
typedef edmplugin::PluginFactory<DaqBaseReader *()> DaqReaderPluginFactoryU;
#endif

