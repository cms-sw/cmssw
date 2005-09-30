#ifndef DaqReaderPluginFactory_H
#define DaqReaderPluginFactory_H

/** \class DaqReaderPluginFactory
 *  Plugin factory for actual data sources for DaqSource.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include <PluginManager/PluginFactory.h>
#include <IORawData/DaqSource/interface/DaqBaseReader.h>

namespace edm {class ParameterSet;}

class DaqReaderPluginFactory : public seal::PluginFactory<DaqBaseReader *(const edm::ParameterSet&)>{
 public:
  DaqReaderPluginFactory();
  static DaqReaderPluginFactory* get (void);
 private:
  static DaqReaderPluginFactory s_instance;
};
#endif

