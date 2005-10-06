#ifndef DaqSource_DaqReaderPluginFactory_h
#define DaqSource_DaqReaderPluginFactory_h

/** \class DaqReaderPluginFactory
 *  Plugin factory for actual data sources for DaqSource.
 *
 *  $Date: 2005/09/30 08:17:48 $
 *  $Revision: 1.1 $
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

