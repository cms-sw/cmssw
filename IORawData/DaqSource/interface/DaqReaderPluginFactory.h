#ifndef DaqSource_DaqReaderPluginFactory_h
#define DaqSource_DaqReaderPluginFactory_h

/** \class DaqReaderPluginFactory
 *  Plugin factory for actual data sources for DaqSource.
 *
 *  $Date: 2005/10/06 18:23:47 $
 *  $Revision: 1.2 $
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
class DaqReaderPluginFactoryU :public seal::PluginFactory<DaqBaseReader *()>{
 public:
  DaqReaderPluginFactoryU();
  static DaqReaderPluginFactoryU* get (void);
 private:
  static DaqReaderPluginFactoryU s_instance;
};
#endif

