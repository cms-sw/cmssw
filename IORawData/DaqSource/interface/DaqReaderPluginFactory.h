#ifndef DaqSource_DaqReaderPluginFactory_h
#define DaqSource_DaqReaderPluginFactory_h

/** \class DaqReaderPluginFactory
 *  Plugin factory for actual data sources for DaqSource.
 *
 *  $Date: 2007/03/26 15:51:06 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include "FWCore/PluginManager/interface/PluginFactory.h"
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

