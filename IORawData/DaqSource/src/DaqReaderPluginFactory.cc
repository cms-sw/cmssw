/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */


#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>

DaqReaderPluginFactory DaqReaderPluginFactory::s_instance;

DaqReaderPluginFactory::DaqReaderPluginFactory () : 
  seal::PluginFactory<DaqBaseReader *(const edm::ParameterSet&)>("DaqReaderPluginFactory"){}

DaqReaderPluginFactory* DaqReaderPluginFactory::get (){
  return &s_instance; 
}

