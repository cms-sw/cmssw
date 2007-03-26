/** \file
 *
 *  $Date: 2005/10/06 18:23:47 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */


#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>

DaqReaderPluginFactory DaqReaderPluginFactory::s_instance;
DaqReaderPluginFactoryU DaqReaderPluginFactoryU::s_instance;


DaqReaderPluginFactory::DaqReaderPluginFactory () : 
  seal::PluginFactory<DaqBaseReader *(const edm::ParameterSet&)>("DaqReaderPluginFactory"){}
DaqReaderPluginFactoryU::DaqReaderPluginFactoryU () : 
seal::PluginFactory<DaqBaseReader *()>("DaqReaderPluginFactoryU"){}

DaqReaderPluginFactory* DaqReaderPluginFactory::get (){
  return &s_instance; 
}
DaqReaderPluginFactoryU* DaqReaderPluginFactoryU::get (){
  return &s_instance; 
}

