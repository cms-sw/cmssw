#include "CondCore/CondDB/interface/Types.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include <boost/python.hpp>

using namespace boost::python;

namespace {

  void CMSSWInit(){
    edmplugin::PluginManager::Config config;
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }

}

namespace cond {

  Iov_t makeIov( cond::Time_t since, const Hash& payloadId ){
    Iov_t ret;
    ret.since = since;
    ret.payloadId = payloadId;
    return ret;
  }

}

BOOST_PYTHON_MODULE(pluginCondDBV2PyInterface) {

  def("CMSSWInit",&CMSSWInit);

  def("makeIov",&cond::makeIov);

  class_<cond::Iov_t>("Iov", init<>()) 
    .def_readwrite("since", &cond::Iov_t::since ) 
    .def_readwrite("payloadId", &cond::Iov_t::payloadId )
    ;

}
