#ifndef FWCore_Framework_edmodule_mightGet_config_h
#define FWCore_Framework_edmodule_mightGet_config_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     edmodule_mightGet_config
// 
/**\class edmodule_mightGet_config edmodule_mightGet_config.h FWCore/Framework/interface/edmodule_mightGet_config.h

 Description: Injects 'mightGet' PSet into all ed module configurations

 Usage:

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb  2 14:26:40 CST 2012
// $Id: edmodule_mightGet_config.h,v 1.1 2012/02/09 22:12:56 chrjones Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
  class ConfigurationDescriptions;
  void edmodule_mightGet_config(ConfigurationDescriptions&);
}
#endif
