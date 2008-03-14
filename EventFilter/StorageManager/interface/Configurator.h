#if !defined(STOR_CONFIGURATOR_H)
#define STOR_CONFIGURATOR_H

// Created by Markus Klute on 2007 Jan 29.
// $Id: Configurator.h,v 1.1 2007/02/05 11:19:56 klute Exp $

// singleton class 
// holds a boost::shared_ptr of configuration parameter

// should be moved to EventFilter/StorageManager

#include <EventFilter/StorageManager/interface/Parameter.h>
#include <boost/shared_ptr.hpp>
#include "boost/thread/thread.hpp"
#include <string>

namespace stor 
{
  class Configurator
    {
    public:
      static boost::shared_ptr<Configurator> instance();
      boost::shared_ptr<Parameter> getParameter();

    private:
      Configurator();
      boost::shared_ptr<Parameter> param_;

      // manage the singleton instance
      static boost::shared_ptr<Configurator> theInstance_;
      static boost::mutex singletonLock_;
    }; 
}

#endif
