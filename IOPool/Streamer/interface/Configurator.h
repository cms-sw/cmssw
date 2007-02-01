#if !defined(STOR_CONFIGURATOR_H)
#define STOR_CONFIGURATOR_H

// Created by Markus Klute on 2007 Jan 29.
// $Id:$

// singleton class 
// holds a boost::shared_ptr of configuration parameter

// should be moved to EventFilter/StorageManager

#include <IOPool/Streamer/interface/Parameter.h>
#include <boost/shared_ptr.hpp>
#include <string>

namespace stor 
{
  class Configurator
    {
    public:
      static Configurator *instance();
      static void instance(Configurator *);    
      boost::shared_ptr<Parameter> &getParameter();

    private:
      Configurator();
      boost::shared_ptr<Parameter> param;
    }; 
}

#endif

