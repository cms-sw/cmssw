// Created by Markus Klute on 2007 Jan 29.
// $Id:$

// should be moved to EventFilter/StorageManager

#include <IOPool/Streamer/interface/Configurator.h>

using stor::Configurator;
using stor::Parameter;
using boost::shared_ptr;
using std::string;

Configurator *insta = 0;

Configurator::Configurator()
{
  param = shared_ptr<Parameter>(new Parameter());
}


Configurator *Configurator::instance()
{ // not thread save
  if (insta == 0) insta = new Configurator();
  return insta;
}


void Configurator::instance(Configurator *anInstance)
{ // not thread save
  delete insta;
  insta = anInstance;
}


shared_ptr<Parameter> &Configurator::getParameter() 
{ 
  return param;
}


