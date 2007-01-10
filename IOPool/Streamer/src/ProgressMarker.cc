// Created by Markus Klute on 2007 Jan 09.
// $Id:$

#include <IOPool/Streamer/interface/ProgressMarker.h>

using stor::ProgressMarker;
using std::string;

ProgressMarker *inst = 0;

ProgressMarker::ProgressMarker()
{
  reading_     = false;
  writing_     = false;
  processing_  = false;
}


ProgressMarker *ProgressMarker::instance()
{ // not thread save
  if (inst == 0) inst = new ProgressMarker();
  return inst;
}


void ProgressMarker::instance(ProgressMarker *anInstance)
{ // not thread save
  delete inst;
  inst = anInstance;
}


string ProgressMarker::status()
{
  if (processing_) return "Process";
  if (reading_)    return "Input";
  if (writing_)    return "Output";
  return "Idle";
}
