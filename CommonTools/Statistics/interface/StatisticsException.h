#ifndef StatisticsException_H
#define StatisticsException_H

#include "FWCore/Utilities/interface/Exception.h"

  /** 
   *  A class that wraps cms::Exception by deriving from it.
   */

class StatisticsException : public cms::Exception
{
public:
  StatisticsException( const char * reason ) : cms::Exception ( reason ) {};
  StatisticsException( const StatisticsException &ex ) : cms::Exception ( ex ) {};
};

#endif
