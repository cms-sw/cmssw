#ifndef Clustering1DException_H
#define Clustering1DException_H

#include "FWCore/Utilities/interface/Exception.h"

  /** 
   *  A class that wraps cms::Exception by deriving from it.
   */

class Clustering1DException : public cms::Exception
{
public:
  Clustering1DException( const char * reason ) : cms::Exception ( reason ) {};
  Clustering1DException( const Clustering1DException &ex ) : cms::Exception ( ex ) {};
};

#endif
