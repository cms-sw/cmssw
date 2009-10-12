// $Id: Notifier.h,v 1.7 2009/07/20 13:06:10 mommsen Exp $
/// @file: Notifier.h 

#ifndef StorageManager_Notifier_h
#define StorageManager_Notifier_h

#include <string>

#include "xdaq/Application.h"
#include "xdaq/exception/Exception.h"

namespace stor
{

  /**
    Interface class for handling RCMS notifier
    
    $Author: mommsen $
    $Revision: 1.7 $
    $Date: 2009/07/20 13:06:10 $
  */

  class Notifier
  {

  public:

    /**
       Constructor
    */
    Notifier() {}

    /**
       Destructor
    */
    virtual ~Notifier() {};

    /**
       Report new state to RCMS
    */
    virtual void reportNewState( const std::string& stateName ) = 0;

  };

}

#endif // StorageManager_Notifier_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
