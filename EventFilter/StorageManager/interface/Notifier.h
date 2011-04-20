// $Id: Notifier.h,v 1.8.12.1 2011/03/07 11:33:04 mommsen Exp $
/// @file: Notifier.h 

#ifndef EventFilter_StorageManager_Notifier_h
#define EventFilter_StorageManager_Notifier_h

#include <string>

#include "xdaq/Application.h"
#include "xdaq/exception/Exception.h"

namespace stor
{

  /**
    Interface class for handling RCMS notifier
    
    $Author: mommsen $
    $Revision: 1.8.12.1 $
    $Date: 2011/03/07 11:33:04 $
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

} // namespace stor

#endif // EventFilter_StorageManager_Notifier_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
