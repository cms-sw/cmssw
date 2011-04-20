// $Id: XHTMLMonitor.h,v 1.5.14.3 2011/02/28 17:56:15 mommsen Exp $
/// @file: XHTMLMonitor.h 

#ifndef EventFilter_StorageManager_XHTMLMonitor_h
#define EventFilter_StorageManager_XHTMLMonitor_h

#include "boost/thread/mutex.hpp"

namespace stor {

  /**
    Controls the use of XHTMLMaker (xerces is not thread-safe)

    $Author: mommsen $
    $Revision: 1.5.14.3 $
    $Date: 2011/02/28 17:56:15 $
  */
  
  class XHTMLMonitor
  {
    
  public:
    
    /**
      Constructor
    */
    XHTMLMonitor();

    /**
      Destructor
    */
    ~XHTMLMonitor();

  private:

    static boost::mutex xhtmlMakerMutex_;

  };

} // namespace stor

#endif // EventFilter_StorageManager_XHTMLMonitor_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
