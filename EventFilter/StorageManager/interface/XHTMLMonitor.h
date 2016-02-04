// $Id: XHTMLMonitor.h,v 1.5.16.1 2011/03/07 11:33:04 mommsen Exp $
/// @file: XHTMLMonitor.h 

#ifndef EventFilter_StorageManager_XHTMLMonitor_h
#define EventFilter_StorageManager_XHTMLMonitor_h

#include "boost/thread/mutex.hpp"

namespace stor {

  /**
    Controls the use of XHTMLMaker (xerces is not thread-safe)

    $Author: mommsen $
    $Revision: 1.5.16.1 $
    $Date: 2011/03/07 11:33:04 $
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
