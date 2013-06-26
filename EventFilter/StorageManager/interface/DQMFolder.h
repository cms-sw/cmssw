// $Id: DQMFolder.h,v 1.3 2013/04/22 16:19:36 wmtan Exp $
/// @file: DQMFolder.h 

#ifndef EventFilter_StorageManager_DQMFolder_h
#define EventFilter_StorageManager_DQMFolder_h

#include <string>
#include <vector>
#include <map>

#include "TObject.h"


namespace stor 
{
  
  /**
   * A single DQM folder holding several histograms
   *
   * $Author: wmtan $
   * $Revision: 1.3 $
   * $Date: 2013/04/22 16:19:36 $
   */

  class DQMFolder
  {
  public:
    DQMFolder();
    ~DQMFolder();
    void addObjects(const std::vector<TObject *>&);
    void fillObjectVector(std::vector<TObject*>&) const;

  private:

    static std::string getSafeMEName(TObject *object);

    typedef std::map<std::string, TObject*> DQMObjectsMap;
    DQMObjectsMap dqmObjects_;
  }; 

} // namespace stor

#endif // EventFilter_StorageManager_DQMFolder_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
