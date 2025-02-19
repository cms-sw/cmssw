// $Id: ErrorStreamConfigurationInfo.h,v 1.6 2011/03/07 15:31:31 mommsen Exp $
/// @file: ErrorStreamConfigurationInfo.h 

#ifndef EventFilter_StorageManager_ErrorStreamConfigurationInfo_h
#define EventFilter_StorageManager_ErrorStreamConfigurationInfo_h

#include "EventFilter/StorageManager/interface/StreamID.h"

#include <boost/shared_ptr.hpp>

#include <string>
#include <vector>
#include <iosfwd>

namespace stor
{

  /**
     Configuration information for the error stream

     $Author: mommsen $
     $Revision: 1.6 $
     $Date: 2011/03/07 15:31:31 $
  */

  class ErrorStreamConfigurationInfo
  {

  public:

    // Constructor:
    ErrorStreamConfigurationInfo( const std::string& streamLabel,
				  int maxFileSizeMB ):
      streamLabel_( streamLabel ),
      maxFileSizeMB_( maxFileSizeMB ),
      streamId_(0)
    {}

    // Destructor:
    ~ErrorStreamConfigurationInfo() {}

    // Accessors:
    const std::string& streamLabel() const { return streamLabel_; }
    const int maxFileSizeMB() const { return maxFileSizeMB_; }
    StreamID streamId() const { return streamId_; }

    // Comparison:
    bool operator<(const ErrorStreamConfigurationInfo&) const;

    // Set stream Id:
    void setStreamId( StreamID sid ) { streamId_ = sid; }

    // Output:
    friend std::ostream& operator <<
      ( std::ostream&, const ErrorStreamConfigurationInfo& );

  private:

    std::string streamLabel_;
    int maxFileSizeMB_;
    StreamID streamId_;

  };

  typedef std::vector<ErrorStreamConfigurationInfo> ErrStrConfigList;
  typedef boost::shared_ptr<ErrStrConfigList> ErrStrConfigListPtr;

  std::ostream& operator << ( std::ostream&, const ErrorStreamConfigurationInfo& );

} // namespace stor

#endif // EventFilter_StorageManager_ErrorStreamConfigurationInfo_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

