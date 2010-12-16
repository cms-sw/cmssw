// $Id: ErrorStreamConfigurationInfo.h,v 1.4 2009/09/11 21:07:06 elmer Exp $
/// @file: ErrorStreamConfigurationInfo.h 

#ifndef StorageManager_ErrorStreamConfigurationInfo_h
#define StorageManager_ErrorStreamConfigurationInfo_h

#include "EventFilter/StorageManager/interface/StreamID.h"

#include <boost/shared_ptr.hpp>

#include <string>
#include <vector>
#include <iosfwd>

namespace stor
{

  /**
     Configuration information for the error stream

     $Author: elmer $
     $Revision: 1.4 $
     $Date: 2009/09/11 21:07:06 $
  */

  class ErrorStreamConfigurationInfo
  {

  public:

    // Constructor:
    ErrorStreamConfigurationInfo( const std::string& streamLabel,
				  int maxFileSizeMB ):
      _streamLabel( streamLabel ),
      _maxFileSizeMB( maxFileSizeMB ),
      _streamId(0)
    {}

    // Destructor:
    ~ErrorStreamConfigurationInfo() {}

    // Accessors:
    const std::string& streamLabel() const { return _streamLabel; }
    const int maxFileSizeMB() const { return _maxFileSizeMB; }
    StreamID streamId() const { return _streamId; }

    // Comparison:
    bool operator<(const ErrorStreamConfigurationInfo&) const;

    // Set stream Id:
    void setStreamId( StreamID sid ) { _streamId = sid; }

    // Output:
    friend std::ostream& operator <<
      ( std::ostream&, const ErrorStreamConfigurationInfo& );

  private:

    std::string _streamLabel;
    int _maxFileSizeMB;
    StreamID _streamId;

  };

  typedef std::vector<ErrorStreamConfigurationInfo> ErrStrConfigList;
  typedef boost::shared_ptr<ErrStrConfigList> ErrStrConfigListPtr;

  std::ostream& operator << ( std::ostream&, const ErrorStreamConfigurationInfo& );
}

#endif // StorageManager_ErrorStreamConfigurationInfo_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

