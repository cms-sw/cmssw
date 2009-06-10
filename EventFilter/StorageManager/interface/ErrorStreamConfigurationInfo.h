// $Id$

#ifndef ERRORSTREAMCONFIGURATIONINFO_H
#define ERRORSTREAMCONFIGURATIONINFO_H

#include "EventFilter/StorageManager/interface/StreamID.h"

#include <boost/shared_ptr.hpp>

#include <string>
#include <vector>
#include <iostream>

namespace stor
{

  class ErrorStreamConfigurationInfo
  {

  public:

    typedef std::vector<std::string> FilterList;

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
}

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

