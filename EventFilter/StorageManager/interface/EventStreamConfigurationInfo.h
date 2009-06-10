// $Id$

#ifndef EVENTSTREAMCONFIGURATIONINFO_H
#define EVENTSTREAMCONFIGURATIONINFO_H

#include "EventFilter/StorageManager/interface/StreamID.h"

#include <boost/shared_ptr.hpp>

#include <string>
#include <vector>
#include <iostream>

namespace stor
{

  class EventStreamConfigurationInfo
  {

  public:

    typedef std::vector<std::string> FilterList;

    // Constructor:
    EventStreamConfigurationInfo( const std::string& streamLabel,
				  const int maxFileSizeMB,
				  const FilterList& selEvents,
				  const std::string& outputModuleLabel,
				  bool useCompression,
				  unsigned int compressionLevel,
				  unsigned int maxEventSize ):
      _streamLabel( streamLabel ),
      _maxFileSizeMB( maxFileSizeMB ),
      _selEvents( selEvents ),
      _outputModuleLabel( outputModuleLabel ),
      _useCompression( useCompression ),
      _compressionLevel( compressionLevel ),
      _maxEventSize( maxEventSize ),
      _streamId(0)
    {}

    // Destructor:
    ~EventStreamConfigurationInfo() {}

    // Accessors:
    const std::string& streamLabel() const { return _streamLabel; }
    const int maxFileSizeMB() const { return _maxFileSizeMB; }
    const FilterList& selEvents() const { return _selEvents; }
    const std::string& outputModuleLabel() const { return _outputModuleLabel; }
    bool useCompression() const { return _useCompression; }
    unsigned int compressionLevel() const { return _compressionLevel; }
    unsigned int maxEventSize() const { return _maxEventSize; }
    StreamID streamId() const { return _streamId; }

    // Set stream Id:
    void setStreamId( StreamID sid ) { _streamId = sid; }

    // Output:
    friend std::ostream& operator <<
      ( std::ostream&, const EventStreamConfigurationInfo& );

  private:

    std::string _streamLabel;
    int _maxFileSizeMB;
    FilterList _selEvents;
    std::string _outputModuleLabel;
    bool _useCompression;
    unsigned int _compressionLevel;
    unsigned int _maxEventSize;
    StreamID _streamId;

  };

  typedef std::vector<EventStreamConfigurationInfo> EvtStrConfigList;
  typedef boost::shared_ptr<EvtStrConfigList> EvtStrConfigListPtr;

}

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
