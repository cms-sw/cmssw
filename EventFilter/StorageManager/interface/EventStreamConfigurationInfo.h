// $Id: EventStreamConfigurationInfo.h,v 1.5 2009/11/24 16:38:24 mommsen Exp $
/// @file: EventStreamConfigurationInfo.h

#ifndef StorageManager_EventStreamConfigurationInfo_h
#define StorageManager_EventStreamConfigurationInfo_h

#include "EventFilter/StorageManager/interface/StreamID.h"

#include <boost/shared_ptr.hpp>

#include <string>
#include <vector>
#include <iosfwd>

namespace stor
{

  /**
     Configuration information for the event stream

     $Author: mommsen $
     $Revision: 1.5 $
     $Date: 2009/11/24 16:38:24 $
  */

  class EventStreamConfigurationInfo
  {

  public:

    typedef std::vector<std::string> FilterList;

    // Constructor:
    EventStreamConfigurationInfo( const std::string& streamLabel,
                                  const int maxFileSizeMB,
                                  const std::string& newSelEvents,
                                  const FilterList& selEvents,
                                  const std::string& outputModuleLabel,
                                  bool useCompression,
                                  unsigned int compressionLevel,
                                  unsigned int maxEventSize,
                                  double fractionToDisk ):
      _streamLabel( streamLabel ),
      _maxFileSizeMB( maxFileSizeMB ),
      _newSelEvents( newSelEvents ),
      _selEvents( selEvents ),
      _outputModuleLabel( outputModuleLabel ),
      _useCompression( useCompression ),
      _compressionLevel( compressionLevel ),
      _maxEventSize( maxEventSize ),
      _fractionToDisk( fractionToDisk ),
      _streamId(0)
    {}

    // Destructor:
    ~EventStreamConfigurationInfo() {}

    // Accessors:
    const std::string& streamLabel() const { return _streamLabel; }
    const int maxFileSizeMB() const { return _maxFileSizeMB; }
    const std::string& newSelEvents() const { return _newSelEvents; }
    const FilterList& selEvents() const { return _selEvents; }
    const std::string& outputModuleLabel() const { return _outputModuleLabel; }
    bool useCompression() const { return _useCompression; }
    unsigned int compressionLevel() const { return _compressionLevel; }
    unsigned int maxEventSize() const { return _maxEventSize; }
    double fractionToDisk() const { return _fractionToDisk; }
    StreamID streamId() const { return _streamId; }

    // Set stream Id:
    void setStreamId( StreamID sid ) { _streamId = sid; }

    // Output:
    friend std::ostream& operator <<
      ( std::ostream&, const EventStreamConfigurationInfo& );

  private:

    std::string _streamLabel;
    int _maxFileSizeMB;
    std::string _newSelEvents;
    FilterList _selEvents;
    std::string _outputModuleLabel;
    bool _useCompression;
    unsigned int _compressionLevel;
    unsigned int _maxEventSize;
    double _fractionToDisk;
    StreamID _streamId;

  };

  typedef std::vector<EventStreamConfigurationInfo> EvtStrConfigList;
  typedef boost::shared_ptr<EvtStrConfigList> EvtStrConfigListPtr;

  std::ostream& operator << ( std::ostream&, const EventStreamConfigurationInfo& );
}

#endif // StorageManager_EventStreamConfigurationInfo_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
