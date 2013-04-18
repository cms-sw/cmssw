// $Id: EventStreamConfigurationInfo.h,v 1.10.4.1 2011/03/07 11:33:04 mommsen Exp $
/// @file: EventStreamConfigurationInfo.h

#ifndef EventFilter_StorageManager_EventStreamConfigurationInfo_h
#define EventFilter_StorageManager_EventStreamConfigurationInfo_h

#include "IOPool/Streamer/interface/HLTInfo.h"
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
     $Revision: 1.10.4.1 $
     $Date: 2011/03/07 11:33:04 $
  */

  class EventStreamConfigurationInfo
  {

  public:

    // Constructor:
    EventStreamConfigurationInfo( const std::string& streamLabel,
                                  const int maxFileSizeMB,
                                  const std::string& triggerSelection,
                                  const Strings& eventSelection,
                                  const std::string& outputModuleLabel,
                                  double fractionToDisk ):
      streamLabel_( streamLabel ),
      maxFileSizeMB_( maxFileSizeMB ),
      triggerSelection_( triggerSelection ),
      eventSelection_( eventSelection ),
      outputModuleLabel_( outputModuleLabel ),
      fractionToDisk_( fractionToDisk ),
      streamId_(0)
    {}

    // Destructor:
    ~EventStreamConfigurationInfo() {}

    // Accessors:
    const std::string& streamLabel() const { return streamLabel_; }
    const int maxFileSizeMB() const { return maxFileSizeMB_; }
    const std::string& triggerSelection() const { return triggerSelection_; }
    const Strings& eventSelection() const { return eventSelection_; }
    const std::string& outputModuleLabel() const { return outputModuleLabel_; }
    double fractionToDisk() const { return fractionToDisk_; }
    StreamID streamId() const { return streamId_; }

    // Comparison:
    bool operator<(const EventStreamConfigurationInfo&) const;

    // Set stream Id:
    void setStreamId( StreamID sid ) { streamId_ = sid; }

    // Output:
    friend std::ostream& operator <<
      ( std::ostream&, const EventStreamConfigurationInfo& );

  private:

    std::string streamLabel_;
    int maxFileSizeMB_;
    std::string triggerSelection_;
    Strings eventSelection_;
    std::string outputModuleLabel_;
    double fractionToDisk_;
    StreamID streamId_;

  };

  typedef std::vector<EventStreamConfigurationInfo> EvtStrConfigList;
  typedef boost::shared_ptr<EvtStrConfigList> EvtStrConfigListPtr;

  std::ostream& operator << ( std::ostream&, const EventStreamConfigurationInfo& );

} // namespace stor

#endif // EventFilter_StorageManager_EventStreamConfigurationInfo_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
