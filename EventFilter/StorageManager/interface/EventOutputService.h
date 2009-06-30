#ifndef EVENTOUTPUTSERVICE_H
#define EVENTOUTPUTSERVICE_H

// $Id: EventOutputService.h,v 1.3 2008/08/07 11:33:14 loizides Exp $

#include <EventFilter/StorageManager/interface/OutputService.h>
#include <IOPool/Streamer/interface/InitMessage.h>
#include <IOPool/Streamer/src/StreamerFileWriter.h>

namespace edm {

  class EventOutputService : public OutputService
  {
    public:
      EventOutputService(boost::shared_ptr<FileRecord>, InitMsgView const&);
      ~EventOutputService();

      void   writeEvent(const uint8 * const);
      void   report(std::ostream &os, int indentation) const;

    private:
      void   writeHeader(InitMsgView const&);
      void   closeFile();

      boost::shared_ptr<StreamerFileWriter> writer_; // writes streamer and index file
  };

} // edm namespace
#endif
