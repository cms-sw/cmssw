#ifndef FRDOUTPUTSERVICE_H
#define FRDOUTPUTSERVICE_H

// $Id: FRDOutputService.h,v 1.3 2008/08/07 11:33:14 loizides Exp $

#include <EventFilter/StorageManager/interface/OutputService.h>
#include <IOPool/Streamer/interface/FRDEventFileWriter.h>

namespace edm {

  class FRDOutputService : public OutputService
  {
    public:
      FRDOutputService(boost::shared_ptr<FileRecord>);
      ~FRDOutputService();

      void   writeEvent(const uint8 * const);
      void   report(std::ostream &os, int indentation) const;

    private:
      void   closeFile();

      boost::shared_ptr<FRDEventFileWriter> writer_;
  };

} // edm namespace
#endif
