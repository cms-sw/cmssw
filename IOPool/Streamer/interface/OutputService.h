#ifndef OUTPUTSERVICE_H
#define OUTPUTSERVICE_H

// $Id:$

#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/FileRecord.h"
#include "IOPool/Streamer/src/StreamerFileWriter.h"

#include <boost/shared_ptr.hpp>
#include <string>


namespace edm {


  class OutputService
    {
      
    public:
      OutputService(boost::shared_ptr<FileRecord>, InitMsgView const&);
      ~OutputService();

      void   writeEvent(EventMsgView const&);
      double lastEntry() { return file_->lastEntry(); }
      void   report(std::ostream &os, int indentation) const;


    private:
      void   writeHeader(InitMsgView const&);
      void   closeFile();
      double getTimeStamp();

      boost::shared_ptr<StreamerFileWriter> writer_; // writes streamer and index file
      boost::shared_ptr<FileRecord>           file_;   // writes streamer and index file
 };

} // edm namespace
#endif
