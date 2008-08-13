// $Id: FRDEventFileWriter.cc,v 1.17 2007/11/07 06:51:52 wmtan Exp $

#include "IOPool/Streamer/interface/FRDEventFileWriter.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

FRDEventFileWriter::FRDEventFileWriter(edm::ParameterSet const& ps)
{
  initialize(ps.getUntrackedParameter<std::string>("fileName", "testFRDfile.dat"));
}

FRDEventFileWriter::FRDEventFileWriter(std::string const& fileName)
{
  initialize(fileName);
}

FRDEventFileWriter::~FRDEventFileWriter()
{
  ost_->close();
}

void FRDEventFileWriter::doOutputEvent(FRDEventMsgView const& msg)
{
  ost_->write((const char*) msg.startAddress(), msg.size());
  if (ost_->fail()) {
    throw cms::Exception("FRDEventFileWriter", "doOutputEvent")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }

  ost_->flush();
  if (ost_->fail()) {
    throw cms::Exception("FRDEventFileWriter", "doOutputEvent")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }
}

void FRDEventFileWriter::initialize(std::string const& name)
{
  fileName_ = name;
  ost_.reset(new std::ofstream(name.c_str(), std::ios_base::binary | std::ios_base::out));

  if (!ost_->is_open()) {
    throw cms::Exception("FRDEventFileWriter","initialize")
      << "Error opening FED Raw Data event output file: " << name << "\n";
  }
}
