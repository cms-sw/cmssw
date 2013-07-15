// $Id: FRDEventFileWriter.cc,v 1.2 2009/04/24 17:48:10 biery Exp $

#include "IOPool/Streamer/interface/FRDEventFileWriter.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
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
  
  cms::Adler32((const char*) msg.startAddress(), msg.size(), adlera_, adlerb_);
}

void FRDEventFileWriter::doOutputEventFragment(unsigned char* dataPtr,
                                               unsigned long dataSize)
{
  ost_->write((const char*) dataPtr, dataSize);
  if (ost_->fail()) {
    throw cms::Exception("FRDEventFileWriter", "doOutputEventFragment")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }

  ost_->flush();
  if (ost_->fail()) {
    throw cms::Exception("FRDEventFileWriter", "doOutputEventFragment")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }

  cms::Adler32((const char*) dataPtr, dataSize, adlera_, adlerb_);
}

void FRDEventFileWriter::initialize(std::string const& name)
{
  fileName_ = name;
  ost_.reset(new std::ofstream(name.c_str(), std::ios_base::binary | std::ios_base::out));

  if (!ost_->is_open()) {
    throw cms::Exception("FRDEventFileWriter","initialize")
      << "Error opening FED Raw Data event output file: " << name << "\n";
  }

  adlera_ = 1;
  adlerb_ = 0;
}
