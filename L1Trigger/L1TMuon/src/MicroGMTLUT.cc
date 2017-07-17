
#include "../interface/MicroGMTLUT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

l1t::MicroGMTLUT::MicroGMTLUT(l1t::LUT* lut) : m_totalInWidth(0), m_outWidth(0), m_initialized(true)
{
  std::stringstream ss;
  lut->write(ss);
  read(ss);
}

// I/O functions
void
l1t::MicroGMTLUT::save(std::ofstream& output)
{
  write(output);
}

int
l1t::MicroGMTLUT::load(const std::string& inFileName) {
  std::ifstream fstream;
  fstream.open(edm::FileInPath(inFileName.c_str()).fullPath());
  if (!fstream.good()) {
    fstream.close();
    throw cms::Exception("FileOpenError") << "Failed to open LUT file: " << inFileName;
  }
  int readCode = read(fstream);

  m_initialized = true;
  fstream.close();

  return readCode;
}

int
l1t::MicroGMTLUT::lookupPacked(const int input) const
{
  if (m_initialized) {
    return data((unsigned int) input);
  }
  throw cms::Exception("Uninitialized") << "If you're not loading a LUT from file you need to implement lookupPacked.";
  return 0;
}

void
l1t::MicroGMTLUT::initialize()
{
  if (empty()) {
    std::stringstream stream;
    stream <<"#<header> V1 "<<m_totalInWidth<<" "<<m_outWidth<<" </header> "<<std::endl;
    for (int in = 0; in < (1 << m_totalInWidth); ++in) {
      int out = lookupPacked(in);
      stream << in << " " << out << std::endl;
    }
    read(stream);
  }
  m_initialized = true;
}

int
l1t::MicroGMTLUT::checkedInput(unsigned in, unsigned maxWidth) const
{
  unsigned maxIn = (1 << maxWidth) - 1;
  return (in < maxIn ? in : maxIn);
}

