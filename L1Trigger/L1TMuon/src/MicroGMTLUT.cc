
#include "../interface/MicroGMTLUT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// I/O functions
void
l1t::MicroGMTLUT::save(std::ofstream& output)
{
  std::stringstream out;
  headerToStream(out);
  contentsToStream(out);
  out << "}" << std::endl;
  output << out.str();
}

void
l1t::MicroGMTLUT::load(const std::string& inFileName) {
  std::ifstream fstream;
  fstream.open(edm::FileInPath(inFileName.c_str()).fullPath());
  if (!fstream.good()) {
    fstream.close();
    throw cms::Exception("FileOpenError") << "Failed to open LUT file: " << inFileName;
  }


  std::string lineID = "";

  while (!fstream.eof()) {
    // read until either end of file is reached or the CONTENT line is found
    lineID = "";
    fstream >> lineID;
    if (lineID.find("CONTENT_VECTOR") != std::string::npos) break;
  }

  if (fstream.eof()) {
    fstream.close();
    throw cms::Exception("FileOpenError") << "Failed to find LUT contents";
  }

  for (int cntr = 0; cntr < (1 << m_totalInWidth); ++cntr) {
    fstream >> m_contents[cntr];
  }
  m_initialized = true;
  fstream.close();
}

int
l1t::MicroGMTLUT::lookupPacked(const int input) const
{
  // sort to make sure the hash is correct?
  if (m_initialized) {
    return m_contents.at(input);
  }
  throw cms::Exception("Unitialized") << "If you're not loading a LUT from file you need to implement lookupPacked.";
  return 0;
}

void
l1t::MicroGMTLUT::initialize()
{
  for (int in = 0; in < (1 << m_totalInWidth); ++in) {
    int out = lookupPacked(in);
    m_contents[in] = out;
  }
  m_initialized = true;
}

int
l1t::MicroGMTLUT::checkedInput(unsigned in, unsigned maxWidth) const
{
  unsigned maxIn = (1 << maxWidth) - 1;
  return (in < maxIn ? in : maxIn);
}

void
l1t::MicroGMTLUT::contentsToStream(std::stringstream& stream)
{
  stream << "\"CONTENT\" : [ ";

  int maxVal = (1<<m_totalInWidth);
  for (int in = 0; in < maxVal; ++in) {
    stream << lookupPacked(in);
    if (in != maxVal - 1) {
      stream << ", ";
    }
  }
  stream << "] " << std::endl;
}

void
l1t::MicroGMTLUT::headerToStream(std::stringstream& stream) const
{
  stream << "{" << std::endl;
  stream << "\"NAME\"      : \"Name of the LUT\","<< std::endl;
  stream << "\"INSTNACES\" : \"List (space separated) of instances of this LUT (differing contents but same in/output)\"," << std::endl;
  stream << "\"INPUTS\"    : \"List (space separated) of inputs in format <input_name>(<input_width>)\"," << std::endl;
  stream << "\"OUTPUTS\"   : \"List (space separated) of outputs in format <output_name>(<output_width>)\"," << std::endl;
  stream << "\"IPBUS_ADD\" : \"Address for access via IPBus\"," << std::endl;
  stream << "\"CONTENT_X\" : \"List (space separated) of outputs from packed int for zero-indexed instance X\"," << std::endl;
}
