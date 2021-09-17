#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"

#include <memory>

using namespace std;
//
// -- Constructor
//
SiPixelConfigWriter::SiPixelConfigWriter() {}
//
// -- Destructor
//
SiPixelConfigWriter::~SiPixelConfigWriter() {}
//
// -- Initialize XML
//
bool SiPixelConfigWriter::init() {
  assert(!"No longer implemented.");
  return true;
}
//
// -- Add an Element
//
void SiPixelConfigWriter::createLayout(string &name) {}
//
// -- Add an Element
//
void SiPixelConfigWriter::createRow() {}
//
// -- Add an Element with Children
//
void SiPixelConfigWriter::createColumn(string &element, string &name) {}
//
// -- Write to File
//
void SiPixelConfigWriter::write(string &fname) {}
