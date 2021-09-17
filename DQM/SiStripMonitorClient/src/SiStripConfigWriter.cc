#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include <memory>

using namespace std;

//
// -- Constructor
//
SiStripConfigWriter::SiStripConfigWriter() {}
//
// -- Destructor
//
SiStripConfigWriter::~SiStripConfigWriter() {}
//
// -- Initialize XML
//
bool SiStripConfigWriter::init(std::string main) {
  assert(!"No longer implemented.");
  return true;
}
//
// -- Add an Element to the top node
//
void SiStripConfigWriter::createElement(std::string tag) {}
//
// -- Add an Element to the top node
//
void SiStripConfigWriter::createElement(std::string tag, std::string name) {}
//
// -- Add a child to the last element
//
void SiStripConfigWriter::createChildElement(std::string tag, std::string name) {}
//
// -- Add a child to the last element
//
void SiStripConfigWriter::createChildElement(std::string tag,
                                             std::string name,
                                             std::string att_name,
                                             std::string att_val) {}
//
// -- Add a child to the last element
//
void SiStripConfigWriter::createChildElement(std::string tag,
                                             std::string name,
                                             std::string att_name1,
                                             std::string att_val1,
                                             std::string att_name2,
                                             std::string att_val2) {}
//
// -- Add a child to the last element
//
void SiStripConfigWriter::createChildElement(std::string tag,
                                             std::string name,
                                             std::string att_name1,
                                             std::string att_val1,
                                             std::string att_name2,
                                             std::string att_val2,
                                             std::string att_name3,
                                             std::string att_val3) {}
//
// -- Write to File
//
void SiStripConfigWriter::write(std::string fname) {}
