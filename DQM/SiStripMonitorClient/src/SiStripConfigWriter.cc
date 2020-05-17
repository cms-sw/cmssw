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
bool SiStripConfigWriter::init(const std::string& main) {
  assert(!"No longer implemented.");
  return true;
}
//
// -- Add an Element to the top node
//
void SiStripConfigWriter::createElement(const std::string& tag) {}
//
// -- Add an Element to the top node
//
void SiStripConfigWriter::createElement(const std::string& tag, const std::string& name) {}
//
// -- Add a child to the last element
//
void SiStripConfigWriter::createChildElement(const std::string& tag, const std::string& name) {}
//
// -- Add a child to the last element
//
void SiStripConfigWriter::createChildElement(const std::string& tag,
                                             const std::string& name,
                                             const std::string& att_name,
                                             const std::string& att_val) {}
//
// -- Add a child to the last element
//
void SiStripConfigWriter::createChildElement(const std::string& tag,
                                             const std::string& name,
                                             const std::string& att_name1,
                                             const std::string& att_val1,
                                             const std::string& att_name2,
                                             const std::string& att_val2) {}
//
// -- Add a child to the last element
//
void SiStripConfigWriter::createChildElement(const std::string& tag,
                                             const std::string& name,
                                             const std::string& att_name1,
                                             const std::string& att_val1,
                                             const std::string& att_name2,
                                             const std::string& att_val2,
                                             const std::string& att_name3,
                                             const std::string& att_val3) {}
//
// -- Write to File
//
void SiStripConfigWriter::write(const std::string& fname) {}
