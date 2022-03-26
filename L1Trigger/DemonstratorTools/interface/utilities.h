
#ifndef L1Trigger_DemonstratorTools_utilities_h
#define L1Trigger_DemonstratorTools_utilities_h

#include <iosfwd>

#include "L1Trigger/DemonstratorTools/interface/BoardData.h"
#include "L1Trigger/DemonstratorTools/interface/FileFormat.h"

namespace l1t::demo {

  // Simple function that converts string to file format enum (for e.g. CMSSW configs)
  FileFormat parseFileFormat(const std::string&);

  BoardData read(const std::string& filePath, const FileFormat);

  BoardData read(std::istream&, const FileFormat);

  void write(const BoardData&, const std::string& filePath, const FileFormat);

  void write(const BoardData&, std::ostream&, const FileFormat);

}  // namespace l1t::demo

#endif