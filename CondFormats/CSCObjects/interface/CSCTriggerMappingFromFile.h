#ifndef CondFormats_CSCTriggerMappingFromFile_h
#define CondFormats_CSCTriggerMappingFromFile_h

/** 
 * \class CSCTriggerMappingFromFile
 * \author Lindsey Gray
 * A concrete CSCTriggerSimpleMapping to read mapping from Ascii file.
 */

#include <CondFormats/CSCObjects/interface/CSCTriggerSimpleMapping.h>
#include <string>

class CSCTriggerMappingFromFile : public CSCTriggerSimpleMapping {
public:
  /// Constructor
  explicit CSCTriggerMappingFromFile(std::string filename);
  CSCTriggerMappingFromFile() {}

  /// Destructor
  ~CSCTriggerMappingFromFile() override;

  /// Fill mapping store
  void fill(void) override;

private:
  std::string filename_;
};

#endif
