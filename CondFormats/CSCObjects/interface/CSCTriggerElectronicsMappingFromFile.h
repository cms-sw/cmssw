#ifndef CondFormats_CSCTriggerElectronicsMappingFromFile_h
#define CondFormats_CSCTriggerElectronicsMappingFromFile_h

/** 
 * \class CSCTriggerElectronicsMappingFromFile
 * \author Lindsey Gray
 * A concrete CSCTriggerElectronicsMapping to read mapping from Ascii file.
 */

#include <CondFormats/CSCObjects/interface/CSCTriggerElectronicsMapping.h>
#include <string>

class CSCTriggerElectronicsMappingFromFile : public CSCTriggerElectronicsMapping {
public:
  /// Constructor
  explicit CSCTriggerElectronicsMappingFromFile(std::string filename);
  CSCTriggerElectronicsMappingFromFile() {}

  /// Destructor
  ~CSCTriggerElectronicsMappingFromFile() override;

  /// Fill mapping store
  void fill(void) override;

private:
  std::string filename_;
};

#endif
