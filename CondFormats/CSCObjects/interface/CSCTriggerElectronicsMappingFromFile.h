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
   explicit CSCTriggerElectronicsMappingFromFile( std::string filename );
   CSCTriggerElectronicsMappingFromFile() {}

  /// Destructor
   virtual ~CSCTriggerElectronicsMappingFromFile();

  /// Fill mapping store
   virtual void fill( void );

 private: 
   std::string filename_;

};

#endif
