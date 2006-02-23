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
   explicit CSCTriggerMappingFromFile( std::string filename );
   CSCTriggerMappingFromFile() {}

  /// Destructor
   virtual ~CSCTriggerMappingFromFile();

  /// Fill mapping store
   virtual void fill( void );

 private: 
   std::string filename_;

};

#endif
