#ifndef CondFormats_CSCReadoutMappingFromFile_h
#define CondFormats_CSCReadoutMappingFromFile_h

/** 
 * \class CSCReadoutMappingFromFile
 * \author Tim Cox
 * A concrete CSCReadoutMappingForSLiceTest to read mapping from Ascii file.
 */

#include <CondFormats/CSCObjects/interface/CSCReadoutMappingForSliceTest.h>
#include <string>

class CSCReadoutMappingFromFile : public CSCReadoutMappingForSliceTest {
 public:

  /// Constructor
   explicit CSCReadoutMappingFromFile( std::string filename );

  /// Destructor
   virtual ~CSCReadoutMappingFromFile();

  /// Fill mapping store
   virtual void fill( void );

 private: 
   std::string filename_;

};

#endif
