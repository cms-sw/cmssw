#ifndef CondFormats_CSCReadoutMappingFromFile_h
#define CondFormats_CSCReadoutMappingFromFile_h

/** 
 * \class CSCReadoutMappingFromFile
 * \author Tim Cox
 * A concrete CSCReadoutMappingForSliceTest to read mapping from Ascii file.
 * Find file from FileInPath of ParameterSet passed from calling E_Producer.
 */

#include <CondFormats/CSCObjects/interface/CSCReadoutMappingForSliceTest.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <string>

class CSCReadoutMappingFromFile : public CSCReadoutMappingForSliceTest {
 public:

  /// Constructor
   explicit CSCReadoutMappingFromFile( const edm::ParameterSet& ps );
   CSCReadoutMappingFromFile() {}

  /// Destructor
   virtual ~CSCReadoutMappingFromFile();

  /// Fill mapping store
   virtual void fill( const edm::ParameterSet& ps );

 private: 
   std::string theMappingFile;

};

#endif
