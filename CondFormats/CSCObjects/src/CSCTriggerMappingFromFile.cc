#include "CondFormats/CSCObjects/interface/CSCTriggerMappingFromFile.h"
#include <iostream>
#include <fstream>
#include <sstream>

CSCTriggerMappingFromFile::CSCTriggerMappingFromFile( std::string filename ) 
  : filename_( filename ) { fill(); }

CSCTriggerMappingFromFile::~CSCTriggerMappingFromFile(){}

void CSCTriggerMappingFromFile::fill( void ) {
  std::ifstream in( filename_.c_str() );
  std::string line;
  const std::string commentFlag = "#";
  if ( !in ) {
    std::cout << "CSCTriggerMappingFromFile: ERROR! Failed to open file containing mapping, " <<
      filename_ << std::endl ;
  }
  else
  {
    std::cout << "CSCTriggerMappingFromFile: opened file containing mapping, " << 
      filename_ << std::endl ;

    while ( getline(in, line) ) { // getline() from <string>
      if ( debugV() ) std::cout << line << std::endl;
      if ( line[0] != commentFlag[0] ) {
        int i1, i2, i3, i4, i5, i6, i7, i8, i9, i10;
	std::istringstream is( line );
        is >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> i8 >> i9 >> i10;
        if ( debugV() ) std::cout << i1 << " " << i2 << " " << i3 << " " << i4 << " " 
				  << i5 << " " << i6 << " " << i7 << " " << i8 << " " 
				  << i9 << " " << i10 << std::endl;
        addRecord( i1, i2, i3, i4, i5, i6, i7, i8, i9 , i10);
      }
    }

  }

  return;
}

