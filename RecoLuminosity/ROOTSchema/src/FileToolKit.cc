#include "RecoLuminosity/ROOTSchema/interface/FileToolKit.h"

// STL Headers
#include <fstream>

// Linux
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// C
#include <cerrno>
#include <cstdio>
#include <cstring>

void FileToolKit::Tokenize(const std::string& str,
			   std::vector< std::string >& tokens,
			   const std::string& delimiters )
{
  using std::string;
  
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  string::size_type pos     = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos)
    {
      // Found a token, add it to the vector.
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      // Skip delimiters.  Note the "not_of"
      lastPos = str.find_first_not_of(delimiters, pos);
      // Find next "non-delimiter"
      pos = str.find_first_of(delimiters, lastPos);
    }
}

int FileToolKit::MakeDir( std::string dirName, mode_t writeMode ){
  
  using std::vector;
  using std::string;

  int errsv = 0;
  
  vector< string > tkDirName;
  string currentDirName = "";

  Tokenize(dirName, tkDirName, "/");

  if(dirName[0] == '/')  currentDirName += "/";
 
  struct stat mStat;
 
  for(unsigned int i = 0; i < tkDirName.size(); ++i ){

    currentDirName += tkDirName[i];
    currentDirName += "/";

    errno = 0;
    stat( currentDirName.c_str(), &mStat );
    errsv = errno;
    
    if( errsv == 2 ){  // No such file or directory 
      
      errno = 0;
      mkdir( currentDirName.c_str(), writeMode);
      errsv = errno;
      if( errno == 0 ){
	errno = 0;
	chmod( currentDirName.c_str(), writeMode);
	errsv = errno;
      }
    }
  }
  return errsv;
}

void FileToolKit::MakeEmptyWebPage( const std::string &fileName, 
				    const std::string &title ){

  std::fstream fileStream;

  fileStream.open( fileName.c_str(), std::fstream::out);

  fileStream << "<html>"   << std::endl;
  fileStream << "<title>"  << std::endl;
  fileStream << title      << std::endl;
  fileStream << "</title>" << std::endl; 
  fileStream << "<body>"   << std::endl;
  fileStream << "</body>"  << std::endl;
  fileStream << "</html>"  << std::endl; 

  fileStream.close();
}

void FileToolKit::InsertLineAfter( const std::string &fileName, 
				   const std::string &newLine,
				   const std::string &searchLine){

  bool bMatch = false;

  std::vector< std::string > fileContents;
  char lineBuffer[256];

  std::fstream fileStream;

  // Read file into memory and insert new line.
  fileStream.open(fileName.c_str(), std::fstream::in );
  while( fileStream.good() ){
    fileStream.getline( lineBuffer, 256 );

    fileContents.push_back( lineBuffer );

    if( strcmp( lineBuffer, searchLine.c_str() ) == 0 ){
      fileContents.push_back( newLine );
      bMatch = true;
    }
  }
  fileStream.close();

  // If search line was found, write file from buffer.
  if(bMatch){
    std::fstream fileStream2;

    fileStream2.open( fileName.c_str(), std::fstream::out );

    for(unsigned int iline = 0; iline < fileContents.size(); ++iline)
      fileStream2 << fileContents[iline] << std::endl;
    
    fileStream2.close();  

    //rename( ( fileName  ).c_str(), fileName.c_str() );
  }
}

void FileToolKit::InsertLineBefore( const std::string &fileName, 
				    const std::string &newLine,
				    const std::string &searchLine ){

  bool bMatch = false;

  std::vector< std::string > fileContents;
  char lineBuffer[256];

  std::fstream fileStream;

  fileStream.open(fileName.c_str());
  while( fileStream.good() ){
    fileStream.getline( lineBuffer, 256 );
    if(strcmp(lineBuffer, searchLine.c_str()) == 0){
      fileContents.push_back( newLine );
      bMatch = true;
    }
    fileContents.push_back( lineBuffer );
  }
  fileStream.close();

  if(bMatch){
    std::fstream fileStream2;

    fileStream2.open( fileName.c_str(), std::fstream::out  );

    for(unsigned int iline = 0; iline < fileContents.size(); ++iline){
      fileStream2 << fileContents[iline] << std::endl;
    }
    
    fileStream2.close();  

    //rename( (fileName ).c_str(), fileName.c_str());
  }
}

bool FileToolKit::fileExists( const std::string &fileName){
  
  std::fstream filestr;

  filestr.open (fileName.c_str(), std::fstream::in );
  if (filestr.is_open()){
    filestr.close();
    return true;
  }
  else
  {
    return false;
  }
}
