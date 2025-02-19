#include "CondCore/DBCommon/interface/FileUtils.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include <fstream>
#include <sstream>

bool cond::FileReader::read(const std::string& fileName){
  /**struct stat st;
  if (stat(inputFileName.c_str(), &st) < 0){
    seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
    log << seal::Msg::Error << "File \"" << inputFileName << "\" not found." << seal::flush;
    return false;
  }

  std::vector<char> buf(st.st_size, 0);
  int fd = open(inputFileName.c_str(), O_RDONLY);
  if (fd < 0){
    seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
    log << seal::Msg::Error << "File \"" << inputFileName << "\" cannot be open." << seal::flush;
    close(fd);
    return false;
  }
  
  if (read(fd, &buf[0], st.st_size) != st.st_size){
    seal::MessageStream log( this, this->name(),seal::Msg::Verbose );
    log << seal::Msg::Error << "File \"" << inputFileName << "\" cannot be open for reading." << seal::flush;
    close(fd);
   return false;
   }
  std::string content(&buf[0], &buf[0]+st.st_size);
  **/  
  std::ifstream inputFile;
  inputFile.open (fileName.c_str());
  if(!inputFile.good()){
    std::stringstream msg;
    msg << "File \"" << fileName << "\" cannot be open.";
    inputFile.close();
    throw cond::Exception(msg.str());
  }
  // get pointer to associated buffer object
  std::filebuf* pbuf=inputFile.rdbuf();
  // get file size using buffer's members
  long size=pbuf->pubseekoff (0,std::ios::end,std::ios::in);
  pbuf->pubseekpos (0,std::ios::in);
  // allocate memory to contain file data
  char* buffer=new char[size+1];
  // get file data  
  pbuf->sgetn (buffer,size);
  inputFile.close();
  buffer[size]=0;
  m_content += buffer;
  delete [] buffer;
  return true;
}
