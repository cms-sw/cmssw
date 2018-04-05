#include "CondCore/CondDB/interface/FileUtils.h"
#include "CondCore/CondDB/interface/Exception.h"
#include <fstream>
#include <sstream>

bool cond::FileReader::read(const std::string& fileName){
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
