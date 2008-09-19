#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/FileUtils.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CoralCommon/Cipher.h"
#include <sstream>
#include <fstream>

static char DecodingKeySeparator(';');

bool cond::DecodingKey::readFromFile(const std::string& keyFileName){
  cond::FileReader reader;
  reader.read(keyFileName);
  std::string content = reader.content();
  readFromString(content);
  return true;
}

bool cond::DecodingKey::readFromString(const std::string& content){
  std::string decodedContent = coral::Cipher::u64dec(content);
  size_t pos = decodedContent.find(DecodingKeySeparator);
  if(pos==std::string::npos || pos==0){
    std::stringstream msg;
    msg << "Key File content is invalid.";
    throw cond::Exception(msg.str());    
  }
  m_password = decodedContent.substr(0,pos);
  m_dataFileName = decodedContent.substr(pos+1);
  return true;
}

bool cond::DecodingKey::validatePassword(const std::string& password){
  if(password.find(DecodingKeySeparator)!=std::string::npos){
    std::stringstream msg;
    msg << "Invalid character ';' found in password string.";
    throw cond::Exception(msg.str());    
  }
  return true;
}

bool cond::DecodingKey::createFile(const std::string& password, const std::string& dataFileName, const std::string& keyFileName){
  std::string content("");
  validatePassword(password);
  if(dataFileName.find(DecodingKeySeparator)!=std::string::npos){
    std::stringstream msg;
    msg << "Invalid character ';' found in data file name string.";
    throw cond::Exception(msg.str());    
  }
  content.append(password).append(1,DecodingKeySeparator).append(dataFileName);
  std::string encodedContent = coral::Cipher::u64enc(content);
  std::ofstream keyFile;
  keyFile.open(keyFileName.c_str());
  if(!keyFile.good()){
    std::stringstream msg;
    msg << "Cannot open the key file \""<<keyFileName<<"\"";
    keyFile.close();
    throw cond::Exception(msg.str());
  }
  keyFile << encodedContent;
  keyFile.flush();
  keyFile.close();
  return true;
}

