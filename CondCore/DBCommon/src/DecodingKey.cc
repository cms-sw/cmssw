#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/FileUtils.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CoralCommon/Cipher.h"
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <pwd.h>

static char DecodingKeySeparator(';');

bool cond::DecodingKey::readUserKey(const std::string& keyFileName){
  return readFromFile(getUserName(), keyFileName);
}

bool cond::DecodingKey::readUserKeyString(const std::string& content){
  return readFromString(getUserName(), content);
}

bool cond::DecodingKey::readFromFile(const std::string& password, const std::string& keyFileName){
  cond::FileReader reader;
  reader.read(keyFileName);
  return readFromString(password,reader.content());
}

bool cond::DecodingKey::readFromString(const std::string& password, const std::string& content){
  std::string decodedContent = coral::Cipher::decode(content,password);
  size_t pos = decodedContent.find(DecodingKeySeparator);
  if(pos==std::string::npos || pos==0){
    std::stringstream msg;
    msg << "Provided Key is invalid.";
    throw cond::Exception(msg.str());    
  }
  m_key = decodedContent.substr(0,pos);
  m_dataSource = decodedContent.substr(pos+1);
  return true;
}

bool cond::DecodingKey::validateKey(const std::string& key){
  if(key.find(DecodingKeySeparator)!=std::string::npos){
    std::stringstream msg;
    msg << "Invalid character ';' found in key string.";
    throw cond::Exception(msg.str());    
  }
  return true;
}


std::string cond::DecodingKey::getUserName(){
  std::string userName("");
  struct passwd* userp = ::getpwuid(::getuid());
  if(userp) {
    char* uName = userp->pw_name;
    if(uName){
      userName += uName;
    }
  }
  if(userName.empty()){
    std::stringstream msg;
    msg << "Cannot determine login name.";
    throw cond::Exception(msg.str());     
  }
  return userName;
}

bool cond::DecodingKey::createFile(const std::string& password, const std::string& key,
                                   const std::string& dataSource, const std::string& keyFileName){
  if(password.empty()){
    std::stringstream msg;
    msg << "Provided password is empty.";
    throw cond::Exception(msg.str());    
  }
  std::string content("");
  validateKey(key);
  if(dataSource.find(DecodingKeySeparator)!=std::string::npos){
    std::stringstream msg;
    msg << "Invalid character ';' found in data file name string.";
    throw cond::Exception(msg.str());    
  }
  content.append(key).append(1,DecodingKeySeparator).append(dataSource);
  std::string encodedContent = coral::Cipher::encode(content,password);
  std::ofstream keyFile;
  keyFile.open(keyFileName.c_str());
  if(!keyFile.good()){
    keyFile.close();
    std::stringstream msg;
    msg << "Cannot open the key file \""<<keyFileName<<"\"";
    throw cond::Exception(msg.str());
  }
  keyFile << encodedContent;
  keyFile.flush();
  keyFile.close();
  return true;
}

