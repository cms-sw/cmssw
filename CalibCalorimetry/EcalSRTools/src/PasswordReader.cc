#include "PasswordReader.h"

#include <iostream>
#include <fstream>
#include <algorithm>

#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

void PasswordReader::readPassword(const std::string& fileName,
                                  const std::string& user,
                                  std::string& password){
  ifstream f(fileName.c_str());
  if(!f.good()){
    throw cms::Exception("File")
      << "Failed to open file " << fileName << " for reading condition "
      "database password\n";
  }
  string line;
  bool found = false;
  int nstatements = 0; //number of lines other than empty and comment lines
  while(f.good() && !found){
    size_t pos = 0;
    getline(f, line);
    trim(line, " \t");
    if(line[0]=='#' || line.empty()){//comment line
      continue;
    }
    ++nstatements;
    string u = tokenize(line, ":/ \t", pos);
    if(u == user){//user found
      password = tokenize(line, ":/ \t", pos);
      found = true;
    }
  }
  if(!found && nstatements==1){//login not found in the file
    //let's check if file does not contain a single password, w/o login
    f.clear();
    f.seekg(0, ios::beg);
    getline(f,line);
    trim(line, " \t");
    if(line.find_first_of(": \t")==string::npos){//no login/password delimiter
      //looks like a single password file
      password = line;
      found = true;
    }
  }
  if(!found){
    throw cms::Exception("Database")
      << " Password for condition database user '" << user << "' not found in"
      << " password file " << fileName << "\n";
  }
}

std::string PasswordReader::tokenize(const string& s,
                                     const string& delim,
                                     size_t& pos) const{
  size_t pos0 = pos;
  size_t len = s.size();
  //eats delimeters at beginning of the string
  while(pos0<s.size() && find(delim.begin(), delim.end(), s[pos0])!=delim.end()){
    ++pos0;
  }
  if(pos0>=len || pos0==string::npos) return "";
  pos = s.find_first_of(delim, pos0);
  return s.substr(pos0, (pos>0?pos:s.size())-pos0);
}

std::string PasswordReader::trim(const std::string& s,
                                 const std::string& chars) const{
  std::string::size_type pos0 = s.find_first_not_of(chars);
  if(pos0==std::string::npos){
    pos0=0;
  }
  string::size_type pos1 = s.find_last_not_of(chars)+1;
  if(pos1==std::string::npos){
    pos1 = pos0;
  }
  return s.substr(pos0, pos1-pos0);
}
