#include "RecoParticleFlow/PFClusterTools/interface/IO.h"
#include "RecoParticleFlow/PFClusterTools/interface/Utils.h"
#include <cstring>

using namespace std;
using namespace pftools;

const unsigned IO::sLinesize = 1000;

IO::IO(const char* filepattern) :  fCurline(0) {
  // loop on all files matching pattern, store each non empty line in 
  // fAllLines for efficient searches later.
  cout<<endl;
  cout<<"------ Reading User Parameters : "<<filepattern<<endl;

  vector<string> files = Utils::Glob(filepattern);
  
  if( files.empty() ) {
    string err = "IO::IO : no files verify pattern ";
    err += filepattern;
    throw err;
  }

  for(unsigned i=0; i<files.size(); i++) {
    ParseFile(files[i].c_str());
  }
  cout<<"------ Reading User Parameters : DONE ---------"<<endl;
  cout<<endl;

}

bool IO::ParseFile(const char* filename) {
  cout<<"file : "<<filename<<"\t\t";

  std::ifstream in(filename);
  if( !in.good() ) {
    cout<<"unreadable"<<endl;
    return false;
  }

  char data[sLinesize];
  char s[sLinesize];
  int pos=0;

  do { 
    in.seekg(pos);
    in.getline(s,sLinesize);
    
    pos = in.tellg();     
 
    if(string(s).empty()) {
      continue; // remove empty lines
    }

    istringstream lin(s);  

    string tag;
    lin>>tag;

    if(!strncmp(tag.c_str(),"//",2)) continue; // remove commented lines can be done better ...

    lin.get(data,sLinesize);
    
    fAllLines.push_back(pair<string, string>(tag, data));
  } while(in.good());
  
  if(in.eof()) {
    cout<<"ok"<<endl;
    return true;
  }
  else {
    cout<<"error"<<endl;
    return false;
  }
}

void IO::Dump(ostream& out) const {
  for (unsigned i=0; i<fAllLines.size(); i++) {
    out<<fAllLines[i].first<< "\t" << fAllLines[i].second << endl; 
  } 
}

ostream& operator<<(ostream& out, IO& io) {
  if(!out) return out;
  io.Dump(out);
  return out;
}


string IO::GetLineData(const char* tag, const char* key) const {
  // if tag matches several option lines, data is the data corresponding
  // to the last tag

  char data[sLinesize];
  bool found = false;
  for(unsigned i=0; i<fAllLines.size(); i++) {
    if( !fnmatch(fAllLines[i].first.c_str(), tag, 0) ) { 
      istringstream in(fAllLines[i].second);
      string readkey; in>>readkey;
      
      if(readkey == key) {
        //      data.erase();
        //      string skey = key;
        //      int start = pos+skey.size();
        //      data.assign(fAllLines[i].second, start, data.size()-start);
        found=true;
        in.get(data,sLinesize);
      }
    }
  }
  if(found) return string(data);
  else return string();
}

string IO::GetNextLineData(const char* tag, const char* key)  {

  if(fCurtag != tag || fCurkey != key) {
    // not the same request
    fCurline = 0;
    fCurtag = tag;
    fCurkey = key;
  }
  // cout<<fCurline<<" "<<fCurtag<<" "<<fCurkey<<endl;

  char data[sLinesize];
  bool found = false;
  for(unsigned i=fCurline; i<fAllLines.size(); i++) {
    if( !fnmatch(fAllLines[i].first.c_str(), tag, 0) ) { 
      istringstream in(fAllLines[i].second);
      string readkey; in>>readkey;
      
      if(readkey == key) {
        found=true;
        in.get(data,sLinesize);
        fCurline=i+1;
        break;
      }
    }
  }
  if(found) return string(data);
  else return string();
}


bool IO::GetOpt(const char* tag, const char* key, string& value) const {
  string data = GetLineData(tag,key);
  
  char cstr[sLinesize];
  istringstream in(data);  
  in.get(cstr,sLinesize);


  value = cstr;
  if(!value.empty()) {
    // remove leading spaces
    int pos = value.find_first_not_of(" \t");
    value = value.substr(pos);
    // remove trailing spaces
    pos = value.find_last_not_of(" \t");
    value = value.substr(0,pos+1);
  }
  if(!value.empty()) return true;
  else return false;
}











