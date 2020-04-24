#include <Riostream.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TKey.h>
#include <TH1.h>
#include <string>
#include <vector>
void parseStructure(TDirectory* td1, TDirectory* td2);
void split(const string& str, vector<string>& tokens, const string& delimiters);
void setPath(string& path, TDirectory* topDir);
bool goToDir(TDirectory* top_dir, string dname);

void sistrip_reduce_file(string fname1, string fname2)
{


  TFile* file1 = new TFile(fname1.c_str());
  if (!file1 || !file1->IsOpen()) return;

  TDirectory* topDir1 = dynamic_cast<TDirectory*>( file1->Get("DQMData"));
  goToDir(topDir1, "SiStrip");
  TDirectory* stripDir = gDirectory;

  TFile* file2 = new TFile(fname2.c_str(), "RECREATE");
  if (!file2 || !file2->IsOpen()) return;

  TDirectory* topDir2 = file2->mkdir("DQMData");
  if (!topDir2) {
    cout << " Can not create directory structure in " << fname2 << endl;
    return;
  }

  parseStructure(stripDir, topDir2);

  file1->Close();

  file2->Write();
  file2->Close();
}

void parseStructure(TDirectory* td1, TDirectory* td2) {
  string dir_name = td1->GetTitle();
  string dir_path = td1->GetPath();
  //  cout << "ParseStructure: dir_name=" << dir_name << ", dir_path=" << dir_path << endl;
  dir_path = dir_path.substr(dir_path.find("DQMData")+8);
  setPath(dir_path, td2);
  TIter next(td1->GetListOfKeys());
  TKey *key;
  while  ( (key = dynamic_cast<TKey*>(next())) ) 
    {
      string clName(key->GetClassName());
      if (clName == "TDirectoryFile") {
        string name(key->GetName());
        if (name.find("forward_") == string::npos  && 
            name.find("backward_") == string::npos &&
            name.find("ring_") == string::npos) {
	  td1->cd(name.c_str());
	  TDirectory *curr_dir = gDirectory; // dynamic_cast<TDirectory*>(obj);
	  parseStructure(curr_dir, td2);
        } else return;
      } else if (clName == "TObjString") {
	//	cout << clName << "  " << key->GetName() << endl;
        TObject* obj = key->ReadObj();
        obj->Write();
      } else {
	key->ReadObj();
      }
    }
}
//
// -- Split a given string into a number of strings using given
//    delimiters and fill a vector with splitted strings
//
void split(const string& str, vector<string>& tokens, const string& delimiters) {
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // Find first "non-delimiter".
  string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (string::npos != pos || string::npos != lastPos)  {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));

    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);

    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}
void setPath(string& path, TDirectory* topDir) {

  TDirectory* cdir = dynamic_cast<TDirectory*> (topDir->GetDirectory(path.c_str()));
  if (!cdir) {
    vector<string> names;
    string tag = "/";
    split(path, names, tag);
    cdir = topDir;
    for (unsigned int it = 0; it < names.size();  it++) {
      string name = names[it];
      if (name.size() != 0) {
	TDirectory* td  = dynamic_cast<TDirectory*> (cdir->Get(name.c_str()));
        if (!td) td = cdir->mkdir(name.c_str());
	cdir = td;
      }
    }     
  }
  cdir->cd();
  //cdir->pwd();
}
bool goToDir(TDirectory* top_dir, string dname){
  string dir_name = top_dir->GetTitle();
  if (dir_name == dname) {
    return true;
  } else {
    TIter next(top_dir->GetListOfKeys());
    TKey *key;
    while  ( (key = dynamic_cast<TKey*>(next())) ) {
      string clName(key->GetClassName());
      if (clName == "TDirectoryFile") {
        TDirectory *curr_dir = dynamic_cast<TDirectory*>(key->ReadObj());
        dname = curr_dir->GetName();
        if (dname.find("Reference") == 0) continue;
        curr_dir->cd();
        if (goToDir(curr_dir, dname)) return true;
        curr_dir->cd("..");
      }
    }
  }
  return false;
}
