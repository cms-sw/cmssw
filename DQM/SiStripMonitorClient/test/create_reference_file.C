#include <Riostream.h>
#include <TDOMParser.h>
#include <TXMLNode.h>
#include <TXMLAttr.h>
#include <TFile.h>
#include <TH1.h>
#include <string>
#include <vector>
void parseXML(TXMLNode *node, TFile* file1, TDirectory* ts, string store_path);
void split(const string& str, vector<string>& tokens, const string& delimiters);
void setPath(TH1* th1, string& path, TDirectory* topDir);
bool goToDir(TDirectory* top_dir, string dname);

void create_reference_file(string fname1, string fname2, string type)
{
  TFile* file1 = new TFile(fname1.c_str());
  if (!file1) return;

  // Get the path name of the required directory specified by type
  TDirectory* td1 = dynamic_cast<TDirectory*>( file1->Get("DQMData"));
  if (!goToDir(td1, type)) {
    cout << " Required Directory does not exist! ";
    return;
  }

  TDirectory* rDir = gDirectory;
  string store_path;
  store_path = rDir->GetPath();
  store_path += "/Run summary";

  // Open the new file and create DQMData directory

  TFile* file2 = new TFile(fname2.c_str(), "RECREATE");
  TDirectory* td2 =   file2->mkdir("DQMData");
  if (!td2) {
    cout << " Can not create directory structure in " << fname2 << endl;
    return;
  }

  // Create XML parser
  TDOMParser *domParser = new TDOMParser();

  domParser->SetValidate(false); // do not validate with DTD
  int icode = domParser->ParseFile("sistrip_plot_layout.xml");
 
  if (icode != 0) return;

  TXMLNode *node = domParser->GetXMLDocument()->GetRootNode();

  parseXML(node, file1, td2, store_path);


  delete domParser;
  file1->Close();
  file2->Write();
  file2->Close();
}

void parseXML(TXMLNode *node, TFile* file1, TDirectory* td, string store_path)
{
  for ( ; node; node = node->GetNextNode()) {
    if (node->GetNodeType() == TXMLNode::kXMLElementNode) { // Element Node
      string node_name = node->GetNodeName();
      cout << node->GetNodeName() << " : ";
      if (node->HasAttributes()) {
	TList* attrList = node->GetAttributes();
	TIter next(attrList);
	TXMLAttr *attr;
	while ((attr =(TXMLAttr*)next())) {
	  string attr_name  = attr->GetName();
	  string attr_value = attr->GetValue();
	  cout << attr_name << " : " << attr_value << endl;
          if (node_name == "monitorable") {
            string path, fname;
            path = store_path + attr_value.substr(attr_value.find("SiStrip")+7);
            TH1F* th1 = dynamic_cast<TH1F*> ( file1->Get(path.c_str()));
            if (th1) {
              cout << " copying " << th1->GetName() << " to " << attr_value << endl;
              setPath(th1, attr_value, td);
            }
            else {
              cout << "\n Requested Histogram does not exist !!! " << endl;
              cout << " ==> " << path << endl;
              cout << " ==> Please check the root file : "<< file1->GetName() << " !! "<< endl;
            }
          }
	}
      }
    }
    if (node->GetNodeType() == TXMLNode::kXMLTextNode) { // Text node
      cout << node->GetContent();
    }
    if (node->GetNodeType() == TXMLNode::kXMLCommentNode) { //Comment node
      cout << "Comment: " << node->GetContent();
    }
    
    parseXML(node->GetChildren(), file1, td, store_path);
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
void setPath(TH1* th1, string& path, TDirectory* topDir) {

  TDirectory* temp_dir  =  dynamic_cast<TDirectory*> (topDir->Get(path.c_str()));
  if (!temp_dir) {
    vector<string> names;
    string tag = "/";
    split(path, names, tag);
    temp_dir = topDir;
    for (unsigned int it = 0; it < names.size()-1;  it++) {
      string name = names[it];
      if (name.size() != 0) {
	TDirectory* td  = dynamic_cast<TDirectory*> (temp_dir->Get(name.c_str()));
        if (!td) td = temp_dir->mkdir(name.c_str());
        if (temp_dir) {
          td->cd();
          temp_dir = td;
        }
      }
    }     
    th1->SetDirectory(temp_dir);
    topDir->cd();
  }
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
        string name = curr_dir->GetName();
        if (name == "Reference") continue;
	curr_dir->cd();
	if (goToDir(curr_dir, dname)) return true;
	curr_dir->cd("..");
      }
    }
  }
  return false;
}

