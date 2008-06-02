#include <Riostream.h>
#include <TDOMParser.h>
#include <TXMLNode.h>
#include <TXMLAttr.h>
#include <TFile.h>
#include <TH1.h>
#include <string>
#include <vector>
void parseXML(TXMLNode *node, TFile* file1, TDirectory* ts, bool flag, string store_path);
void split(const string& str, vector<string>& tokens, const string& delimiters);
void setPath(TH1* th1, string& path, TDirectory* topDir);

void create_reference_file(string fname1, string fname2, string irun)
{
  TFile* file1 = new TFile(fname1.c_str());
  if (!file1) return;

  bool online = false;
  if (fname1.find("SiStripWebInterface") != string::npos) online = true;

  cout << fname1 << " " << online << endl;
  TFile* file2 = new TFile(fname2.c_str(), "RECREATE");

  TDirectory* td =   file2->mkdir("DQMData");
  if (!td) {
    cout << " Can not create directory structure in " << fname2 << endl;
    return;
  }

  TDOMParser *domParser = new TDOMParser();

  domParser->SetValidate(false); // do not validate with DTD
  int icode = domParser->ParseFile("sistrip_plot_layout.xml");
 
  if (icode != 0) return;

  TXMLNode *node = domParser->GetXMLDocument()->GetRootNode();

  string store_path;
  store_path = "DQMData/Run "+ irun + "/SiStrip/Run summary";
  parseXML(node, file1, td, online, store_path);

  delete domParser;
  file1->Close();
  file2->Write();
  file2->Close();
}

void parseXML(TXMLNode *node, TFile* file1, TDirectory* td, bool flag, string store_path)
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
    
    parseXML(node->GetChildren(), file1, td, flag,store_path);
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

