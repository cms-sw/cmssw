#include <Riostream.h>
#include <TDOMParser.h>
#include <TXMLNode.h>
#include <TXMLAttr.h>
#include <TFile.h>
#include <TH1.h>


void create_reference(const char* fname1, const char* fname2)
{
  TFile* file1 = new TFile(fname1);
  if (!file1) return;

  TFile* file2 = new TFile(fname2, "RECREATE");

  TDOMParser *domParser = new TDOMParser();

  domParser->SetValidate(false); // do not validate with DTD
  int icode = domParser->ParseFile("sistrip_plot_layout.xml");
 
  if (icode != 0) return;

  TXMLNode *node = domParser->GetXMLDocument()->GetRootNode();

  TDirectory* td;
  parseXML(node, file1, file2, td);

  delete domParser;
  file1->Close();
  file2->Write();
  file2->Close();
}

void parseXML(TXMLNode *node, TFile* file1, TFile* file2, TDirectory* td)
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
	  cout << attr_name << " : " << attr_value;
	  if (node_name == "layout") {
	    td =  dynamic_cast<TDirectory*> (file2->Get(attr_name.c_str()));
	    if (!td) {
               td = file2->mkdir(attr_value.c_str());
            }
	  }
          if (node_name == "monitorable") {
            string path = attr_value.replace(0,13,"DQMData");
            TH1F* th1 = dynamic_cast<TH1F*> ( file1->Get(path.c_str()));
	    th1->SetDirectory(td);
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
    
    parseXML(node->GetChildren(), file1, file2, td);
  }
}


