#include "L1Trigger/L1TMuonEndCap/interface/ForestHelper.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <iostream>
#include <cassert>

using namespace l1t;
using namespace std;

const ForestHelper *  ForestHelper::readFromEventSetup(const L1TMuonEndCapForest * es){
  return new ForestHelper(es);
}

ForestHelper *  ForestHelper::readAndWriteFromEventSetup(const L1TMuonEndCapForest * es){
  ForestHelper * x = new ForestHelper(es);
  x->useCopy();
  return x;
}

ForestHelper::ForestHelper(L1TMuonEndCapForest * w) {
  write_ = w; 
  check_write(); 
  we_own_write_ = false;
  //write_->m_version = VERSION; 
  read_ = write_; 
}

ForestHelper::ForestHelper(const L1TMuonEndCapForest * es) {read_ = es; write_=NULL;}

void ForestHelper::useCopy(){
  write_ = new L1TMuonEndCapForest(*read_);
  we_own_write_ = true;
  read_  = write_;
}

ForestHelper::~ForestHelper() {
  if (we_own_write_ && write_) delete write_;
}


// print all the L1 GT stable parameters
void ForestHelper::print(std::ostream& myStr) const {
    myStr << "\nL1T EndCap  Parameters \n" << std::endl;


    for (int mode=0; mode<16; mode++){
      if (mode != 15) continue;

      auto it = read_->forest_map_.find(mode);
      if (it == read_->forest_map_.end())
	continue;

      const DForest & dforest = read_->forest_coll_[it->second];
      
      int count = 0;
      for (auto itree = dforest.begin(); itree != dforest.end(); itree++){
	const DTree & tree = *itree;
	cout << "DUMP: ***** Tree " << count << " with size " << tree.size() << "\n";
	for (unsigned index=0; index<tree.size(); index++){
	  const DTreeNode & node = tree[index];
	  cout << "node " << index << " l: " << node.ileft << " r: " << node.iright << "svar: " << node.splitVar << " sval: " << node.splitVal << " fit: " << node.fitVal << "\n";
	}


	count++;
      }
    }
}



void ForestHelper::initializeFromXML(const char * dirname, const std::vector<int> & modes, int ntrees){

  //cout << "DEBUG: starting initializeFromXML...\n";

  assert(write_->forest_coll_.size() == 0);

  for(int i =0; i < (int) modes.size(); i++){
    int mode = modes[i];
    //cout << "DEBUG: initializing Decision Forest for mode=" << mode << "\n";

    //DForest tmp;
    write_->forest_coll_.push_back(DForest());
    DForest & dforest = write_->forest_coll_[i];
    write_->forest_map_[mode]=i;

    std::stringstream ss;
    ss << dirname << "/" << mode;
    std::string directory;
    ss >> directory;

    for(int j=0; j < ntrees; j++){   
      std::stringstream ss;
      ss << directory << "/" << j << ".xml";
      std::string filename;
      ss >> filename;
      //cout << "DEBUG: loading tree " << filename << "\n";	 

      
      dforest.push_back(DTree());
      DTree & dtree = dforest[j];
      dtree.push_back(DTreeNode());


      // First create the engine.
      TXMLEngine* xml = new TXMLEngine();

      // Now try to parse xml file.
      XMLDocPointer_t xmldoc = xml->ParseFile(edm::FileInPath(filename.c_str()).fullPath().c_str());
      if (xmldoc==0){
	delete xml;
	continue;
      }
      // Get access to main node of the xml file.
      XMLNodePointer_t mainnode = xml->DocGetRootElement(xmldoc);
      
      loadTreeFromXMLRecursive(xml, mainnode, dtree, 0);

      //cout << "DEBUG: parsed tree of size " << dtree.size() << " for mode " << mode << "\n";

      xml->FreeDoc(xmldoc);
      delete xml;

     }
  }
}

double ForestHelper::evaluate(int mode, const std::vector<double> & data) const {
  auto it = read_->forest_map_.find(mode);
  if (it == read_->forest_map_.end())
    return 0;

  const DForest & dforest = read_->forest_coll_[it->second];
  
  double sum = 0;
  for (auto itree = dforest.begin(); itree != dforest.end(); itree++){
    double x = evalTreeRecursive(data, *itree, 0);
    //cout << "forest eval to " << x << "\n";
    sum += x;
  }
  return sum;  
}




void ForestHelper::loadTreeFromXMLRecursive(TXMLEngine* xml, XMLNodePointer_t xnode, DTree & tree, unsigned index) 
{
  assert(tree.size() > index); 
  //cout << "DEBUG:  recursive call at index " << index <<"\n";

  // Get the split information from xml.
  XMLAttrPointer_t attr = xml->GetFirstAttr(xnode);
  std::vector<std::string> splitInfo(3);
  for(unsigned int i=0; i<3; i++)
    {
      splitInfo[i] = xml->GetAttrValue(attr); 
      attr = xml->GetNextAttr(attr);  
    }
  
  // Convert strings into numbers.
  std::stringstream converter;
  Int_t splitVar;
  Double_t splitVal;
  Double_t fitVal;  

  converter << splitInfo[0];
  converter >> splitVar;
  converter.str("");
  converter.clear();
  
  converter << splitInfo[1];
  converter >> splitVal;
  converter.str("");
  converter.clear();
  
  converter << splitInfo[2];
  converter >> fitVal;
  converter.str("");
  converter.clear();
  
  //cout << "fitval:  " << fitVal << "\n";

  // Store gathered splitInfo into the node object.
  tree[index].splitVar = splitVar;
  tree[index].splitVal = splitVal;
  tree[index].fitVal   = fitVal;

  // Get the xml daughters of the current xml node. 
  XMLNodePointer_t xleft = xml->GetChild(xnode);
  XMLNodePointer_t xright = xml->GetNext(xleft);

  assert( ((xleft!=0)&&(xright!=0)) || ((xleft==0)&&(xright==0)) );
  
  // This seems potentially problematic, but leaving for now until
  // bitwise equivalence is demonstrated...
  if(xleft == 0 || xright == 0) return;

  // append two more nodes at end of tree and update this indices in this node:
  tree[index].ileft  = tree.size();
  tree[index].iright = tree[index].ileft + 1;
  tree.push_back(DTreeNode());
  tree.push_back(DTreeNode());

  // recursively handle the next two nodes:
  loadTreeFromXMLRecursive(xml, xleft, tree, tree[index].ileft);
  loadTreeFromXMLRecursive(xml, xright, tree, tree[index].iright);
}

double ForestHelper::evalTreeRecursive(const std::vector<double> & data, const DTree & tree, int index) const{
  const DTreeNode & node = tree[index];
  if ((node.ileft == 0) && (node.iright==0)){
    //cout << "terminal node:  fitVal:  " << node.fitVal << "\n";
    return node.fitVal;
  }
  assert(data.size() > (unsigned) node.splitVar);
  // by convention, nodes are either not terminal or fully terminal
  assert(node.ileft > 0);
  assert(node.iright > 0);

  //cout << "NODE:  svar: " << node.splitVar << " sval: " << node.splitVal << " data: " << data[node.splitVar] << "fit:  " << node.fitVal << "\n";
  
  if (data[node.splitVar] < node.splitVal){
    //cout << "going left to " << node.ileft << "\n";
    return evalTreeRecursive(data, tree, node.ileft);
  } else {
    //cout << "going right to " << node.iright << "\n";
    return evalTreeRecursive(data, tree, node.iright);
  }
}
