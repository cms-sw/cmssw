// $Id: SprRootReader.cc,v 1.5 2007/10/30 18:56:14 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRootReader.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprEmptyFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPreFilter.hh"

#include <TFile.h>
#include <TTree.h>
#include <TLeaf.h>
#include <TObjArray.h>

#include <stdlib.h>
#include <utility>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <memory>
#include <algorithm>
#include <iterator>

using namespace std;

SprRootReader::SprRootReader(SprPreFilter* filter)
  : 
  SprAbsReader(filter),
  treeNames_(),
  treeClasses_(),
  leafNames_(),
  fileObjects_(),
  hasSpecialClassifier_(false),
  classifierVarName_()
{}

// parses the text file to read names of root files
// defers reading of those to readRootObjects()
SprAbsFilter* SprRootReader::read(const char* filename)
{
  ifstream file(filename);
  if (not file) {
    cerr << "Unable to open " << filename << endl;
    return 0;
  }
  
  string line;
  double weight = 1.0;
  
  // if the weight is never set, we can save some time
  bool weightHasChanged = false;
  cout << "Parsing File: " << filename << endl;
  
  while (getline(file, line)) {
    if (line.find('#') != string::npos) {
      line.erase(line.find_first_of('#'));
    }
    if (line.find_first_not_of(' ') == string::npos)
      continue;

    istringstream inString(line);
    vector<string> lineFields;
    string fieldDummy;
    while (inString >> fieldDummy)
      lineFields.push_back(fieldDummy);
    
    assert( lineFields.size() > 1 );

    if      (lineFields.at(0) == "Tree:") {
      assert( treeNames_.empty() );
      copy(&lineFields[1],&lineFields[lineFields.size()],
	   back_inserter(treeNames_));
      assert( !treeNames_.empty() );
    }
    else if (lineFields.at(0) == "TreeClass:") {
      assert( treeClasses_.empty() );
      for( unsigned int i=1;i<lineFields.size();i++ )
	treeClasses_.push_back(atoi(lineFields[i].c_str()));
      if( treeNames_.size() != treeClasses_.size() ) {
	cerr << "If you supply TreeClass, you must supply as many " 
	     << "tree classes as you supplied trees, one per tree." << endl;
	return 0;
      }
    }      
    else if (lineFields.at(0) == "ClassVariable:") {
      //Accept variable name of TrueClass	
      if(hasSpecialClassifier_){
	cout<<"WARNING - True class variable was already chosen as "
	    <<classifierVarName_<<" will be overwritten to "
	    <<lineFields.at(1)
	    <<"\nPlease change your Run File"<<endl;
      }	  	  
      hasSpecialClassifier_ = true;
      classifierVarName_ = lineFields.at(1);
    } 
    else if (lineFields.at(0) == "WeightVariable:") {
      //Accept variable name of TrueClass
      weightHasChanged = true;
      assert( weightLeafNames_.empty() );
      copy(&lineFields[1],&lineFields[lineFields.size()],
	   back_inserter(weightLeafNames_)); 
    } 
    else if (lineFields.at(0) == "Leaves:") {
      copy(&lineFields[1],&lineFields[lineFields.size()],
	   back_inserter(leafNames_)); 
    } 
    else if (lineFields.at(0) == "Weight:") {
      weightHasChanged = true;
      istringstream s(lineFields.at(1));
      s >> weight;
    } 
    else if (lineFields.at(0) == "File:") {

      assert( lineFields.size() > 1 );

      FileInfo thisFile;
      thisFile.name = lineFields.at(1);

      thisFile.start = 0;
      thisFile.end = -1;

      if( lineFields.size() > 2 ) {
	istringstream 
	  dummyIn(string(lineFields.at(2),
			 0,
			 lineFields.at(2).find_first_of('-')));
	if( !(dummyIn >> thisFile.start) ) {
	  thisFile.start = 0;
	}
	dummyIn.clear();
	dummyIn.str(string(lineFields.at(2), 
			   lineFields.at(2).find_first_of('-')+1, 
			   string::npos));
	if( !(dummyIn >> thisFile.end) ) {
	  thisFile.end = -1;
	}
	dummyIn.clear();
      }

      thisFile.fileClass = 0;
      
      if( lineFields.size() > 3 ) {
	istringstream dummyIn(lineFields.at(3));
	if (not (dummyIn >> thisFile.fileClass)) {
	  thisFile.fileClass = 0;
	  cout << dummyIn.get();
	}
      }
      
      thisFile.weight = weight;
      fileObjects_.push_back(thisFile);
      
      cout << "Found file: " << thisFile.name
	   << " start: " << thisFile.start
	   << " end: " << thisFile.end
	   << " class: " << thisFile.fileClass
	   << " weight: " << thisFile.weight
	   << endl;
    }
  }
  
  if(hasSpecialClassifier_){
    cout << "True class value is given by leaf " 
	 << classifierVarName_ << endl;
  }

  if(weightLeafNames_.size()){
    cout<<"A variable determined weight has been chosen, the value"
	<<" assigned to ";
    for(unsigned int i = 0; i < weightLeafNames_.size(); i++){
      if(i%5 == 0) cout<<"\n\t";
      if(i == 0)  cout<<weightLeafNames_[i];
      else cout<<" * "<<weightLeafNames_[i];
    }
    cout<<"\n will be used for the weight."<<endl;
  }
  
  return readRootObjects(weightHasChanged);
}

SprAbsFilter* SprRootReader::readRootObjects(bool needToCalcWeights)
{
  auto_ptr<SprData> data(new SprData);
  vector<double> weights;

  // set up pre-filter vars
  if( filter_!=0 && !filter_->setVars(leafNames_) ) {
    cerr << "Unable to apply pre-filter requirements." << endl;
    return 0;
  }
  
  // get a new list of variables
  vector<string> transformed;
  if( filter_ != 0 ) {
    if( !filter_->transformVars(leafNames_,transformed) ) {
      cerr << "Pre-filter is unable to transform variables." << endl;
      return 0;
    }
  }
  if( transformed.empty() ) transformed = leafNames_; 
  
  // set up data vars
  if( !data->setVars(transformed) ) {
    cerr << "Unable to set variable list for input data." << endl;
    return 0;
  }

  // loop over files
  for( vector<FileInfo>::const_iterator fileIter = fileObjects_.begin(); 
       fileIter != fileObjects_.end(); ++fileIter) {
    TFile f(fileIter->name.c_str());

    // loop over trees
    for( vector<string>::const_iterator treeIter = treeNames_.begin();
	 treeIter != treeNames_.end(); ++treeIter ) {
      TTree* tree = dynamic_cast<TTree*>(f.Get(treeIter->c_str()));
      if( tree == 0 ) {
	cout<< "Tree " << treeIter->c_str() << " not found in file "
	    << fileIter->name.c_str() << endl;
	continue;
      }
      int istart = fileIter->start;
      int iend   = fileIter->end;
      if( iend < 0 ) iend = tree->GetEntries();
      cout << "Reading File: " << fileIter->name.c_str()
	   << " for Tree: " << treeIter->c_str()
	   << " (" << iend-istart << " events)" << endl;
      map<string, TLeaf*> leaves;

      // leaves
      for (vector<string>::const_iterator leafIter = leafNames_.begin(); 
	   leafIter != leafNames_.end(); ++leafIter) {

	TLeaf* tempLeaf = tree->GetLeaf(leafIter->c_str());
	
	if(tempLeaf == 0){
	  cerr << "No Leaf associated with variable "
	       << leafIter->c_str() << " ...aborting." <<endl;
	  abort();
	}
	leaves.insert(make_pair(*leafIter, tempLeaf));
      }

      // events
      for (int iEvent=istart; iEvent<iend; ++iEvent) {
	if( tree->GetEntry(iEvent) <= 0 ) {
	  cerr << "Unable to read event " << iEvent 
	       << " from tree " << treeIter->c_str() 
	       << " in file " << fileIter->name.c_str() 
	       << ". Aborting event loop." << endl;
	  break;
	}
	vector<double> row;
	for (vector<string>::const_iterator leafIter = leafNames_.begin(); 
	     leafIter != leafNames_.end(); ++leafIter) {
	  // Take always the first entry
	  row.push_back(leaves[*leafIter]->GetValue(0));
	}

	// get class
	int assignedClass = fileIter->fileClass;
	// TreeClass overrides fileClass
	if( !treeClasses_.empty() ) {
	  int nTree = treeIter - treeNames_.begin();
	  assignedClass = treeClasses_[nTree];
	}
	// special ClassVariable overrides fileClass and TreeClass
	if( hasSpecialClassifier_ ) {
	  TLeaf* classLeaf = tree->GetLeaf(classifierVarName_.c_str());
	  if( classLeaf == 0 ) {
	    cerr << "No Leaf associated with classifier variable. Aborting."
		 << endl;
	    abort();
	  }
	  else {
	    assignedClass = (int) classLeaf->GetValue(0);
	  }
	}
    
	float assignedWeight = fileIter->weight;
	for (unsigned int i = 0; i < weightLeafNames_.size(); i++) {   
	  TLeaf* tempLeaf = tree->GetLeaf(weightLeafNames_[i].c_str());
	  if(tempLeaf == 0){
	    cerr<<"No Leaf associated with variable "
		<<weightLeafNames_[i]<<" - probably"
		<<" a typo - please fix this"<<endl;
	    abort();
	  }
	  else {
	    assignedWeight *= (float) tempLeaf->GetValue(0);
	  } 
	}

	// pass filter?
	if( filter_!=0 && !filter_->pass(assignedClass,row) ) continue;
	
	// compute user-defined class
	if( filter_!=0 ) {
	  pair<int,bool> computedClass = filter_->computeClass(row);
	  if( computedClass.second ) 
	    assignedClass = computedClass.first;
	}
	
	// transform coordinates
	if( filter_ != 0 ) {
	  vector<double> vNew;
	  if( filter_->transformCoords(row,vNew) )
	    data->insert(assignedClass,vNew);
	  else {
	    cerr << "Pre-filter is unable to transform coordinates." << endl;
	    return 0;
	  }
	}
	else
	  data->insert(assignedClass,row);
	
	weights.push_back(assignedWeight);
      }
    }
    f.Close();
  }

  // exit
  if (needToCalcWeights)
    return new SprEmptyFilter(data.release(), weights, true);
  return new SprEmptyFilter(data.release(), true);
}


bool SprRootReader::chooseVars(const std::set<std::string>& vars)
{
  cerr << "Unable to choose variables: "
       << "SprRootReader::chooseVars() not implemented." << endl;
  return false;
}


bool SprRootReader::chooseAllBut(const std::set<std::string>& vars)
{
  cerr << "Unable to choose variables: "
       << "SprRootReader::chooseAllBut() not implemented." << endl;
  return false;
}


void SprRootReader::chooseAll()
{
  vector<FileInfo>::iterator fileIter = fileObjects_.begin();
  TFile f(fileIter->name.c_str());
  TTree* tree = dynamic_cast<TTree*>(f.Get(treeNames_[0].c_str()));
    
  if(tree == 0) {
    cerr << "Tree " << treeNames_[0] << " not found in file "
	 << fileIter->name.c_str() << endl;
    cerr << "No variables will be selected." << endl;
    return;
  }
  
  TObjArray* leafArray = tree->GetListOfLeaves();
  TIter leafIter(leafArray);
  leafIter.Reset();
  
  leafNames_.clear();
  TLeaf* thisLeaf = 0;
  while( (thisLeaf = (TLeaf*)leafIter.Next()) != 0 ){
    leafNames_.push_back(thisLeaf->GetName());
  }
}
