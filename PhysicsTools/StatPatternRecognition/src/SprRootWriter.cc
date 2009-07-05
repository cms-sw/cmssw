//$Id: SprRootWriter.cc,v 1.3 2007/10/12 19:25:01 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprRootWriter.hh"
#include <TFile.h>

#include <stdlib.h>
#include <iostream>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <algorithm>

using namespace std;


SprRootWriter::~SprRootWriter() 
{
  delete [] data_;
  delete tuple_;
}


bool SprRootWriter::init(const char* filename)
{
  // init
  fname_ = filename;
  string cmd;

  // check if file exists, delete and issue a warning
  struct stat buf;
  if( stat(fname_.c_str(),&buf) == 0 ) {
    cerr << "Warning: file " << fname_.c_str() << " will be deleted." << endl;
    cmd = "rm -f ";
    cmd += fname_.c_str();
    if( system(cmd.c_str()) != 0 ) {
      cerr << "Attempt to delete file " << fname_.c_str() 
	   << " terminated with error " << errno << endl;
      return false;
    }
  }

  // exit
  return true;
}


int SprRootWriter::SetBranches()
{
  // initial check
  if(setBranches_) {
    cerr <<"DANGER - already initialized - branch structure in danger - ABORT"
	 <<endl;
    abort();
  }
  if( data_ != 0 ) {
    cerr << "Root data has been already filled - abort."<< endl;
    abort();
  }

  // make data pointer
  int size = axes_.size()+3;
  data_ = new Float_t[size];

  // make axis names
  TString values = "index/F:classification/F:weight/F";
  for(unsigned int i = 0; i < axes_.size(); i++) {
    values += ":";
    string temp = axes_[i];
    replace(temp.begin(),temp.end(),'/','_');
    values += temp.c_str();
    values += "/F";
  }

  // book tree
  if( tuple_ != 0 ) {
    cerr << "Root tree has been already booked - abort." << endl;
    abort();
  }
  tuple_ = new TTree("ClassRecord", "Classification Filling Information");
  tuple_->Branch("Vars", data_, values);

  // exit
  setBranches_ = true;
  return 1;
}


bool SprRootWriter::write(int cls, unsigned index, double weight,
			  const std::vector<double>& v, 
			  const std::vector<double>& f)
{
  if(!setBranches_)
    SetBranches();

  // check vector sizes
  unsigned int check = v.size() + f.size();
  if(check != axes_.size()){
    cerr << "Dimensionality of input vector unequal to dimensionality " 
	 << "of tuple: " << v.size() << " " << f.size()
	 << " " << axes_.size() << endl;
    return false;
  }

  // fill data array  
  data_[0] = index;
  data_[1] = cls;
  data_[2] = weight;
  for(unsigned int i = 0; i < v.size(); i++)
      data_[i+3] = (Float_t) v[i];
  for(unsigned int i = 0; i < f.size(); i++)
    data_[i+3+v.size()] = (Float_t) f[i];

  // fill and save the root tree
  tuple_->Fill();
  if(index%1000 == 0) tuple_->AutoSave("SaveSelf");

  // exit
  return true;
}


bool SprRootWriter::close()
{
  TFile file(fname_.c_str(), "recreate");
  file.cd();
  tuple_->Write();
  return true;
}


