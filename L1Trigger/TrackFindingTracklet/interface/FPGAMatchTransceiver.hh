//This class implementes the projection tranceiver
#ifndef FPGAMATCHTRANSCEIVER_H
#define FPGAMATCHTRANSCEIVER_H

#include "FPGAProcessBase.hh"

using namespace std;

class FPGAMatchTransceiver:public FPGAProcessBase{

public:

  FPGAMatchTransceiver(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="matchout1"||
	output=="matchout2"||
	output=="matchout3"||
	output=="matchout4"||
	output=="matchout5"){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      outputmatches_.push_back(tmp);
      return;
    }
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="proj1in"||
	input=="proj2in"||
	input=="proj3in"||
	input=="proj4in"||
	input=="proj5in"||
	input=="proj6in"||
	input=="proj7in"||
	input=="proj8in"||
	input=="proj9in"||
	input=="proj10in"||
	input=="proj11in"||
	input=="proj12in"||
	input=="proj13in"||
	input=="proj14in"||
	input=="proj15in"||
	input=="proj16in"
	){
      FPGAFullMatch* tmp=dynamic_cast<FPGAFullMatch*>(memory);
      assert(tmp!=0);
      inputmatches_.push_back(tmp);
      return;
    }

    assert(0);
  }

  //this->inputmatches_ to other->outputmatches_ 

  void execute(FPGAMatchTransceiver* other){
    int count=0;
    //assert(inputmatches_.size()==3);
    for(unsigned int i=0;i<inputmatches_.size();i++){
      //cout << "InputMatch name : "<<inputmatches_[i]->getName() <<endl;
      //cout << getName()<<" output match size : "<<other->outputmatches_.size()<<endl;
      for(unsigned int l=0;l<inputmatches_[i]->nMatches();l++){
	FPGATracklet* tracklet=inputmatches_[i]->getMatch(l).first;
	//cout << "FPGAMatchTransceiver "<<getName()<<" seed layer and disk : "<<tracklet->layer()<<" "<<tracklet->disk()<<endl;
	int nMatches=0;
	for(unsigned int j=0;j<other->outputmatches_.size();j++){
	  assert(other->outputmatches_[j]!=0);
	  //cout << "OutputMatch name : "<<outputmatches_[j]->getName() <<endl;
	  string subname=outputmatches_[j]->getName().substr(3,4);
	  //cout << "FPGAMatchTransceiver "<<getName()<<" target subname = "<<subname<<endl;
	  if ((subname=="D1L1"&&tracklet->layer()==1&&abs(tracklet->disk())==1)||
	      (subname=="D1L2"&&tracklet->layer()==2)||  //dangerous to only check layer
	      (subname=="D1D2"&&tracklet->layer()==0&&abs(tracklet->disk())==1)||
	      (subname=="D3D4"&&tracklet->layer()==0&&abs(tracklet->disk())==3)||
	      (subname=="L1L2"&&tracklet->layer()==1&&tracklet->disk()==0)||
	      (subname=="L3L4"&&tracklet->layer()==3&&tracklet->disk()==0)||
	      (subname=="L5L6"&&tracklet->layer()==5&&tracklet->disk()==0)
	      ) {
	    other->outputmatches_[j]->addMatch(inputmatches_[i]->getMatch(l));
	    count++;
	    nMatches++;
	  }
	}
	assert(nMatches==1);
      }
    }
    if (writeMatchTransceiver) {
      static ofstream out("matchtransceiver.txt");
      out << getName() << " " 
	  << count << endl;
    }
  }
  

private:

  vector<FPGAFullMatch*> inputmatches_;

  vector<FPGAFullMatch*> outputmatches_;

};

#endif
