#ifndef L1TTRACKS_H
#define L1TTRACKS_H

#include <iostream>
#include <assert.h>

#include "L1TTrack.hh"


using namespace std;


class L1TTracks{

public:

  L1TTracks(){

  }

  void addTrack(const L1TTrack& aTrack){
    tracks_.push_back(aTrack);
  }

  void addTracks(const L1TTracks& tracks){
    for(unsigned int i=0;i<tracks.tracks_.size();i++){
      tracks_.push_back(tracks.tracks_[i]);
    }
  }

  unsigned int size() const { return tracks_.size(); }

  L1TTrack& get(unsigned int i) { return tracks_[i];}

  vector<int> NduplicatesPerTrack() {

    vector<int> NdupPerTrack;
    vector<int> Ndups((int)tracks_.size());
    
    for (unsigned int i=0; i<tracks_.size(); i++) {
      Ndups[i] = 0;

      for (unsigned int j=i+1; j<tracks_.size(); j++) {
	if (tracks_[i].overlap(tracks_[j])) Ndups[i]++;
      }

      NdupPerTrack.push_back(Ndups[i]);
    }
    
    return NdupPerTrack;
  }
  
  L1TTracks purged() {

    vector<bool> deleted(tracks_.size());
    for(unsigned int i=0;i<tracks_.size();i++){
      deleted[i]=false;
    }
    for(unsigned int i=0;i<tracks_.size();i++){
      if (deleted[i]) continue;
      for(unsigned int j=i+1;j<tracks_.size();j++){
	if (deleted[j]) continue;
	if (tracks_[i].overlap(tracks_[j])) {
	  //double fractioni=0.0;
	  //tracks_[i].simtrackid(fractioni);
	  //double fractionj=0.0;
	  //tracks_[j].simtrackid(fractionj);
	  //if (fabs(tracks_[i].rinv())>fabs(tracks_[j].rinv())) {
	  //if (fractioni<fractionj) {
	  if ((tracks_[i].npixelstrip()<2)&&(tracks_[j].npixelstrip()>=2)){
	    deleted[i]=true;
	    continue;
	  }
	  if ((tracks_[j].npixelstrip()<2)&&(tracks_[i].npixelstrip()>=2)){
	    deleted[j]=true;
	    continue;
	  }
	  if ((tracks_[i].npixelstrip()<3)&&(tracks_[j].npixelstrip()>=3)){
	    deleted[i]=true;
	    continue;
	  }
	  if ((tracks_[j].npixelstrip()<3)&&(tracks_[i].npixelstrip()>=3)){
	    deleted[j]=true;
	    continue;
	  }
	  if ((tracks_[i].npixelstrip()<4)&&(tracks_[j].npixelstrip()>=4)){
	    deleted[i]=true;
	    continue;
	  }
	  if ((tracks_[j].npixelstrip()<4)&&(tracks_[i].npixelstrip()>=4)){
	    deleted[j]=true;
	    continue;
	  }



	  if (tracks_[i].chisqdof()>tracks_[j].chisqdof()) {
	    deleted[i]=true;
	  } else {
	    deleted[j]=true;
	  }
	  continue;
	}
      }
      if (deleted[i]) continue;
    }      

    L1TTracks tmp;
	
    for(unsigned int i=0;i<tracks_.size();i++){
      if (!deleted[i]){
	tmp.addTrack(tracks_[i]);
      }
    }

    return tmp;
    
	  

   
  }
    

  void clean(){
    tracks_.clear();
  }

private:

  vector<L1TTrack> tracks_;

};



#endif



