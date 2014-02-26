#include "../interface/TrackFitter.h"

TrackFitter::~TrackFitter(){
  clean();
}

TrackFitter::TrackFitter(){
  nb_layers = 4;
  sec_phi = 0;
}

TrackFitter::TrackFitter(int nb){
  nb_layers = nb;
  sec_phi = 0;
}

vector<Pattern*> TrackFitter::getFilteredPatterns(){
  vector<Pattern*> copy;
  for(unsigned int i=0;i<patterns.size();i++){
    copy.push_back(new Pattern(*patterns[i]));
  }
  return copy;
}

vector<Track*> TrackFitter::getTracks(){
  vector<Track*> copy;
  for(unsigned int i=0;i<tracks.size();i++){
    copy.push_back(new Track(*tracks[i]));
  }
  return copy;
}

void TrackFitter::clean(){
  for(unsigned int i=0;i<patterns.size();i++){
    delete patterns[i];
  }
  patterns.clear();

 for(unsigned int i=0;i<tracks.size();i++){
    delete tracks[i];
  }
  tracks.clear();
}

void TrackFitter::addPattern(Pattern* p){
  patterns.push_back(new Pattern(*p));
}

void TrackFitter::setPhiRotation(double rot){
  sec_phi = rot;
}

double TrackFitter::getPhiRotation(){
  return sec_phi;
}

void TrackFitter::setSectorID(int id){
  sector_id = id;
}
