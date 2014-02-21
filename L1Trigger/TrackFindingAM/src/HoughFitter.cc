#include "../interface/HoughFitter.h"

HoughFitter::HoughFitter():TrackFitter(0){
  cuts = new HoughCut();
  ch = new ComputerHough(cuts);
  h_x= new float[1024];
  h_y= new float[1024];
  h_z= new float[1024];
  h_layer= new unsigned int[1024];
  ch->DefaultCuts();
}

HoughFitter::HoughFitter(int nb):TrackFitter(nb){
  cuts = new HoughCut();
  ch = new ComputerHough(cuts);
  h_x= new float[1024];
  h_y= new float[1024];
  h_z= new float[1024];
  h_layer= new unsigned int[1024];
  ch->DefaultCuts();
}

HoughFitter::~HoughFitter(){
  delete ch;
  delete cuts;
  delete [] h_x;
  delete [] h_y;
  delete [] h_z;
  delete [] h_layer;
}

void HoughFitter::initialize(){

}

void HoughFitter::mergePatterns(){
  //cout<<"Merging of patterns not implemented"<<endl;
}

void HoughFitter::mergeTracks(){
  unsigned int index = 0;
  vector<Track*>::iterator it = tracks.begin();
  while(it!=tracks.end()){
    Track* newTrack = *it;
    bool found = false;
    for(unsigned int i=0;i<index;i++){
      Track* ref = tracks[i];
      float dpt,dphi,dz,deta;
      dpt = fabs(newTrack->getCurve()-ref->getCurve());
      dphi = fabs(newTrack->getPhi0()-ref->getPhi0());
      dz = fabs(newTrack->getZ0()-ref->getZ0());
      deta = fabs(newTrack->getEta0()-ref->getEta0());
      found = (deta<0.02) &&
	(dphi<0.005) &&
	(dpt<0.1) &&
	(dz<0.3);
      if(found)
	break;
    }
    if(found)
      tracks.erase(it);
    else{
      index++;
      it++;
    }
  }
}

void HoughFitter::fit(vector<Hit*> hits){
  if(hits.size()>1024){
    cout<<"ERROR : too many stubs for fitting!"<<endl;
    return;
  }

  memset(h_x,0,1024*sizeof(float));
  memset(h_y,0,1024*sizeof(float));
  memset(h_z,0,1024*sizeof(float));
  memset(h_layer,0,1024*sizeof(unsigned int));
 
  set<int> layers;
  bool barrel = true;

  //Collect the stubs
  for(unsigned int j=0;j<hits.size();j++){
    h_x[j]=hits[j]->getX();
    h_y[j]=hits[j]->getY();
    h_z[j]=hits[j]->getZ();

    // we tag the stubs we want to use for the linear regression (pixel stubs)
    uint16_t zinfo=0;
    unsigned int layer = hits[j]->getLayer();
    if (layer<=7) 
      zinfo=1;//pixel on barrel
    if (layer>10 && hits[j]->getLadder()<9)
      zinfo=2;//pixel on endcap
    h_layer[j]= layer | (zinfo<<16) | (hits[j]->getID()<<18);
    //check if we are in barrel, hybrid or endcap sector
    if(layer>10){
      barrel = false;
      layers.insert(layer);
    }
  }

  int sectorType = 0;
  if(!barrel){
    float sum=0;
    for(set<int>::iterator it=layers.begin();it!=layers.end();it++){
      sum+=*it;
    }
    sum=sum/(float)layers.size();
    if((sum>=11 && sum<=12) || (sum>=18 && sum<=19))
      sectorType=1;
    else
      sectorType=2;
  }

  int sectorValue = 0;
  if(sectorType==0)
    sectorValue=24+(sector_id%4);//barrel
  if(sectorType==1)
    sectorValue=8+(sector_id%4);//hybrid
  if(sectorType==2)
    sectorValue=0+(sector_id%4);//endcap

  //Actual computing
  ch->ComputeOneShot(sectorValue,hits.size(),h_x,h_y,h_z,h_layer);
  std::vector<mctrack_t> &v=ch->getCandidates();

  //format the results and feed the tracks vector
  for (std::vector<mctrack_t>::iterator it=v.begin();it!=v.end();it++){
    Track* fit_track = new Track((*it).pt, 0, (*it).phi, (*it).eta, (*it).z0);
    for(unsigned int i=0;i<(*it).layers.size();i++){
      unsigned int value = (*it).layers[i];
      short id = value>>18;
      fit_track->addStubIndex(id);
    }
    tracks.push_back(fit_track);
  }
  
}

void HoughFitter::fit(){

  vector<Hit*> activatedHits;

  //////// Get the list of unique stubs from all the patterns ///////////
  set<int> ids;
  int total=0;
  for(unsigned int i=0;i<patterns.size();i++){
    vector<Hit*> allHits = patterns[i]->getHits();
    total+=allHits.size();
    for(unsigned int j=0;j<allHits.size();j++){
      pair<set<int>::iterator,bool> result = ids.insert(allHits[j]->getID());
      if(result.second==true)
	activatedHits.push_back(allHits[j]);
    }
  }

  fit(activatedHits);
 
}

TrackFitter* HoughFitter::clone(){
  HoughFitter* fit = new HoughFitter(nb_layers);
  fit->setPhiRotation(sec_phi);
  fit->setSectorID(sector_id);
  return fit;
}
