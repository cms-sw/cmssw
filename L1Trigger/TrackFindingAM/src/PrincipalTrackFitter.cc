#include "../interface/PrincipalTrackFitter.h"

PrincipalTrackFitter::PrincipalTrackFitter():TrackFitter(){
  threshold = 1000;
}

PrincipalTrackFitter::PrincipalTrackFitter(int nb, int t):TrackFitter(nb){
  threshold = t;
  sec_phi = 0;
}

PrincipalTrackFitter::~PrincipalTrackFitter(){
  for(map<string, FitParams*>::iterator itr = params.begin(); itr != params.end(); ++itr){
    delete itr->second;
  }
}

void PrincipalTrackFitter::initialize(){

}

void PrincipalTrackFitter::mergePatterns(){
  //cout<<"Merging of patterns not implemented"<<endl;
}

void PrincipalTrackFitter::mergeTracks(){
  //cout<<"Merging of Tracks not implemented"<<endl;
}

void PrincipalTrackFitter::fit(vector<Hit*> hits){
  //cout<<"fit(vector<Hit*>) not implemented"<<endl;
}

void PrincipalTrackFitter::fit(){
  for(unsigned int i=0;i<patterns.size();i++){
    ostringstream oss;
    oss<<std::setfill('0');
    Pattern* p = patterns[i];
    for(int j=0;j<nb_layers;j++){
      oss<<setw(2)<<j;
      oss<<setw(2)<<p->getLayerStrip(j)->getPhi();
      if(j!=nb_layers-1)
	oss<<"-";
    }
    cout<<"ladder : "<<oss.str()<<endl;
    map<string, FitParams*>::iterator it = params.find(oss.str());
    if(it==params.end()){//not found
      cout<<"parametres de fit non trouves"<<endl;
    }
    else{
      FitParams* fit = it->second;
      vector<Hit*> active_hits = p->getHits();
      double coords[active_hits.size()*3];
      double coords_PCA[active_hits.size()*3];
      //cout<<sec_phi<<endl;
      for(unsigned int i=0;i<active_hits.size();i++){
	coords[i*3]=active_hits[i]->getX()*cos(sec_phi)+active_hits[i]->getY()*sin(sec_phi);
	coords[i*3+1]=-active_hits[i]->getX()*sin(sec_phi)+active_hits[i]->getY()*cos(sec_phi);
	coords[i*3+2]=active_hits[i]->getZ();
	cout<<*active_hits[i]<<endl;
      }
      double chi2 = fit->get_chi_square(coords,4);
      cout<<"Erreur : "<<chi2<<endl;
      fit->x2p(coords,coords_PCA);
      Track* fit_track = fit->getTrack(coords_PCA);
      fit_track->setPhi0(fit_track->getPhi0()+sec_phi);//correction de la rotation en PHI
      cout<<"PT estime de la trace : "<<fit_track->getCurve()<<endl;
      cout<<"PHI estime de la trace : "<<fit_track->getPhi0()<<endl;
      cout<<"D0 estime de la trace : "<<fit_track->getD0()<<endl;
      cout<<"ETA estime de la trace : "<<fit_track->getEta0()<<endl;
      cout<<"Z0 estime de la trace : "<<fit_track->getZ0()<<endl;
      if(fit_track->getCurve()<-200)
	fit_track->setCurve(1.0);
      tracks.push_back(fit_track);
    }
    cout<<endl;
  }
  //  FitParams* fp;
  //map<string, FitParams*>::iterator it = params.find(oss.str());
  // if(it==params.end()){//not found
}

TrackFitter* PrincipalTrackFitter::clone(){
  PrincipalTrackFitter* fit = new PrincipalTrackFitter(nb_layers,threshold);
  fit->setPhiRotation(sec_phi);
  for(map<string, FitParams*>::iterator itr = params.begin(); itr != params.end(); ++itr){
    fit->params[itr->first]=new FitParams(*(itr->second));
  }
  
  return fit;
}

void PrincipalTrackFitter::addTrackForPrincipal(int* tracker, double* coord){
  //build the map string
  ostringstream oss;
  oss<<std::setfill('0');
  for(int i=0;i<nb_layers;i++){
    oss<<setw(2)<<tracker[i*3];
    oss<<setw(2)<<tracker[i*3+1];
    if(i!=nb_layers-1)
      oss<<"-";
  }
  FitParams* fp;
  map<string, FitParams*>::iterator it = params.find(oss.str());
  if(it==params.end()){//not found
    fp=new FitParams(nb_layers, threshold);
    params[oss.str()]=fp;
  }
  else{
    fp=it->second;
  }

  //rotation according to sec_phi
  for(int i=0;i<nb_layers;i++){
    double x = coord[i*3];
    double y = coord[i*3+1];
    coord[i*3]=x*cos(sec_phi)+y*sin(sec_phi);
    coord[i*3+1]=-x*sin(sec_phi)+y*cos(sec_phi);
  }

  //cout<<oss.str()<<endl;
  fp->addDataForPrincipal(coord);
  
}

void PrincipalTrackFitter::addTrackForMultiDimFit(int* tracker, double* coord, double* val){
  //build the map string
  ostringstream oss;
  oss<<std::setfill('0');
  for(int i=0;i<nb_layers;i++){
    oss<<setw(2)<<tracker[i*3];
    oss<<setw(2)<<tracker[i*3+1];
    if(i!=nb_layers-1)
      oss<<"-";
  }
  FitParams* fp;
  map<string, FitParams*>::iterator it = params.find(oss.str());
  if(it==params.end()){//not found
    cout<<"No sub sector found for multi dim fit!"<<endl;
    return;
  }
  else{
    fp=it->second;
  }

  //rotation according to sec_phi
  for(int i=0;i<nb_layers;i++){
    double x = coord[i*3];
    double y = coord[i*3+1];
    coord[i*3]=x*cos(sec_phi)+y*sin(sec_phi);
    coord[i*3+1]=-x*sin(sec_phi)+y*cos(sec_phi);
  }

  double phi0 = val[1]-sec_phi; // correction phi0
  val[0] = 100/(2*((val[0]/(0.3*3.833))*100)*pow(cos(phi0),3)); // correction PT
  val[1] = tan(phi0);
  val[2] = val[2]/cos(phi0);
  val[3] = sinh(val[3])/cos(phi0);

  //cout<<oss.str()<<endl;
  fp->addDataForMultiDimFit(coord, val);
  
}

bool PrincipalTrackFitter::hasPrincipalParams(){
  //bool complete=true;
  //int total=0;
  //int ok = 0;
  for(map<string, FitParams*>::iterator itr = params.begin(); itr != params.end(); ++itr){
    //total++;
    if(!itr->second->hasPrincipalParams()){
      //complete=false;
      //      cout<<"manque : "<<itr->first<<endl;
      return false;
    }
    //else
    //  ok++;
  }
  return true;
  //cout<<"Principal : "<<ok<<"/"<<total<<endl;
  //return complete;
}

void PrincipalTrackFitter::forcePrincipalParamsComputing(){
  for(map<string, FitParams*>::iterator itr = params.begin(); itr != params.end(); ++itr){
    if(itr->second->getNbPrincipalTracks()<2)
      params.erase(itr);
    else{
      if(!itr->second->hasPrincipalParams()){
	itr->second->forcePrincipalParamsComputing();
      }
    }
  }
}

bool PrincipalTrackFitter::hasMultiDimFitParams(){
  //bool complete=true;
  //int total=0;
  //int ok = 0;
  for(map<string, FitParams*>::iterator itr = params.begin(); itr != params.end(); ++itr){
    //total++;
    if(!itr->second->hasMultiDimFitParams()){
      //complete=false;
      //cout<<"manque : "<<itr->first<<endl;
      return false;
    }
    //else
    //  ok++;
  }
  return true;
  //cout<<"MultiDimFit : "<<ok<<"/"<<total<<endl;
  //return complete;
}

void PrincipalTrackFitter::forceMultiDimFitParamsComputing(){
  for(map<string, FitParams*>::iterator itr = params.begin(); itr != params.end(); ++itr){
    if(!itr->second->hasMultiDimFitParams()){
      itr->second->forceMultiDimFitParamsComputing();
    }
  }
}
