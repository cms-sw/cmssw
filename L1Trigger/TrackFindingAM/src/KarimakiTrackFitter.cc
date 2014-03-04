#include "../interface/KarimakiTrackFitter.h"

KarimakiTrackFitter::KarimakiTrackFitter():TrackFitter(){
}

KarimakiTrackFitter::KarimakiTrackFitter(int nb):TrackFitter(nb){
}

KarimakiTrackFitter::~KarimakiTrackFitter(){
}

void KarimakiTrackFitter::initialize(){

}

void KarimakiTrackFitter::mergePatterns(){
  //cout<<"Merging of patterns not implemented"<<endl;
}

void KarimakiTrackFitter::mergeTracks(){
  //cout<<"Merging of Tracks not implemented"<<endl;
}

void KarimakiTrackFitter::fit(){
  for(unsigned int i=0;i<patterns.size();i++){
    fitPattern(patterns[i]);
  }
}

void KarimakiTrackFitter::fit(vector<Hit*> hits){
  cout<<"fit(vector<Hit*>) not implemented"<<endl;
}


void KarimakiTrackFitter::fitPattern(Pattern* p){
  double parZX[2][2];
  double resZX[2];
  double invZX[2][2];
  double detZX = 0;

  for (int i=0;i<2;++i){
    resZX[i] = 0.;
    for (int j=0;j<2;++j)
      parZX[i][j] = 0.;
    for (int j=0;j<2;++j) 
      invZX[i][j] = 0.;
  }

  double kappa = 0.;
  double delta = 0.;
  double phi   = 0.;

  double c_xx = 0.;
  double c_xy = 0.;
  double c_yy = 0.;
  double c_xr = 0.;
  double c_yr = 0.;
  double c_rr = 0.;

  int n_miss = 0;
  int n_mult = 0;
  int ngc = 0;
  int ng = 0;

  double x,y,z,rr;
 
  double wght=1.;

  double pr[5];

  double s_x     = 0.;
  double s_y     = 0.;
  double s_xy    = 0.;
  double s_xx    = 0.;
  double s_yy    = 0.;
  double s_rr    = 0.;
  double s_xrr   = 0.;
  double s_yrr   = 0.;
  double s_rrrr  = 0.;
  double s_0     = 0.;

  for (int i=0;i<5;++i)  
    pr[i] = 0.;

  for (int i=0;i<nb_layers;++i){
    vector<Hit*> active_hits = p->getHits(i);//list of hits in the pattern for this layer

    ng = active_hits.size();
    
    if (ng==0) 
      n_miss++;
    if (ng>1)
      n_mult++;   
    
    if (ng!=1) 
      continue;
    
    ngc++;
    
    (active_hits[0]->getLayer()<8)? wght = 1. : wght = 5000.;
    
    //cout<<*active_hits[0]<<endl;

    x = active_hits[0]->getX()*cos(sec_phi)+active_hits[0]->getY()*sin(sec_phi);
    y = -active_hits[0]->getX()*sin(sec_phi)+active_hits[0]->getY()*cos(sec_phi);
    z = active_hits[0]->getZ();
    
    rr = (x*x+y*y);
    
    parZX[0][0] += rr/wght;
    parZX[1][1] += 1/wght;
    parZX[1][0] += sqrt(rr)/wght;
    
    resZX[0] += sqrt(rr)*z/wght;
    resZX[1] += z/wght;
    
    s_x    += x;
    s_y    += y;
    s_xy   += x*y;
    s_xx   += x*x;
    s_yy   += y*y;
    s_rr   += rr;
    s_xrr  += x*rr;
    s_yrr  += y*rr;
    s_rrrr += rr*rr;
    s_0    += 1;
  }
  
  s_x /= s_0; 
  s_y /= s_0; 
  s_xy /= s_0; 
  s_xx /= s_0; 
  s_yy /= s_0; 
  s_rr /= s_0; 
  s_xrr /= s_0; 
  s_yrr /= s_0; 
  s_rrrr /= s_0; 
  
  c_xx = s_xx - s_x*s_x;
  c_xy = s_xy - s_x*s_y; 
  c_yy = s_yy - s_y*s_y; 
  c_xr = s_xrr - s_x*s_rr; 
  c_yr = s_yrr - s_y*s_rr; 
  c_rr = s_rrrr - s_rr*s_rr; 

  double q_1 = c_rr*c_xy - c_xr*c_yr;
  double q_2 = c_rr*(c_xx-c_yy) - c_xr*c_xr + c_yr*c_yr;

  phi   = 0.5*atan(2*q_1/q_2);
  kappa = (sin(phi)*c_xr-cos(phi)*c_yr)/c_rr;
  delta = -kappa*s_rr+sin(phi)*s_x-cos(phi)*s_y;

  detZX = parZX[0][0]*parZX[1][1]-parZX[1][0]*parZX[1][0];
 
  invZX[0][0] =  parZX[1][1]/detZX; 
  invZX[1][0] = -parZX[1][0]/detZX; 
  invZX[1][1] =  parZX[0][0]/detZX; 

  if (n_miss<=1 && n_mult==0){    
    pr[0] = 0.01/(2*kappa/sqrt(1-4*delta*kappa))*(0.3*3.833);
    pr[1] = phi;
    pr[2] = -2*delta/(1+sqrt(1-4*delta*kappa));// TODO : pourquoi signe oppose?
    pr[3] = asinh((invZX[0][0]*resZX[0] + invZX[1][0]*resZX[1])); 
    pr[4] = invZX[1][0]*resZX[0] + invZX[1][1]*resZX[1]; 
    
    Track* fit_track = new Track(pr[0], pr[2], pr[1], pr[3], pr[4]);
    fit_track->setPhi0(fit_track->getPhi0()+sec_phi);//correction de la rotation en PHI
    //cout<<"PT estime de la trace : "<<fit_track->getCurve()<<endl;
    //cout<<"PHI estime de la trace : "<<fit_track->getPhi0()<<endl;
    //cout<<"D0 estime de la trace : "<<fit_track->getD0()<<endl;
    //cout<<"ETA estime de la trace : "<<fit_track->getEta0()<<endl;
    //cout<<"Z0 estime de la trace : "<<fit_track->getZ0()<<endl;
    tracks.push_back(fit_track);
  }
}

TrackFitter* KarimakiTrackFitter::clone(){
  KarimakiTrackFitter* fit = new KarimakiTrackFitter(nb_layers);
  fit->setPhiRotation(sec_phi);
  return fit;
}
