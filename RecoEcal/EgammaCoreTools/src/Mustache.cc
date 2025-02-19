#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "TMath.h"
using namespace std;
using namespace reco;

void Mustache::MustacheID(const reco::SuperCluster& sc, int & nclusters, float & EoutsideMustache)
{
  CaloClusterPtrVector clusters;

  for(CaloCluster_iterator iter = sc.clustersBegin(); iter != sc.clustersEnd(); ++iter){
    clusters.push_back(*iter);
  }

  MustacheID(clusters, nclusters, EoutsideMustache);
}

void Mustache::MustacheID(CaloClusterPtrVector& clusters, int & nclusters, float & EoutsideMustache) {
  std::vector<const CaloCluster*> myClusters;
  unsigned myNclusters(clusters.size());
  for(unsigned icluster=0;icluster<myNclusters;++icluster) {
    myClusters.push_back(&(*clusters[icluster]));
  }
  MustacheID(myClusters,nclusters,EoutsideMustache);
}

void Mustache::MustacheID(std::vector<const CaloCluster*>clusters, int & nclusters, float & EoutsideMustache)
{

  nclusters = 0;
  EoutsideMustache = 0;
  
  unsigned int ncl = clusters.size();
  if(!ncl) return;

  //loop over all clusters to find the one with highest energy
  float emax = 0;
  int imax = -1;
  for(unsigned int i=0; i<ncl; ++i){
    float e = (*clusters[i]).energy();
    if(e > emax){
      emax = e;
      imax = i;
    }
  }

  if(imax<0) return;
  float eta0 = (*clusters[imax]).eta();
  float phi0 = (*clusters[imax]).phi();
  
  //==== Parameters for Mustache ID  =====================
  float p00 = -0.107537;
  float p01 = 0.590969;
  float p02 = -0.076494;
  float p10 = -0.0268843;
  float p11 = 0.147742;
  float p12 = -0.0191235;
  
  float w00 = -0.00571429;
  float w01 = -0.002;
  float w10 = 0.0135714;
  float w11 = 0.001;

  float deta, dphi;
  float upper_cut, lower_cut;
  float b_upper, b_lower;
  float a_upper, a_lower;
  float curv_low, curv_up;
  float midpoint;

  for(unsigned int k=0; k<ncl; k++){
    deta = 0.0; 
    dphi = 0.0; 
    upper_cut = 0.0;
    lower_cut = 0.0;
    b_upper = 0.0;
    b_lower = 0.0;
    a_upper = 0.0;
    a_lower = 0.0;
    curv_low = 0.0;
    curv_up = 0.0;
    midpoint = 0.0;  

    deta = sin(phi0)*((*clusters[k]).eta()-eta0);	
    dphi = (*clusters[k]).phi()-phi0;
    //if(dphi> 3.1415927) dphi -= 6.2832;
    if(dphi> TMath::Pi()) dphi -= 2* TMath::Pi();
    if(dphi<-1* TMath::Pi()) dphi +=2* TMath::Pi();
    
    //2 parabolas (upper and lower) 
    //of the form: y = a*x*x + b      
    
    //b comes from a fit to the width
    //and has a slight dependence on Et on the upper edge
    b_lower = w00*sin(eta0)*eta0 + w01 / sqrt(log10((*clusters[k]).energy())+1.1);
    b_upper = w10*sin(eta0)*eta0 + w11 / sqrt(log10((*clusters[k]).energy())+1.1);
    
    //here make an adjustment to the width for the offset from 0.
    midpoint = b_upper - (b_upper-b_lower)/2.;
    b_lower = b_lower - midpoint;
    b_upper = b_upper - midpoint;
    
    //the curvature comes from a parabolic 
    //fit for many slices in eta given a 
    //slice -0.1 < log10(Et) < 0.1
    curv_up = p00*pow(eta0*sin(eta0),2)+p01*eta0*sin(eta0)+p02;
    curv_low = p10*pow(eta0*sin(eta0),2)+p11*eta0*sin(eta0)+p12;
    
    //solving for the curviness given the width of this particular point
    a_lower = (1/(4*curv_low))-fabs(b_lower);
    a_upper = (1/(4*curv_up))-fabs(b_upper);
    
    upper_cut =(1./(4.*a_upper))*pow(dphi,2)+b_upper;
    lower_cut =(1./(4.*a_lower))*pow(dphi,2)+b_lower;
    
  
    if (!(deta < upper_cut && deta > lower_cut)){
      nclusters++;
      EoutsideMustache += (*clusters[k]).energy();
    }
    
  }
}

void Mustache::MustacheClust(std::vector<CaloCluster> clusters, std::vector<unsigned int>& insideMust, std::vector<unsigned int>& outsideMust){  
  unsigned int ncl = clusters.size();
  if(!ncl) return;
  
  //loop over all clusters to find the one with highest energy
  float emax = 0;
  int imax = -1;
  for(unsigned int i=0; i<ncl; ++i){
    float e = (clusters[i]).energy();
    if(e > emax){
      emax = e;
      imax = i;
    }
  }
  
  if(imax<0) return;
  float eta0 = (clusters[imax]).eta();
  float phi0 = (clusters[imax]).phi();
  
  //==== Parameters for Mustache ID  =====================
  float p00 = -0.107537;
  float p01 = 0.590969;
  float p02 = -0.076494;
  float p10 = -0.0268843;
  float p11 = 0.147742;
  float p12 = -0.0191235;
  
  float w00 = -0.00571429;
  float w01 = -0.002;
  float w10 = 0.0135714;
  float w11 = 0.001;
  
  float deta, dphi;
  float upper_cut, lower_cut;
  float b_upper, b_lower;
  float a_upper, a_lower;
  float curv_low, curv_up;
  float midpoint;
  
  for(unsigned int k=0; k<ncl; k++){
    deta = 0.0; 
    dphi = 0.0; 
    upper_cut = 0.0;
    lower_cut = 0.0;
    b_upper = 0.0;
    b_lower = 0.0;
    a_upper = 0.0;
    a_lower = 0.0;
    curv_low = 0.0;
    curv_up = 0.0;
    midpoint = 0.0;  

    deta = sin(phi0)*((clusters[k]).eta()-eta0);	
    dphi = (clusters[k]).phi()-phi0;
    //if(dphi> 3.1415927) dphi -= 6.2832;
    if(dphi> TMath::Pi()) dphi -= 2* TMath::Pi();
    if(dphi<-1* TMath::Pi()) dphi +=2* TMath::Pi();
    
    //2 parabolas (upper and lower) 
    //of the form: y = a*x*x + b      
    
    //b comes from a fit to the width
    //and has a slight dependence on Et on the upper edge
    b_lower = w00*sin(eta0)*eta0 + w01 / sqrt(log10((clusters[k]).energy())+1.1);
    b_upper = w10*sin(eta0)*eta0 + w11 / sqrt(log10((clusters[k]).energy())+1.1);
    
    //here make an adjustment to the width for the offset from 0.
    midpoint = b_upper - (b_upper-b_lower)/2.;
    b_lower = b_lower - midpoint;
    b_upper = b_upper - midpoint;
    
    //the curvature comes from a parabolic 
    //fit for many slices in eta given a 
    //slice -0.1 < log10(Et) < 0.1
    curv_up = p00*pow(eta0*sin(eta0),2)+p01*eta0*sin(eta0)+p02;
    curv_low = p10*pow(eta0*sin(eta0),2)+p11*eta0*sin(eta0)+p12;
    
    //solving for the curviness given the width of this particular point
    a_lower = (1/(4*curv_low))-fabs(b_lower);
    a_upper = (1/(4*curv_up))-fabs(b_upper);
    
    upper_cut =(1./(4.*a_upper))*pow(dphi,2)+b_upper;
    lower_cut =(1./(4.*a_lower))*pow(dphi,2)+b_lower;
    
    //return indices of Clusters outside the Mustache
    if (!(deta < upper_cut && deta > lower_cut)){
      outsideMust.push_back(k);
    }
    else{//return indices of Clusters inside the Mustache
      insideMust.push_back(k);
    }
  }
}

void Mustache::FillMustacheVar(std::vector<CaloCluster>clusters){
  Energy_In_Mustache_=0;
  Energy_Outside_Mustache_=0;
  LowestClusterEInMustache_=0;
  excluded_=0;
  included_=0;
  std::multimap<float, unsigned int>OrderedClust;
  std::vector<unsigned int> insideMust;
  std::vector<unsigned int> outsideMust;
  MustacheClust(clusters, insideMust, outsideMust);
  included_=insideMust.size(); excluded_=outsideMust.size();
  for(unsigned int i=0; i<insideMust.size(); ++i){
    unsigned int index=insideMust[i];
    Energy_In_Mustache_=clusters[index].energy()+Energy_In_Mustache_;
    OrderedClust.insert(make_pair(clusters[index].energy(), index));
  }
  for(unsigned int i=0; i<outsideMust.size(); ++i){
    unsigned int index=outsideMust[i];
    Energy_Outside_Mustache_=clusters[index].energy()+Energy_Outside_Mustache_;
    Et_Outside_Mustache_=clusters[index].energy()*sin(clusters[index].position().theta())
      +Et_Outside_Mustache_;
  }
  std::multimap<float, unsigned int>::iterator it;
  it=OrderedClust.begin();
  unsigned int lowEindex=(*it).second; 
  LowestClusterEInMustache_=clusters[lowEindex].energy();
  
}
