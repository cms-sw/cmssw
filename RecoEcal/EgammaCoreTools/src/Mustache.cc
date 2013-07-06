#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "TMath.h"
#include "TVector2.h"
using namespace std;

namespace reco {  
  namespace MustacheKernel {    
    bool inMustache(float maxEta, float maxPhi, 
		    float ClustE, float ClusEta, float ClusPhi){
      bool inMust=false;
      float eta0 = maxEta;
      float phi0 = maxPhi;      
      
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
      
      float deta=sin(phi0)*(ClusEta-eta0);
      float dphi=TVector2::Phi_mpi_pi(maxPhi-ClusPhi);
      //2 parabolas (upper and lower) 
      //of the form: y = a*x*x + b
      
      //b comes from a fit to the width
      //and has a slight dependence on E on the upper edge
      float b_upper= w10*sin(eta0)*eta0 + w11 / sqrt(log10(ClustE)+1.1);
      
      float b_lower=w00*sin(eta0)*eta0 + w01 / sqrt(log10(ClustE)+1.1);
      
      //here make an adjustment to the width for the offset from 0.
      float midpoint=  b_upper - (b_upper-b_lower)/2.;
      b_upper = b_upper - midpoint;
      b_lower = b_lower - midpoint;
      
      
      //the curvature comes from a parabolic 
      //fit for many slices in eta given a 
      //slice -0.1 < log10(Et) < 0.1
      float curv_up=p00*pow(eta0*sin(eta0),2)+p01*eta0*sin(eta0)+p02;
      float curv_low=p10*pow(eta0*sin(eta0),2)+p11*eta0*sin(eta0)+p12;
      
      //solving for the curviness given the width of this particular point
      float a_upper=(1/(4*curv_up))-fabs(b_upper);
      float a_lower = (1/(4*curv_low))-fabs(b_lower);
      
      float upper_cut=(1./(4.*a_upper))*pow(dphi,2)+b_upper; 
      float lower_cut=(1./(4.*a_lower))*pow(dphi,2)+b_lower;
      
      if(deta < upper_cut && deta > lower_cut) inMust=true;
      
      return inMust;
    }    
  }
  
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
  
  void Mustache::MustacheID(std::vector<const CaloCluster*>& clusters, int & nclusters, float & EoutsideMustache)
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
    
    for(unsigned int k=0; k<ncl; k++){
      bool inMust=MustacheKernel::inMustache(eta0, phi0, (*clusters[k]).energy(), (*clusters[k]).eta(), (*clusters[k]).phi());
      
      if (!(inMust)){
	nclusters++;
	EoutsideMustache += (*clusters[k]).energy();
      }
      
    }
  }
  
  void Mustache::MustacheClust(std::vector<CaloCluster>& clusters, std::vector<unsigned int>& insideMust, std::vector<unsigned int>& outsideMust){  
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
    
    
    for(unsigned int k=0; k<ncl; k++){
      
      bool inMust=MustacheKernel::inMustache(eta0, phi0, (clusters[k]).energy(), (clusters[k]).eta(), (clusters[k]).phi());
      //return indices of Clusters outside the Mustache
      if (!(inMust)){
	outsideMust.push_back(k);
      }
      else{//return indices of Clusters inside the Mustache
	insideMust.push_back(k);
      }
    }
  }
  
  void Mustache::FillMustacheVar(std::vector<CaloCluster>& clusters){
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
}
