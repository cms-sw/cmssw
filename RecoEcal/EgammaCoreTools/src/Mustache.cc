#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "TMath.h"
#include "TVector2.h"
using namespace std;

namespace reco {  
  namespace MustacheKernel {    
    bool inMustache(const float maxEta, const float maxPhi, 
		    const float ClustE, const float ClusEta, 
		    const float ClusPhi){
      //bool inMust=false;
      //float eta0 = maxEta;
      //float phi0 = maxPhi;      
      
      const float p00 = -0.107537;
      const float p01 = 0.590969;
      const float p02 = -0.076494;
      const float p10 = -0.0268843;
      const float p11 = 0.147742;
      const float p12 = -0.0191235;
      
      const float w00 = -0.00571429;
      const float w01 = -0.002;
      const float w10 = 0.0135714;
      const float w11 = 0.001;
      
      const float sineta0 = std::sin(maxEta);
      const float eta0xsineta0 = maxEta*sineta0;
      
      
      //2 parabolas (upper and lower) 
      //of the form: y = a*x*x + b
      
      //b comes from a fit to the width
      //and has a slight dependence on E on the upper edge
      const float ClustEt = ClustE/std::cosh(ClusEta);
      const float sqrt_log10_clustE = std::sqrt(std::log10(ClustEt)+1.1);
      // rishi's original code
      /*float b_upper= w10*eta0xsineta0 + w11 / sqrt_log10_clustE;      
	float b_lower=w00*eta0xsineta0 + w01 / sqrt_log10_clustE;      
      //here make an adjustment to the width for the offset from 0.
      float midpoint=  b_upper - (b_upper-b_lower)/2.;
      b_upper = b_upper - midpoint;
      b_lower = b_lower - midpoint;
      */
      
      // midpoint = (up + lo) / 2
      // so b_upper = (up - lo) / 2
      // and b_lower = (lo - up) / 2 = -b_upper

      const float b_upper =  0.5*( eta0xsineta0*(w10 - w00) + 
				   (w11-w01)/sqrt_log10_clustE );

      //the curvature comes from a parabolic 
      //fit for many slices in eta given a 
      //slice -0.1 < log10(Et) < 0.1
      const float curv_up=eta0xsineta0*(p00*eta0xsineta0+p01)+p02;
      const float curv_low=eta0xsineta0*(p10*eta0xsineta0+p11)+p12;
      
      //solving for the curviness given the width of this particular point
      const float a_upper=(1/(4*curv_up))-fabs(b_upper);
      const float a_lower = (1/(4*curv_low))-fabs(b_upper);
      
      const float dphi=TVector2::Phi_mpi_pi(ClusPhi-maxPhi);
      const double dphi2 = dphi*dphi;
      const float upper_cut=(1./(4.*a_upper))*dphi2+b_upper; 
      const float lower_cut=(1./(4.*a_lower))*dphi2-b_upper;
      
      //if(deta < upper_cut && deta > lower_cut) inMust=true;
      
      const float deta=sineta0*(ClusEta-maxEta);
      return (deta < upper_cut && deta > lower_cut);
    }    
  }
  
  void Mustache::MustacheID(const reco::SuperCluster& sc, 
			    int & nclusters, 
			    float & EoutsideMustache) {
    MustacheID(sc.clustersBegin(),sc.clustersEnd(), 
	       nclusters, EoutsideMustache);
  }
  
  void Mustache::MustacheID(const CaloClusterPtrVector& clusters, 
			    int & nclusters, 
			    float & EoutsideMustache) {    
    MustacheID(clusters.begin(),clusters.end(),nclusters,EoutsideMustache);
  }
  
  void Mustache::MustacheID(const std::vector<const CaloCluster*>& clusters, 
			    int & nclusters, 
			    float & EoutsideMustache) {
    MustacheID(clusters.cbegin(),clusters.cend(),nclusters,EoutsideMustache);
  }

  template<class RandomAccessPtrIterator>
  void Mustache::MustacheID(const RandomAccessPtrIterator& begin, 
			    const RandomAccessPtrIterator& end,
			    int & nclusters, 
			    float & EoutsideMustache) {    
    nclusters = 0;
    EoutsideMustache = 0;
    
    unsigned int ncl = end-begin;
    if(!ncl) return;
    
    //loop over all clusters to find the one with highest energy
    RandomAccessPtrIterator icl = begin;
    RandomAccessPtrIterator clmax = end;
    float emax = 0;
    for( ; icl != end; ++icl){
      const float e = (*icl)->energy();
      if(e > emax){
	emax = e;
	clmax = icl;
      }
    }
    
    if(end == clmax) return;
    
    float eta0 = (*clmax)->eta();
    float phi0 = (*clmax)->phi();
    

    bool inMust = false;
    icl = begin;
    for( ; icl != end; ++icl ){
      inMust=MustacheKernel::inMustache(eta0, phi0, 
					(*icl)->energy(), 
					(*icl)->eta(), 
					(*icl)->phi());
      
      nclusters += (int)!inMust;
      EoutsideMustache += (!inMust)*((*icl)->energy()); 
    }
  }
  
  void Mustache::MustacheClust(const std::vector<CaloCluster>& clusters, 
			       std::vector<unsigned int>& insideMust, 
			       std::vector<unsigned int>& outsideMust){  
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
      
      bool inMust=MustacheKernel::inMustache(eta0, phi0, 
					     (clusters[k]).energy(), 
					     (clusters[k]).eta(), 
					     (clusters[k]).phi());
      //return indices of Clusters outside the Mustache
      if (!(inMust)){
	outsideMust.push_back(k);
      }
      else{//return indices of Clusters inside the Mustache
	insideMust.push_back(k);
      }
    }
  }
  
  void Mustache::FillMustacheVar(const std::vector<CaloCluster>& clusters){
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
