#include "PFMultiDepthClusterizer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "Math/GenVector/VectorUtil.h"

#include "vdt/vdtMath.h"

#include <iterator>



PFMultiDepthClusterizer::
PFMultiDepthClusterizer(const edm::ParameterSet& conf) :
  PFClusterBuilderBase(conf) 
{
  
  _allCellsPosCalc.reset(NULL);
  if( conf.exists("allCellsPositionCalc") ) {
    const edm::ParameterSet& acConf = 
      conf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = 
      acConf.getParameter<std::string>("algoName");
    PosCalc* accalc = 
      PFCPositionCalculatorFactory::get()->create(algoac, acConf);
    _allCellsPosCalc.reset(accalc);
  }

  nSigmaEta_ = pow(conf.getParameter<double>("nSigmaEta"),2);
  nSigmaPhi_ = pow(conf.getParameter<double>("nSigmaPhi"),2);

}

void PFMultiDepthClusterizer::
buildClusters(const reco::PFClusterCollection& input,
	      const std::vector<bool>& seedable,
	      reco::PFClusterCollection& output) {

  std::vector<double> etaRMS(input.size(),0.0);
  std::vector<double> phiRMS(input.size(),0.0);

  //We need to sort the clusters for smaller to larger depth
  //  for (unsigned int i=0;i<input.size();++i)
  //   printf(" cluster%f %f \n",input[i].depth(),input[i].energy());



  //calculate cluster shapes
  calculateShowerShapes(input,etaRMS,phiRMS);

  //link
  std::vector<ClusterLink> links = link(input,etaRMS,phiRMS); 
  //  for (const auto& link: links)
  //    printf("link %d %d %f %f\n",link.from(),link.to(),link.dR(),link.dZ());


 
  std::vector<bool> mask(input.size(),false);
  std::vector<bool> linked(input.size(),false);

  //prune
  std::vector<ClusterLink> prunedLinks;
  if (links.size())
    prunedLinks =  prune(links,linked);

  //printf("Pruned links\n")
  //  for (const auto& link: prunedLinks)
  //  printf("link %d %d %f %f\n",link.from(),link.to(),link.dR(),link.dZ());



  //now we need to build clusters 
  for (unsigned int i=0;i<input.size();++i) {
    //if not masked
    if( mask[i])
      continue;
    //if not linked just spit it out
    if (!linked[i]) {
      output.push_back(input[i]);
      //      printf("Added single cluster with energy =%f \n",input[i].energy());
      mask[i] = true;
      continue;
    }


    //now business: if  linked and not  masked gather clusters
    reco::PFCluster cluster = input[i];
    mask[i] = true;
    expandCluster(cluster,i, mask,input,prunedLinks);
    _allCellsPosCalc->calculateAndSetPosition(cluster);
    output.push_back(cluster);
    //    printf("Added linked cluster with energy =%f\n",cluster.energy());
    
  }
}


void PFMultiDepthClusterizer::
calculateShowerShapes(const reco::PFClusterCollection& clusters,std::vector<double>& etaRMS,std::vector<double>& phiRMS) {
  float etaSum;
  float phiSum;

  //shower shapes. here do not use the fractions 

  for( unsigned int i=0;i<clusters.size();++i ) {
    const reco::PFCluster& cluster = clusters[i]; 
    etaSum=0.0;phiSum=0.0;
    for (const auto& frac : cluster.recHitFractions()) {
      etaSum +=frac.fraction()*frac.recHitRef()->energy()*fabs(frac.recHitRef()->position().Eta() -cluster.position().Eta() );
      phiSum +=frac.fraction()*frac.recHitRef()->energy()*fabs(deltaPhi(frac.recHitRef()->position().Phi(),cluster.position().Phi()));
    }
    //protection for single line : assign ~ tower
    etaRMS[i] = std::max(etaSum/cluster.energy(),0.1);
    phiRMS[i] = std::max(phiSum/cluster.energy(),0.1);
    
  }

}


std::vector<PFMultiDepthClusterizer::ClusterLink>   PFMultiDepthClusterizer::
link(const reco::PFClusterCollection& clusters ,const std::vector<double>& etaRMS,const std::vector<double>& phiRMS) {

  std::vector<ClusterLink> links;
  //loop on all pairs
  for (unsigned int i=0;i<clusters.size();++i)
    for (unsigned int j=0;j<clusters.size();++j) {
      if (i==j )
	continue;

      const reco::PFCluster& cluster1 = clusters[i]; 
      const reco::PFCluster& cluster2 = clusters[j]; 

      float dz = (cluster2.depth() - cluster1.depth());

      //Do not link at the same layer and only link inside out!
      if (dz<0.0 || fabs(dz)<0.2)
	continue;

      float deta =(cluster1.position().Eta()-cluster2.position().Eta())*(cluster1.position().Eta()-cluster2.position().Eta())/(etaRMS[i]*etaRMS[i]+etaRMS[j]*etaRMS[j]);
      float dphi = deltaPhi(cluster1.position().Phi(),cluster2.position().Phi())*deltaPhi(cluster1.position().Phi(),cluster2.position().Phi())/(phiRMS[i]*phiRMS[i]+phiRMS[j]*phiRMS[j]);

      //      printf("Testing Link %d -> %d (%f %f %f %f ) \n",i,j,deta,dphi,cluster1.position().Eta()-cluster2.position().Eta(),deltaPhi(cluster1.position().Phi(),cluster2.position().Phi()));

      if (deta<nSigmaEta_ && dphi<nSigmaPhi_ ) 
	links.push_back(ClusterLink(i,j,deta+dphi,fabs(dz),cluster1.energy()+cluster2.energy()));
    }

      return links;
}

std::vector<PFMultiDepthClusterizer::ClusterLink> PFMultiDepthClusterizer::
prune(std::vector<ClusterLink>& links,std::vector<bool>& linkedClusters) {
      std::vector<ClusterLink> goodLinks ;
      std::vector<bool> mask(links.size(),false);

      for (unsigned int i=0;i<links.size()-1;++i) {
	if (mask[i])
	  continue;
	for (unsigned int j=i+1;j<links.size();++j) {

	  if (mask[j])
	    continue;
	  
	  const ClusterLink& link1  = links[i];
	  const ClusterLink& link2  = links[j];

	  if (link1.to() == link2.to()) {//found two links going to the same spot,kill one  
	    //first prefer nearby layers
	    if (link1.dZ() < link2.dZ()) {
	      mask[j]=true;
	    }
	    else if (link1.dZ() > link2.dZ()) {
	      mask[i] = true;
	    }
	    else if (fabs(link1.dZ()-link2.dZ())<0.2) { //if same layer-pick based on transverse compatibility
	      if (link1.dR()<link2.dR()) {
		mask[j]=true;
	      }
	      else if (link1.dR()>link2.dR()) {
		mask[i] = true;
	      }
	      else {
		//same distance as well -> can happen !!!!! Pick the highest SUME 
		if (link1.energy()<link2.energy())
		  mask[i] = true;
		else
		  mask[j] = true;
		  
	      }
	    }
	  }
	}
      }

      for (unsigned int i=0;i<links.size();++i) {
	if (mask[i])
	  continue;
	goodLinks.push_back(links[i]);
	linkedClusters[links[i].from()]=true;
	linkedClusters[links[i].to()]=true;
      }


      return goodLinks;       
}	


void 
PFMultiDepthClusterizer::
absorbCluster(reco::PFCluster& main, const reco::PFCluster& added) {
  
  double e1=0.0;
  double e2=0.0;

  //find seeds
  for ( const auto& fraction : main.recHitFractions())
    if (fraction.recHitRef()->detId() == main.seed()) {
      e1=fraction.recHitRef()->energy();
    }

  for ( const auto& fraction : added.recHitFractions()) {
    main.addRecHitFraction(fraction); 
    if (fraction.recHitRef()->detId() == added.seed()) {
      e2=fraction.recHitRef()->energy();
    }
  }
  if (e2>e1)
    main.setSeed(added.seed());

}


void 
PFMultiDepthClusterizer::
expandCluster(reco::PFCluster& cluster,unsigned int point,std::vector<bool>& mask,const reco::PFClusterCollection& clusters,  const std::vector<ClusterLink>& links) {
  for (const auto& link : links) {
    if (link.from() == point) {
      //found link that starts from this guy if not masked absorb
      if (!mask[link.from()]) {
	absorbCluster(cluster,clusters[link.from()]);
	mask[link.from()] = true;
      }

      if (!mask[link.to()]) {
	absorbCluster(cluster,clusters[link.to()]);
	mask[link.to()]=true;
	expandCluster(cluster,link.to(),mask,clusters,links);
      }
    }
    if (link.to() == point) {
      //found link that starts from this guy if not masked absorb
      if (!mask[link.to()]) {
	absorbCluster(cluster,clusters[link.to()]);
	mask[link.to()] = true;
      }

      if (!mask[link.from()]) {
	absorbCluster(cluster,clusters[link.from()]);
	mask[link.from()]=true;
	expandCluster(cluster,link.from(),mask,clusters,links);
      }
    }



  }

}
