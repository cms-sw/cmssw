#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalSuperClusterAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "Math/GenVector/VectorUtil.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFHcalSuperClusterInit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <stdexcept>
#include <string>
#include <sstream>

using namespace std;
using namespace reco;

unsigned PFHcalSuperClusterAlgo::prodNum_ = 1;

//for debug only 
//#define PFLOW_DEBUG

PFHcalSuperClusterAlgo::PFHcalSuperClusterAlgo() :
  pfClusters_( new vector<reco::PFCluster> ),
  pfSuperClusters_( new vector<reco::PFSuperCluster> ),
  debug_(false) 
{

}

void
PFHcalSuperClusterAlgo::write() {
}
void PFHcalSuperClusterAlgo::doClustering( const PFClusterHandle& clustersHandle, const PFClusterHandle& clustersHOHandle ) {
  const reco::PFClusterCollection& clusters = * clustersHandle;
  const reco::PFClusterCollection& clustersHO = * clustersHOHandle;

  // cache the Handle to the clusters
  clustersHandle_ = clustersHandle;
  clustersHOHandle_ = clustersHOHandle;

  // perform clustering
  doClusteringWorker( clusters, clustersHO );
}

void PFHcalSuperClusterAlgo::doClustering( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO ) {
  // using clusters without a Handle, clear to avoid a stale member
  clustersHandle_.clear();
  clustersHOHandle_.clear();

  // perform clustering
  doClusteringWorker( clusters, clustersHO );
}


// calculate cluster position: Rachel Myers, July 2012
std::pair<double, double> PFHcalSuperClusterAlgo::calculatePosition(const reco::PFCluster& cluster)
{
  double numeratorEta = 0.0;
  double numeratorPhi = 0.0;
  double denominator = 0.0;
  double posEta = 0.0;
  double posPhi = 0.0;
  double w0_ = 4.2;
  const double clusterEnergy = cluster.energy();
  if (cluster.energy()>0.0 && cluster.recHitFractions().size()>0) {
    const std::vector <reco::PFRecHitFraction >& pfhitsandfracs = cluster.recHitFractions();
    for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
      const reco::PFRecHitRef rechit = it->recHitRef();
      double hitEta = rechit->positionREP().Eta();
      double hitPhi = rechit->positionREP().Phi();
      //Change into the -pi to +pi angular range
      while (hitPhi > +Geom::pi()) { hitPhi -= Geom::twoPi(); }
      while (hitPhi < -Geom::pi()) { hitPhi += Geom::twoPi(); }
      double hitEnergy = rechit->energy();
      const double w = std::max(0.0, w0_ + log(hitEnergy / clusterEnergy));
      denominator += w;
      numeratorEta += w*hitEta;
      numeratorPhi += w*hitPhi;
    }
    posEta = numeratorEta/denominator;
    posPhi = numeratorPhi/denominator;
  }

  pair<double, double> posEtaPhi(posEta,posPhi);

  return posEtaPhi;
}

// calculate cluster width: Rachel Myers, July 2012
std::pair<double, double> PFHcalSuperClusterAlgo::calculateWidths(const reco::PFCluster& cluster)
{
  double numeratorEtaEta = 0;
  //  double numeratorEtaPhi = 0;
  double numeratorPhiPhi = 0;
  double denominator     = 0;
  double widthEta = 0.0;
  double widthPhi = 0.0;

  double w0_ = 4.2;

  const double clusterEta = calculatePosition(cluster).first;
  const double clusterPhi = calculatePosition(cluster).second;
  const double clusterEnergy = cluster.energy();
  if(cluster.energy()>0.0 && cluster.recHitFractions().size()>0) {
    double hitEta, hitPhi, hitEnergy, dEta, dPhi; 
    const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = cluster.recHitFractions();
    for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
      const reco::PFRecHitRef rechit = it->recHitRef();
      //      rechit->calculatePositionREP();
      hitEta = rechit->positionREP().Eta();
      hitPhi = rechit->positionREP().Phi();
      hitEnergy = rechit->energy();
      dEta  = hitEta - clusterEta;
      dPhi  = hitPhi - clusterPhi;

      while (dPhi > +Geom::pi()) { dPhi -= Geom::twoPi(); }
      while (dPhi < -Geom::pi()) { dPhi += Geom::twoPi(); }


      const double w = std::max(0.0, w0_ + log(hitEnergy / clusterEnergy));

      denominator += w;
      numeratorEtaEta += w * dEta * dEta;
      //      numeratorEtaPhi += w * dEta * dPhi;
      numeratorPhiPhi += w * dPhi * dPhi;
    }

    double covEtaEta = numeratorEtaEta / denominator;
    //    double covEtaPhi_ = numeratorEtaPhi / denominator;
    double covPhiPhi = numeratorPhiPhi / denominator;

    widthEta = sqrt(abs(covEtaEta));
    widthPhi = sqrt(abs(covPhiPhi));
  }
  pair<double, double> widthEtaPhi(widthEta,widthPhi);
  return widthEtaPhi;
}
// do clustering with new widths, positions, parameters, merging conditions: Rachel Myers, July 2012
void PFHcalSuperClusterAlgo::doClusteringWorker( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO ) {

  double dRcut=0.17;
  double dEtacut = 0.0;
  double dPhicut = 0.0;
  double etaScale = 1.0;
  double phiScale = 0.5;
  //  double dRcut=0.30;

  if ( pfClusters_.get() )
    pfClusters_->clear();
  else 
    pfClusters_.reset( new std::vector<reco::PFCluster> );

  if ( pfSuperClusters_.get() )
    pfSuperClusters_->clear();
  else 
    pfSuperClusters_.reset( new std::vector<reco::PFSuperCluster> );

  // compute cluster depth index
  std::vector< unsigned > clusterdepth(clusters.size());
  //  cout << " clusters: " << clusters.size() <<endl; 
  int mclustersHB=0;
  int mclustersHE=0;
  for (unsigned short ic=0; ic<clusters.size();++ic) {
    if( clusters[ic].layer() == PFLayer::HCAL_BARREL1) mclustersHB++;
    if( clusters[ic].layer() == PFLayer::HCAL_ENDCAP) mclustersHE++;
  }
  for (unsigned short ic=0; ic<clusters.size();++ic)
    {
      if( clusters[ic].layer() == PFLayer::HCAL_BARREL1
	  || clusters[ic].layer() == PFLayer::HCAL_ENDCAP ) { //Hcal case

        const std::vector< std::pair<DetId, float> > & hitsandfracs =
	  clusters[ic].hitsAndFractions();
        unsigned clusterdepthfirst=0;
        for(unsigned ihandf=0; ihandf<hitsandfracs.size(); ihandf++) {
          unsigned depth = ((HcalDetId)hitsandfracs[ihandf].first).depth();
	  //          cout << " depth parameter from clustering: " << depth <<endl; 
          clusterdepth[ic] = depth;         
          if( ihandf>0 && depth!=clusterdepthfirst) cout << " Problem with cluster depth: " << depth << " not equal to " << clusterdepthfirst <<endl;
	  else if( ihandf==0 ) clusterdepthfirst = depth;
        }
	//        delete hitsandfracs;
      }
    }
  std::vector< unsigned > clusterdepthHO(clustersHO.size());
  //  cout << " HO clusters: " << clustersHO.size() <<endl; 
  double hcaleta1=0.0;
  double hcalphi1=0.0;
  double hcaleta2=0.0;
  double hcalphi2=0.0;
  double dR = 0.0;
  double dEta = 0.0;
  double dPhi = 0.0;
  for (unsigned short ic=0; ic<clustersHO.size();++ic)
    {
      if( clustersHO[ic].layer() == PFLayer::HCAL_BARREL2) { //HO case

        const std::vector< std::pair<DetId, float> > & hitsandfracs =
	  clustersHO[ic].hitsAndFractions();
        unsigned clusterdepthfirst=0;
        for(unsigned ihandf=0; ihandf<hitsandfracs.size(); ihandf++) {
          unsigned depth = ((HcalDetId)hitsandfracs[ihandf].first).depth();
	  //          cout << " depth parameter from HO clustering: " << depth <<endl; 
          clusterdepthHO[ic] = depth;          
          if( ihandf>0 && depth!=clusterdepthfirst) cout << " Problem with HO cluster depth: " << depth << " not equal to " << clusterdepthfirst <<endl;
          else if( ihandf==0 ) clusterdepthfirst = depth;
        }
	//        delete hitsandfracs;
      }
    }

  std::vector< unsigned > imerge(clusters.size());
  std::vector< unsigned > imergeHO(clustersHO.size());
  std::vector< bool > lmerge(clusters.size());
  std::vector< bool > lmergeHO(clustersHO.size());

  hcaleta1=0.0;
  hcalphi1=0.0;
  hcaleta2=0.0;
  hcalphi2=0.0;
  dR = 0.0;
  dEta = 0.0;
  dPhi = 0.0;

  //    cout << " setting up cluster merging indices "<<endl;
  for (unsigned short ic1=0; ic1<clusters.size();++ic1) {
    lmerge[ic1]=false;
  }
  for (unsigned short ic1=0; ic1<clustersHO.size();++ic1) {
    lmergeHO[ic1]=false;
  }
  for (unsigned short ic1=0; ic1<clusters.size();++ic1) {
    if( clusterdepth[ic1]==1 ){
      hcaleta1 = calculatePosition(clusters[ic1]).first;
      hcalphi1 = calculatePosition(clusters[ic1]).second;
      for (unsigned short ic2=0; ic2<clusters.size();++ic2) {
	hcaleta2 = calculatePosition(clusters[ic2]).first;
	hcalphi2 = calculatePosition(clusters[ic2]).second;
	dR = deltaR( hcaleta1, hcalphi1, hcaleta2, hcalphi2 );
	dEta = abs(hcaleta1 - hcaleta2);
	dPhi = abs(deltaPhi(hcalphi1, hcalphi2));
	double w1 = calculateWidths(clusters[ic1]).first;
	if (w1 == 0) w1 = 0.087;
	double w2 = calculateWidths(clusters[ic2]).first;
	if (w2 == 0) w2 = 0.087;
	double w3 = calculateWidths(clusters[ic1]).second;
	if (w3 < 0.087) w3 = 0.087;
	double w4 = calculateWidths(clusters[ic2]).second;
	if (w4 < 0.087) w4 = 0.087;
	double etawidth = sqrt(w1*w1 + w2*w2);
	double phiwidth = sqrt(w3*w3 + w4*w4);
	dEtacut = etaScale*etawidth;
	dPhicut = phiScale*phiwidth;
	if( clusterdepth[ic2]==2 ){
	  //            cout << " depth 1-2 dR = " << dR <<endl;
	  if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
	    imerge[ic2]=ic1;
	    lmerge[ic2]=true;
	  }
	} else if( clusterdepth[ic2]==3 ){
	  //            cout << " depth 1-3 dR = " << dR <<endl;
	  if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut) ) {
	    imerge[ic2]=ic1;
	    lmerge[ic2]=true;
	  }
	}
      }
    } else if( clusterdepth[ic1]==2 ){
      hcaleta1 = calculatePosition(clusters[ic1]).first;
      hcalphi1 = calculatePosition(clusters[ic1]).second;
      for (unsigned short ic2=0; ic2<clusters.size();++ic2) {
	hcaleta2 = calculatePosition(clusters[ic2]).first;
	hcalphi2 = calculatePosition(clusters[ic2]).second;
	dEta = abs(hcaleta1 - hcaleta2);
	dPhi = abs(deltaPhi(hcalphi1, hcalphi2));
	dR = deltaR( hcaleta1, hcalphi1, hcaleta2, hcalphi2 );
	double w1 = calculateWidths(clusters[ic1]).first;
	if (w1 == 0) w1 = 0.087;
	double w2 = calculateWidths(clusters[ic2]).first;
	if (w2 == 0) w2 = 0.087;
	double w3 = calculateWidths(clusters[ic1]).second;
	if (w3 < 0.087) w3 = 0.087;
	double w4 = calculateWidths(clusters[ic2]).second;
	if (w4 < 0.087) w4 = 0.087;
	double etawidth = sqrt(w1*w1+w2*w2);
	double phiwidth = sqrt(w3*w3+w4*w4);
	dEtacut = etaScale*etawidth;
	dPhicut = phiScale*phiwidth;
	if( clusterdepth[ic2]==3 ){
	  if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
	    imerge[ic2]=ic1;
	    lmerge[ic2]=true;
	  }
	} else if( clusterdepth[ic2]==4 ){
	  //            cout << " depth 2-4 dR = " << dR <<endl;
	  if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
	    imerge[ic2]=ic1;
	    lmerge[ic2]=true;
	  }
	}
      }
    } else if( clusterdepth[ic1]==3 ){
      hcaleta1 = calculatePosition(clusters[ic1]).first;
      hcalphi1 = calculatePosition(clusters[ic1]).second;
      for (unsigned short ic2=0; ic2<clusters.size();++ic2) {
	hcaleta2 = calculatePosition(clusters[ic2]).first;
	hcalphi2 = calculatePosition(clusters[ic2]).second;
	dEta = abs(hcaleta1 - hcaleta2);
	dPhi = abs(deltaPhi(hcalphi1, hcalphi2));
	dR = deltaR( hcaleta1, hcalphi1, hcaleta2, hcalphi2 );
	double w1 = calculateWidths(clusters[ic1]).first;
	if (w1 == 0) w1 = 0.087;
	double w2 = calculateWidths(clusters[ic2]).first;
	if (w2 == 0) w2 = 0.087;
	double w3 = calculateWidths(clusters[ic1]).second;
	if (w3 < 0.087) w3 = 0.087;
	double w4 = calculateWidths(clusters[ic2]).second;
	if (w4 < 0.087) w4 = 0.087;
	double etawidth = sqrt(w1*w1+w2*w2);
	double phiwidth = sqrt(w3*w3+w4*w4);
	dEtacut = etaScale*etawidth;
	dPhicut = phiScale*phiwidth;
	if( clusterdepth[ic2]==4 ){
	  //            cout << " depth 3-4 dR = " << dR <<endl;
	  if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
	    imerge[ic2]=ic1;
	    lmerge[ic2]=true;
	  }
	} else if( clusterdepth[ic2]==5 ){
	  //            cout << " depth 3-5 dR = " << dR <<endl;
	  if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
	    imerge[ic2]=ic1;
	    lmerge[ic2]=true;
	  }
	}
      }
      for (unsigned short ic2=0; ic2<clustersHO.size();++ic2) {
	hcaleta2 = calculatePosition(clustersHO[ic2]).first;
	hcalphi2 = calculatePosition(clustersHO[ic2]).second;
	dEta = abs(hcaleta1 - hcaleta2);
	dPhi = abs(deltaPhi(hcalphi1, hcalphi2));
	dR = deltaR( hcaleta1, hcalphi1, hcaleta2, hcalphi2 );
	double w1 = calculateWidths(clusters[ic1]).first;
	if (w1 == 0) w1 = 0.087;
	double w2 = calculateWidths(clustersHO[ic2]).first;
	if (w2 == 0) w2 = 0.087;
	double w3 = calculateWidths(clusters[ic1]).second;
	if (w3 < 0.087) w3 = 0.087;
	double w4 = calculateWidths(clustersHO[ic2]).second;
	if (w4 < 0.087) w4 = 0.087;
	double etawidth = sqrt(w1*w1+w2*w2);
	double phiwidth = sqrt(w3*w3+w4*w4);
	dEtacut = etaScale*etawidth;
	dPhicut = phiScale*phiwidth;
	if( clusterdepthHO[ic2]==5 ){
	  //            cout << " depth 3-HO dR = " << dR <<endl;
	  if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
	    imergeHO[ic2]=ic1;
	    lmergeHO[ic2]=true;
	  }
	}
      }
    } else if( clusterdepth[ic1]==4 ){
      hcaleta1 = calculatePosition(clusters[ic1]).first;
      hcalphi1 = calculatePosition(clusters[ic1]).second;
      for (unsigned short ic2=0; ic2<clusters.size();++ic2) {
	hcaleta2 = calculatePosition(clusters[ic2]).first;
	hcalphi2 = calculatePosition(clusters[ic2]).second;
	dEta = abs(hcaleta1 - hcaleta2);
	dPhi = abs(deltaPhi(hcalphi1, hcalphi2));
	dR = deltaR( hcaleta1, hcalphi1, hcaleta2, hcalphi2 );
	double w1 = calculateWidths(clusters[ic1]).first;
	if (w1 < 0.087) w1 = 0.087;
	double w2 = calculateWidths(clusters[ic2]).first;
	if (w2 < 0.087) w2 = 0.087;
	double w3 = calculateWidths(clusters[ic1]).second;
	if (w3 < 0.087) w3 = 0.087;
	double w4 = calculateWidths(clusters[ic2]).second;
	if (w4 < 0.087) w4 = 0.087;
	double etawidth = sqrt(w1*w1+w2*w2);
	double phiwidth = sqrt(w3*w3+w4*w4);
	dEtacut = etaScale*etawidth;
	dPhicut = phiScale*phiwidth;
	if( clusterdepth[ic2]==5 ){
	  //            cout << " depth 4-5 dR = " << dR <<endl;
	  if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
	    imerge[ic2]=ic1;
	    lmerge[ic2]=true;
	  }
	}
      }
    } else if( clusterdepth[ic1]==5 ){
      hcaleta1 = calculatePosition(clusters[ic1]).first;
      hcalphi1 = calculatePosition(clusters[ic1]).second;
    }
  }

  // start a supercluster with a depth=1 cluster, then loop on all other
  // clusters to add to cluster list, then for each cluster to add, loop
  // on remaining clusters to check 2nd level of addition, repeat for all layers
  
  // need to add HO cluster logic
  edm::PtrVector< reco::PFCluster >  mergeclusters;
  std::vector< bool >  lmergeclusters(clusters.size());
  for (unsigned short id=0; id<4;++id) {
    //    cout << " merging with starting depth: "<<id<<endl;
    for (unsigned short ic1=0; ic1<clusters.size();++ic1) {
      if( clusterdepth[ic1]==(unsigned)(1+id) ){
        if(!lmerge[ic1]) {
          for (unsigned short ic=0; ic<clusters.size();++ic) {
            lmergeclusters[ic]=false;
          }
	  //          mergeclusters.push_back(clusters[ic1]);
          for (unsigned short ic2=0; ic2<clusters.size();++ic2) {
            if( clusterdepth[ic2]==(unsigned)(2+id) ){
              if( imerge[ic2]==ic1 && lmerge[ic2] ) {
		//                mergeclusters.push_back(clusters[ic2]);
                lmergeclusters[ic2]=true;
                for (unsigned short ic3=0; ic3<clusters.size();++ic3) {
                  if( clusterdepth[ic3]==(unsigned)(3+id) ){
                    if( imerge[ic3]==ic2 && lmerge[ic3] ) {
		      //                      mergeclusters.push_back(clusters[ic3]);
                      lmergeclusters[ic3]=true;
                      for (unsigned short ic4=0; ic4<clusters.size();++ic4) {
                        if( clusterdepth[ic4]==(unsigned)(4+id) ){
                          if( imerge[ic4]==ic3 && lmerge[ic4] ) {
			    //                            mergeclusters.push_back(clusters[ic4]);
                            lmergeclusters[ic4]=true;
                            for (unsigned short ic5=0; ic5<clusters.size();++ic5) {
                              if( clusterdepth[ic5]==(unsigned)(5+id) ){
				//                                if( imerge[ic5]==ic4 && lmerge[ic5] ) mergeclusters.push_back(clusters[ic5]);
                                if( imerge[ic5]==ic4 && lmerge[ic5] ) lmergeclusters[ic5]=true;
                              }
                            }
                          } else if( clusterdepth[ic4]==(unsigned)(5+id) ){
			    //                            if( imerge[ic4]==ic3 && lmerge[ic4] ) mergeclusters.push_back(clusters[ic4]);
                            if( imerge[ic4]==ic3 && lmerge[ic4] ) lmergeclusters[ic4]=true;
                          }
                        }
                      }
                    } else if( clusterdepth[ic3]==(unsigned)(4+id) ){
                      if( imerge[ic3]==ic2 && lmerge[ic3] ) {
			//                        mergeclusters.push_back(clusters[ic3]);
                        lmergeclusters[ic3]=true;
                        for (unsigned short ic4=0; ic4<clusters.size();++ic4) {
                          if( clusterdepth[ic4]==(unsigned)(5+id) ){
			    //                            if( imerge[ic4]==ic3 && lmerge[ic4] ) mergeclusters.push_back(clusters[ic4]);
                            if( imerge[ic4]==ic3 && lmerge[ic4] ) lmergeclusters[ic4]=true;
                          }
                        }
                      }
                    }
                  } if( clusterdepth[ic3]==(unsigned)(5+id) ){
		    //                    if( imerge[ic3]==ic2 && lmerge[ic3] ) mergeclusters.push_back(clusters[ic3]);
                    if( imerge[ic3]==ic2 && lmerge[ic3] ) lmergeclusters[ic3]=true;
                  }
                }
              }
            } else if( clusterdepth[ic2]==(unsigned)(3+id) ){
              if( imerge[ic2]==ic1 && lmerge[ic2] ) {
		//                mergeclusters.push_back(clusters[ic2]);
                lmergeclusters[ic2]=true;
                for (unsigned short ic3=0; ic3<clusters.size();++ic3) {
                  if( clusterdepth[ic3]==(unsigned)(4+id) ){
                    if( imerge[ic3]==ic2 && lmerge[ic3] ) {
		      //                      mergeclusters.push_back(clusters[ic3]);
                      lmergeclusters[ic3]=true;
                      for (unsigned short ic4=0; ic4<clusters.size();++ic4) {
                        if( clusterdepth[ic4]==(unsigned)(5+id) ){
			  //                          if( imerge[ic4]==ic3 && lmerge[ic4] ) mergeclusters.push_back(clusters[ic4]);
                          if( imerge[ic4]==ic3 && lmerge[ic4] ) lmergeclusters[ic4]=true;
                        }
                      }
                    } else if( clusterdepth[ic3]==(unsigned)(5+id) ){
		      //                      if( imerge[ic3]==ic2 && lmerge[ic3] ) mergeclusters.push_back(clusters[ic3]);
                      if( imerge[ic3]==ic2 && lmerge[ic3] ) lmergeclusters[ic3]=true;
                    }
                  }
                }
              }
            }
          }
          mergeclusters.push_back( PFClusterPtr( clustersHandle_, ic1));
          for (unsigned short ic=0; ic<clusters.size();++ic) {
            if(lmergeclusters[ic]) {
              mergeclusters.push_back( PFClusterPtr(clustersHandle_, ic ) );
            }
          }
          if(mergeclusters.size()>0) {
	    //            cout << " number of clusters to merge: " <<mergeclusters.size()<<endl;
            reco::PFSuperCluster ipfsupercluster(mergeclusters);
	    PFHcalSuperClusterInit init;
	    init.initialize( ipfsupercluster, clusters_);
            pfSuperClusters_->push_back(ipfsupercluster);
            pfClusters_->push_back((reco::PFCluster)ipfsupercluster);
            mergeclusters.clear();
          }
          
        }
      } // end of depth 1+id initiated logic
    }
  }

  /*
    for (unsigned short ic=0; ic<clusters.size();++ic) {
    mergeclusters.clear();
    mergeclusters.push_back(clusters[ic]);
    reco::PFSuperCluster ipfsupercluster(mergeclusters);
    pfSuperClusters_->push_back(ipfsupercluster);
    pfClusters_->push_back((reco::PFCluster)ipfsupercluster);
    //    pfClusters_->push_back(clusters[ic]);
    }
  */

  clusterdepth.clear();
  clusterdepthHO.clear();
  imerge.clear();
  imergeHO.clear();
  lmerge.clear();
  lmergeHO.clear();
  mergeclusters.clear();

}
ostream& operator<<(ostream& out,const PFHcalSuperClusterAlgo& algo) { 
  if(!out) return out;
  out<<"PFSuperClusterAlgo parameters : "<<endl;
  out<<"-----------------------------------------------------"<<endl;
  
  out<<endl;
  out<<algo.pfClusters_->size()<<" clusters:"<<endl;
  
  for(unsigned i=0; i<algo.pfClusters_->size(); i++) {
    out<<(*algo.pfClusters_)[i]<<endl;
    
    if(!out) return out;
  }
  
  out<<algo.pfSuperClusters_->size()<<" superclusters:"<<endl;
    
  for(unsigned i=0; i<algo.pfSuperClusters_->size(); i++) {
    out<<(*algo.pfSuperClusters_)[i]<<endl;
    
    if(!out) return out;
  }   
  return out;
}
