#include "RecoParticleFlow/PFClusterProducer/interface/PFSuperClusterAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Math/GenVector/VectorUtil.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TROOT.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <stdexcept>
#include <string>
#include <sstream>

using namespace std;

unsigned PFSuperClusterAlgo::prodNum_ = 1;

//for debug only 
//#define PFLOW_DEBUG

PFSuperClusterAlgo::PFSuperClusterAlgo() :
  pfClusters_( new vector<reco::PFCluster> ),
  pfSuperClusters_( new vector<reco::PFSuperCluster> ),
  debug_(false) 
{
  file_ = 0;

    dR12HB = 0;
    dR13HB = 0;
    dR23HB = 0;
    dR3HOHB = 0;
    dR12HE = 0;
    dR13HE = 0;
    dR23HE = 0;
    dR24HE = 0;
    dR34HE = 0;
    dR35HE = 0;
    dR45HE = 0;
    dEta12HB = 0;
    dEta13HB = 0;
    dEta23HB = 0;
    dEta3HOHB = 0;
    dEta12HE = 0;
    dEta13HE = 0;
    dEta23HE = 0;
    dEta24HE = 0;
    dEta34HE = 0;
    dEta35HE = 0;
    dEta45HE = 0;
    dPhi12HB = 0;
    dPhi13HB = 0;
    dPhi23HB = 0;
    dPhi3HOHB = 0;
    dPhi12HE = 0;
    dPhi13HE = 0;
    dPhi23HE = 0;
    dPhi24HE = 0;
    dPhi34HE = 0;
    dPhi35HE = 0;
    dPhi45HE = 0;
    normalized12HB = 0;
    normalized13HB = 0;
    normalized23HB = 0;
    normalized3HOHB = 0;
    normalized12HE = 0;
    normalized13HE = 0;
    normalized23HE = 0;
    normalized24HE = 0;
    normalized34HE = 0;
    normalized35HE = 0;
    normalized45HE = 0;
    nclustersHB = 0;
    nclustersHE = 0;
    nclustersHO = 0;
    mergeclusters1HB = 0;
    mergeclusters2HB = 0;
    mergeclusters3HB = 0;
    mergeclusters1HE = 0;
    mergeclusters2HE = 0;
    mergeclusters3HE = 0;
    mergeclusters4HE = 0;
    hitsHB = 0;
    hitsHE = 0;
    etaPhi = 0;
    etaPhiHits = 0;
    etaPhiHits1HB = 0;
    etaPhiHits2HB = 0;
    etaPhiHits3HB = 0;
    etaPhiHits1HE = 0;
    etaPhiHits2HE = 0;
    etaPhiHits3HE = 0;
    etaPhiHits4HE = 0;
    etaPhiHits5HE = 0;
    hitTime1HB = 0;
    hitTime2HB = 0;
    hitTime3HB = 0;
    hitTime1HE = 0;
    hitTime2HE = 0;
    hitTime3HE = 0;
    hitTime4HE = 0;
    hitTime5HE = 0;
    etaWidth1HB = 0;
    etaWidth2HB = 0;
    etaWidth3HB = 0;
    etaWidth1HE = 0;
    etaWidth2HE = 0;
    etaWidth3HE = 0;
    etaWidth4HE = 0;
    etaWidth5HE = 0;
    phiWidth1HB = 0;
    phiWidth2HB = 0;
    phiWidth3HB = 0;
    phiWidth1HE = 0;
    phiWidth2HE = 0;
    phiWidth3HE = 0;
    phiWidth4HE = 0;
    phiWidth5HE = 0;
    etaWidthSuperClusterHB = 0;
    phiWidthSuperClusterHB = 0;
    etaWidthSuperClusterHE = 0;
    phiWidthSuperClusterHE = 0;
    sizeSuperClusterHB = 0;
    sizeSuperClusterHE = 0;


//  void setHistos(TFile* file) {

//  file_=file;
//  file_ = TFile::Open("superclusteralgo.root", "recreate");

  if (file_) {
    file_->cd();
    dR12HB = new TH1F("dR12HB", "dR 1-2 HB", 50, 0.0, 1.0);
    dR13HB = new TH1F("dR13HB", "dR 1-3 HB", 50, 0.0, 1.0);
    dR23HB = new TH1F("dR23HB", "dR 2-3 HB", 50, 0.0, 1.0);
    dR3HOHB = new TH1F("dR3HOHB", "dR 3-HO HB", 50, 0.0, 1.0);
    dR12HE = new TH1F("dR12HE", "dR 1-2 HE", 50, 0.0, 1.0);
    dR13HE = new TH1F("dR13HE", "dR 1-3 HE", 50, 0.0, 1.0);
    dR23HE = new TH1F("dR23HE", "dR 2-3 HE", 50, 0.0, 1.0);
    dR24HE = new TH1F("dR24HE", "dR 2-4 HE", 50, 0.0, 1.0);
    dR34HE = new TH1F("dR34HE", "dR 3-4 HE", 50, 0.0, 1.0);
    dR35HE = new TH1F("dR35HE", "dR 3-5 HE", 50, 0.0, 1.0);
    dR45HE = new TH1F("dR45HE", "dR 4-5 HE", 50, 0.0, 1.0);
    dEta12HB = new TH1F("dEta12HB", "dEta 1-2 HB", 50, 0.0, 1.0);
    dEta13HB = new TH1F("dEta13HB", "dEta 1-3 HB", 50, 0.0, 1.0);
    dEta23HB = new TH1F("dEta23HB", "dEta 2-3 HB", 50, 0.0, 1.0);
    dEta3HOHB = new TH1F("dEta3HOHB", "dEta 3-HO HB", 50, 0.0, 1.0);
    dEta12HE = new TH1F("dEta12HE", "dEta 1-2 HE", 50, 0.0, 1.0);
    dEta13HE = new TH1F("dEta13HE", "dEta 1-3 HE", 50, 0.0, 1.0);
    dEta23HE = new TH1F("dEta23HE", "dEta 2-3 HE", 50, 0.0, 1.0);
    dEta24HE = new TH1F("dEta24HE", "dEta 2-4 HE", 50, 0.0, 1.0);
    dEta34HE = new TH1F("dEta34HE", "dEta 3-4 HE", 50, 0.0, 1.0);
    dEta35HE = new TH1F("dEta35HE", "dEta 3-5 HE", 50, 0.0, 1.0);
    dEta45HE = new TH1F("dEta45HE", "dEta 4-5 HE", 50, 0.0, 1.0);
    dPhi12HB = new TH1F("dPhi12HB", "dPhi 1-2 HB", 50, 0.0, 1.0);
    dPhi13HB = new TH1F("dPhi13HB", "dPhi 1-3 HB", 50, 0.0, 1.0);
    dPhi23HB = new TH1F("dPhi23HB", "dPhi 2-3 HB", 50, 0.0, 1.0);
    dPhi3HOHB = new TH1F("dPhi3HOHB", "dPhi 3-HO HB", 50, 0.0, 1.0);
    dPhi12HE = new TH1F("dPhi12HE", "dPhi 1-2 HE", 50, 0.0, 1.0);
    dPhi13HE = new TH1F("dPhi13HE", "dPhi 1-3 HE", 50, 0.0, 1.0);
    dPhi23HE = new TH1F("dPhi23HE", "dPhi 2-3 HE", 50, 0.0, 1.0);
    dPhi24HE = new TH1F("dPhi24HE", "dPhi 2-4 HE", 50, 0.0, 1.0);
    dPhi34HE = new TH1F("dPhi34HE", "dPhi 3-4 HE", 50, 0.0, 1.0);
    dPhi35HE = new TH1F("dPhi35HE", "dPhi 3-5 HE", 50, 0.0, 1.0);
    dPhi45HE = new TH1F("dPhi45HE", "dPhi 4-5 HE", 50, 0.0, 1.0);
    normalized12HB = new TH1F("normalized12HB", "Normalized dEta 1-2 HB", 50, 0.0, 10.0);
    normalized13HB = new TH1F("normalized13HB", "Normalized dEta 1-3 HB", 50, 0.0, 10.0);
    normalized23HB = new TH1F("normalized23HB", "Normalized dEta 2-3 HB", 50, 0.0, 10.0);
    normalized3HOHB = new TH1F("normalized3HOHB", "Normalized dEta 3-HO HB", 50, 0.0, 10.0);
    normalized12HE = new TH1F("normalized12HE", "Normalized dEta 1-2 HB", 50, 0.0, 10.0);
    normalized13HE = new TH1F("normalized13HE", "Normalized dEta 1-3 HB", 50, 0.0, 10.0);
    normalized23HE = new TH1F("normalized23HE", "Normalized dEta 2-3 HB", 50, 0.0, 10.0);
    normalized24HE = new TH1F("normalized24HE", "Normalized dEta 2-4 HB", 50, 0.0, 10.0);
    normalized34HE = new TH1F("normalized34HE", "Normalized dEta 3-4 HB", 50, 0.0, 10.0);
    normalized35HE = new TH1F("normalized35HE", "Normalized dEta 3-5 HB", 50, 0.0, 10.0);
    normalized45HE = new TH1F("normalized45HE", "Normalized dEta 4-5 HB", 50, 0.0, 10.0);
    nclustersHB = new TH1F("nclustersHB", "number of HB clusters", 1000,-0.5, 999.5);
    nclustersHE = new TH1F("nclustersHE", "number of HE clusters", 1000,-0.5, 999.5);
    nclustersHO = new TH1F("nclustersHO", "number of HO clusters", 100,-0.5, 99.5);
    mergeclusters1HB = new TH1F("mergeclusters1HB", "number of merge clusters in depth-1 initiated HB superclusters", 100,-0.5, 99.5);
    mergeclusters2HB = new TH1F("mergeclusters2HB", "number of merge clusters in depth-2 initiated HB superclusters", 100,-0.5, 99.5);
    mergeclusters3HB = new TH1F("mergeclusters3HB", "number of merge clusters in depth-3 initiated HB superclusters", 100,-0.5, 99.5);
    mergeclusters1HE = new TH1F("mergeclusters1HE", "number of merge clusters in depth-1 initiated HE superclusters", 100,-0.5, 99.5);
    mergeclusters2HE = new TH1F("mergeclusters2HE", "number of merge clusters in depth-2 initiated HE superclusters", 100,-0.5, 99.5);
    mergeclusters3HE = new TH1F("mergeclusters3HE", "number of merge clusters in depth-3 initiated HE superclusters", 100,-0.5, 99.5);
    mergeclusters4HE = new TH1F("mergeclusters4HE", "number of merge clusters in depth-4 initiated HE superclusters", 100,-0.5, 99.5);
    hitsHB = new TH1F("hitsHB", "number of hits in HB supercluster", 1000, -0.5, 999.5);
    hitsHE = new TH1F("hitsHE", "number of hits in HE supercluster", 1000, -0.5, 999.5);
    etaPhi = new TH2F("etaPhi", "eta-phi of superclusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    etaPhiHits = new TH2F("etaPhiHits", "eta-phi hits of superclusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    etaPhiHits1HB = new TH2F("etaPhiHits1HB", "eta-phi hits of HB depth-1 clusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    etaPhiHits2HB = new TH2F("etaPhiHits2HB", "eta-phi hits of HB depth-2 clusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    etaPhiHits3HB = new TH2F("etaPhiHits3HB", "eta-phi hits of HB depth-3 clusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    etaPhiHits1HE = new TH2F("etaPhiHits1HE", "eta-phi hits of HE depth-1 clusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    etaPhiHits2HE = new TH2F("etaPhiHits2HE", "eta-phi hits of HE depth-2 clusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    etaPhiHits3HE = new TH2F("etaPhiHits3HE", "eta-phi hits of HE depth-3 clusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    etaPhiHits4HE = new TH2F("etaPhiHits4HE", "eta-phi hits of HE depth-4 clusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    etaPhiHits5HE = new TH2F("etaPhiHits5HE", "eta-phi hits of HE depth-5 clusters",70,-3.05, 3.05, 74, -3.20, 3.20);
    hitTime1HB = new TH1F("hitTime1HB", "arrival time of hits in HB depth-1 clusters", 400, -78.4, 77.6);
    hitTime2HB = new TH1F("hitTime2HB", "arrival time of hits in HB depth-2 clusters", 400, -78.4, 77.6);
    hitTime3HB = new TH1F("hitTime3HB", "arrival time of hits in HB depth-3 clusters", 400, -78.4, 77.6);
    hitTime1HE = new TH1F("hitTime1HE", "arrival time of hits in HE depth-1 clusters", 400, -78.4, 77.6);
    hitTime2HE = new TH1F("hitTime2HE", "arrival time of hits in HE depth-2 clusters", 400, -78.4, 77.6);
    hitTime3HE = new TH1F("hitTime3HE", "arrival time of hits in HE depth-3 clusters", 400, -78.4, 77.6);
    hitTime4HE = new TH1F("hitTime4HE", "arrival time of hits in HE depth-4 clusters", 400, -78.4, 77.6);
    hitTime5HE = new TH1F("hitTime5HE", "arrival time of hits in HE depth-5 clusters", 400, -78.4, 77.6);
    etaWidth1HB = new TH1F("etaWidth1HB", "eta width of HB depth-1 clusters", 20, 0.0, 1.0);
    etaWidth2HB = new TH1F("etaWidth2HB", "eta width of HB depth-2 clusters", 20, 0.0, 1.0);
    etaWidth3HB = new TH1F("etaWidth3HB", "eta width of HB depth-3 clusters", 20, 0.0, 1.0);
    etaWidth1HE = new TH1F("etaWidth1HE", "eta width of HE depth-1 clusters", 20, 0.0, 1.0);
    etaWidth2HE = new TH1F("etaWidth2HE", "eta width of HE depth-2 clusters", 20, 0.0, 1.0);
    etaWidth3HE = new TH1F("etaWidth3HE", "eta width of HE depth-3 clusters", 20, 0.0, 1.0);
    etaWidth4HE = new TH1F("etaWidth4HE", "eta width of HE depth-4 clusters", 20, 0.0, 1.0);
    etaWidth5HE = new TH1F("etaWidth5HE", "eta width of HE depth-5 clusters", 20, 0.0, 1.0);
    phiWidth1HB = new TH1F("phiWidth1HB", "phi width of HB depth-1 clusters", 20, 0.0, 1.0);
    phiWidth2HB = new TH1F("phiWidth2HB", "phi width of HB depth-2 clusters", 20, 0.0, 1.0);
    phiWidth3HB = new TH1F("phiWidth3HB", "phi width of HB depth-3 clusters", 20, 0.0, 1.0);
    phiWidth1HE = new TH1F("phiWidth1HE", "phi width of HE depth-1 clusters", 20, 0.0, 1.0);
    phiWidth2HE = new TH1F("phiWidth2HE", "phi width of HE depth-2 clusters", 20, 0.0, 1.0);
    phiWidth3HE = new TH1F("phiWidth3HE", "phi width of HE depth-3 clusters", 20, 0.0, 1.0);
    phiWidth4HE = new TH1F("phiWidth4HE", "phi width of HE depth-4 clusters", 20, 0.0, 1.0);
    phiWidth5HE = new TH1F("phiWidth5HE", "phi width of HE depth-5 clusters", 20, 0.0, 1.0);
    etaWidthSuperClusterHB = new TH1F("etaWidthSuperClusterHB", "eta width of HB superclusters", 20, 0.0, 1.0);
    phiWidthSuperClusterHB = new TH1F("phiWidthSuperClusterHB", "phi width of HB superclusters", 20, 0.0, 1.0);
    etaWidthSuperClusterHE = new TH1F("etaWidthSuperClusterHE", "eta width of HE superclusters", 20, 0.0, 1.0);
    phiWidthSuperClusterHE = new TH1F("phiWidthSuperClusterHE", "phi width of HE superclusters", 20, 0.0, 1.0);
    sizeSuperClusterHB = new TH1F("sizeSuperClusterHB", "size of HB superclusters", 10, 0.0, 10.0);
    sizeSuperClusterHE = new TH1F("sizeSuperClusterHE", "size of HE superclusters", 10, 0.0, 10.0);
}
}

void
PFSuperClusterAlgo::write() {

  if ( file_ ) {
    file_->cd();
    dR12HB->Write();
    dR13HB->Write();
    dR23HB->Write();
    dR3HOHB->Write();
    dR12HE->Write();
    dR13HE->Write();
    dR23HE->Write();
    dR24HE->Write();
    dR34HE->Write();
    dR35HE->Write();
    dR45HE->Write();
    dEta12HB->Write();
    dEta13HB->Write();
    dEta23HB->Write();
    dEta3HOHB->Write();
    dEta12HE->Write();
    dEta13HE->Write();
    dEta23HE->Write();
    dEta24HE->Write();
    dEta34HE->Write();
    dEta35HE->Write();
    dEta45HE->Write();
    dPhi12HB->Write();
    dPhi13HB->Write();
    dPhi23HB->Write();
    dPhi3HOHB->Write();
    dPhi12HE->Write();
    dPhi13HE->Write();
    dPhi23HE->Write();
    dPhi24HE->Write();
    dPhi34HE->Write();
    dPhi35HE->Write();
    dPhi45HE->Write();
    normalized12HB->Write();
    normalized13HB->Write();
    normalized23HB->Write();
    normalized3HOHB->Write();
    normalized12HE->Write();
    normalized13HE->Write();
    normalized23HE->Write();
    normalized24HE->Write();
    normalized34HE->Write();
    normalized35HE->Write();
    normalized45HE->Write();
    nclustersHB->Write();
    nclustersHE->Write();
    nclustersHO->Write();
    mergeclusters1HB->Write();
    mergeclusters2HB->Write();
    mergeclusters3HB->Write();
    mergeclusters1HE->Write();
    mergeclusters2HE->Write();
    mergeclusters3HE->Write();
    mergeclusters4HE->Write();
    hitsHB->Write();
    hitsHE->Write();
    etaPhi->Write();
    etaPhiHits->Write();
    etaPhiHits1HB->Write();
    etaPhiHits2HB->Write();
    etaPhiHits3HB->Write();
    etaPhiHits1HE->Write();
    etaPhiHits2HE->Write();
    etaPhiHits3HE->Write();
    etaPhiHits4HE->Write();
    etaPhiHits5HE->Write();
    hitTime1HB->Write();
    hitTime2HB->Write();
    hitTime3HB->Write();
    hitTime1HE->Write();
    hitTime2HE->Write();
    hitTime3HE->Write();
    hitTime4HE->Write();
    hitTime5HE->Write();
    etaWidth1HB->Write();
    etaWidth2HB->Write();
    etaWidth3HB->Write();
    etaWidth1HE->Write();
    etaWidth2HE->Write();
    etaWidth3HE->Write();
    etaWidth4HE->Write();
    etaWidth5HE->Write();
    phiWidth1HB->Write();
    phiWidth2HB->Write();
    phiWidth3HB->Write();
    phiWidth1HE->Write();
    phiWidth2HE->Write();
    phiWidth3HE->Write();
    phiWidth4HE->Write();
    phiWidth5HE->Write();
    etaWidthSuperClusterHB->Write();
    phiWidthSuperClusterHB->Write();
    etaWidthSuperClusterHE->Write();
    phiWidthSuperClusterHE->Write();
    sizeSuperClusterHB->Write();
    sizeSuperClusterHE->Write();

    file_->Write();
    cout << "Supercluster Benchmark output written to file " << file_->GetName() << endl;
    file_->Close();
  }

}


void PFSuperClusterAlgo::doClustering( const PFClusterHandle& clustersHandle, const PFClusterHandle& clustersHOHandle ) {
  const reco::PFClusterCollection& clusters = * clustersHandle;
  const reco::PFClusterCollection& clustersHO = * clustersHOHandle;

  // cache the Handle to the clusters
  clustersHandle_ = clustersHandle;
  clustersHOHandle_ = clustersHOHandle;

  // perform clustering
  doClusteringWorker( clusters, clustersHO );
}

void PFSuperClusterAlgo::doClustering( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO ) {
  // using clusters without a Handle, clear to avoid a stale member
  clustersHandle_.clear();
  clustersHOHandle_.clear();

  // perform clustering
  doClusteringWorker( clusters, clustersHO );
}

// calculate cluster position: Rachel Myers, July 2012
std::pair<double, double> PFSuperClusterAlgo::calculatePosition(const reco::PFCluster& cluster)
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
      if (hitPhi > + Geom::pi()) { hitPhi = Geom::twoPi() - hitPhi; }
      if (hitPhi < - Geom::pi()) { hitPhi = Geom::twoPi() - hitPhi; }
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
std::pair<double, double> PFSuperClusterAlgo::calculateWidths(const reco::PFCluster& cluster)
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
    const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = cluster.recHitFractions();
    for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
      const reco::PFRecHitRef rechit = it->recHitRef();
//      rechit->calculatePositionREP();
      double hitEta = rechit->positionREP().Eta();
      double hitPhi = rechit->positionREP().Phi();
      double hitEnergy = rechit->energy();
      double dEta  = hitEta - clusterEta;
      double dPhi  = hitPhi - clusterPhi;

      if (dPhi > + Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }
      if (dPhi < - Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }

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
void PFSuperClusterAlgo::doClusteringWorker( const reco::PFClusterCollection& clusters, const reco::PFClusterCollection& clustersHO ) {

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
  if ( file_ ) {
    nclustersHB->Fill(mclustersHB);
    nclustersHE->Fill(mclustersHE);
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
          if( ihandf==0 ) clusterdepthfirst = depth;
          if( ihandf>0 ) {
            if(depth!=clusterdepthfirst) cout << " Problem with cluster depth: " << depth << " not equal to " << clusterdepthfirst <<endl;
          }
        }
//        delete hitsandfracs;
      }
    }
  std::vector< unsigned > clusterdepthHO(clustersHO.size());
//  cout << " HO clusters: " << clustersHO.size() <<endl; 
  if ( file_ ) {
     nclustersHO->Fill(clustersHO.size());
  }
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
          if( ihandf==0 ) clusterdepthfirst = depth;
          if( ihandf>0 ) {
            if(depth!=clusterdepthfirst) cout << " Problem with HO cluster depth: " << depth << " not equal to " << clusterdepthfirst <<endl;
          }
        }
//        delete hitsandfracs;
      }
    }

  std::vector< unsigned > imerge(clusters.size());
  std::vector< unsigned > imergeHO(clustersHO.size());
  std::vector< bool > lmerge(clusters.size());
  std::vector< bool > lmergeHO(clustersHO.size());

  double hcaleta1=0.0;
  double hcalphi1=0.0;
  double hcaleta2=0.0;
  double hcalphi2=0.0;
  double dR = 0.0;
  double dEta = 0.0;
  double dPhi = 0.0;

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
        if(clusters[ic1].layer()==PFLayer::HCAL_BARREL1) {
          if (file_) {
            etaPhiHits1HB->Fill(hcaleta1, hcalphi1, clusters[ic1].hitsAndFractions().size());
          }
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
            const reco::PFRecHitRef rechit = it->recHitRef();
            double time  = rechit->time();
            if (file_) {
              if(rechit->energy()>1.0) hitTime1HB->Fill(time);
            }
          }
        }
        if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          if (file_) {

            etaPhiHits1HE->Fill(hcaleta1, hcalphi1, clusters[ic1].hitsAndFractions().size());
          }
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
            const reco::PFRecHitRef rechit = it->recHitRef();
            double time  = rechit->time();
            if (file_) {
              if(rechit->energy()>1.0) hitTime1HE->Fill(time);
            }
          }
        }
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
          double etawidth = sqrt(pow(w1,2.0)+pow(w2,2.0));
          double phiwidth = sqrt(pow(w3,2.0)+pow(w4,2.0));
          dEtacut = etaScale*etawidth;
          dPhicut = phiScale*phiwidth;
          if( clusterdepth[ic2]==2 ){
//            cout << " depth 1-2 dR = " << dR <<endl;
            if ( file_ ) {
              if ( abs(hcaleta1) <1.479 ) {
                 dR12HB->Fill(dR);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta12HB->Fill(dEta);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi12HB->Fill(dPhi);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized12HB->Fill(abs(dEta/etawidth));
              } else {
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta12HE->Fill(dEta);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi12HE->Fill(dPhi);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized12HE->Fill(abs(dEta/etawidth));
                 dR12HE->Fill(dR);
              }
            }
            if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
              imerge[ic2]=ic1;
              lmerge[ic2]=true;
            }
          } else if( clusterdepth[ic2]==3 ){
//            cout << " depth 1-3 dR = " << dR <<endl;
            if ( file_ ) {
              if ( abs(hcaleta1) <1.479 ) {
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta13HB->Fill(dEta);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi13HB->Fill(dPhi);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized13HB->Fill(abs(dEta/etawidth));
                 dR13HB->Fill(dR);
              } else {
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta13HE->Fill(dEta);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi13HE->Fill(dPhi);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized13HE->Fill(abs(dEta/etawidth));
                 dR13HE->Fill(dR);
              }
            }
            if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut) ) {
              imerge[ic2]=ic1;
              lmerge[ic2]=true;
            }
          }
        }
      } else if( clusterdepth[ic1]==2 ){
        hcaleta1 = calculatePosition(clusters[ic1]).first;
        hcalphi1 = calculatePosition(clusters[ic1]).second;
        if(clusters[ic1].layer()==PFLayer::HCAL_BARREL1) {
          if (file_) {
            etaPhiHits2HB->Fill(hcaleta1, hcalphi1, clusters[ic1].hitsAndFractions().size());
          }
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
            const reco::PFRecHitRef rechit = it->recHitRef();
            double time  = rechit->time();
            if (file_) {
              if(rechit->energy()>1.0) hitTime2HB->Fill(time);
            }
          }
        }
        if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          if (file_) {
            etaPhiHits2HE->Fill(hcaleta1, hcalphi1, clusters[ic1].hitsAndFractions().size());
          }
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
            const reco::PFRecHitRef rechit = it->recHitRef();
            double time  = rechit->time();
            if (file_) {
              if(rechit->energy()>1.0) hitTime2HE->Fill(time);
            }
          }
        }
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
          double etawidth = sqrt(pow(w1,2.0)+pow(w2,2.0));
          double phiwidth = sqrt(pow(w3,2.0)+pow(w4,2.0));
          dEtacut = etaScale*etawidth;
          dPhicut = phiScale*phiwidth;
          if( clusterdepth[ic2]==3 ){
//            cout << " depth 2-3 dR = " << dR <<endl;
            if ( file_ ) {
              if ( abs(hcaleta1) <1.479 ) {
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta23HB->Fill(dEta);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi23HB->Fill(dPhi);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized23HB->Fill(abs(dEta/etawidth));
                 dR23HB->Fill(dR);
              } else {
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta23HE->Fill(dEta);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi23HE->Fill(dPhi);
                 if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized23HE->Fill(abs(dEta/etawidth));
                 dR23HE->Fill(dR);
              }
            }
            if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
              imerge[ic2]=ic1;
              lmerge[ic2]=true;
            }
          } else if( clusterdepth[ic2]==4 ){
//            cout << " depth 2-4 dR = " << dR <<endl;
            if ( file_ ) {
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta24HE->Fill(dEta);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi24HE->Fill(dPhi);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized24HE->Fill(abs(dEta/etawidth));
              dR24HE->Fill(dR);
            }
            if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
              imerge[ic2]=ic1;
              lmerge[ic2]=true;
            }
          }
        }
      } else if( clusterdepth[ic1]==3 ){
        hcaleta1 = calculatePosition(clusters[ic1]).first;
        hcalphi1 = calculatePosition(clusters[ic1]).second;
        if(clusters[ic1].layer()==PFLayer::HCAL_BARREL1) {
          if ( file_ ) {
            etaPhiHits3HB->Fill(hcaleta1, hcalphi1, clusters[ic1].hitsAndFractions().size());
          }
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
            const reco::PFRecHitRef rechit = it->recHitRef();
            double time  = rechit->time();
            if ( file_ ) {
              if(rechit->energy()>1.0) hitTime3HB->Fill(time);
            }
          }
        }
        if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          if ( file_ ) {
            etaPhiHits3HE->Fill(hcaleta1, hcalphi1, clusters[ic1].hitsAndFractions().size());
          }
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
            const reco::PFRecHitRef rechit = it->recHitRef();
            double time  = rechit->time();
            if ( file_ ) {
              if(rechit->energy()>1.0) hitTime3HE->Fill(time);
            }
          }
        }
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
          double etawidth = sqrt(pow(w1,2.0)+pow(w2,2.0));
          double phiwidth = sqrt(pow(w3,2.0)+pow(w4,2.0));
          dEtacut = etaScale*etawidth;
          dPhicut = phiScale*phiwidth;
          if( clusterdepth[ic2]==4 ){
//            cout << " depth 3-4 dR = " << dR <<endl;
            if ( file_ ) {
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta34HE->Fill(dEta);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi34HE->Fill(dPhi);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized34HE->Fill(abs(dEta/etawidth));
              dR34HE->Fill(dR);
            }
            if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
              imerge[ic2]=ic1;
              lmerge[ic2]=true;
            }
          } else if( clusterdepth[ic2]==5 ){
//            cout << " depth 3-5 dR = " << dR <<endl;
            if ( file_ ) {
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta35HE->Fill(dEta);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi35HE->Fill(dPhi);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized35HE->Fill(abs(dEta/etawidth));
              dR35HE->Fill(dR);
            }
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
          double etawidth = sqrt(pow(w1,2.0)+pow(w2,2.0));
          double phiwidth = sqrt(pow(w3,2.0)+pow(w4,2.0));
          dEtacut = etaScale*etawidth;
          dPhicut = phiScale*phiwidth;
          if( clusterdepthHO[ic2]==5 ){
//            cout << " depth 3-HO dR = " << dR <<endl;
            if ( file_ ) {
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clustersHO[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clustersHO[ic2].energy() > 1.0)) dEta3HOHB->Fill(dEta);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clustersHO[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clustersHO[ic2].energy() > 1.0)) dPhi3HOHB->Fill(dPhi);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clustersHO[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clustersHO[ic2].energy() > 1.0)) normalized3HOHB->Fill(abs(dEta/etawidth));
              dR3HOHB->Fill(dR);
            }
            if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
              imergeHO[ic2]=ic1;
              lmergeHO[ic2]=true;
            }
          }
        }
      } else if( clusterdepth[ic1]==4 ){
        hcaleta1 = calculatePosition(clusters[ic1]).first;
        hcalphi1 = calculatePosition(clusters[ic1]).second;
        if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          if ( file_ ) {
            etaPhiHits4HE->Fill(hcaleta1, hcalphi1, clusters[ic1].hitsAndFractions().size());
          }
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
            const reco::PFRecHitRef rechit = it->recHitRef();
            double time  = rechit->time();
            if ( file_ ) {
              if(rechit->energy()>1.0) hitTime4HE->Fill(time);
            }
          }
        }
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
          double etawidth = sqrt(pow(w1,2.0)+pow(w2,2.0));
          double phiwidth = sqrt(pow(w3,2.0)+pow(w4,2.0));
          dEtacut = etaScale*etawidth;
          dPhicut = phiScale*phiwidth;
          if( clusterdepth[ic2]==5 ){
//            cout << " depth 4-5 dR = " << dR <<endl;
            if ( file_ ) {
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dEta45HE->Fill(dEta);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) dPhi45HE->Fill(dPhi);
              if ((clusters[ic1].recHitFractions().size() > 3.0) && (clusters[ic2].recHitFractions().size() > 3.0) && (clusters[ic1].energy() > 1.0) && (clusters[ic2].energy() > 1.0)) normalized45HE->Fill(abs(dEta/etawidth));
              dR45HE->Fill(dR);
            }
            if((dR<dRcut) || (dEta<dEtacut && dPhi<dPhicut)) {
              imerge[ic2]=ic1;
              lmerge[ic2]=true;
            }
          }
        }
      } else if( clusterdepth[ic1]==5 ){
        hcaleta1 = calculatePosition(clusters[ic1]).first;
        hcalphi1 = calculatePosition(clusters[ic1]).second;
        if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          if ( file_ ) {
            etaPhiHits5HE->Fill(hcaleta1, hcalphi1, clusters[ic1].hitsAndFractions().size());
          }
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
            const reco::PFRecHitRef rechit = it->recHitRef();
            double time  = rechit->time();
            if ( file_ ) {
              if(rechit->energy()>1.0) hitTime5HE->Fill(time);
            }
          }
        }
      }
    }

    // start a supercluster with a depth=1 cluster, then loop on all other
    // clusters to add to cluster list, then for each cluster to add, loop
    // on remaining clusters to check 2nd level of addition, repeat for all layers

  // need to add HO cluster logic
  std::vector< reco::PFCluster >  mergeclusters;
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
          mergeclusters.push_back(clusters[ic1]);
          for (unsigned short ic=0; ic<clusters.size();++ic) {
            if(lmergeclusters[ic]) {
              mergeclusters.push_back(clusters[ic]);
            }
          }
          if(mergeclusters.size()>0) {
//            cout << " number of clusters to merge: " <<mergeclusters.size()<<endl;
            if ( file_ ) {
               if ( clusters[ic1].layer() == PFLayer::HCAL_BARREL1) {
                  if( clusterdepth[ic1]==1 ){
                     mergeclusters1HB->Fill(mergeclusters.size());
                  } else if( clusterdepth[ic1]==2 ){
                     mergeclusters2HB->Fill(mergeclusters.size());
                  } else if( clusterdepth[ic1]==3 ){
                     mergeclusters3HB->Fill(mergeclusters.size());
                  }
               } else if ( clusters[ic1].layer() == PFLayer::HCAL_ENDCAP) {
                  if( clusterdepth[ic1]==1 ){
                     mergeclusters1HE->Fill(mergeclusters.size());
                  } else if( clusterdepth[ic1]==2 ){
                     mergeclusters2HE->Fill(mergeclusters.size());
                  } else if( clusterdepth[ic1]==3 ){
                     mergeclusters3HE->Fill(mergeclusters.size());
                  } else if( clusterdepth[ic1]==4 ){
                     mergeclusters4HE->Fill(mergeclusters.size());
                  }
               } else {
                  cout << " unknown cluster layer: " << clusters[ic1].layer() <<endl;
               }
            }
            reco::PFSuperCluster ipfsupercluster(mergeclusters);
            if ( file_ ) {
              double hcaleta  = ipfsupercluster.positionREP().Eta();
              double hcalphi  = ipfsupercluster.positionREP().Phi();
              double widthEta = calculateWidths(ipfsupercluster).first;
              double widthPhi = calculateWidths(ipfsupercluster).second;
              etaPhi->Fill(hcaleta, hcalphi);
              const std::vector< std::pair<DetId, float> > & hitsandfracs =
                    ipfsupercluster.hitsAndFractions();
              etaPhiHits->Fill(hcaleta, hcalphi, hitsandfracs.size());
              if ( ipfsupercluster.layer() == PFLayer::HCAL_BARREL1) {
                etaWidthSuperClusterHB->Fill(widthEta);
                phiWidthSuperClusterHB->Fill(widthPhi);
                hitsHB->Fill(hitsandfracs.size());
                sizeSuperClusterHB->Fill(ipfsupercluster.size());
              } else if ( ipfsupercluster.layer() == PFLayer::HCAL_ENDCAP) {
                hitsHE->Fill(hitsandfracs.size());
                etaWidthSuperClusterHE->Fill(widthEta);
                phiWidthSuperClusterHE->Fill(widthPhi);
                sizeSuperClusterHE->Fill(ipfsupercluster.size());
              } else {
                 cout << " unknown supercluster layer: " << ipfsupercluster.layer() <<endl;
              }
            }
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
// do widths: Rachel Myers, July 2012    
  for (unsigned short ic1=0; ic1<clusters.size();++ic1) {
      if( clusterdepth[ic1]==1 ){
        if(clusters[ic1].layer()==PFLayer::HCAL_BARREL1) {
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
            double widthEta = calculateWidths(clusters[ic1]).first; 
            double widthPhi = calculateWidths(clusters[ic1]).second;
            if ( file_ ) {
              etaWidth1HB->Fill(widthEta);
              phiWidth1HB->Fill(widthPhi);
            }
          }
        }
       if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
          double widthEta = calculateWidths(clusters[ic1]).first;
          double widthPhi = calculateWidths(clusters[ic1]).second;
          if ( file_ ) {
            etaWidth1HE->Fill(widthEta);
            phiWidth1HE->Fill(widthPhi);
          }
          }
         }
        } else if( clusterdepth[ic1]==2 ){
        if(clusters[ic1].layer()==PFLayer::HCAL_BARREL1) {
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
          double widthEta = calculateWidths(clusters[ic1]).first;
          double widthPhi = calculateWidths(clusters[ic1]).second;
          if ( file_ ) {
            etaWidth2HB->Fill(widthEta);
            phiWidth2HB->Fill(widthPhi);
          }
          }
        }
        if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
          double widthEta = calculateWidths(clusters[ic1]).first;
          double widthPhi = calculateWidths(clusters[ic1]).second;
          if ( file_ ) {
            etaWidth2HE->Fill(widthEta);
            phiWidth2HE->Fill(widthPhi);
          }
          }
        }
        }  else if( clusterdepth[ic1]==3 ){
        if(clusters[ic1].layer()==PFLayer::HCAL_BARREL1) {
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
          double widthEta = calculateWidths(clusters[ic1]).first;
          double widthPhi = calculateWidths(clusters[ic1]).second;
          if ( file_ ) {
            etaWidth3HB->Fill(widthEta);
            phiWidth3HB->Fill(widthPhi);
          }
          }
        }
        if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
          double widthEta = calculateWidths(clusters[ic1]).first;
          double widthPhi = calculateWidths(clusters[ic1]).second;
          if ( file_ ) {
            etaWidth3HE->Fill(widthEta);
            phiWidth3HE->Fill(widthPhi);
          }
          }
        }
      } else if( clusterdepth[ic1]==4 ){
        if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
          double widthEta = calculateWidths(clusters[ic1]).first;
          double widthPhi = calculateWidths(clusters[ic1]).second;
          if ( file_ ) {
            etaWidth4HE->Fill(widthEta);
            phiWidth4HE->Fill(widthPhi);
          }
          }
        }
      } else if( clusterdepth[ic1]==5 ){
        if(clusters[ic1].layer()==PFLayer::HCAL_ENDCAP) {
          const std::vector< reco::PFRecHitFraction >& pfhitsandfracs = clusters[ic1].recHitFractions();
          for (std::vector<reco::PFRecHitFraction>::const_iterator it = pfhitsandfracs.begin(); it != pfhitsandfracs.end(); ++it) {
          double widthEta = calculateWidths(clusters[ic1]).first;
          double widthPhi = calculateWidths(clusters[ic1]).second;
          if ( file_ ) {
            etaWidth5HE->Fill(widthEta);
            phiWidth5HE->Fill(widthPhi);
          }
         }
        }
      }
    }
}
ostream& operator<<(ostream& out,const PFSuperClusterAlgo& algo) { 
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
