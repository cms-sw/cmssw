// ROOT includes
#include <Math/VectorUtil.h>

#include "RecoHI/HiEgammaAlgos/interface/TrackIsoCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/Vector3D.h"

using namespace edm;
using namespace reco;
using namespace std;

TrackIsoCalculator::TrackIsoCalculator (const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::Handle<reco::TrackCollection> trackLabel, const std::string trackQuality)
{
  recCollection = trackLabel;
  trackQuality_ = trackQuality;
}

double TrackIsoCalculator::getTrackIso(const reco::Photon cluster, const double x, const double threshold, const double innerDR)
{
  double TotalPt = 0;

  for(reco::TrackCollection::const_iterator
	recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
  {
    bool goodtrack = recTrack->quality(reco::TrackBase::qualityByName(trackQuality_));
    if(!goodtrack) continue;

    double pt = recTrack->pt();
    double dR2 = reco::deltaR2(cluster, *recTrack);
    if(dR2 >= (0.01 * x*x))
      continue;
    if(dR2 < innerDR*innerDR)
      continue;
    if(pt > threshold)
      TotalPt = TotalPt + pt;
  }

  return TotalPt;
}

double TrackIsoCalculator::getBkgSubTrackIso(const reco::Photon cluster, const double x, const double threshold, const double innerDR)
{
  double SClusterEta = cluster.eta();
  double TotalPt = 0;

  TotalPt = 0;

  for(reco::TrackCollection::const_iterator
	recTrack = recCollection->begin(); recTrack!= recCollection->end(); recTrack++)
  {
    bool goodtrack = recTrack->quality(reco::TrackBase::qualityByName(trackQuality_));
    if(!goodtrack) continue;

    double pt = recTrack->pt();
    double eta2 = recTrack->eta();
    double dEta = fabs(eta2-SClusterEta);
    double dR2 = reco::deltaR2(cluster, *recTrack);
    if(dEta >= 0.1 * x)
      continue;
    if(dR2 < innerDR*innerDR)
      continue;

    if(pt > threshold)
      TotalPt = TotalPt + pt;
  }

  double Tx = getTrackIso(cluster,x,threshold,innerDR);
  double CTx = (Tx - TotalPt / 40.0 * x)*(1/(1-x/40.));

  return CTx;
}
