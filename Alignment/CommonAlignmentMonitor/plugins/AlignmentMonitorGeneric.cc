#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "Alignment/CommonAlignmentMonitor/plugins/AlignmentMonitorGeneric.h"

#include <TString.h>

AlignmentMonitorGeneric::AlignmentMonitorGeneric(const edm::ParameterSet& cfg):
  AlignmentMonitorBase(cfg)
{
}

void AlignmentMonitorGeneric::book()
{
  TrackerAlignableId idMap;

  std::vector<std::string> residNames; // names of residual histograms

  residNames.push_back("x hit residuals pos track");
  residNames.push_back("x hit residuals neg track");
  residNames.push_back("y hit residuals pos track");
  residNames.push_back("y hit residuals neg track");

  const std::vector<Alignable*>& alignables = pStore()->alignables();

  unsigned int nAlignable = alignables.size();
  unsigned int nResidName = residNames.size();

  for (unsigned int i = 0; i < nAlignable; ++i)
  {
    const Alignable* ali = alignables[i];

    Hist1Ds& hists = m_resHists[ali];

    hists.resize(nResidName, 0);

    TrackerAlignableId::UniqueId id = idMap.alignableUniqueId(ali);

    for (unsigned int n = 0; n < nResidName; ++n)
    {
      const std::string& name = residNames[n];

      TString histName(name.c_str());
      histName += Form("_%s_%d", idMap.alignableTypeIdToName(id.second).c_str(), id.first);
      histName.ReplaceAll(" ", "");

      TString histTitle(name.c_str());
      histTitle += Form(" for %s with ID %d (subdet %d)",
			idMap.alignableTypeIdToName(id.second).c_str(),
			id.first, DetId(id.first).subdetId());

      TH1F* hist = new TH1F(histName, histTitle, nBin_, -5., 5.);
      hists[n] = static_cast<TH1F*>( add("/iterN/" + name + '/', hist) );
    }
  }

  m_trkHists.resize(6, 0);

  m_trkHists[0] = new TH1F("pt"  , "track p_{t} (GeV)" , nBin_,   0.0, 10.0);
  m_trkHists[1] = new TH1F("eta" , "track #eta"        , nBin_, - 3.0,  3.0);
  m_trkHists[2] = new TH1F("phi" , "track #phi"        , nBin_, -M_PI, M_PI);
  m_trkHists[3] = new TH1F("d0"  , "track d0 (cm)"     , nBin_, - 0.1,  0.1);
  m_trkHists[4] = new TH1F("dz"  , "track dz (cm)"     , nBin_, -10.0, 10.0);
  m_trkHists[5] = new TH1F("chi2", "track #chi^{2}/dof", nBin_,   0.0, 10.0);

  for (unsigned int h = 0; h < m_trkHists.size(); ++h)
    m_trkHists[h] = static_cast<TH1F*>( add("/iterN/", m_trkHists[h]) );
}

void AlignmentMonitorGeneric::event(const edm::EventSetup&,
				    const ConstTrajTrackPairCollection& tracks)
{
  static TrajectoryStateCombiner tsoscomb;

  for (unsigned int t = 0; t < tracks.size(); ++t)
  {
    const reco::Track* track = tracks[t].second;

    float charge = tracks[t].second->charge();

    const std::vector<TrajectoryMeasurement>& meass
      = tracks[t].first->measurements();

    for (unsigned int m = 0; m < meass.size(); ++m)
    {
      const TrajectoryMeasurement& meas = meass[m];
      const TransientTrackingRecHit& hit = *meas.recHit();

      if ( hit.isValid() )
      {
	const Alignable* ali = pNavigator()->alignableFromDetId( hit.geographicalId() );

	while (ali) {
	  std::map<const Alignable*, Hist1Ds>::iterator h = m_resHists.find(ali);
	  if ( h != m_resHists.end() )
	    {
	      TrajectoryStateOnSurface tsos = tsoscomb( meas.forwardPredictedState(), meas.backwardPredictedState() );
	      
	      align::LocalVector res = tsos.localPosition() - hit.localPosition();
	      LocalError err1 = tsos.localError().positionError();
	      LocalError err2 = hit.localPositionError();
	      
	      float errX = std::sqrt( err1.xx() + err2.xx() );
	      float errY = std::sqrt( err1.yy() + err2.yy() );
	      
	      h->second[charge > 0 ? 0 : 1]->Fill(res.x() / errX);
	      h->second[charge > 0 ? 2 : 3]->Fill(res.y() / errY);
	    }
	  ali = ali->mother();
	}
      }
    }

    m_trkHists[0]->Fill( track->pt() );
    m_trkHists[1]->Fill( track->eta() );
    m_trkHists[2]->Fill( track->phi() );
    m_trkHists[3]->Fill( track->d0() );
    m_trkHists[4]->Fill( track->dz() );
    m_trkHists[5]->Fill( track->normalizedChi2() );
  }
}

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorGeneric, "AlignmentMonitorGeneric");
  
