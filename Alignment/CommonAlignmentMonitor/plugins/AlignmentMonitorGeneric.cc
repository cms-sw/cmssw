#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

#include "Alignment/CommonAlignmentMonitor/plugins/AlignmentMonitorGeneric.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h> 
#include "TObject.h" 

#include <TString.h>

AlignmentMonitorGeneric::AlignmentMonitorGeneric(const edm::ParameterSet& cfg):
  AlignmentMonitorBase(cfg, "AlignmentMonitorGeneric")
{
}

void AlignmentMonitorGeneric::book()
{
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

    align::ID id = ali->id();
    align::StructureType type = ali->alignableObjectId();

    for (unsigned int n = 0; n < nResidName; ++n)
    {
      const std::string& name = residNames[n];

      TString histName(name.c_str());
      histName += Form("_%s_%d", AlignableObjectId::idToString(type), id);
      histName.ReplaceAll(" ", "");

      TString histTitle(name.c_str());
      histTitle += Form(" for %s with ID %d (subdet %d)",
			AlignableObjectId::idToString(type),
			id, DetId(id).subdetId());

      hists[n] = book1D(std::string("/iterN/") + std::string(name) + std::string("/"), std::string(histName), std::string(histTitle), nBin_, -5., 5.);
    }
  }

  m_trkHists.resize(6, 0);
  
  m_trkHists[0] = book1D("/iterN/", "pt"  , "track p_{t} (GeV)" , nBin_,   0.0,100.0);
  m_trkHists[1] = book1D("/iterN/", "eta" , "track #eta"        , nBin_, - 3.0,  3.0);
  m_trkHists[2] = book1D("/iterN/", "phi" , "track #phi"        , nBin_, -M_PI, M_PI);
  m_trkHists[3] = book1D("/iterN/", "d0"  , "track d0 (cm)"     , nBin_, -0.02, 0.02);
  m_trkHists[4] = book1D("/iterN/", "dz"  , "track dz (cm)"     , nBin_, -20.0, 20.0);
  m_trkHists[5] = book1D("/iterN/", "chi2", "track #chi^{2}/dof", nBin_,   0.0, 20.0);

}

void AlignmentMonitorGeneric::event(const edm::Event &iEvent,
				    const edm::EventSetup&,
				    const ConstTrajTrackPairCollection& tracks)
{
  TrajectoryStateCombiner tsoscomb;

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
  
