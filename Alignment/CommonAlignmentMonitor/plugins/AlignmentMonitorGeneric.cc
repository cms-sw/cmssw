#include "TH3F.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "Alignment/CommonAlignmentMonitor/plugins/AlignmentMonitorGeneric.h"

AlignmentMonitorGeneric::AlignmentMonitorGeneric(const edm::ParameterSet& cfg):
  AlignmentMonitorBase(cfg)
{
}

void AlignmentMonitorGeneric::book()
{
  static TrackerAlignableId idMap;

  const std::vector<Alignable*>& alignables = pStore()->alignables();

  unsigned int nAlignable = alignables.size();

  for (unsigned int i = 0; i < nAlignable; ++i)
  {
    const Alignable* ali = alignables[i];

    TrackerAlignableId::UniqueId id = idMap.alignableUniqueId(ali);

    std::ostringstream name;
    std::ostringstream title;

    name << "hitResidual" << id.first;
    title << "Hit residual for " << idMap.alignableTypeIdToName(id.second)
	  << " with ID " << id.first;

    TH3F*& his = m_residuals[ali]
      = new TH3F(name.str().c_str(), title.str().c_str(),
		 100, -5., 5.,
		 100, -5., 5.,
		 100, -5., 5.);

    add("/iterN/", his);
  }
}

void AlignmentMonitorGeneric::event(const edm::EventSetup&,
				    const ConstTrajTrackPairCollection& tracks)
{
  static TrajectoryStateCombiner tsoscomb;

  for (unsigned int t = 0; t < tracks.size(); ++t)
  {
    const std::vector<TrajectoryMeasurement>& meass
      = tracks[t].first->measurements();

    for (unsigned int m = 0; m < meass.size(); ++m)
    {
      const TrajectoryMeasurement& meas = meass[m];
      const TransientTrackingRecHit& hit = *meas.recHit();

      if ( hit.isValid() )
      {
	const Alignable* ali = pNavigator()->alignableFromDetId( hit.geographicalId() );

	std::map<const Alignable*, TH3F*>::iterator h = m_residuals.find(ali);

	while ( h == m_residuals.end() && ( ali = ali->mother() ) )
	  h = m_residuals.find(ali);

	if ( h != m_residuals.end() )
	{
	  TrajectoryStateOnSurface tsos = tsoscomb( meas.forwardPredictedState(), meas.backwardPredictedState() );

	  align::LocalVector dr = tsos.localPosition() - hit.localPosition();

	  h->second->Fill( dr.x(), dr.y(), dr.z() );
	}
      }
    }
  }
}

#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"

DEFINE_EDM_PLUGIN(AlignmentMonitorPluginFactory, AlignmentMonitorGeneric, "AlignmentMonitorGeneric");
