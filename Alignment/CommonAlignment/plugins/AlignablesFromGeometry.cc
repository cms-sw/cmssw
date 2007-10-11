#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "Alignment/CommonAlignment/interface/AlignSetup.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Alignment/CommonAlignment/plugins/AlignablesFromGeometry.h"

AlignablesFromGeometry::AlignablesFromGeometry(const edm::ParameterSet& cfg):
  theTrackerCfg( cfg.getUntrackedParameter<edm::ParameterSet>("tracker", edm::ParameterSet() ) ),
  theMuonDetCfg( cfg.getUntrackedParameter<edm::ParameterSet>("muonDet", edm::ParameterSet() ) )
{
}

void AlignablesFromGeometry::analyze(const edm::Event&,
				     const edm::EventSetup&)
{
  if ( !theTrackerCfg.empty() )
  {
    const TrackerGeometry& tracker = AlignSetup<TrackerGeometry>::find();

    detsToAlignables( tracker.detsPXB(), theTrackerCfg.getParameter<std::string>("pixelBarrel") );
    detsToAlignables( tracker.detsPXF(), theTrackerCfg.getParameter<std::string>("pixelEndcap") );
    detsToAlignables( tracker.detsTIB(), theTrackerCfg.getParameter<std::string>("innerBarrel") );
    detsToAlignables( tracker.detsTID(), theTrackerCfg.getParameter<std::string>("innerEndcap") );
    detsToAlignables( tracker.detsTOB(), theTrackerCfg.getParameter<std::string>("outerBarrel") );
    detsToAlignables( tracker.detsTEC(), theTrackerCfg.getParameter<std::string>("outerEndcap") );
  }

  if ( !theMuonDetCfg.empty() )
  {
    const  DTGeometry& muonDT  = AlignSetup< DTGeometry>::find();
    const CSCGeometry& muonCSC = AlignSetup<CSCGeometry>::find();

    detsToAlignables( muonDT .dets(), theMuonDetCfg.getParameter<std::string>("muonBarrel") );
    detsToAlignables( muonCSC.dets(), theMuonDetCfg.getParameter<std::string>("muonEndcap") );
  }
}

void AlignablesFromGeometry::detsToAlignables(const DetContainer& dets,
					      const std::string& name)
{
  unsigned int nDet = dets.size();

  align::Alignables& alis = AlignSetup<align::Alignables>::get(name);

  alis.reserve(nDet);

  for (unsigned int i = 0; i < nDet; ++i)
  {
    SiStripDetId detId = dets[i]->geographicalId();

    if ( !detId.glued() ) // not a component of glued det
    {
      alis.push_back( new AlignableDet(dets[i]) );
    }
  }
//   buildAlignableMap(alis);
}

void AlignablesFromGeometry::buildAlignableMap(const align::Alignables& dets)
{
  typedef std::map<align::ID, Alignable*> AlignMap;

  AlignMap& allDets = AlignSetup<AlignMap>::get("allDets"); // for AlignableNavigator

  unsigned int nDet = dets.size();

  for (unsigned int i = 0; i < nDet; ++i)
  {
    Alignable* det = dets[i];

    if (!allDets.insert( std::make_pair(det->id(), det) ).second) // insert fails
    {
      throw cms::Exception("AlignablesFromGeometryError")
	<< "Det of id " << det->id() << " already exists in Alignable Map.";
    }
  }
}
