#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotSimTracks.h"
#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotUtils.h"
#include "QCDAnalysis/ChargedHadronSpectra/interface/HitInfo.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// Ecal
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

using namespace std;

/*****************************************************************************/
struct sortByPabs
{
  bool operator() (const PSimHit& a, const PSimHit& b) const
  {
    return (a.pabs() > b.pabs());
  }
};

/*****************************************************************************/
struct sortByTof
{
  bool operator() (const PSimHit& a, const PSimHit& b) const
  {
    return (a.tof() < b.tof());
  }
};

/*****************************************************************************/
PlotSimTracks::PlotSimTracks
  (const edm::EventSetup& es, ofstream& file_) : file(file_)
{
  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> trackerHandle;
  es.get<TrackerDigiGeometryRecord>().get(trackerHandle);
  theTracker = trackerHandle.product();
 
  // Get calorimetry
  edm::ESHandle<CaloGeometry> calo;
  es.get<CaloGeometryRecord>().get(calo);
  theCaloGeometry = (const CaloGeometry*)calo.product();
}

/*****************************************************************************/
PlotSimTracks::~PlotSimTracks()
{
}

/*****************************************************************************/
void PlotSimTracks::printSimTracks(const edm::Event& ev, const edm::EventSetup& es)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Tracker
  edm::Handle<TrackingParticleCollection> simTrackHandle;
  ev.getByLabel("mix",            simTrackHandle);
  const TrackingParticleCollection* simTracks = simTrackHandle.product();

  // Ecal
  edm::Handle<edm::PCaloHitContainer>      simHitsBarrel;
  ev.getByLabel("g4SimHits", "EcalHitsEB", simHitsBarrel);
   
//  edm::Handle<edm::PCaloHitContainer>      simHitsPreshower;
//  ev.getByLabel("g4SimHits", "EcalHitsES", simHitsPreshower);

  edm::Handle<edm::PCaloHitContainer>      simHitsEndcap;
  ev.getByLabel("g4SimHits", "EcalHitsEE", simHitsEndcap);

// FIXME
/*
  {
  edm::Handle<edm::SimTrackContainer>  simTracks;
  ev.getByType<edm::SimTrackContainer>(simTracks);
  std::cerr << " SSSSS " << simTracks.product()->size() << std::endl;

  for(edm::SimTrackContainer::const_iterator t = simTracks.product()->begin();
                                             t!= simTracks.product()->end(); t++)
  {
    std::cerr << " simTrack " << t - simTracks.product()->begin()
         << " " << t->type()
         << " " << t->charge()
         << " " << t->vertIndex()
         << " " << t->genpartIndex()
         << " " << t->momentum().x()
         << " " << t->momentum().y()
         << " " << t->momentum().z()
         << std::endl;
  } 

  }
*/

  const CaloSubdetectorGeometry* geom;

  // Utilities
  PlotUtils plotUtils;

  file << ", If[st, {RGBColor[0.5,0.5,0.5]";

  for(TrackingParticleCollection::const_iterator simTrack = simTracks->begin();
                                                 simTrack!= simTracks->end();
                                                 simTrack++)
  {
    std::vector<PSimHit> simHits;

#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
    simHits = simTrack->trackPSimHit();
#endif

    // reorder with help of momentum
    sort(simHits.begin(), simHits.end(), sortByPabs());

    for(std::vector<PSimHit>::const_iterator simHit = simHits.begin();
                                        simHit!= simHits.end(); simHit++)
    {
      DetId id = DetId(simHit->detUnitId());

      if(theTracker->idToDetUnit(id) != 0)
      {  
      GlobalPoint  p1 =
        theTracker->idToDetUnit(id)->toGlobal(simHit->localPosition()); 
      GlobalVector v1 =
        theTracker->idToDetUnit(id)->toGlobal(simHit->localDirection());

      // simHit
      file << ", Point[{" << p1.x() << "," << p1.y() << ",(" << p1.z() << "-zs)*mz}]"
           << std::endl;
      file << ", Text[StyleForm[\"s\", URL->\"SimHit | Ekin="
           << simTrack->energy() - simTrack->mass()
           << " GeV | parent: source="
           << simTrack->parentVertex()->nSourceTracks() 
           << " daughter=" << simTrack->parentVertex()->nDaughterTracks()
           << HitInfo::getInfo(*simHit, tTopo) << "\"], {"
           << p1.x() << "," << p1.y() << ",(" << p1.z() << "-zs)*mz}, {1,1}]"
           << std::endl;

      // det
      double x = theTracker->idToDet(id)->surface().bounds().width() /2;
      double y = theTracker->idToDet(id)->surface().bounds().length()/2;
      double z = 0.;
  
      GlobalPoint p00 =  theTracker->idToDet(id)->toGlobal(LocalPoint(-x,-y,z));
      GlobalPoint p01 =  theTracker->idToDet(id)->toGlobal(LocalPoint(-x, y,z));
      GlobalPoint p10 =  theTracker->idToDet(id)->toGlobal(LocalPoint( x,-y,z));
      GlobalPoint p11 =  theTracker->idToDet(id)->toGlobal(LocalPoint( x, y,z));

      if(theTracker->idToDet(id)->subDetector() ==
           GeomDetEnumerators::PixelBarrel ||
         theTracker->idToDet(id)->subDetector() ==
           GeomDetEnumerators::PixelEndcap)
        file << ", If[sd, {RGBColor[0.6,0.6,0.6], ";
      else
        file << ", If[sd, {RGBColor[0.8,0.8,0.8], ";

      file       <<"Line[{{"<< p00.x()<<","<<p00.y()<<",("<<p00.z()<<"-zs)*mz}, "
                       <<"{"<< p01.x()<<","<<p01.y()<<",("<<p01.z()<<"-zs)*mz}, "
                       <<"{"<< p11.x()<<","<<p11.y()<<",("<<p11.z()<<"-zs)*mz}, "
                       <<"{"<< p10.x()<<","<<p10.y()<<",("<<p10.z()<<"-zs)*mz}, "
                       <<"{"<< p00.x()<<","<<p00.y()<<",("<<p00.z()<<"-zs)*mz}}]}]"
        << std::endl;

      if(simHit == simHits.begin()) // vertex to first point
      {
        GlobalPoint p0(simTrack->vertex().x(),
                       simTrack->vertex().y(),
                       simTrack->vertex().z());
        plotUtils.printHelix(p0,p1,v1, file, simTrack->charge());
      }

      if(simHit+1 != simHits.end()) // if not last
      {
        DetId id = DetId((simHit+1)->detUnitId());
        GlobalPoint  p2 =
          theTracker->idToDetUnit(id)->toGlobal((simHit+1)->localPosition());
        GlobalVector v2 =
          theTracker->idToDetUnit(id)->toGlobal((simHit+1)->localDirection());

        plotUtils.printHelix(p1,p2,v2, file, simTrack->charge());
      }

      // Continue to Ecal
      if(simHit+1 == simHits.end()) // if last
      {
        DetId id = DetId(simHit->detUnitId());
        GlobalPoint p =
          theTracker->idToDetUnit(id)->toGlobal(simHit->localPosition());

        // EB
        geom = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);

        for(edm::PCaloHitContainer::const_iterator
              simHit = simHitsBarrel->begin();
              simHit!= simHitsBarrel->end(); simHit++)
	  if(simHit->geantTrackId() == static_cast<int>(simTrack->g4Track_begin()->trackId()) && //the sign of trackId tells whether there was a match  
           simHit->energy() > 0.060)
        {
          EBDetId detId(simHit->id());
          const CaloCellGeometry* cell = geom->getGeometry(detId);

          if(cell != 0)
          file << ", Line[{{" << p.x()
                       << "," << p.y()
                       << ",(" << p.z() <<"-zs)*mz}"
               << ", {" << cell->getPosition().x() << ","
                        << cell->getPosition().y() << ",("
                        << cell->getPosition().z() << "-zs)*mz}}]" << std::endl;
        }

        // ES

        // EE
        geom = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);

        for(edm::PCaloHitContainer::const_iterator
              simHit = simHitsEndcap->begin();
              simHit!= simHitsEndcap->end(); simHit++)
	  if(simHit->geantTrackId() == static_cast<int>(simTrack->g4Track_begin()->trackId()) && //the sign of trackId tells whether there was a match
           simHit->energy() > 0.060)
        {
          EEDetId detId(simHit->id());
          const CaloCellGeometry* cell = geom->getGeometry(detId);

          if(cell != 0)
          file << ", Line[{{" << p.x()
                       << "," << p.y()
                       << ",(" << p.z() << "-zs)*mz}"
               << ", {" << cell->getPosition().x() << ","
                        << cell->getPosition().y() << ",("
                        << cell->getPosition().z() << "-zs)*mz}}]" << std::endl;
        }
      }
      }
    }
  }

  file << "}]";
}

