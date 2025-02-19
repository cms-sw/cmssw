#include "QCDAnalysis/ChargedHadronSpectra/interface/PlotEcalRecHits.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

using namespace std;

/*****************************************************************************/
PlotEcalRecHits::PlotEcalRecHits
  (const edm::EventSetup& es, ofstream& file_) : file(file_)
{
  // Get calorimetry
  edm::ESHandle<CaloGeometry> calo;
  es.get<CaloGeometryRecord>().get(calo);
  theCaloGeometry = (const CaloGeometry*)calo.product();
}

/*****************************************************************************/
PlotEcalRecHits::~PlotEcalRecHits()
{
}

/*****************************************************************************/
void PlotEcalRecHits::printEcalRecHit
  (const CaloCellGeometry* cell, float energy)
{ 
  float x = energy;
  if(x > 1) x = 1.;
  float red   = fabs(2*x-0.5);
  float green = sin(180*x);
  float blue  = cos( 90*x);
  
  file << ", PointSize[" << energy*0.01 << "]";
  file << ", RGBColor[" << red << "," << green << "," << blue << "]";
  file << ", Point[{" << cell->getPosition().x() << ","
                      << cell->getPosition().y() << ",("
                      << cell->getPosition().z() << "-zs)*mz}]" << std::endl;

  const CaloCellGeometry::CornersVec & p(cell->getCorners()) ;

  file << ", If[sd, {RGBColor[0.5,0.5,0.5]";

  file << ", Line[{";
  file << "{" << p[0].x() << "," <<p[0].y() << ",(" << p[0].z() << "-zs)*mz}, ";
  file << "{" << p[1].x() << "," <<p[1].y() << ",(" << p[1].z() << "-zs)*mz}, ";
  file << "{" << p[2].x() << "," <<p[2].y() << ",(" << p[2].z() << "-zs)*mz}, ";
  file << "{" << p[3].x() << "," <<p[3].y() << ",(" << p[3].z() << "-zs)*mz}, ";
  file << "{" << p[0].x() << "," <<p[0].y() << ",(" << p[0].z() << "-zs)*mz}";
  file << "}]" << std::endl;

  file << ", Line[{";
  file << "{" << p[4].x() << "," <<p[4].y() << ",(" << p[4].z() << "-zs)*mz}, ";
  file << "{" << p[5].x() << "," <<p[5].y() << ",(" << p[5].z() << "-zs)*mz}, ";
  file << "{" << p[6].x() << "," <<p[6].y() << ",(" << p[6].z() << "-zs)*mz}, ";
  file << "{" << p[7].x() << "," <<p[7].y() << ",(" << p[7].z() << "-zs)*mz}, ";
  file << "{" << p[4].x() << "," <<p[4].y() << ",(" << p[4].z() << "-zs)*mz}";
  file << "}]" << std::endl;
  
  file << ", Line[{";
  file << "{" << p[0].x() << "," <<p[0].y() << ",(" << p[0].z() << "-zs)*mz}, ";
  file << "{" << p[4].x() << "," <<p[4].y() << ",(" << p[4].z() << "-zs)*mz}";
  file << "}]" << std::endl;
  
  file << ", Line[{";
  file << "{" << p[1].x() << "," <<p[1].y() << ",(" << p[1].z() << "-zs)*mz}, ";
  file << "{" << p[5].x() << "," <<p[5].y() << ",(" << p[5].z() << "-zs)*mz}";
  file << "}]" << std::endl;
  
  file << ", Line[{";
  file << "{" << p[2].x() << "," <<p[2].y() << ",(" << p[2].z() << "-zs)*mz}, ";
  file << "{" << p[6].x() << "," <<p[6].y() << ",(" << p[6].z() << "-zs)*mz}";
  file << "}]" << std::endl;
  
  file << ", Line[{";
  file << "{" << p[3].x() << "," <<p[3].y() << ",(" << p[3].z() << "-zs)*mz}, ";
  file << "{" << p[7].x() << "," <<p[7].y() << ",(" << p[7].z() << "-zs)*mz}";
  file << "}]" << std::endl;

  file << "}]" << std::endl;
}

/*****************************************************************************/
void PlotEcalRecHits::printEcalRecHits(const edm::Event& ev)
{
//  const float minEnergy = 0.060;

  edm::Handle<EBRecHitCollection>              recHitsBarrel;
  ev.getByLabel("ecalRecHit", "EcalRecHitsEB", recHitsBarrel);
  edm::Handle<EERecHitCollection>              recHitsEndcap;
  ev.getByLabel("ecalRecHit", "EcalRecHitsEE", recHitsEndcap);
/*
  edm::Handle<ESRecHitCollection>              recHitsPreshower;
  ev.getByLabel("ecalPreshowerRecHit", "EcalRecHitsES", recHitsPreshower);
*/

  LogTrace("MinBiasTracking")
       << " [EventPlotter] ecal barrel/endcap :"
       << " " << recHitsBarrel->size()
       << "/" << recHitsEndcap->size();
//       << "/" << recHitsPreshower->size() << std::endl;

  const CaloSubdetectorGeometry * geom;

  file << ", If[er, {RGBColor[0.0,0.0,0.0]";

  // Barrel
  geom = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  for(EBRecHitCollection::const_iterator recHit = recHitsBarrel->begin();
                                         recHit!= recHitsBarrel->end();
                                         recHit++)
  {
    EBDetId detId(recHit->id());
    const CaloCellGeometry* cell = geom->getGeometry(detId);
    if(cell != 0)
      printEcalRecHit(cell, recHit->energy());
  }

  // Endcap
  geom = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
  for(EERecHitCollection::const_iterator recHit = recHitsEndcap->begin();
                                         recHit!= recHitsEndcap->end();
                                         recHit++)
  {
    EEDetId detId(recHit->id());
    const CaloCellGeometry* cell = geom->getGeometry(detId);
    if(cell != 0)
      printEcalRecHit(cell, recHit->energy());
  }

  // Preshower
/*
  geom = theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,EcalPreshower);
  for(ESRecHitCollection::const_iterator recHit = recHitsPreshower->begin();
                                         recHit!= recHitsPreshower->end();
                                         recHit++)
  {
    ESDetId detId(recHit->id()); 
    const CaloCellGeometry* cell = geom->getGeometry(detId);
    printEcalRecHit(cell, recHit->energy() * 1e+3);
  }
*/
  file << "}]";
}

