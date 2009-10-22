#include <DQM/CastorMonitor/interface/CastorRecHitsValidation.h>
#include <DataFormats/HcalDetId/interface/HcalCastorDetId.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "SimG4CMS/Calo/interface/CaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"


#include <string>
#include <map>
#include <vector>

using namespace cms;
using namespace edm;
using namespace std;



CastorRecHitsValidation::CastorRecHitsValidation(){
}


CastorRecHitsValidation::~CastorRecHitsValidation(){
}

void CastorRecHitsValidation::reset(){
}


void CastorRecHitsValidation::setup(const edm::ParameterSet& ps, DQMStore* dbe){

 CastorBaseMonitor::setup(ps,dbe);
 baseFolder_ = rootFolder_+"CastorRecHitsV";

 cout << "CastorRecHitsValidation::setup (start)" << endl;

 
 if ( m_dbe !=NULL ) {  
 
m_dbe->setCurrentFolder("CastorRecHitsV/CastorRecHitsTask");

    //*************** rec hits ***************************   
histo = "CastorRecHits in modules";
    meCastorRecHitsModule_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 14, 0.5, 14.5);
histo = "CastorRecHits Energy in modules";
    meCastorRecHitsEmodule_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 14, 0.5, 14.5);
histo = "CastorRecHits in sectors";
    meCastorRecHitsSector_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 16, 0.5, 16.5);
histo = "CastorRecHits Energy in sectors";
    meCastorRecHitsEsector_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 16, 0.5, 16.5);
histo = "CastorRecHits Energy";
    meCastorRecHitsEnergy_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 100, 0., 100.);
histo = "CastorRecHits Total Energy";
    meCastorRecHitsTotalEnergy_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 100, 0., 100.);

//histo = "CastorRecHits X cell position";
//    meCastorRecHitsX_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 1000, -150., 150.);
//histo = "CastorRecHits Y cell position";
//    meCastorRecHitsY_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 1000, -150., 150.);
//histo = "CastorRecHits Z cell position";
//    meCastorRecHitsZ_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 10000, -14000., -16200.);
//histo = "CastorRecHits XY profile";
//    meCastorRecHitsXY_ = m_dbe->book2D(histo.c_str(), histo.c_str(), 500, -150., 150., 500, -150., 150.);
//histo = "CastorRecHits z-side";
//    meCastorRecHitsZside_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 4, -2., 2.);
//histo = "CastorRecHits #eta";
//    meCastorRecHitsEta_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 10, -7.0, -5.0);
//histo = "CastorRecHits #phi";
//   meCastorRecHitsPhi_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 31, -3.14, 3.14);
//histo = "CastorRecHits X cell position (from calo)";
// meCastorRecHitsGlobalX_ = m_dbe->book1D(histo.c_str(), histo.c_str(), 500, -150., 150.);
histo = "CastorRecHits occupancy";
    meCastorRecHitsOccupancy_ = m_dbe->book2D(histo.c_str(), histo.c_str(), 16, 0.5, 16.5, 14, 0.5, 14.5);

}
   
 else{
  if(fVerbosity>0) cout << "CastorRecHitsValidation::setup - NO DQMStore service" << endl; 
 }
   
if(fVerbosity>0) cout << "CastorRecHitsValidation::setup (end)" << endl;

return;
}


void CastorRecHitsValidation::processEvent(const CastorRecHitCollection& castorHits){
 
  int module = -1;
  int sector = -1;
  double energy = 0.;
  //double simhitEnergy = 0.;
  double total_energy = 0.;

//get the geometry record 
//edm::ESHandle<CaloGeometry> calo;
//c.get<CaloGeometryRecord>().get(calo);
//const CaloGeometry * theCaloGeometry;
//theCaloGeometry = (const CaloGeometry*)calo.product();



    // loop over RecHits 
    for (CastorRecHitCollection::const_iterator recHit = castorHits.begin(); recHit != castorHits.end() ; ++recHit) {

HcalCastorDetId castorid = HcalCastorDetId(recHit->id());
//std::cout << "castor id = " << castorid << std::endl;

CastorRecHitCollection::const_iterator rh = castorHits.find(castorid);

module = castorid.module();
sector = castorid.sector();
//int zside = castorid.zside();

//const CaloCellGeometry* cellGeometry = calo->getSubdetectorGeometry(castorid)->getGeometry(castorid) ;
//get position (default in cm) -> front face centers??
//double x = cellGeometry->getPosition ().x() ;
//double y = cellGeometry->getPosition ().y() ;
//double z = cellGeometry->getPosition ().z() ;
//std::cout << "( X = " << x << ", Y = " << y << ", Z = " << z << " )" << std::endl;
//try a different way directly for the calo 
//            ( result is the same )
//const GlobalPoint& pos=calo->getPosition(castorid);
//double eta = pos.eta();
//double phi = pos.phi();
//double globalZ = pos.z();
//double globalY = pos.y();
//double globalX = pos.x();
//std::cout << "( X = " << globalX << ", Y = " << globalY << ", z = " << globalZ << " )" << std::endl;


energy = rh->energy();
total_energy += rh->energy(); // not really correct...

// -------
// this is for a check... get one sector, EM  
// x,y,z are most likely the front face centers of the readout units
// 
/*
  if (sector == 1 && module < 3) { 
double xEM1 = cellGeometry->getPosition ().x() ;
double yEM1 = cellGeometry->getPosition ().y() ;

meCastorRecHitsXYEMsector1_->Fill(xEM1*10, yEM1*10);
}

 if (sector == 1 ) { 
double x1 = cellGeometry->getPosition ().x() ; 
double y1 = cellGeometry->getPosition ().y() ; 
meCastorRecHitsXYsector1_->Fill(x1*10, y1*10);
}
*/
//------

 //fill histograms
meCastorRecHitsModule_->Fill(module);
meCastorRecHitsEmodule_->Fill(module, energy);
meCastorRecHitsSector_->Fill(sector);
meCastorRecHitsEsector_->Fill(sector, energy);
meCastorRecHitsOccupancy_->Fill(sector, module);
meCastorRecHitsEnergy_->Fill(energy);
meCastorRecHitsTotalEnergy_->Fill(total_energy);
// X,Y,Z dimensions in mm
//meCastorRecHitsX_->Fill(x*10);
//meCastorRecHitsY_->Fill(y*10);
//meCastorRecHitsZ_->Fill(z*10);
//meCastorRecHitsXY_->Fill(x*10,y*10);
//meCastorRecHitsZside_->Fill(zside);
//meCastorRecHitsEta_->Fill(eta);
//meCastorRecHitsPhi_->Fill(phi);
//meCastorRecHitsGlobalX_->Fill(globalX*10);
      }

    

} 
 
