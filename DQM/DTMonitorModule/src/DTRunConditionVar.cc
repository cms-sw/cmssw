/******* \class DTRunConditionVar *******
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Paolo Bellan, Antonio Branca
 * $date   : 23/09/2011 15:42:04 CET $
 * $Revision: 1.1 $
 *
 * Modification:
 *
 *********************************/

#include "DTRunConditionVar.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/Common/interface/RefToBase.h" 

#include <TMath.h>
#include <cmath>

using namespace std;
using namespace edm;

DTRunConditionVar::DTRunConditionVar(const ParameterSet& pSet):
  // Get the debug parameter for verbose output
  debug(pSet.getUntrackedParameter<bool>("debug",false)),
  nMinHitsPhi(pSet.getUntrackedParameter<int>("nMinHitsPhi")),
  maxAnglePhiSegm(pSet.getUntrackedParameter<double>("maxAnglePhiSegm")),
  thedt4DSegments_(pSet.getParameter<InputTag>("recoSegments"))
{
  //  LogVerbatim("DTDQM|DTRunConditionVar|DTRunConditionVar")
  //    << "DTRunConditionVar: constructor called";

  // Get the DQM needed services
  theDbe = Service<DQMStore>().operator->();

}

DTRunConditionVar::~DTRunConditionVar()
{
  LogTrace("DTDQM|DTMonitorModule|DTRunConditionVar")
    << "DTRunConditionVar: destructor called";

  // free memory
}



void DTRunConditionVar::beginJob() {

  LogTrace("DTDQM|DTMonitorModule|DTRunConditionVar")
    << "DTRunConditionVar: beginOfJob";

  for(int wheel=-2;wheel<=2;wheel++){
    for(int sec=1; sec<=14; sec++) {
      for(int stat=1; stat<=4; stat++) {

        bookChamberHistos(DTChamberId(wheel,stat,sec),"VDrift_FromSegm",50,25.,75.);
        bookChamberHistos(DTChamberId(wheel,stat,sec),"T0_FromSegm",75,-75.,75.);

      }
    }
  }


  return;
}





void DTRunConditionVar::beginRun(const Run& run, const EventSetup& setup)
{
  // Get the DT Geometry
  setup.get<MuonGeometryRecord>().get(dtGeom);

  return;
}




void DTRunConditionVar::endJob()
{
  LogTrace("DTDQM|DTMonitorModule|DTRunConditionVar")
    << "DTRunConditionVar: endOfJob";

  return;
}



void DTRunConditionVar::analyze(const Event & event,
    const EventSetup& eventSetup)
{

  LogTrace("DTDQM|DTMonitorModule|DTRunConditionVar") <<
    "--- [DTRunConditionVar] Event analysed #Run: " <<
    event.id().run() << " #Event: " << event.id().event() << endl;

  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the segment collection from the event
  Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(thedt4DSegments_, all4DSegments); 

  // Loop over the segments
  for(DTRecSegment4DCollection::const_iterator segment  = all4DSegments->begin();
      segment != all4DSegments->end(); ++segment){

    // Get the chamber from the setup
    DTChamberId DTid = (DTChamberId) segment->chamberId();
    uint32_t indexCh = DTid.rawId();

    // Fill v-drift values
    if( (*segment).hasPhi() ) {

      int nHitsPhi = (*segment).phiSegment()->degreesOfFreedom()+2;
      double xdir = (*segment).phiSegment()->localDirection().x();      
      double ydir = (*segment).phiSegment()->localDirection().y();      
      double zdir = (*segment).phiSegment()->localDirection().z();      

      double anglePhiSegm = fabs(atan(xdir/zdir))*180./TMath::Pi();

      cout<<sqrt( pow(xdir,2) + pow(ydir,2) + pow(zdir,2) )<<"\t"<<xdir<<"\t"<<ydir<<"\t"<<zdir<<"\t"<<anglePhiSegm<<endl;

      if( nHitsPhi >= nMinHitsPhi && anglePhiSegm <= maxAnglePhiSegm ) {

        double segmentVDrift = segment->phiSegment()->vDrift();
        double segmentT0 = segment->phiSegment()->t0();

        if(segmentT0 != -999 ) (chamberHistos[indexCh])["T0_FromSegm"]->Fill(segmentT0);
        if( segmentVDrift > 0.00 ) (chamberHistos[indexCh])["VDrift_FromSegm"]->Fill(segmentVDrift);

      }
    }

    //    if( (*segment).hasZed() ){
    //      double segmentVDrift = segment->zSegment()->vDrift();
    //      double segmentT0 = segment->zSegment()->t0();
    //
    //
    //      if(segmentT0 != -999 ) ht0[sector-1]->Fill(segmentT0);      
    //      if( segmentVDrift > 0.00 ) hvd[sector-1]->Fill(segmentVDrift);
    //
    //    }
  } //end loop on segment

} //end analyze




void DTRunConditionVar::bookChamberHistos(const DTChamberId& dtCh, string histoType, int nbins, float min, float max) {

  int wh = dtCh.wheel();		
  int sc = dtCh.sector();	
  int st = dtCh.station();
  stringstream wheel; wheel << wh;	
  stringstream station; station << st;	
  stringstream sector; sector << sc;	

  string bookingFolder = "DT/02-Segments/Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" + station.str();
  string histoTag      = "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  theDbe->setCurrentFolder(bookingFolder);

  LogTrace ("DTDQM|DTMonitorModule|DTRunConditionVar") 
    << "[DTRunConditionVar]: booking histos in " << bookingFolder << endl;

  string histoName = histoType  +  histoTag;
  string histoLabel = histoType;

  (chamberHistos[dtCh.rawId()])[histoType] = 
    theDbe->book1D(histoName,histoLabel,nbins,min,max);

}
