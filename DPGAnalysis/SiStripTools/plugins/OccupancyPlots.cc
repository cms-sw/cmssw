// -*- C++ -*-
//
// Package:    OccupancyPlots
// Class:      OccupancyPlots
// 
/**\class OccupancyPlots OccupancyPlots.cc myTKAnalyses/DigiInvestigator/src/OccupancyPlots.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Oct 27 17:37:53 CET 2008
// $Id: OccupancyPlots.cc,v 1.4 2013/04/05 14:39:20 venturia Exp $
//
//


// system include files
#include <memory>

// user include files

#include <vector>
#include <map>
#include <limits>

#include "TProfile.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"
#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

//
// class decleration
//

class OccupancyPlots : public edm::EDAnalyzer {
   public:
      explicit OccupancyPlots(const edm::ParameterSet&);
      ~OccupancyPlots();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void endJob() ;

      // ----------member data ---------------------------

  std::vector<edm::InputTag> m_multiplicityMaps;
  std::vector<edm::InputTag> m_occupancyMaps;
  edm::FileInPath m_fp;

  RunHistogramManager m_rhm;
  std::map<unsigned int,DetIdSelector> m_wantedsubdets;
  std::map<unsigned int,DetIdSelector> m_wantedsubdetslbl;

  TProfile** m_avemultiplicity;
  TProfile** m_aveoccupancy;

  TH1F** m_nchannels_ideal;
  TH1F** m_nchannels_real;

  TProfile** m_averadius;
  TProfile** m_avez;
  TProfile** m_avex;
  TProfile** m_avey;
  TProfile** m_zavedr;
  TProfile** m_zavedz;
  TProfile** m_zavedrphi;
  TProfile** m_yavedr;
  TProfile** m_yavedz;
  TProfile** m_yavedrphi;
  TProfile** m_xavedr;
  TProfile** m_xavedz;
  TProfile** m_xavedrphi;

  TProfile** m2_averadius;
  TProfile** m2_avez;
  TProfile** m2_avex;
  TProfile** m2_avey;
  TProfile** m2_zavedr;
  TProfile** m2_zavedz;
  TProfile** m2_zavedrphi;
  TProfile** m2_yavedr;
  TProfile** m2_yavedz;
  TProfile** m2_yavedrphi;
  TProfile** m2_xavedr;
  TProfile** m2_xavedz;
  TProfile** m2_xavedrphi;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
OccupancyPlots::OccupancyPlots(const edm::ParameterSet& iConfig):
  m_multiplicityMaps(iConfig.getParameter<std::vector<edm::InputTag> >("multiplicityMaps")),
  m_occupancyMaps(iConfig.getParameter<std::vector<edm::InputTag> >("occupancyMaps")),
  m_fp(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt"))),
  m_rhm(), m_wantedsubdets(), m_wantedsubdetslbl()
{
   //now do what ever initialization is needed

  m_avemultiplicity = m_rhm.makeTProfile("avemult","Average Multiplicty",6000,0.5,6000.5);
  m_aveoccupancy = m_rhm.makeTProfile("aveoccu","Average Occupancy",6000,0.5,6000.5);

  m_nchannels_ideal = m_rhm.makeTH1F("nchannels_ideal","Number of channels (ideal)",6000,0.5,6000.5);
  m_nchannels_real = m_rhm.makeTH1F("nchannels_real","Number of channels (real)",6000,0.5,6000.5);

  m_averadius = m_rhm.makeTProfile("averadius","Average Module Radius",6000,0.5,6000.5);
  m_avez = m_rhm.makeTProfile("avez","Average Module z coordinate",6000,0.5,6000.5);
  m_avex = m_rhm.makeTProfile("avex","Average Module x coordinate",6000,0.5,6000.5);
  m_avey = m_rhm.makeTProfile("avey","Average Module y coordinate",6000,0.5,6000.5);

  m_zavedr = m_rhm.makeTProfile("zavedr","Average z unit vector dr",6000,0.5,6000.5);
  m_zavedz = m_rhm.makeTProfile("zavedz","Average z unit vector dz",6000,0.5,6000.5);
  m_zavedrphi = m_rhm.makeTProfile("zavedrphi","Average z unit vector drphi",6000,0.5,6000.5);
  m_xavedr = m_rhm.makeTProfile("xavedr","Average x unit vector dr",6000,0.5,6000.5);
  m_xavedz = m_rhm.makeTProfile("xavedz","Average x unit vctor dz",6000,0.5,6000.5);
  m_xavedrphi = m_rhm.makeTProfile("xavedrphi","Average Module x unit vector drphi",6000,0.5,6000.5);
  m_yavedr = m_rhm.makeTProfile("yavedr","Average y unit vector dr",6000,0.5,6000.5);
  m_yavedz = m_rhm.makeTProfile("yavedz","Average y unit vector dz",6000,0.5,6000.5);
  m_yavedrphi = m_rhm.makeTProfile("yavedrphi","Average y unit vector drphi",6000,0.5,6000.5);

  m2_averadius = m_rhm.makeTProfile("averadius2","Average2 Module Radius",6000,0.5,6000.5);
  m2_avez = m_rhm.makeTProfile("avez2","Average2 Module z coordinate",6000,0.5,6000.5);
  m2_avex = m_rhm.makeTProfile("avex2","Average2 Module x coordinate",6000,0.5,6000.5);
  m2_avey = m_rhm.makeTProfile("avey2","Average2 Module y coordinate",6000,0.5,6000.5);

  m2_zavedr = m_rhm.makeTProfile("zavedr2","Average2 z unit vector dr",6000,0.5,6000.5);
  m2_zavedz = m_rhm.makeTProfile("zavedz2","Average2 z unit vector dz",6000,0.5,6000.5);
  m2_zavedrphi = m_rhm.makeTProfile("zavedrphi2","Average2 z unit vector drphi",6000,0.5,6000.5);
  m2_xavedr = m_rhm.makeTProfile("xavedr2","Average2 x unit vector dr",6000,0.5,6000.5);
  m2_xavedz = m_rhm.makeTProfile("xavedz2","Average2 x unit vctor dz",6000,0.5,6000.5);
  m2_xavedrphi = m_rhm.makeTProfile("xavedrphi2","Average2 Module x unit vector drphi",6000,0.5,6000.5);
  m2_yavedr = m_rhm.makeTProfile("yavedr2","Average2 y unit vector dr",6000,0.5,6000.5);
  m2_yavedz = m_rhm.makeTProfile("yavedz2","Average2 y unit vector dz",6000,0.5,6000.5);
  m2_yavedrphi = m_rhm.makeTProfile("yavedrphi2","Average2 y unit vector drphi",6000,0.5,6000.5);

  std::vector<edm::ParameterSet> wantedsubdets_ps = iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedSubDets");
  LogTrace("OccupancyPlots") << "wantedsubdets_ps size" << wantedsubdets_ps.size();

  for(std::vector<edm::ParameterSet>::const_iterator wsdps = wantedsubdets_ps.begin();wsdps!=wantedsubdets_ps.end();++wsdps) {

    unsigned int detsel = wsdps->getParameter<unsigned int>("detSelection");
    std::vector<std::string> selstr = wsdps->getUntrackedParameter<std::vector<std::string> >("selection");
    std::string sellbl = wsdps->getParameter<std::string>("detLabel");

    LogTrace("OccupancyPlots") << "Label     >>" << sellbl;
    for( auto sel : selstr) 
      LogTrace("OccupancyPlots") << "Selection >>" << sel;


    m_wantedsubdets[detsel]=DetIdSelector(selstr);
    m_wantedsubdetslbl[detsel] = DetIdSelector(sellbl, "words");

    LogTrace("OccupancyPlots") << ">>>>>>>>>" ;
  }


}


OccupancyPlots::~OccupancyPlots()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
OccupancyPlots::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  

  for(std::vector<edm::InputTag>::const_iterator map = m_multiplicityMaps.begin();map!=m_multiplicityMaps.end();++map) {

    Handle<std::map<unsigned int, int> > mults;
    iEvent.getByLabel(*map,mults);
  
    for(std::map<unsigned int,int>::const_iterator mult=mults->begin();mult!=mults->end();mult++) {
      if(m_avemultiplicity && *m_avemultiplicity) (*m_avemultiplicity)->Fill(mult->first,mult->second);
    }
  }



  for(std::vector<edm::InputTag>::const_iterator map = m_occupancyMaps.begin();map!=m_occupancyMaps.end();++map) {

    Handle<std::map<unsigned int, int> > occus;
    iEvent.getByLabel(*map,occus);
  
    for(std::map<unsigned int,int>::const_iterator occu=occus->begin();occu!=occus->end();occu++) {
      if(m_aveoccupancy && *m_aveoccupancy) (*m_aveoccupancy)->Fill(occu->first,occu->second);
    }
  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
OccupancyPlots::beginJob()
{

}

void
OccupancyPlots::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {

  m_rhm.beginRun(iRun);

}

void
OccupancyPlots::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {

  
  //  edm::ESHandle<GlobalTrackingGeometry> trkgeo;
  //  iSetup.get<GlobalTrackingGeometryRecord>().get("",trkgeo);
  edm::ESHandle<TrackerGeometry> trkgeo;
  iSetup.get<TrackerDigiGeometryRecord>().get("",trkgeo);

  // Test new TrackerGeometry features
  LogDebug("IsThereTest") << "Test of TrackerGeometry::isThere";
  LogTrace("IsThereTest") << " is there PixelBarrel: " << trkgeo->isThere(GeomDetEnumerators::PixelBarrel);
  LogTrace("IsThereTest") << " is there PixelEndcap: " << trkgeo->isThere(GeomDetEnumerators::PixelEndcap);
  LogTrace("IsThereTest") << " is there P1PXB: " << trkgeo->isThere(GeomDetEnumerators::P1PXB);
  LogTrace("IsThereTest") << " is there P1PXEC: " << trkgeo->isThere(GeomDetEnumerators::P1PXEC);
  LogTrace("IsThereTest") << " is there P2PXEC: " << trkgeo->isThere(GeomDetEnumerators::P2PXEC);
  LogTrace("IsThereTest") << " is there TIB: " << trkgeo->isThere(GeomDetEnumerators::TIB);
  LogTrace("IsThereTest") << " is there TID: " << trkgeo->isThere(GeomDetEnumerators::TID);
  LogTrace("IsThereTest") << " is there TOB: " << trkgeo->isThere(GeomDetEnumerators::TOB);
  LogTrace("IsThereTest") << " is there TEC: " << trkgeo->isThere(GeomDetEnumerators::TEC);
  LogTrace("IsThereTest") << " is there P2OTB: " << trkgeo->isThere(GeomDetEnumerators::P2OTB);
  LogTrace("IsThereTest") << " is there P2OTEC: " << trkgeo->isThere(GeomDetEnumerators::P2OTEC);

  edm::ESHandle< TrackerTopology > tTopoHandle;
  iSetup.get< IdealGeometryRecord >().get(tTopoHandle);
  const TrackerTopology* tTopo(tTopoHandle.product());

  const Local2DPoint center(0.,0.);
  const Local3DPoint locz(0.,0.,1.);
  const Local3DPoint locx(1.,0.,0.);
  const Local3DPoint locy(0.,1.,0.);
  const GlobalPoint origin(0.,0.,0.);

  TrackingGeometry::DetIdContainer detunits = trkgeo->detUnitIds();

  for(TrackingGeometry::DetIdContainer::const_iterator det = detunits.begin(); det!=detunits.end(); ++det) {

    if(det->det()!=DetId::Tracker) continue;

    edm::LogInfo("DetIdFromGeometry") << det->rawId();
    edm::LogInfo("OccupancyPlots") << ">>>>>" << tTopo->print(det->rawId());

    GlobalPoint position = trkgeo->idToDet(*det)->toGlobal(center);
    GlobalPoint zpos = trkgeo->idToDet(*det)->toGlobal(locz);
    GlobalPoint xpos = trkgeo->idToDet(*det)->toGlobal(locx);
    GlobalPoint ypos = trkgeo->idToDet(*det)->toGlobal(locy);
    GlobalVector posvect = position - origin;
    GlobalVector dz = zpos - position;
    GlobalVector dx = xpos - position;
    GlobalVector dy = ypos - position;

    double dzdr = posvect.perp()>0 ? (dz.x()*posvect.x()+dz.y()*posvect.y())/posvect.perp() : 0. ;
    double dxdr = posvect.perp()>0 ? (dx.x()*posvect.x()+dx.y()*posvect.y())/posvect.perp() : 0. ;
    double dydr = posvect.perp()>0 ? (dy.x()*posvect.x()+dy.y()*posvect.y())/posvect.perp() : 0. ;

    double dzdrphi = posvect.perp()>0 ? (dz.y()*posvect.x()-dz.x()*posvect.y())/posvect.perp() : 0. ;
    double dxdrphi = posvect.perp()>0 ? (dx.y()*posvect.x()-dx.x()*posvect.y())/posvect.perp() : 0. ;
    double dydrphi = posvect.perp()>0 ? (dy.y()*posvect.x()-dy.x()*posvect.y())/posvect.perp() : 0. ;

     for(std::map<unsigned int,DetIdSelector>::const_iterator sel=m_wantedsubdets.begin();sel!=m_wantedsubdets.end();++sel) {

       if(sel->second.isSelected(*det)) {
         LogTrace("OccupancyPlots") << sel->first << " is selected with bits";  
	 // average positions
	 if(m_averadius && *m_averadius) (*m_averadius)->Fill(sel->first,position.perp());
	 if(m_avez && *m_avez) (*m_avez)->Fill(sel->first,position.z());
	 if(m_avex && *m_avex) (*m_avex)->Fill(sel->first,position.x());
	 if(m_avey && *m_avey) (*m_avey)->Fill(sel->first,position.y());
	 if(m_zavedr && *m_zavedr) (*m_zavedr)->Fill(sel->first,dzdr);
	 if(m_zavedz && *m_zavedz) (*m_zavedz)->Fill(sel->first,dz.z());
	 if(m_zavedrphi && *m_zavedrphi) (*m_zavedrphi)->Fill(sel->first,dzdrphi);
	 if(m_xavedr && *m_xavedr) (*m_xavedr)->Fill(sel->first,dxdr);
	 if(m_xavedz && *m_xavedz) (*m_xavedz)->Fill(sel->first,dx.z());
	 if(m_xavedrphi && *m_xavedrphi) (*m_xavedrphi)->Fill(sel->first,dxdrphi);
	 if(m_yavedr && *m_yavedr) (*m_yavedr)->Fill(sel->first,dydr);
	 if(m_yavedz && *m_yavedz) (*m_yavedz)->Fill(sel->first,dy.z());
	 if(m_yavedrphi && *m_yavedrphi) (*m_yavedrphi)->Fill(sel->first,dydrphi);
       
       } else {
         LogTrace("OccupancyPlots") << sel->first << " is NOT selected with bits";
       }
     }


     for(std::map<unsigned int,DetIdSelector>::const_iterator sel=m_wantedsubdetslbl.begin();sel!=m_wantedsubdetslbl.end();++sel) {

       if(sel->second.isSelectedByWords(*det, tTopo)) {
         LogTrace("OccupancyPlots") << sel->first << " is selected with labels";
         // average positions
         if(m2_averadius && *m2_averadius) (*m2_averadius)->Fill(sel->first,position.perp());
         if(m2_avez && *m2_avez) (*m2_avez)->Fill(sel->first,position.z());
         if(m2_avex && *m2_avex) (*m2_avex)->Fill(sel->first,position.x());
         if(m2_avey && *m2_avey) (*m2_avey)->Fill(sel->first,position.y());
         if(m2_zavedr && *m2_zavedr) (*m2_zavedr)->Fill(sel->first,dzdr);
         if(m2_zavedz && *m2_zavedz) (*m2_zavedz)->Fill(sel->first,dz.z());
         if(m2_zavedrphi && *m2_zavedrphi) (*m2_zavedrphi)->Fill(sel->first,dzdrphi);
         if(m2_xavedr && *m2_xavedr) (*m2_xavedr)->Fill(sel->first,dxdr);
         if(m2_xavedz && *m2_xavedz) (*m2_xavedz)->Fill(sel->first,dx.z());
         if(m2_xavedrphi && *m2_xavedrphi) (*m2_xavedrphi)->Fill(sel->first,dxdrphi);
         if(m2_yavedr && *m2_yavedr) (*m2_yavedr)->Fill(sel->first,dydr);
         if(m2_yavedz && *m2_yavedz) (*m2_yavedz)->Fill(sel->first,dy.z());
         if(m2_yavedrphi && *m2_yavedrphi) (*m2_yavedrphi)->Fill(sel->first,dydrphi); 
 
       } else {
         LogTrace("OccupancyPlots") << sel->first << " is NOT selected with labels";
       }
         
     }

    if((*m_averadius)->Integral() != (*m2_averadius)->Integral()){
      edm::LogError("OccupancyPlots") << "qui abbiamo fatto merda!\n"  
                      << (*m_averadius)->Integral() << " dets selected with bits"
                      << (*m2_averadius)->Integral() << " dets selected with label";
      return;
    }

  }
  

  edm::LogInfo("OccupancyPlots") << (*m_averadius)->Integral() << " dets selected with bits";
  edm::LogInfo("OccupancyPlots") << (*m2_averadius)->Integral() << " dets selected with label";

  edm::ESHandle<SiStripQuality> quality;
  iSetup.get<SiStripQualityRcd>().get("",quality);


   SiStripDetInfoFileReader * reader=edm::Service<SiStripDetInfoFileReader>().operator->();

   const std::vector<uint32_t>& detids = reader->getAllDetIds();

   for(std::vector<uint32_t>::const_iterator detid=detids.begin();detid!=detids.end();++detid) {

     int nchannideal = reader->getNumberOfApvsAndStripLength(*detid).first*128;
     //     int nchannreal = reader->getNumberOfApvsAndStripLength(*detid).first*128;
     int nchannreal = 0;
     for(int strip = 0; strip < nchannideal; ++strip) {
       if(!quality->IsStripBad(*detid,strip)) ++nchannreal;
     }


     for(std::map<unsigned int,DetIdSelector>::const_iterator sel=m_wantedsubdets.begin();sel!=m_wantedsubdets.end();++sel) {

       if(sel->second.isSelected(*detid)) {
	 if(m_nchannels_ideal && *m_nchannels_ideal) (*m_nchannels_ideal)->Fill(sel->first,nchannideal);
	 if(m_nchannels_real && *m_nchannels_real) (*m_nchannels_real)->Fill(sel->first,nchannreal);
       }
     }

   }


  edm::ESHandle<SiPixelQuality> pxlquality;
  iSetup.get<SiPixelQualityRcd>().get("",pxlquality);


   SiPixelDetInfoFileReader pxlreader(m_fp.fullPath());

   const std::vector<uint32_t>& pxldetids = pxlreader.getAllDetIds();

   for(std::vector<uint32_t>::const_iterator detid=pxldetids.begin();detid!=pxldetids.end();++detid) {

     int nchannideal = pxlreader.getDetUnitDimensions(*detid).first*pxlreader.getDetUnitDimensions(*detid).second;
     int nchannreal = 0;
     if(!pxlquality->IsModuleBad(*detid)) {
       nchannreal = pxlreader.getDetUnitDimensions(*detid).first*pxlreader.getDetUnitDimensions(*detid).second;
     }
     /*
     int nchannreal = 0;
     for(int strip = 0; strip < nchannideal; ++strip) {
       if(!quality->IsStripBad(*detid,strip)) ++nchannreal;
     }
     */

     for(std::map<unsigned int,DetIdSelector>::const_iterator sel=m_wantedsubdets.begin();sel!=m_wantedsubdets.end();++sel) {

       if(sel->second.isSelected(*detid)) {
	 if(m_nchannels_ideal && *m_nchannels_ideal) (*m_nchannels_ideal)->Fill(sel->first,nchannideal);
	 if(m_nchannels_real && *m_nchannels_real) (*m_nchannels_real)->Fill(sel->first,nchannreal);
       }
     }

   }

}
// ------------ method called once each job just after ending the event loop  ------------
void 
OccupancyPlots::endJob() {
}
//define this as a plug-in
DEFINE_FWK_MODULE(OccupancyPlots);
