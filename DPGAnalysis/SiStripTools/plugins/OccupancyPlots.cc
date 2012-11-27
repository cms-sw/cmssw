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
// $Id: OccupancyPlots.cc,v 1.1 2012/03/26 17:13:02 venturia Exp $
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
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  virtual void endJob() ;

      // ----------member data ---------------------------

  std::vector<edm::InputTag> m_multiplicityMaps;
  std::vector<edm::InputTag> m_occupancyMaps;

  RunHistogramManager m_rhm;
  std::map<unsigned int,DetIdSelector> m_wantedsubdets;

  TProfile** m_avemultiplicity;
  TProfile** m_aveoccupancy;

  TH1F** m_nchannels_ideal;
  TH1F** m_nchannels_real;

  TProfile** m_averadius;
  TProfile** m_avez;


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
  m_rhm(), m_wantedsubdets()
{
   //now do what ever initialization is needed

  m_avemultiplicity = m_rhm.makeTProfile("avemult","Average Multiplicty",6000,0.5,6000.5);
  m_aveoccupancy = m_rhm.makeTProfile("aveoccu","Average Occupancy",6000,0.5,6000.5);

  m_nchannels_ideal = m_rhm.makeTH1F("nchannels_ideal","Number of channels (ideal)",6000,0.5,6000.5);
  m_nchannels_real = m_rhm.makeTH1F("nchannels_real","Number of channels (real)",6000,0.5,6000.5);

  m_averadius = m_rhm.makeTProfile("averadius","Average Module Radius",6000,0.5,6000.5);
  m_avez = m_rhm.makeTProfile("avez","Average Module z coordinate",6000,0.5,6000.5);

  std::vector<edm::ParameterSet> wantedsubdets_ps = iConfig.getParameter<std::vector<edm::ParameterSet> >("wantedSubDets");

  for(std::vector<edm::ParameterSet>::const_iterator wsdps = wantedsubdets_ps.begin();wsdps!=wantedsubdets_ps.end();++wsdps) {

    unsigned int detsel = wsdps->getParameter<unsigned int>("detSelection");
    std::vector<std::string> selstr = wsdps->getUntrackedParameter<std::vector<std::string> >("selection");
    m_wantedsubdets[detsel]=DetIdSelector(selstr);

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

  const Local2DPoint center(0.,0.);

  TrackingGeometry::DetIdContainer detunits = trkgeo->detUnitIds();

  for(TrackingGeometry::DetIdContainer::const_iterator det = detunits.begin(); det!=detunits.end(); ++det) {

    if(det->det()!=DetId::Tracker) continue;

    edm::LogInfo("DetIdFromGeometry") << det->rawId();

    GlobalPoint position = trkgeo->idToDet(*det)->toGlobal(center);

     for(std::map<unsigned int,DetIdSelector>::const_iterator sel=m_wantedsubdets.begin();sel!=m_wantedsubdets.end();++sel) {

       if(sel->second.isSelected(*det)) {
	 edm::LogInfo("SelectedDetId") << sel->first;
	 // average positions
	 if(m_averadius && *m_averadius) (*m_averadius)->Fill(sel->first,position.perp());
	 if(m_avez && *m_avez) (*m_avez)->Fill(sel->first,position.z());
       }
     }
  }
  

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


   edm::FileInPath fp("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt");
     
   SiPixelDetInfoFileReader pxlreader(fp.fullPath());

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
