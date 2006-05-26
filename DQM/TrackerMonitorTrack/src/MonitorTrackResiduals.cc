// -*- C++ -*-
//
// Package:    TrackerMonitorTrack
// Class:      MonitorTrackResiduals
// 
/**\class MonitorTrackResiduals MonitorTrackResiduals.cc DQM/TrackerMonitorTrack/src/MonitorTrackResiduals.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Israel Goitom
//         Created:  Fri May 26 14:12:01 CEST 2006
// $Id$
//
//


// system include files

// user include files
#include "DQM/TrackerMonitorTrack/interface/MonitorTrackResiduals.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MonitorTrackResiduals::MonitorTrackResiduals(const edm::ParameterSet& iConfig)
{
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  conf_ = iConfig;
}


MonitorTrackResiduals::~MonitorTrackResiduals()
{
 
}

void MonitorTrackResiduals::beginJob(edm::EventSetup const& iSetup)
{
  using namespace edm;
  using namespace std;

  dbe->setCurrentFolder("SiStrip/Detectors");

  // take from eventSetup the SiStripStructure object - here will use SiStripDetControl later on
    edm::ESHandle<SiStripDetCabling> tkmechstruct;
    iSetup.get<SiStripDetCablingRcd>().get(tkmechstruct);

  // get list of active detectors from SiStripStructure - this will change and be taken from a SiStripDetControl constobject 

    vector<uint32_t> activeDets;
    activeDets.clear(); // just in case
    tkmechstruct->addActiveDetectorsRawIds(activeDets);

  // use SiStripSubStructure for selecting certain regions
  SiStripSubStructure substructure;

  vector<uint32_t> DetIds = activeDets;
  vector<uint32_t> TIBDetIds;
  vector<uint32_t> TOBDetIds;
  vector<uint32_t> TIDDetIds;
  vector<uint32_t> TECDetIds;

  substructure.getTIBDetectors(activeDets, TIBDetIds); // this adds rawDetIds to SelectedDetIds
  substructure.getTOBDetectors(activeDets, TOBDetIds);    // this adds rawDetIds to SelectedDetIds
  substructure.getTIDDetectors(activeDets, TIDDetIds); // this adds rawDetIds to SelectedDetIds
  substructure.getTECDetectors(activeDets, TECDetIds); // this adds rawDetIds to SelectedDetIds

  
  vector<uint32_t>::const_iterator detid_begin = TIBDetIds.begin();
  vector<uint32_t>::const_iterator detid_end = TIBDetIds.end() -1;


  int detBegin=(*detid_begin)&0x1ffffff;
  int detEnd=(*detid_end)&0x1ffffff;

  HitResidual["TIB"] = dbe->book2D("TIBHitRes", "TIB Hit residuals",  TIBDetIds.size(), detBegin, detEnd, 50, 0, .2);

  detid_begin = TOBDetIds.begin();
  detid_end = TOBDetIds.end()-1;

  detBegin=(*detid_begin)&0x1ffffff;
  detEnd=(*detid_end)&0x1ffffff;

  HitResidual["TOB"] = dbe->book2D("TOBHitRes", "TOB Hit residuals", TOBDetIds.size(), detBegin, detEnd, 50, 0, .2);

  detid_begin = TIDDetIds.begin();
  detid_end = TIDDetIds.end()-1;

  detBegin=(*detid_begin)&0x1ffffff;
  detEnd=(*detid_end)&0x1ffffff;

  HitResidual["TID"] = dbe->book2D("TIDHitRes", "TID Hit residuals", TIDDetIds.size(), detBegin, detEnd, 50, 0, .2);

  detid_begin = TECDetIds.begin();
  detid_end = TECDetIds.end()-1;

  detBegin=(*detid_begin)&0x1ffffff;
  detEnd=(*detid_end)&0x1ffffff;

  HitResidual["TEC"] = dbe->book2D("TECHitRes", "TEC Hit residuals", TECDetIds.size(), detBegin, detEnd, 50, 0, .2);

}

void MonitorTrackResiduals::endJob(void)
{
  dbe->showDirStructure();
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe->save(outputFileName);
  }
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
MonitorTrackResiduals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}

//define this as a plug-in
//DEFINE_FWK_MODULE(MonitorTrackResiduals)
