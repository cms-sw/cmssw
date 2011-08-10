// -*- C++ -*-
//
// Package:    PSTools
// Class:      HVTkMapCreator
// 
/**\class HVTkMapCreator HVTkMapCreator.cc myTKAnalyses/PSTools/plugins/HVTkMapCreator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Sat Oct 10 11:56:00 CEST 2009
//
//


// system include files
#include <memory>

// user include files

#include <iostream>
#include <fstream>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h" 
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

//
// class decleration
//

class HVTkMapCreator : public edm::EDAnalyzer {
 public:
    explicit HVTkMapCreator(const edm::ParameterSet&);
    ~HVTkMapCreator();


   private:
      virtual void beginJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  //const std::string _hvfile;
  const std::string _hvreassfile;

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
HVTkMapCreator::HVTkMapCreator(const edm::ParameterSet& iConfig):
  //_hvfile(iConfig.getParameter<std::string>("hvChannelFile")),
  _hvreassfile(iConfig.getParameter<std::string>("hvReassChannelFile"))

{
   //now do what ever initialization is needed
}


HVTkMapCreator::~HVTkMapCreator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HVTkMapCreator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

}

void 
HVTkMapCreator::beginRun(const edm::Run& iRun, const edm::EventSetup&)
{
  TrackerMap hvmap,hvmapreassigned;

  //TkHistoMap hvhisto("TID_7131_off","TID_7131_off",-1);
  TkHistoMap hvhistoreassigned("HVMap_new","HVMap_new",-1);
  
  //ifstream hvdata(_hvfile.c_str());
  ifstream hvreassdata(_hvreassfile.c_str());

  // HV channel map filling

  unsigned int detid;
  std::string channel1;
  /*
  while(hvdata >> detid >> channel1) {
    double cha =0.;
    if(channel1=="channel002") cha = 1.;
    if(channel1=="channel003") cha = -1.;
    hvhisto.fill(detid,cha);
    }*/

  //int napv2, napv3;
  /*
  while (hvdata >> detid >> napv2 >> napv3 ) {
    
    if(napv2+napv3>0) {
      double asim = double(napv2-napv3)/double(napv2+napv3);
      hvhisto.fill(detid,asim);
    }
  }
  */
  std::string channel2;

  while(hvreassdata >> detid >> channel2) {
    double cha =0.;
    if(channel2=="channel002") cha = 1.;
    if(channel2=="channel003") cha = -6.;
    if (channel2=="channel001") cha= -0.7;
    if (channel2 !="channel000") hvhistoreassigned.fill(detid,cha);
  }



  //hvmap.setPalette(1);
  hvmapreassigned.setPalette(1);

  //hvhisto.dumpInTkMap(&hvmap);
  hvhistoreassigned.dumpInTkMap(&hvmapreassigned);

  //std::string hvname = "HVMAP_old.png";
  //hvmap.save(true,0,0,hvname);
  std::string hvnamereassigned = "HVMAP_new.png";
  hvmapreassigned.save(true,0,0,hvnamereassigned);

  std::string rootmapname = "HVChannels_TKMap.root";
  //hvhisto.save(rootmapname);
  hvhistoreassigned.save(rootmapname);
}

void 
HVTkMapCreator::endRun(const edm::Run& iRun, const edm::EventSetup&)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
HVTkMapCreator::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HVTkMapCreator::endJob() {}


//define this as a plug-in
DEFINE_FWK_MODULE(HVTkMapCreator);
