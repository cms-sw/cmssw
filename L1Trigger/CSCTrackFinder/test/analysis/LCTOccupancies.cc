
// system include files
#include <memory>
#include <cmath>
#include <iostream>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"//
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"

#include "L1Trigger/CSCTrackFinder/test/src/RunSRLUTs.h"

#include "TH1F.h"

using namespace std;
using namespace edm;

class LCTOccupancies : public edm::EDAnalyzer {
   public:
      explicit LCTOccupancies(const edm::ParameterSet&);
      ~LCTOccupancies();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  //TH1F* rateHist;

	edm::InputTag lctsTag;	
  	edm::Service<TFileService> fs;
        csctf_analysis::RunSRLUTs* runSRLUTs;
	TH1F* hMPCLink;
	TH1F* hLocalPhi;
	TH1F* hPhi;
	TH1F* hEta;
	TH1F* hPhiPacked;
	TH1F* hEtaPacked;
	TH1F* hBx;

	TH1F* hSector;
	TH1F* hStation;
	TH1F* hEndcap;
	TH1F* hSubSector;

	TH1F* hOccStation1SubSec1;
	TH1F* hOccStation1SubSec2;
	TH1F* hOccStation2;
	TH1F* hOccStation3;
	TH1F* hOccStation4;
	TH1F* hOccMax;
	TH1F* hOccMaxNo0;
	TH1F* hStubsTotal;
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
LCTOccupancies::LCTOccupancies(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
   runSRLUTs = new csctf_analysis::RunSRLUTs();
   lctsTag= iConfig.getParameter<edm::InputTag>("lctsTag");
}



LCTOccupancies::~LCTOccupancies()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   delete runSRLUTs;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
LCTOccupancies::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  ///////////////////
  //Setup Stuff//////
  ///////////////////

  std::vector<csctf::TrackStub> stubs;
  std::vector<csctf::TrackStub>::const_iterator stub;

  //MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi>    "csctfunpacker"          ""        "CsctfFilter"   
  edm::Handle<CSCCorrelatedLCTDigiCollection> lctDigiColls;
  iEvent.getByLabel(lctsTag,lctDigiColls);

  std::vector<csctf::TrackStub>* trackStubs = new std::vector<csctf::TrackStub>;
  runSRLUTs->makeTrackStubs(lctDigiColls.product(),trackStubs);

  std::vector<csctf::TrackStub>::const_iterator ts = trackStubs->begin();
  std::vector<csctf::TrackStub>::const_iterator tsEnd = trackStubs->end();

  std::vector<int> occStation1SubSec1(12,0) ;
  std::vector<int> occStation1SubSec2(12,0) ;
  std::vector<int> occStation2(12,0) ;
  std::vector<int> occStation3(12,0) ;
  std::vector<int> occStation4(12,0) ;


  for (;ts != tsEnd; ts++)
  {
	//std::cout << "etaValue: \t" <<ts->etaValue()<< std::endl;
	//std::cout << "phiValue: \t" <<ts->phiValue()<< std::endl;
	//std::cout << "eta: \t" <<ts->etaPacked()<< std::endl;
	//std::cout << "phi: \t" <<ts->phiPacked()<< std::endl;
	//std::cout << "cscid: \t" <<ts->cscid()<< std::endl;
	//std::cout << "subsector: \t" <<ts->subsector()<< std::endl;
	//std::cout << "sector: \t" <<ts->sector()<< std::endl;
	//std::cout << "station: \t" <<ts->station()<< std::endl;
	//std::cout << "endcap: \t" <<ts->endcap()<< std::endl;
	//std::cout << "bx: \t" <<ts->BX()<< std::endl;
	//std::cout << "MPCLink: \t" <<ts->getMPCLink()<< std::endl;
	//std::cout << std::endl;

	unsigned sector = ts->sector()-1;

	hMPCLink->Fill(ts->getMPCLink());
	hLocalPhi->Fill(ts->phiValue());
	hPhi->Fill(ts->phiValue()+15.0*M_PI/180+(sector)*60.0*M_PI/180); 
	hPhiPacked->Fill(ts->phiPacked());
	hEta->Fill(ts->etaValue());
	hEtaPacked->Fill(ts->etaPacked());
	hBx->Fill(ts->BX());
        int station = ts->station();
        int subsector = ts->subsector();
	if (ts->endcap()==2)
	{
	  station = -station;
	  sector = sector+6;
	}
	hSector->Fill(sector+1);
	hStation->Fill(station);
	hSubSector->Fill(subsector);
        //std::cout << "my sector packed: " << sector << std::endl << std::endl;
	if (station==1)
	{
	  if(subsector==1)
		occStation1SubSec1[sector]++;
	  else
		occStation1SubSec2[sector]++;
	}
	if (station==2)
		occStation2[sector]++;
	if (station==3)
		occStation3[sector]++;
	if (station==4)
		occStation4[sector]++;
  }

  int maxOcc = 0;

  for (unsigned iSector=0; iSector<12; iSector++)
  {
    hOccStation1SubSec1->Fill(occStation1SubSec1[iSector]);
    hOccStation1SubSec2->Fill(occStation1SubSec2[iSector]);
    hOccStation2->Fill(occStation2[iSector]);
    hOccStation3->Fill(occStation3[iSector]);
    hOccStation4->Fill(occStation4[iSector]);

    if (occStation1SubSec1[iSector]>maxOcc)
	maxOcc = occStation1SubSec1[iSector];
    if (occStation1SubSec2[iSector]>maxOcc)
	maxOcc = occStation1SubSec2[iSector];
    if (occStation2[iSector]>maxOcc)
	maxOcc = occStation2[iSector];
    if (occStation3[iSector]>maxOcc)
	maxOcc = occStation3[iSector];
    if (occStation4[iSector]>maxOcc)
	maxOcc = occStation4[iSector];
  }
  hOccMax->Fill(maxOcc);
  if(maxOcc>0)
    hOccMaxNo0->Fill(maxOcc);

  hStubsTotal->Fill(trackStubs->size());

}


// ------------ method called once each job just before starting event loop  ------------
void 
LCTOccupancies::beginJob()
{
  hMPCLink=fs->make<TH1F>("MPCLink","Stub MPC Link Number",5,-1,4);
  hMPCLink->GetXaxis()->SetTitle("MPC Link");
  hMPCLink->GetYaxis()->SetTitle("Counts");
  hLocalPhi=fs->make<TH1F>("LocalPhi","Stub Local #phi",4096,0,2.0*62.0*M_PI/180.0);//62.0*M_PI/180.0
  hLocalPhi->GetXaxis()->SetTitle("Local #phi");
  hLocalPhi->GetYaxis()->SetTitle("Counts");
  hPhi=fs->make<TH1F>("Phi","Stub #phi",100,0,3.2);
  hPhi->GetXaxis()->SetTitle("#phi");
  hPhi->GetYaxis()->SetTitle("Counts");
  hPhiPacked=fs->make<TH1F>("PhiPacked","Stub #phi Packed",4096,0,4096);//4096
  hPhiPacked->GetXaxis()->SetTitle("#phi Packed");
  hPhiPacked->GetYaxis()->SetTitle("Counts");
  hEta=fs->make<TH1F>("Eta","Stub |#eta|",128,0.9,2.4);
  hEta->GetXaxis()->SetTitle("|#eta|");
  hEta->GetYaxis()->SetTitle("Counts");
  hEtaPacked=fs->make<TH1F>("EtaPacked","Stub #eta Packed",128,0,128);//128
  hEtaPacked->GetXaxis()->SetTitle("#eta Packed");
  hEtaPacked->GetYaxis()->SetTitle("Counts");
  hBx=fs->make<TH1F>("Bx","Stub Bx Number",15,0,15);
  hBx->GetXaxis()->SetTitle("bx");
  hBx->GetYaxis()->SetTitle("Counts");

  hStation=fs->make<TH1F>("Station","Stub Station",9,-4.5,4.5);
  hStation->GetXaxis()->SetTitle("Station");
  hStation->GetYaxis()->SetTitle("Counts");
  hSector=fs->make<TH1F>("Sector","Stub Sector",12,0.5,12.5);
  hSector->GetXaxis()->SetTitle("Sector");
  hSector->GetYaxis()->SetTitle("Counts");
  hSubSector=fs->make<TH1F>("SubSector","Stub SubSector",3,0,3);
  hSubSector->GetXaxis()->SetTitle("SubSector");
  hSubSector->GetYaxis()->SetTitle("Counts");

  hOccStation1SubSec1=fs->make<TH1F>("OccStation1SubSec1","Stub Occupancy, Station 1, Subsector 1",5,-0.5,4.5);
  hOccStation1SubSec1->GetXaxis()->SetTitle("Stub Occupancy, Station 1, Subsector 1, Summed over Sectors");
  hOccStation1SubSec1->GetYaxis()->SetTitle("Counts");
  hOccStation1SubSec2=fs->make<TH1F>("OccStation1SubSec2","Stub Occupancy, Station 1, Subsector 2",5,-0.5,4.5);
  hOccStation1SubSec2->GetXaxis()->SetTitle("Stub Occupancy, Station 1, Subsector 1, Summed over Sectors");
  hOccStation1SubSec2->GetYaxis()->SetTitle("Counts");
  hOccStation2=fs->make<TH1F>("OccStation2","Stub Occupancy, Station 2",5,-0.5,4.5);
  hOccStation2->GetXaxis()->SetTitle("Stub Occupancy, Station 2, Summed over Sectors");
  hOccStation2->GetYaxis()->SetTitle("Counts");
  hOccStation3=fs->make<TH1F>("OccStation3","Stub Occupancy, Station 3",5,-0.5,4.5);
  hOccStation3->GetXaxis()->SetTitle("Stub Occupancy, Station 3, Summed over Sectors");
  hOccStation3->GetYaxis()->SetTitle("Counts");
  hOccStation4=fs->make<TH1F>("OccStation4","Stub Occupancy, Station 4",5,-0.5,4.5);
  hOccStation4->GetXaxis()->SetTitle("Stub Occupancy, Station 4, Summed over Sectors");
  hOccStation4->GetYaxis()->SetTitle("Counts");
  hOccMax=fs->make<TH1F>("OccMax","Maximum Stub Occupancy",5,-0.5,4.5);
  hOccMax->GetXaxis()->SetTitle("Maximum Stub Occupancy of Stations, Sectors, and Subsectors");
  hOccMax->GetYaxis()->SetTitle("Counts");
  hOccMaxNo0=fs->make<TH1F>("OccMaxNo0","Maximum Stub Occupancy",4,0.5,4.5);
  hOccMaxNo0->GetXaxis()->SetTitle("Maximum Stub Occupancy of Stations, Sectors, and Subsectors");
  hOccMaxNo0->GetYaxis()->SetTitle("Counts");
  hStubsTotal=fs->make<TH1F>("StubsTotal","N Stubs",20,0,20);
  hStubsTotal->GetXaxis()->SetTitle("N Stubs Unpacked in Event");
  hStubsTotal->GetYaxis()->SetTitle("Counts");
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LCTOccupancies::endJob() 
{
  TH1F* hOccStation1AddSubSec=fs->make<TH1F>("OccStation1AddSubSec","Stub Occupancy, Station 1",5,-0.5,4.5);
  hOccStation1AddSubSec->GetXaxis()->SetTitle("Stub Occupancy, Station 1, Summed over Sectors, Subsectors");
  hOccStation1AddSubSec->GetYaxis()->SetTitle("Counts");
  TH1F* hOccAddStations=fs->make<TH1F>("OccAddStation","Stub Occupancy",5,-0.5,4.5);
  hOccAddStations->GetXaxis()->SetTitle("Stub Occupancy, Summed over Stations, Sectors, Subsectors");
  hOccAddStations->GetYaxis()->SetTitle("Counts");

  hOccStation1AddSubSec->Add(hOccStation1SubSec1);
  hOccStation1AddSubSec->Add(hOccStation1SubSec2);
  hOccAddStations->Add(hOccStation1SubSec1);
  hOccAddStations->Add(hOccStation1SubSec2);
  hOccAddStations->Add(hOccStation2);
  hOccAddStations->Add(hOccStation3);
  hOccAddStations->Add(hOccStation4);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LCTOccupancies);
