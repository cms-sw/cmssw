// -*- C++ -*-
//
// Package:   EcalBxOrbitNumberGrapher 
// Class:     EcalBxOrbitNumberGrapher 
// 
/**\class EcalBxOrbitNumberGrapher EcalBxOrbitNumberGrapher.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalBxOrbitNumberGrapher.cc,v 1.3 2011/10/10 09:05:21 eulisse Exp $
//
//

#include "CaloOnlineTools/EcalTools/plugins/EcalBxOrbitNumberGrapher.h"

using namespace cms;
using namespace edm;
using namespace std;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalBxOrbitNumberGrapher::EcalBxOrbitNumberGrapher(const edm::ParameterSet& iConfig) :
  digiProducer_(iConfig.getParameter<std::string>("RawDigis")),
  runNum_(-1),
  fileName_ (iConfig.getUntrackedParameter<std::string>("fileName", std::string("ecalURechHitHists")))
{
  
  
}


EcalBxOrbitNumberGrapher::~EcalBxOrbitNumberGrapher()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EcalBxOrbitNumberGrapher::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
   using namespace cms;
  //int ievt = iEvent.id().event();
  int orbit = -100;
  int bx = -100;
  int numorbiterrors = 0;
  bool orbiterror = false;

  
  edm::Handle<EcalRawDataCollection> DCCHeaders;
  iEvent.getByLabel(digiProducer_, DCCHeaders);
  if (!DCCHeaders.isValid()) {
	edm::LogError("BxOrbitNumber") << "can't get the product for EcalRawDataCollection";
  }

  //-----------------BX STuff here
   for ( EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin();headerItr != DCCHeaders->end(); 
	  ++headerItr ) {
    headerItr->getEventSettings();
    int myorbit = headerItr->getOrbit();
    int mybx = headerItr->getBX();
    
    if ( orbit == -100 )
    {
      orbit = myorbit;
    }
    else if (orbit != myorbit)
    {
       std::cout << " NOOOO This header has a conflicting orbit OTHER " << orbit << " new " << myorbit  << std::endl;
       orbiterror = true; 
       numorbiterrors++;
       orbitErrorBxDiffPlot_->Fill(myorbit - orbit);
    }
    
    if ( bx == -100 )
    {
      bx = mybx;
    }
    else if (bx != mybx)
    {
       std::cout << " NOOOO This header has a conflicting bx OTHER " << bx << " new " << mybx  << std::endl; 
    }
    //LogDebug("EcalTimingCosmic") << " Lambda " << lambda; //hmm... this isn't good, I should keep a record of the wavelength in the headers as an inactive SM might have a different wavelength for this field and make this not go through.
  }
  
  if ( (bx != -100) & ( orbit != -100) )
  {
     std::cout << " Interesting event Orbit " << orbit << " BX " << bx << std::endl;
     bxnumberPlot_->Fill(bx);
     if (orbiterror) 
     {
        orbitErrorPlot_->Fill(bx);	
     }
  }
  numberofOrbitDiffPlot_->Fill(numorbiterrors);
  
  
  if(runNum_==-1)
  {
    runNum_ = iEvent.id().run();
  }
}


// insert the hist map into the map keyed by FED number
void EcalBxOrbitNumberGrapher::initHists(int FED)
{
 
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalBxOrbitNumberGrapher::beginJob()
{
  bxnumberPlot_ = new TH1F("bxnumber", "BX number of interexting events",3600, 0., 3600.);
  orbitErrorPlot_ = new TH1F("bxOfOrbitDiffs", "BX number of interexting events with orbit changes",3600, 0., 3600.);
  orbitErrorBxDiffPlot_ = new TH1F("orbitErrorDiffPlot","Orbit Difference of those HEADERS that have a difference",20,-10.,10.);
  numberofOrbitDiffPlot_ = new TH1F("numberOfOrbitDiffsPlot","Number of Orbit Differences",54,0., 54.);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalBxOrbitNumberGrapher::endJob()
{
  using namespace std;
  fileName_ += ".bx.root";

  TFile root_file_(fileName_.c_str() , "RECREATE");

  bxnumberPlot_->Write();
  orbitErrorPlot_->Write();
  numberofOrbitDiffPlot_->Write();
  orbitErrorBxDiffPlot_->Write();
  root_file_.Close();

}



