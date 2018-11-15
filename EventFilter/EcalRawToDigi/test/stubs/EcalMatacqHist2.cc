#include "EcalMatacqHist2.h"

#include "TH1D.h"
#include "TProfile.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <FWCore/Utilities/interface/Exception.h>


EcalMatacqHist2::EcalMatacqHist2(const edm::ParameterSet& ps):
  iEvent(0){
  outFileName= ps.getUntrackedParameter<std::string>("outputRootFile", "matacqHist.root");
  nTimePlots = ps.getUntrackedParameter<int>("nTimePlots", 10);
  firstTimePlotEvent = ps.getUntrackedParameter<int>("firstTimePlotEvent",
						     1);
  hTTrigMin = ps.getUntrackedParameter<double>("hTTrigMin", 0.);
  hTTrigMax = ps.getUntrackedParameter<double>("hTTrigMax", 2000.);
  matacqProducer_ = ps.getParameter<std::string>("matacqProducer");
  TDirectory* dsave = gDirectory;
  outFile = std::unique_ptr<TFile> (new TFile(outFileName.c_str(), "RECREATE"));
  if(outFile->IsZombie()){
    std::cout << "EcalMatacqHist2: Failed to create file " << outFileName
	 << " No histogram will be created.\n";
  }

  hTTrig =  new TH1D("tTrig", "Trigger time in ns",
		      100,
		      hTTrigMin,
		      hTTrigMax);
  dsave->cd();
}

EcalMatacqHist2::~EcalMatacqHist2(){
  if(!outFile->IsZombie()){
    TDirectory* dsave = gDirectory;
    outFile->cd();
    for(std::vector<TProfile>::iterator it = profiles.begin();
       it != profiles.end();
       ++it){
      it->Write();
    }
    if(hTTrig!=0) hTTrig->Write();
    dsave->cd();
  }
}

void
EcalMatacqHist2:: analyze( const edm::Event & e, const  edm::EventSetup& c){
  ++iEvent;
  if(outFile->IsZombie()) return;
  TDirectory* dsave = gDirectory;
  outFile->cd();

  edm::TimeValue_t t = e.time().value();

  time_t ts = t >>32;
  time_t tus = t & 0xFFFFFFFF;
  char buf[256];
  strftime(buf, sizeof(buf), "%F %R %S s", localtime(&ts));
   
  std::cerr << std::flush;
  std::cout << "---- > Event data: " << buf
	    << " " << tus << "us" << std::endl;
  
  // retrieving MATACQ digis:
  edm::Handle<EcalMatacqDigiCollection> digiColl;
  if(e.getByLabel(matacqProducer_, "", digiColl)){
    unsigned iCh=0;
    for(EcalMatacqDigiCollection::const_iterator it = digiColl->begin();
	it!=digiColl->end(); ++it, ++iCh){
      
      const EcalMatacqDigi& digis = *it;

      std::cout << "Matacq digi size: " << digis.size() << std::endl;
      
      if(digis.size()==0) continue;
      
      if(iEvent >= firstTimePlotEvent
	 && iEvent < firstTimePlotEvent + nTimePlots){
      int nSamples = digis.size();
      std::stringstream title;
      std::stringstream name;
      name << "matacq" << digis.chId() << "_"
	   << std::setfill('0') << std::setw(4) << iEvent;
      title << "Matacq channel " <<  digis.chId() << ", event " << iEvent
	    << ", Ts = " << digis.ts()*1.e9 << "ns";
      float tTrig_s = digis.tTrig();
      if(tTrig_s<999.){
	title << ", t_trig = " << tTrig_s * 1.e9 << "ns";
      }
      TH1D h1(name.str().c_str(), title.str().c_str(),
	      nSamples, -.5, -.5+nSamples);
      for(int i=0; i<digis.size(); ++i){
	h1.Fill(i, digis.adcCount(i));
      }
      h1.Write();
    }
    
    //profile
    //init:
    if(iCh>=profiles.size()){ //profile not yet allocated for this matacq ch.
      std::stringstream profTitle;
      profTitle << "Matacq channel " <<  digis.chId()
		<< " profile";
      std::stringstream profileName;
      profileName << "matacq" << digis.chId();
      profiles.push_back(TProfile(profileName.str().c_str(),
				  profTitle.str().c_str(),
				  digis.size(),
				  -.5,
				  -.5+digis.size(),
				  "I"));
      profiles.back().SetDirectory(0);//mem. management done by std::vector
      profChId.push_back(digis.chId());
    }
    
    for(int i=0; i<digis.size(); ++i){
      if(profChId[iCh]==digis.chId()){
	profiles[iCh].Fill(i, digis.adcCount(i));
      } else{
	throw cms::Exception("EcalMatacqHist2",
			     "Order or number of matacq channels is not the "
			     "same in the different event. Such a "
			     "configuration is not supported by "
			     "EcalMatacqHist2");
      }
      hTTrig->Fill(digis.tTrig()*1.e9);
    }
    }
  } else{
    edm::LogInfo("No matacq digi found");
  }
  dsave->cd();
} // analyze


DEFINE_FWK_MODULE(EcalMatacqHist2);
