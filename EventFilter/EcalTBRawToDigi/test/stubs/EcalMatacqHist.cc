#include "EcalMatacqHist.h"

#include "TProfile.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <FWCore/Utilities/interface/Exception.h>


EcalMatacqHist::EcalMatacqHist(const edm::ParameterSet& ps):
  iEvent(0){
  outFileName= ps.getUntrackedParameter<std::string>("outputRootFile", "matacqHist.root");
  nTimePlots = ps.getUntrackedParameter<int>("nTimePlots", 10);
  firstTimePlotEvent = ps.getUntrackedParameter<int>("firstTimePlotEvent",
						     1);
  hTTrigMin = ps.getUntrackedParameter<double>("hTTrigMin", 0.);
  hTTrigMax = ps.getUntrackedParameter<double>("hTTrigMax", 2000.);
  TDirectory* dsave = gDirectory;
  outFile = std::unique_ptr<TFile> (new TFile(outFileName.c_str(), "RECREATE"));
  if(outFile->IsZombie()){
    std::cout << "EcalMatacqHist: Failed to create file " << outFileName
	 << " No histogram will be created.\n";
  }

  hTTrig =  new TH1D("tTrig", "Trigger time in ns",
		      100,
		      hTTrigMin,
		      hTTrigMax);
  dsave->cd();
}

EcalMatacqHist::~EcalMatacqHist(){
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
EcalMatacqHist:: analyze( const edm::Event & e, const  edm::EventSetup& c){
  ++iEvent;
  if(outFile->IsZombie()) return;
  TDirectory* dsave = gDirectory;
  outFile->cd();
  // retrieving MATACQ digis:
  edm::Handle<EcalMatacqDigiCollection> digiColl;
  e.getByLabel("ecalEBunpacker", digiColl);

  unsigned iCh=0;
  for(EcalMatacqDigiCollection::const_iterator it = digiColl->begin();
      it!=digiColl->end(); ++it, ++iCh){
   
    const EcalMatacqDigi& digis = *it;

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
	throw cms::Exception("EcalMatacqHist",
			     "Order or number of matacq channels is not the "
			     "same in the different event. Such a "
			     "configuration is not supported by "
			     "EcalMatacqHist");
      }
      hTTrig->Fill(digis.tTrig()*1.e9);
    }
  }
  dsave->cd();
} // analyze


DEFINE_FWK_MODULE(EcalMatacqHist);
