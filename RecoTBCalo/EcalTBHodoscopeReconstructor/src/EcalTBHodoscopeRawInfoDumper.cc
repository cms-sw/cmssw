#include "RecoTBCalo/EcalTBHodoscopeReconstructor/interface/EcalTBHodoscopeRawInfoDumper.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <TFile.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
    
EcalTBHodoscopeRawInfoDumper::EcalTBHodoscopeRawInfoDumper(edm::ParameterSet const& ps)
{
  rawInfoCollection_ = ps.getParameter<std::string>("rawInfoCollection");
  rawInfoProducer_   = ps.getParameter<std::string>("rawInfoProducer");
  rootfile_          = ps.getUntrackedParameter<std::string>("rootfile","ecalHodoscopeRawInfoPlots.root");
}

EcalTBHodoscopeRawInfoDumper::~EcalTBHodoscopeRawInfoDumper() {
}

//========================================================================
void
EcalTBHodoscopeRawInfoDumper::beginJob() {
//========================================================================

  char histoName[100];
  char histoTitle[100];
  
  for (int i=0; i<4; i++)
    {
      sprintf(histoName,"h_numberOfFiredHits_%d",i);
      sprintf(histoTitle,"NumberOfFiredHits Plane %d",i);
      h_numberOfFiredHits_[i]=new TH1F(histoName,histoTitle,10,0.,10.);
    }

  for (int i=0; i<4; i++)
    {
      sprintf(histoName,"h_firedHits_%d",i);
      sprintf(histoTitle,"firedHits Plane %d",i);
      h_firedHits_[i]=new TH1F(histoName,histoTitle,64,-0.5,63.5);
    }  
}

//========================================================================
void
EcalTBHodoscopeRawInfoDumper::endJob() {
//========================================================================

  TFile f(rootfile_.c_str(),"RECREATE");
  
  for (int i=0; i<4; i++)
    h_numberOfFiredHits_[i]->Write();
  
  for (int i=0; i<4; i++)
    h_firedHits_[i]->Write();

  f.Close();
}

void EcalTBHodoscopeRawInfoDumper::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  // Get input
   edm::Handle<EcalTBHodoscopeRawInfo> ecalRawHodoscope;  
   const EcalTBHodoscopeRawInfo* hodoscopeRawInfo = 0;
   //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
   e.getByLabel( rawInfoProducer_, ecalRawHodoscope);
   if (!ecalRawHodoscope.isValid()) {
     edm::LogError("EcalTBHodoscopeRecInfoError") << "Error! can't get the product " << rawInfoCollection_.c_str() ;
   } else {
     hodoscopeRawInfo = ecalRawHodoscope.product();
   }

   if (hodoscopeRawInfo)
     for (int i=0; i<4; i++)
       {
	 std::ostringstream str;
	 str << " Hits " ;
	 std::vector<int> firedHits;
	 h_numberOfFiredHits_[i]->Fill((*hodoscopeRawInfo)[i].numberOfFiredHits());
	 for (int j=0;j<64;j++)
	   if ((*hodoscopeRawInfo)[i][j])
	     {
	       h_firedHits_[i]->Fill(j);
	       firedHits.push_back(j);
	       str << j << " " ;
	     }
	 LogDebug("EcalTBHodoscope") << "Looking plane " << i << " number of hits " << (*hodoscopeRawInfo)[i].numberOfFiredHits() << str.str();
       }
  // Create empty output
} 


