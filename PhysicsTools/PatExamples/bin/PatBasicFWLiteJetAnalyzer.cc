#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <TH1F.h>
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "FWCore/ParameterSetReader/interface/ProcessDescImpl.h"


int main(int argc, char* argv[])
{
  // ----------------------------------------------------------------------
  // First Part:
  //
  //  * enable FWLite
  //  * book the histograms of interest
  //  * open the input file
  // ----------------------------------------------------------------------

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();

  // only allow one argument for this simple example which should be the
  // the python cfg file
  if ( argc < 2 ) {
    std::cout << "Usage : " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }
 
  // get the python configuration
  ProcessDescImpl builder(argv[1]);
  const edm::ParameterSet& fwliteParameters = builder.processDesc()->getProcessPSet()->getParameter<edm::ParameterSet>("FWLiteParams");

  // now get each parameter
  std::string   input_ ( fwliteParameters.getParameter<std::string  >("inputFile"  ) );
  std::string   output_( fwliteParameters.getParameter<std::string  >("outputFile" ) );
  edm::InputTag jets_  ( fwliteParameters.getParameter<edm::InputTag>("jets") );

  // book a set of histograms
  fwlite::TFileService fs = fwlite::TFileService(output_);
  TFileDirectory theDir = fs.mkdir("analyzeBasicPat");
  TH1F* jetPt_  = theDir.make<TH1F>("jetPt", "pt",    100,  0.,300.);
  TH1F* jetEta_ = theDir.make<TH1F>("jetEta","eta",   100, -3.,  3.);
  TH1F* jetPhi_ = theDir.make<TH1F>("jetPhi","phi",   100, -5.,  5.);
  TH1F* disc_   = theDir.make<TH1F>("disc", "Discriminant", 100, 0.0, 10.0);
  TH1F* constituentPt_ = theDir.make<TH1F>("constituentPt", "Constituent pT", 100, 0, 300.0);
 
  // open input file (can be located on castor)
  TFile* inFile = TFile::Open(input_.c_str());

  // ----------------------------------------------------------------------
  // Second Part:
  //
  //  * loop the events in the input file
  //  * receive the collections of interest via fwlite::Handle
  //  * fill the histograms
  //  * after the loop close the input file
  // ----------------------------------------------------------------------

  // loop the events
  unsigned int iEvent=0;
  fwlite::Event ev(inFile);
  for(ev.toBegin(); !ev.atEnd(); ++ev, ++iEvent){
    edm::EventBase const & event = ev;

    // break loop after end of file is reached
    // or after 1000 events have been processed
    if( iEvent==1000 ) break;
   
    // simple event counter
    if(iEvent>0 && iEvent%1==0){
      std::cout << "  processing event: " << iEvent << std::endl;
    }

    // Handle to the jet collection
    edm::Handle<std::vector<pat::Jet> > jets;
    edm::InputTag jetLabel( argv[3] );
    event.getByLabel(jets_, jets);
   
    // loop jet collection and fill histograms
    for(unsigned i=0; i<jets->size(); ++i){
      // basic kinematics
      jetPt_ ->Fill( (*jets)[i].pt()  );
      jetEta_->Fill( (*jets)[i].eta() );
      jetPhi_->Fill( (*jets)[i].phi() );
      // access tag infos
      reco::SecondaryVertexTagInfo const *svTagInfos = (*jets)[i].tagInfoSecondaryVertex("secondaryVertex");
      if( svTagInfos != nullptr ) {
	if( svTagInfos->nVertices() > 0 ){
	  disc_->Fill( svTagInfos->flightDistance(0).value() );
	}
      }
      // access calo towers
      std::vector<reco::PFCandidatePtr> const & pfConstituents =  (*jets)[i].getPFConstituents();
      for( std::vector<reco::PFCandidatePtr>::const_iterator ibegin=pfConstituents.begin(), iend=pfConstituents.end(), iconstituent=ibegin; iconstituent!=iend; ++iconstituent){
	constituentPt_->Fill( (*iconstituent)->pt() );
      }
    }
  } 
  // close input file
  inFile->Close();
  
  // ----------------------------------------------------------------------
  // Third Part:
  //
  //  * never forget to free the memory of objects you created
  // ----------------------------------------------------------------------

  // in this example there is nothing to do
 
  // that's it!
  return 0;
}

