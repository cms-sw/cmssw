// -*- C++ -*-
//
// Package:    evtCounter
// Class:      evtCounter
//
/**\class evtCounter evtCounter.cc CmsHi/evtCounter/src/evtCounter.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Yong Kim,32 4-A08,+41227673039,
//         Created:  Fri Oct 29 12:18:14 CEST 2010
// $Id: evtCounter.cc,v 1.4 2011/04/06 21:26:21 kimy Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollectionV.h"
//#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <Math/VectorUtil.h>
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "TNtuple.h"


//
// class declaration
//

class evtCounter : public edm::EDAnalyzer {
public:
  explicit evtCounter(const edm::ParameterSet&);
  ~evtCounter();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------

  edm::Service<TFileService> fs;

  TH1D*  NoE ;
  TNtuple* nt;

  double hf;
  double eb;
  double nPix;
  double cbin;
  double nPixelTracks;
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
evtCounter::evtCounter(const edm::ParameterSet& iConfig)

{

}


evtCounter::~evtCounter()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
evtCounter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  NoE->Fill(0);

  Float_t var[100];
  int idx = 0;
  var[idx] = hf;   idx++;
  var[idx] = eb;   idx++;
  var[idx] = nPix;   idx++;
  var[idx] = nPixelTracks;   idx++;
  var[idx] = (float)cbin;    idx++;
  nt->Fill(var);


}


// ------------ method called once each job just before starting event loop  ------------
void
evtCounter::beginJob()
{
  NoE      = fs->make<TH1D>( "NoE"  , "", 1,  -100., 100. );
  nt       = fs->make<TNtuple>("cent", "centrality info",
			       "hf:eb:nPix:nPixTrk:cbin");
  std::cout<<"done beginjob"<<std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void
evtCounter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(evtCounter);
