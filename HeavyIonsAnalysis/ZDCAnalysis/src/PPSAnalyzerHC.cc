// -*- C++ -*-
//
// Package:    HeavyIonAnalyzer/ZDCAnalysis/PPSAnalyzerHC
// Class:      PPSAnalyzerHC
//
/**\class PPSAnalyzerHC PPSAnalyzerHC.cc HeavyIonAnalyzer/ZDCAnalysis/plugins/PPSAnalyzerHC
   Description: Produced Tree with local tracks from PPS
*/
//
// Original Author:  Michael Pitt, CERN
//         Created:  29-06-2025
//         Modified from PPSAnalyzerHC
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TTree.h"

#include "PPSstruct.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

using reco::TrackCollection;

class PPSAnalyzerHC : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit PPSAnalyzerHC(const edm::ParameterSet&);
  ~PPSAnalyzerHC();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<std::vector<CTPPSLocalTrackLite> > ctppsToken_;
  edm::Service<TFileService> fs;
  TTree* t1;

  MyPPSTracks ppsTracks;

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  edm::ESGetToken<SetupData, SetupRecord> setupToken_;
#endif
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
PPSAnalyzerHC::PPSAnalyzerHC(const edm::ParameterSet& iConfig)
    : ctppsToken_(consumes<std::vector<CTPPSLocalTrackLite> >(iConfig.getParameter<edm::InputTag>("ctppsLocalTracks")))
      {
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  setupDataToken_ = esConsumes<SetupData, SetupRecord>();
#endif
  //now do what ever initialization is needed
}

PPSAnalyzerHC::~PPSAnalyzerHC() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called for each event  ------------
void PPSAnalyzerHC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  edm::Handle<CTPPSLocalTrackLiteCollection> recoPPSTracks;
  iEvent.getByToken(ctppsToken_, recoPPSTracks);

  ppsTracks.n = 0;
  for (unsigned int i = 0; i < MAXPRO; i++) {
    ppsTracks.zside[i] = -99;
    ppsTracks.station[i] = -99;
    ppsTracks.x[i] = -999;
    ppsTracks.y[i] = -999;
  }

  int nhits = 0;
  for (const auto& trk : *recoPPSTracks) {
    CTPPSDetId detid(trk.rpId());
    
    int zside = detid.arm();
    int station = detid.station();
    int rp = detid.rp();

    if (nhits >= MAXPRO)
      break;
    if (!(rp == 3))
      continue;  // only pixels (horizontal tracking stations)

    ppsTracks.zside[nhits] = zside ? -1 : 1; // arm 0 is sector 45 (positive z), arm 1 is sector 56 (negative z)
    ppsTracks.station[nhits] = station; // 0 for near pixel, 2 for far pixel station

    ppsTracks.x[nhits] = trk.x();
    ppsTracks.y[nhits] = trk.y();

    nhits++;
  }  // end loop pps tracks

  ppsTracks.n = nhits;

  t1->Fill();

  // #ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  // // if the SetupData is always needed
  // auto setup = iSetup.getData(setupToken_);
  // // if need the ESHandle to check if the SetupData was there or not
  // auto pSetup = iSetup.getHandle(setupToken_);
  // #endif
}

// ------------ method called once each job just before starting event loop  ------------
void PPSAnalyzerHC::beginJob() {
  t1 = fs->make<TTree>("ppstracks", "ppstracks");

  t1->Branch("n", &ppsTracks.n, "n/I");
  t1->Branch("zside", ppsTracks.zside, "zside[n]/I");
  t1->Branch("station", ppsTracks.station, "station[n]/I");
  t1->Branch("x", ppsTracks.x, "x[n]/F");
  t1->Branch("y", ppsTracks.y, "y[n]/F");

}

// ------------ method called once each job just after ending the event loop  ------------
void PPSAnalyzerHC::endJob() {
  // please remove this method if not needed
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void PPSAnalyzerHC::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PPSAnalyzerHC);