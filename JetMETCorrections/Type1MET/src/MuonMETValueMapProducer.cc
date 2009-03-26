// -*- C++ -*-
//
// Package:    MuonMETValueMapProducer
// Class:      MuonMETValueMapProducer
// 
/**\class MuonMETValueMapProducer MuonMETValueMapProducer.cc JetMETCorrections/Type1MET/src/MuonMETValueMapProducer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Puneeth Kalavase
//         Created:  Sun Mar 15 11:33:20 CDT 2009
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "JetMETCorrections/Type1MET/interface/MuonMETValueMapProducer.h"
#include "JetMETCorrections/Type1MET/interface/MuonMETAlgo.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h" 
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

typedef math::XYZTLorentzVector LorentzVector;
typedef math::XYZPoint Point;


namespace cms {
  MuonMETValueMapProducer::MuonMETValueMapProducer(const edm::ParameterSet& iConfig) {

    using namespace edm;
  
    produces<ValueMap<int> >   ("muCorrFlag");
    produces<ValueMap<double> >("muCorrDepX");
    produces<ValueMap<double> >("muCorrDepY");
  
    //get configuration parameters
    minPt_       = iConfig.getParameter<double>("minPt"       );
    maxEta_      = iConfig.getParameter<double>("maxEta"      );
    isAlsoTkMu_  = iConfig.getParameter<bool>  ("isAlsoTkMu"  );
    maxNormChi2_ = iConfig.getParameter<double>("maxNormChi2" );
    maxd0_       = iConfig.getParameter<double>("maxd0"       );
    minnHits_    = iConfig.getParameter<int>   ("minnHits"    );
    qOverPErr_   = iConfig.getParameter<double>("qOverPErr"   );
    delPtOverPt_ = iConfig.getParameter<double>("delPtOverPt" );
  
    beamSpotInputTag_            = iConfig.getParameter<InputTag>("beamSpotInputTag"         );
    muonInputTag_   = iConfig.getParameter<InputTag>("muonInputTag");
  
    //Parameters from earlier
    useTrackAssociatorPositions_ = iConfig.getParameter<bool>("useTrackAssociatorPositions");
    useHO_                       = iConfig.getParameter<bool>("useHO"                      );
  
    ParameterSet trackAssociatorParams =
      iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
    trackAssociatorParameters_.loadParameters(trackAssociatorParams);
    trackAssociator_.useDefaultPropagator();
  
    towerEtThreshold_ = iConfig.getParameter<double>("towerEtThreshold");
    useRecHits_     = iConfig.getParameter<bool>("useRecHits");
  
  }


  MuonMETValueMapProducer::~MuonMETValueMapProducer()
  {
 
    // do anything here that needs to be done at desctruction time
    // (e.g. close files, deallocate resources etc.)

  }


  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void MuonMETValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
    using namespace edm;
    using namespace reco;
  
    //get the Muon collection
    Handle<View<reco::Muon> > muons;
    iEvent.getByLabel(muonInputTag_,muons);

    //use the BeamSpot
    Handle<BeamSpot> beamSpotH;
    iEvent.getByLabel(beamSpotInputTag_, beamSpotH);
    bool haveBeamSpot = true;
    if(!beamSpotH.isValid() )
      haveBeamSpot = false;

    //get the Bfield
    edm::ESHandle<MagneticField> magneticField;
    iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
    //get the B-field at the origin
    double bfield = magneticField->inTesla(GlobalPoint(0.,0.,0.)).z();

    //make a ValueMap of ints => flags for 
    //met correction. The values and meanings of the flags are :
    // flag==0 --->    The muon is not used to correct the MET by default
    // flag==1 --->    The muon is used to correct the MET. The Global pt is used.
    // flag==2 --->    The muon is used to correct the MET. The tracker pt is used.
    // flag==3 --->    The muon is used to correct the MET. The standalone pt is used.
    // In general, the flag should never be 3. You do not want to correct the MET using
    // the pt measurement from the standalone system (unless you really know what you're 
    // doing
    
    std::auto_ptr<ValueMap<int> >    vm_flag( new ValueMap<int> ());
    std::auto_ptr<ValueMap<double> > vm_delx( new ValueMap<double> ());
    std::auto_ptr<ValueMap<double> > vm_dely( new ValueMap<double> ());
    uint nMuons = muons->size();
    std::vector<int>     v_flag;
    std::vector<double>  v_delx;
    std::vector<double>  v_dely;

    for (unsigned int iMu=0; iMu<nMuons; iMu++) {

      const reco::Muon* mu = &(*muons)[iMu];
      double deltax = 0.0;
      double deltay = 0.0;
        
      TrackRef mu_track;
      if(mu->isGlobalMuon()) {
	mu_track = mu->globalTrack();
      } else if(mu->isTrackerMuon()) {
	mu_track = mu->innerTrack();
      } else 
	mu_track = mu->outerTrack();
    
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup,
							  trackAssociator_.getFreeTrajectoryState(iSetup, *mu_track),
							  trackAssociatorParameters_);
      MuonMETAlgo alg;
      alg.GetMuDepDeltas(mu, info,
			  useTrackAssociatorPositions_, useRecHits_,
			  useHO_, towerEtThreshold_, 
			  deltax, deltay, bfield);

      v_delx.push_back(deltax);
      v_dely.push_back(deltay);
    
      //now we have to figure out the flags
      int flag = 0;
      //have to be a global muon!
      if(!mu->isGlobalMuon()) {
	v_flag.push_back(flag);
	continue;
      }
    
      //if we require that the muon be both a global muon and tkmuon
      //but the muon fails the tkmuon requirement, we fail it
      if(!mu->isTrackerMuon() && isAlsoTkMu_) {
	v_flag.push_back(flag);
	continue;
      }

      //if we have gotten here, we only have global muons
        
      TrackRef globTk = mu->globalTrack();
      TrackRef siTk   = mu->innerTrack();
        
      if(globTk->pt() < minPt_ || fabs(globTk->eta()) > maxEta_) {
	v_flag.push_back(flag);
	continue;
      }
      if(globTk->chi2()/globTk->ndof() > maxNormChi2_) {
	v_flag.push_back(flag);
	continue;
      }
      if(fabs(globTk->dxy(beamSpotH->position())) > fabs(maxd0_) ) {
	v_flag.push_back(flag);
	continue;
      }
      if(siTk->numberOfValidHits() < minnHits_) {
	v_flag.push_back(flag);
	continue;
      }

      //if we've gotten here. the global muon has passed all the tests
      //all that remains is to see which pt we need to use to correct the MET
      double delpt = fabs(globTk->pt() - siTk->pt());
      if(delpt/(siTk->pt()) < delPtOverPt_ && globTk->qoverpError() < qOverPErr_)
	v_flag.push_back(1);
      else 
	v_flag.push_back(2);
    }
    
    ValueMap<int>::Filler flagFiller(*vm_flag);
    ValueMap<double>::Filler delXFiller(*vm_delx);
    ValueMap<double>::Filler delYFiller(*vm_dely);
   
    flagFiller.insert(muons, v_flag.begin(), v_flag.end());
    delXFiller.insert(muons, v_delx.begin(), v_delx.end());
    delYFiller.insert(muons, v_dely.begin(), v_dely.end());
    flagFiller.fill();
    delXFiller.fill();
    delYFiller.fill();

    iEvent.put(vm_flag, "muCorrFlag");
    iEvent.put(vm_delx, "muCorrDepX");
    iEvent.put(vm_dely, "muCorrDepY");
    
  }
  
  // ------------ method called once each job just before starting event loop  ------------
  void MuonMETValueMapProducer::beginJob()
  {
  }

  // ------------ method called once each job just after ending the event loop  ------------
  void MuonMETValueMapProducer::endJob() {
  }
}
