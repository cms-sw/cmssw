#include "DQM/PhysicsHWW/interface/MuonMaker.h"

typedef math::XYZTLorentzVectorF LorentzVector;
typedef math::XYZPoint Point;

using namespace std;
using namespace reco;
using namespace edm;


MuonMaker::MuonMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector iCollector) {

  Muon_                     = iCollector.consumes<edm::View<reco::Muon> > (iConfig.getParameter<edm::InputTag>("muonsInputTag"));
  MuonShower_               = iCollector.consumes<edm::ValueMap<reco::MuonShower> > (iConfig.getParameter<edm::InputTag>("muonShower"));
  thePVCollection_          = iCollector.consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexInputTag"));
  PFCandidateCollection_    = iCollector.consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandsInputTag"));
  BeamSpot_                 = iCollector.consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotTag"));
  MuonCollection_           = iCollector.consumes<reco::MuonCollection> (iConfig.getParameter<edm::InputTag>("muonsInputTag"));

}


void MuonMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  hww.Load_mus_sta_d0();
  hww.Load_mus_sta_z0corr();
  hww.Load_mus_sta_p4();
  hww.Load_mus_gfit_chi2();
  hww.Load_mus_gfit_ndof();
  hww.Load_mus_gfit_validSTAHits();
  hww.Load_mus_trkKink();
  hww.Load_mus_type();
  hww.Load_mus_goodmask();
  hww.Load_mus_charge();
  hww.Load_mus_nmatches();
  hww.Load_mus_caloCompatibility();
  hww.Load_mus_segmCompatibility();
  hww.Load_mus_p4();
  hww.Load_mus_numberOfMatchedStations();
  hww.Load_mus_pid_TMLastStationTight();
  hww.Load_mus_pid_PFMuon();
  hww.Load_mus_e_em();
  hww.Load_mus_e_had();
  hww.Load_mus_e_ho();
  hww.Load_mus_e_emS9();
  hww.Load_mus_e_hadS9();
  hww.Load_mus_e_hoS9();
  hww.Load_mus_iso_ecalvetoDep();
  hww.Load_mus_iso_hcalvetoDep();
  hww.Load_mus_iso03_sumPt();
  hww.Load_mus_iso03_emEt();
  hww.Load_mus_iso03_hadEt();
  hww.Load_mus_iso05_sumPt();
  hww.Load_mus_iso05_emEt();
  hww.Load_mus_iso05_hadEt();
  hww.Load_mus_trk_p4();
  hww.Load_mus_vertex_p4();
  hww.Load_mus_trkidx();
  hww.Load_mus_d0();
  hww.Load_mus_chi2();
  hww.Load_mus_ndof();
  hww.Load_mus_validHits();
  hww.Load_mus_ptErr();
  hww.Load_mus_isoR03_pf_ChargedHadronPt();
  hww.Load_mus_isoR03_pf_NeutralHadronEt();
  hww.Load_mus_isoR03_pf_PhotonEt();
  hww.Load_mus_isoR03_pf_PUPt();

  bool validToken;

  ///////////////
  // Get Muons //
  ///////////////

  Handle<View<Muon> > muon_h;
  validToken = iEvent.getByToken( Muon_ , muon_h );
  if(!validToken) return;

  
  /////////////////////////////////
  // Get Muon Shower Information //
  /////////////////////////////////

  Handle<ValueMap<MuonShower> > showerMap;
  validToken = iEvent.getByToken( MuonShower_ , showerMap );
  if(!validToken) return;


  //////////////////
  // Get Vertices //
  //////////////////

  edm::Handle<reco::VertexCollection> vertexHandle;
  validToken = iEvent.getByToken( thePVCollection_ , vertexHandle );  
  if(!validToken) return;


  ///////////////////////
  // Get PF Candidates //
  ///////////////////////

  edm::Handle<reco::PFCandidateCollection> pfCand_h;
  validToken = iEvent.getByToken( PFCandidateCollection_ , pfCand_h );
  if(!validToken) return;


  //////////////
  // Beamspot //
  //////////////

  Handle<reco::BeamSpot> beamspot_h;
  validToken = iEvent.getByToken(BeamSpot_, beamspot_h);
  if(!validToken) return;
  const reco::BeamSpot &beamSpotreco = *(beamspot_h.product());
  const Point beamSpot = Point(beamSpotreco.x0(), beamSpotreco.y0(), beamSpotreco.z0());

  
  ////////////////////////// 
  // Cosmic Compatibility //
  //////////////////////////

  Handle<MuonCollection> muons;
  validToken = iEvent.getByToken( MuonCollection_, muons );
  if(!validToken) return;

  ///////////
  // Muons // 
  ///////////
  
  unsigned int muonIndex = 0;
  View<Muon>::const_iterator muons_end = muon_h->end();  // Iterator
  for ( View<Muon>::const_iterator muon = muon_h->begin(); muon != muons_end; ++muon ) {

    // References
    const RefToBase<Muon>         muonRef                 = muon_h->refAt(muonIndex); 
    const TrackRef                globalTrack             = muon->globalTrack();
    const TrackRef                siTrack                 = muon->innerTrack();
    const TrackRef                staTrack                = muon->outerTrack();
    const MuonQuality             quality                 = muon->combinedQuality();
    const VertexCollection*       vertexCollection        = vertexHandle.product();

    // Iterators
    VertexCollection::const_iterator firstGoodVertex = vertexCollection->end();


    /////////
    // STA //
    /////////

    hww.mus_sta_d0()            .push_back( staTrack.isNonnull()  ? staTrack->d0()                   :  -9999.        );
    hww.mus_sta_z0corr()        .push_back( staTrack.isNonnull()  ? staTrack->dz(beamSpot)           :  -9999.        );
    hww.mus_sta_p4()            .push_back( staTrack.isNonnull()  ? LorentzVector( staTrack->px() , staTrack->py() , staTrack->pz() , staTrack->p() ) : LorentzVector(0, 0, 0, 0) );


    ////////////
    // Global //
    ////////////

    hww.mus_gfit_chi2()         .push_back( globalTrack.isNonnull() ? globalTrack->chi2()               :  -9999.        );
    hww.mus_gfit_ndof()         .push_back( globalTrack.isNonnull() ? globalTrack->ndof()               :  -9999         );
    hww.mus_gfit_validSTAHits() .push_back( globalTrack.isNonnull() ? globalTrack->hitPattern().numberOfValidMuonHits()    : -9999         );

    //////////////////
    // Muon Quality //
    //////////////////

    hww.mus_trkKink()           .push_back( quality.trkKink             );



    // Calculate Overlaps
    int mus_overlap0 = -1, muInd = -1, mus_nOverlaps = 0;
    for ( View<Muon>::const_iterator muonJ = muon_h->begin(); muonJ != muons_end; ++muonJ ) {
      muInd++;
      if ( muonJ != muon ){
        if ( muon::overlap( *muon, *muonJ ) ) {
          if ( mus_overlap0 == -1) mus_overlap0 = muInd;
          mus_nOverlaps++;
        }
      }
    }

    // Calculate Muon position at ECAL
    math::XYZPoint ecal_p( -9999.0, -9999.0, -9999.0 );
    if( muon->isEnergyValid() ) ecal_p = muon->calEnergy().ecal_position;

    // Calculate Mask
    int goodMask = 0;
    for ( int iG = 0; iG < 24; ++iG ) { //overkill here
      if( isGoodMuon( *muon, (muon::SelectionType)iG ) ) goodMask |= (1<<iG);
    }

    
    /////////////////////
    // Muon Quantities //
    /////////////////////

    hww.mus_type()                    .push_back( muon->type()                                              );
    hww.mus_goodmask()                .push_back( goodMask                                                  );
    hww.mus_charge()                  .push_back( muon->charge()                                            );
    hww.mus_nmatches()                .push_back( muon->isMatchesValid() ? muon->numberOfMatches() :  -9999 );
    hww.mus_caloCompatibility()       .push_back( muon->caloCompatibility()                                 );
    hww.mus_segmCompatibility()       .push_back( muon::segmentCompatibility(*muon)                         );
    hww.mus_p4()                      .push_back( LorentzVector( muon->p4()                              )  );
    hww.mus_numberOfMatchedStations() .push_back( muon->numberOfMatchedStations()                            );


    /////////////////////////////
    // Muon Shower Information //
    /////////////////////////////

    const MuonShower muShower = showerMap.isValid() ? (*showerMap)[muonRef] : MuonShower();


    ////////
    // ID //
    ////////

    bool matchIsValid = muon->isMatchesValid();

    hww.mus_pid_TMLastStationTight()     .push_back( matchIsValid ? muon::isGoodMuon( *muon, muon::TMLastStationTight     ) : -9999  );
    hww.mus_pid_PFMuon()                 .push_back( muon->isPFMuon() );

    ////////////
    // Energy //
    ////////////

    bool energyIsValid = muon->isEnergyValid();

    hww.mus_e_em()               .push_back( energyIsValid ? muon->calEnergy().em                                 :  -9999.       );
    hww.mus_e_had()              .push_back( energyIsValid ? muon->calEnergy().had                                :  -9999.       );
    hww.mus_e_ho()               .push_back( energyIsValid ? muon->calEnergy().ho                                 :  -9999.       );
    hww.mus_e_emS9()             .push_back( energyIsValid ? muon->calEnergy().emS9                               :  -9999.       );
    hww.mus_e_hadS9()            .push_back( energyIsValid ? muon->calEnergy().hadS9                              :  -9999.       );
    hww.mus_e_hoS9()             .push_back( energyIsValid ? muon->calEnergy().hoS9                               :  -9999.       );


    ///////////////
    // Isolation //
    ///////////////

    hww.mus_iso_ecalvetoDep()    .push_back( muon->isEnergyValid()    ? muon->isolationR03().emVetoEt       : -9999.        );
    hww.mus_iso_hcalvetoDep()    .push_back( muon->isEnergyValid()    ? muon->isolationR03().hadVetoEt      : -9999.        );
    hww.mus_iso03_sumPt()        .push_back( muon->isIsolationValid() ? muon->isolationR03().sumPt          : -9999.        );
    hww.mus_iso03_emEt()         .push_back( muon->isIsolationValid() ? muon->isolationR03().emEt           : -9999.        );
    hww.mus_iso03_hadEt()        .push_back( muon->isIsolationValid() ? muon->isolationR03().hadEt          : -9999.        );
    hww.mus_iso05_sumPt()        .push_back( muon->isIsolationValid() ? muon->isolationR05().sumPt          : -9999.        );
    hww.mus_iso05_emEt()         .push_back( muon->isIsolationValid() ? muon->isolationR05().emEt           : -9999.        );
    hww.mus_iso05_hadEt()        .push_back( muon->isIsolationValid() ? muon->isolationR05().hadEt          : -9999.        );

    ////////////
    // Tracks //
    ////////////

    hww.mus_trk_p4()             .push_back( siTrack.isNonnull()     ? LorentzVector( siTrack.get()->px() , siTrack.get()->py() , siTrack.get()->pz() , siTrack.get()->p() ) : LorentzVector(     0.0,     0.0,     0.0,     0.0) );
    hww.mus_vertex_p4()          .push_back( siTrack.isNonnull()     ? LorentzVector( siTrack->vx()       , siTrack->vy()       , siTrack->vz()       , 0.0                ) : LorentzVector( -9999.0, -9999.0, -9999.0, -9999.0) );
    hww.mus_trkidx()             .push_back( siTrack.isNonnull()     ? static_cast<int>(siTrack.key())                      : -9999         );
    hww.mus_d0()                 .push_back( siTrack.isNonnull()     ? siTrack->d0()                                        : -9999.        );
    hww.mus_chi2()               .push_back( siTrack.isNonnull()     ? siTrack->chi2()                                      : -9999.        );
    hww.mus_ndof()               .push_back( siTrack.isNonnull()     ? siTrack->ndof()                                      : -9999.        );
    hww.mus_validHits()          .push_back( siTrack.isNonnull()     ? siTrack->numberOfValidHits()                         : -9999         );
    hww.mus_ptErr()              .push_back( siTrack.isNonnull()     ? siTrack->ptError()                                   : -9999.        );


    ////////
    // PF //
    ////////

    MuonPFIsolation pfStructR03 = muon->pfIsolationR03();

    hww.mus_isoR03_pf_ChargedHadronPt()                 .push_back( pfStructR03.sumChargedHadronPt              );
    hww.mus_isoR03_pf_NeutralHadronEt()                 .push_back( pfStructR03.sumNeutralHadronEt              );
    hww.mus_isoR03_pf_PhotonEt()                        .push_back( pfStructR03.sumPhotonEt                     );
    hww.mus_isoR03_pf_PUPt()                            .push_back( pfStructR03.sumPUPt                         );


    muonIndex++;

  } // end loop on muons

} 
