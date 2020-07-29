#include "DQM/TrackingMonitorSource/interface/TrackToTrackValidator.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/Math/interface/deltaR.h"

//
// constructors and destructor
//
TrackToTrackValidator::TrackToTrackValidator(const edm::ParameterSet& iConfig)
  : monitoredTrackInputTag_ ( iConfig.getParameter<edm::InputTag>("monitoredTrack")       )
  , referenceTrackInputTag_ ( iConfig.getParameter<edm::InputTag>("referenceTrack")       )
  , topDirName_       ( iConfig.getParameter<std::string>  ("topDirName")     )
  , dRmin_            ( iConfig.getParameter<double>("dRmin")                 )
{
  initialize_parameter(iConfig);

  //now do what ever initialization is needed
  monitoredTrackToken_      = consumes<reco::TrackCollection>(monitoredTrackInputTag_);
  referenceTrackToken_      = consumes<reco::TrackCollection>(referenceTrackInputTag_);
  monitoredBSToken_         = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("monitoredBeamSpot"));
  referenceBSToken_         = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("referenceBeamSpot"));
  monitoredPVToken_         = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("monitoredPrimaryVertices"));
  referencePVToken_         = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("referencePrimaryVertices"));

  referenceTracksMEs_.label        = referenceTrackInputTag_.label();  
  matchedReferenceTracksMEs_.label = referenceTrackInputTag_.label()+"_matched";  

  monitoredTracksMEs_.label = monitoredTrackInputTag_.label();  
  unMatchedMonitoredTracksMEs_.label = monitoredTrackInputTag_.label()+"_unMatched";  

  matchTracksMEs_.label = "matches";
}


TrackToTrackValidator::~TrackToTrackValidator()
{
}

void
TrackToTrackValidator::beginJob(const edm::EventSetup& iSetup) {
}

void
TrackToTrackValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //
  //  Get Reference Track Info
  //
  edm::Handle<reco::TrackCollection> referenceTracksHandle;
  iEvent.getByToken(referenceTrackToken_,referenceTracksHandle);
  reco::TrackCollection referenceTracks = *referenceTracksHandle;

  edm::Handle<reco::BeamSpot> referenceBSHandle;
  iEvent.getByToken(referenceBSToken_,referenceBSHandle);
  reco::BeamSpot referenceBS = *referenceBSHandle;
  
  edm::Handle<reco::VertexCollection> referencePVHandle;
  iEvent.getByToken(referencePVToken_,referencePVHandle);
  reco::Vertex referencePV = referencePVHandle->at(0);

  //
  //  Get Monitored Track Info
  //
  edm::Handle<reco::TrackCollection> monitoredTracksHandle;
  iEvent.getByToken(monitoredTrackToken_,monitoredTracksHandle);
  reco::TrackCollection monitoredTracks = *monitoredTracksHandle;

  edm::Handle<reco::BeamSpot> monitoredBSHandle;
  iEvent.getByToken(monitoredBSToken_,monitoredBSHandle);
  reco::BeamSpot monitoredBS = *monitoredBSHandle;

  edm::Handle<reco::VertexCollection> monitoredPVHandle;
  iEvent.getByToken(monitoredPVToken_,monitoredPVHandle);
  reco::Vertex monitoredPV = monitoredPVHandle->at(0);


  edm::LogVerbatim("TrackToTrackValidator") << "analyzing "
					   << monitoredTrackInputTag_.process()  << ":"
					   << monitoredTrackInputTag_.label()    << ":"
					   << monitoredTrackInputTag_.instance() << " w.r.t. "
					   << referenceTrackInputTag_.process()  << ":"
					   << referenceTrackInputTag_.label()    << ":"
					   << referenceTrackInputTag_.instance() << " \n";
  
  //
  // Build the dR maps
  //
  idx2idxByDoubleColl monitored2referenceColl;
  fillMap(monitoredTracks,referenceTracks,monitored2referenceColl, dRmin_ );

  idx2idxByDoubleColl reference2monitoredColl;
  fillMap(referenceTracks,monitoredTracks,reference2monitoredColl, dRmin_);


  unsigned int nReferenceTracks(0);       // Counts the number of refernce tracks
  unsigned int nMatchedReferenceTracks(0);// Counts the number of matched refernce tracks
  unsigned int nMonitoredTracks(0);       // Counts the number of monitored tracks
  unsigned int nUnmatchedMonitoredTracks(0);// Counts the number of unmatched monitored tracks


  //
  // loop over reference tracks
  //
  edm::LogVerbatim("TrackToTrackValidator") << "\n# of tracks (reference): " << referenceTracks.size() << "\n";
  for (idx2idxByDoubleColl::const_iterator pItr = reference2monitoredColl.begin(), eItr = reference2monitoredColl.end(); pItr != eItr; ++pItr) {
    
    nReferenceTracks++;       
    int trackIdx = pItr->first;
    reco::Track track = referenceTracks.at(trackIdx);

    fill_generic_tracks_histos(*&referenceTracksMEs_,&track,&referenceBS,&referencePV);
    
    std::map<double,int> trackDRmap = pItr->second;
    if (trackDRmap.size() == 0) {
      (matchedReferenceTracksMEs_.h_dRmin)->Fill(-1.);
      (matchedReferenceTracksMEs_.h_dRmin_l)->Fill(-1.);
      continue;
    }
    
    double dRmin = trackDRmap.begin()->first;
    (referenceTracksMEs_.h_dRmin)->Fill(dRmin);
    (referenceTracksMEs_.h_dRmin_l)->Fill(dRmin);

    bool matched = false;
    if ( dRmin < dRmin_ ) matched = true;
    
    if ( matched ) {
      nMatchedReferenceTracks++;
      fill_generic_tracks_histos(*&matchedReferenceTracksMEs_,&track,&referenceBS,&referencePV);
      (matchedReferenceTracksMEs_.h_dRmin)->Fill(dRmin);
      (matchedReferenceTracksMEs_.h_dRmin_l)->Fill(dRmin);

      int matchedTrackIndex = trackDRmap[dRmin];
      reco::Track matchedTrack = monitoredTracks.at(matchedTrackIndex);
      fill_matching_tracks_histos(*&matchTracksMEs_,&track,&matchedTrack,&referenceBS,&referencePV);
    }
    
  }// Over reference tracks

  //
  // loop over monitoed tracks
  //
  edm::LogVerbatim("TrackToTrackValidator") << "\n# of tracks (monitored): " << monitoredTracks.size() << "\n";
  for (idx2idxByDoubleColl::const_iterator pItr = monitored2referenceColl.begin(), eItr = monitored2referenceColl.end(); pItr != eItr; ++pItr) {
    
    nMonitoredTracks++;   
    int trackIdx = pItr->first;
    reco::Track track = monitoredTracks.at(trackIdx);

    fill_generic_tracks_histos(*&monitoredTracksMEs_,&track,&monitoredBS,&monitoredPV);

    std::map<double,int> trackDRmap = pItr->second;
    if (trackDRmap.size() == 0) {
      (unMatchedMonitoredTracksMEs_.h_dRmin)->Fill(-1.);
      (unMatchedMonitoredTracksMEs_.h_dRmin_l)->Fill(-1.);
      continue;
    }

    double dRmin = trackDRmap.begin()->first;
    (monitoredTracksMEs_.h_dRmin)->Fill(dRmin);
    (monitoredTracksMEs_.h_dRmin_l)->Fill(dRmin);

    bool matched = false;
    if ( dRmin < dRmin_ ) matched = true;

    if ( !matched ) {
      nUnmatchedMonitoredTracks++;
      fill_generic_tracks_histos(*&unMatchedMonitoredTracksMEs_,&track,&monitoredBS,&monitoredPV);
      (unMatchedMonitoredTracksMEs_.h_dRmin)->Fill(dRmin);
      (unMatchedMonitoredTracksMEs_.h_dRmin_l)->Fill(dRmin);
    }

  } // over monitoed tracks

  
  edm::LogVerbatim("TrackToTrackValidator") << "Total reference tracks: "          << nReferenceTracks << "\n"
					    << "Total matched reference tracks: "  << nMatchedReferenceTracks << "\n"
					    << "Total monitored tracks: "          << nMonitoredTracks << "\n"
					    << "Total unMatched monitored tracks: "  << nUnmatchedMonitoredTracks << "\n";
    
}

void 
TrackToTrackValidator::bookHistograms(DQMStore::IBooker & ibooker,
				     edm::Run const & iRun,
				     edm::EventSetup const & iSetup)
{
  
  std::string dir = topDirName_;

  bookHistos(ibooker,referenceTracksMEs_,          "ref",            dir);
  bookHistos(ibooker,matchedReferenceTracksMEs_,   "ref_matched",    dir);

  bookHistos(ibooker,monitoredTracksMEs_,           "mon",    dir);
  bookHistos(ibooker,unMatchedMonitoredTracksMEs_, "mon_unMatched",dir);

  book_matching_tracks_histos(ibooker,matchTracksMEs_,"matches",dir);

}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TrackToTrackValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void
TrackToTrackValidator::fillMap(reco::TrackCollection tracks1, reco::TrackCollection tracks2, idx2idxByDoubleColl& map, float dRMin)
{
  //
  // loop on tracks1
  //
  int i = 0;
  for ( auto track1 : tracks1 ) {
    std::map<double,int> tmp;
    int j = 0;
    float smallest_dR = 1e9;
    int smallest_dR_j = -1;

    //
    // loop on tracks2
    //
    for ( auto track2 : tracks2 ) {
      double dR = reco::deltaR(track1.eta(),track1.phi(),track2.eta(),track2.phi());

      if(dR < smallest_dR){
	smallest_dR = dR;
	smallest_dR_j = j;
      }

      if(dR < dRMin){
	tmp[dR] = j;
      }

      j++;
    }
    
    //
    // If there are no tracks that pass the dR store the smallest (for debugging/validating matching)
    //
    if(!tmp.size())
      tmp[smallest_dR] = smallest_dR_j;
    
    map.push_back(std::make_pair(i,tmp));
    i++;
  }
}

void 
TrackToTrackValidator::bookHistos(DQMStore::IBooker & ibooker, generalME& mes, TString label, std::string & dir){

  book_generic_tracks_histos(ibooker,mes,label,dir);

}

void 
TrackToTrackValidator::book_generic_tracks_histos(DQMStore::IBooker & ibooker, generalME& mes, TString label, std::string & dir){

  ibooker.cd();
  ibooker.setCurrentFolder(dir);

  (mes.h_pt )      = ibooker.book1D(label+"_pt",       "track p_{T}",                               nintPt,  minPt,  maxPt   );
  (mes.h_eta)      = ibooker.book1D(label+"_eta",      "track pseudorapidity",                     nintEta, minEta, maxEta   );
  (mes.h_phi)      = ibooker.book1D(label+"_phi",      "track #phi",                               nintPhi, minPhi, maxPhi   );
  (mes.h_dxy)      = ibooker.book1D(label+"_dxy",      "track transverse dca to beam spot",        nintDxy, minDxy, maxDxy   );
  (mes.h_dz )      = ibooker.book1D(label+"_dz",       "track longitudinal dca to beam spot",       nintDz,  minDz,  maxDz   );
  (mes.h_dxyWRTpv) = ibooker.book1D(label+"_dxyWRTpv", "track transverse dca to primary vertex",   nintDxy, minDxy, maxDxy   );
  (mes.h_dzWRTpv)  = ibooker.book1D(label+"_dzWRTpv",  "track longitudinal dca to primary vertex",  nintDz,  0.1*minDz,  0.1*maxDz   );
  (mes.h_charge)   = ibooker.book1D(label+"_charge",   "track charge",                                   5,   -2,        2   );
  (mes.h_hits  )   = ibooker.book1D(label+"_hits",     "track number of hits",                          35,   -0.5,     34.5 );
  (mes.h_dRmin)    = ibooker.book1D(label+"_dRmin",    "track min dR",                                 100,    0.,       0.01); 
  (mes.h_dRmin_l)  = ibooker.book1D(label+"_dRmin_l",  "track min dR",                                 100,    0.,       0.4); 

  (mes.h_pt_vs_eta)  = ibooker.book2D(label+"_ptVSeta","track p_{T} vs #eta", nintEta, minEta, maxEta, nintPt, minPt, maxPt);

}

void 
TrackToTrackValidator::book_matching_tracks_histos(DQMStore::IBooker & ibooker, matchingME& mes, TString label, std::string & dir){

  ibooker.cd();
  ibooker.setCurrentFolder(dir);

  (mes.h_hits_vs_hits)  = ibooker.book2D(label+"_hits_vs_hits","monitored track # hits vs reference track # hits", 35, -0.5, 34.5, 35,-0.5, 34.5);
  (mes.h_pt_vs_pt)      = ibooker.book2D(label+"_pt_vs_pt",    "monitored track p_{T} vs reference track p_{T}",    nintPt,  minPt,  maxPt,  nintPt,  minPt,  maxPt);
  (mes.h_eta_vs_eta)    = ibooker.book2D(label+"_eta_vs_eta",  "monitored track #eta vs reference track #eta",     nintEta, minEta, maxEta, nintEta, minEta, maxEta);
  (mes.h_phi_vs_phi)    = ibooker.book2D(label+"_phi_vs_phi",  "monitored track #phi vs reference track #phi",     nintPhi, minPhi, maxPhi, nintPhi, minPhi, maxPhi);

  (mes.h_dPt)       = ibooker.book1D(label+"_dPt",       "#Delta track #P_T",                                         ptRes_nbin,    ptRes_rangeMin,         ptRes_rangeMax   );
  (mes.h_dEta)      = ibooker.book1D(label+"_dEta",      "#Delta track #eta",                                         etaRes_nbin,   etaRes_rangeMin,        etaRes_rangeMax   );
  (mes.h_dPhi)      = ibooker.book1D(label+"_dPhi",      "#Delta track #phi",                                         phiRes_nbin,   phiRes_rangeMin,        phiRes_rangeMax   );
  (mes.h_dDxy)      = ibooker.book1D(label+"_dDxy",      "#Delta track transverse dca to beam spot",                  dxyRes_nbin,   dxyRes_rangeMin,        dxyRes_rangeMax   );
  (mes.h_dDz)       = ibooker.book1D(label+"_dDz",       "#Delta track longitudinal dca to beam spot",                dzRes_nbin,    dzRes_rangeMin,         dzRes_rangeMax   );
  (mes.h_dDxyWRTpv) = ibooker.book1D(label+"_dDxyWRTpv", "#Delta track transverse dca to primary vertex ",            dxyRes_nbin,   dxyRes_rangeMin,        dxyRes_rangeMax   );
  (mes.h_dDzWRTpv)  = ibooker.book1D(label+"_dDzWRTpv",  "#Delta track longitudinal dca to primary vertex",           dzRes_nbin,    dzRes_rangeMin,         dzRes_rangeMax   );
  (mes.h_dCharge)   = ibooker.book1D(label+"_dCharge",   "#Delta track charge",                                       5,                       -2.5,                    2.5   );
  (mes.h_dHits)     = ibooker.book1D(label+"_dHits",     "#Delta track number of hits",                               39,                      -19.5,                  19.5   );
}


void
TrackToTrackValidator::fill_generic_tracks_histos(generalME& mes, reco::Track* trk, reco::BeamSpot* bs, reco::Vertex* pv) {

  float pt       = trk->pt();
  float eta      = trk->eta();
  float phi      = trk->phi();
  float dxy      = trk->dxy(bs->position());
  float dz       = trk->dz(bs->position());
  float dxyWRTpv = trk->dxy(pv->position());
  float dzWRTpv  = trk->dz(pv->position());
  float charge   = trk->charge();
  float nhits    = trk->hitPattern().numberOfValidHits();

  (mes.h_pt      ) -> Fill(pt);
  (mes.h_eta     ) -> Fill(eta);
  (mes.h_phi     ) -> Fill(phi);
  (mes.h_dxy     ) -> Fill(dxy);
  (mes.h_dz      ) -> Fill(dz);
  (mes.h_dxyWRTpv) -> Fill(dxyWRTpv);
  (mes.h_dzWRTpv ) -> Fill(dzWRTpv);
  (mes.h_charge  ) -> Fill(charge);
  (mes.h_hits    ) -> Fill(nhits);

  (mes.h_pt_vs_eta) -> Fill(eta,pt);
  
}

void
TrackToTrackValidator::fill_matching_tracks_histos(matchingME& mes, reco::Track* mon, reco::Track* ref, reco::BeamSpot* bs, reco::Vertex* pv) {

  float mon_pt       = mon->pt();
  float mon_eta      = mon->eta();
  float mon_phi      = mon->phi();
  float mon_dxy      = mon->dxy(bs->position());
  float mon_dz       = mon->dz(bs->position());
  float mon_dxyWRTpv = mon->dxy(pv->position());
  float mon_dzWRTpv  = mon->dz(pv->position());
  float mon_charge   = mon->charge();
  float mon_nhits    = mon->hitPattern().numberOfValidHits();

  float ref_pt       = ref->pt();
  float ref_eta      = ref->eta();
  float ref_phi      = ref->phi();
  float ref_dxy      = ref->dxy(bs->position());
  float ref_dz       = ref->dz(bs->position());
  float ref_dxyWRTpv = ref->dxy(pv->position());
  float ref_dzWRTpv  = ref->dz(pv->position());
  float ref_charge   = ref->charge();
  float ref_nhits    = ref->hitPattern().numberOfValidHits();

  (mes.h_hits_vs_hits) -> Fill(ref_nhits,mon_nhits);
  (mes.h_pt_vs_pt    ) -> Fill(ref_pt,   mon_pt);
  (mes.h_eta_vs_eta  ) -> Fill(ref_eta,  mon_eta);
  (mes.h_phi_vs_phi  ) -> Fill(ref_phi,  mon_phi);


  (mes.h_dPt)       -> Fill(ref_pt        - mon_pt);
  (mes.h_dEta)      -> Fill(ref_eta       - mon_eta);
  (mes.h_dPhi)      -> Fill(ref_phi       - mon_phi);
  (mes.h_dDxy)      -> Fill(ref_dxy       - mon_dxy);
  (mes.h_dDz)       -> Fill(ref_dz        - mon_dz);
  (mes.h_dDxyWRTpv) -> Fill(ref_dxyWRTpv  - mon_dxyWRTpv);
  (mes.h_dDzWRTpv)  -> Fill(ref_dzWRTpv   - mon_dzWRTpv);
  (mes.h_dCharge)   -> Fill(ref_charge    - mon_charge);
  (mes.h_dHits)     -> Fill(ref_nhits     - mon_nhits);
  
}

void
TrackToTrackValidator::initialize_parameter(const edm::ParameterSet& iConfig)
{

  const edm::ParameterSet& pset = iConfig.getParameter<edm::ParameterSet>("histoPSet");

  //parameters for _vs_eta plots
  minEta     = pset.getParameter<double>("minEta");
  maxEta     = pset.getParameter<double>("maxEta");
  nintEta    = pset.getParameter<int>("nintEta");
  useFabsEta = pset.getParameter<bool>("useFabsEta");

  //parameters for _vs_pt plots
  minPt    = pset.getParameter<double>("minPt");
  maxPt    = pset.getParameter<double>("maxPt");
  nintPt   = pset.getParameter<int>("nintPt");

  //parameters for _vs_phi plots
  minPhi  = pset.getParameter<double>("minPhi");
  maxPhi  = pset.getParameter<double>("maxPhi");
  nintPhi = pset.getParameter<int>("nintPhi");

  //parameters for _vs_Dxy plots
  minDxy  = pset.getParameter<double>("minDxy");
  maxDxy  = pset.getParameter<double>("maxDxy");
  nintDxy = pset.getParameter<int>("nintDxy");

  //parameters for _vs_Dz plots
  minDz  = pset.getParameter<double>("minDz");
  maxDz  = pset.getParameter<double>("maxDz");
  nintDz = pset.getParameter<int>("nintDz");

  //parameters for resolution plots
  ptRes_rangeMin = pset.getParameter<double>("ptRes_rangeMin");
  ptRes_rangeMax = pset.getParameter<double>("ptRes_rangeMax");
  ptRes_nbin = pset.getParameter<int>("ptRes_nbin");

  phiRes_rangeMin = pset.getParameter<double>("phiRes_rangeMin");
  phiRes_rangeMax = pset.getParameter<double>("phiRes_rangeMax");
  phiRes_nbin     = pset.getParameter<int>("phiRes_nbin");
  
  etaRes_rangeMin = pset.getParameter<double>("etaRes_rangeMin");
  etaRes_rangeMax = pset.getParameter<double>("etaRes_rangeMax");
  etaRes_nbin     = pset.getParameter<int>("etaRes_nbin");
  
  dxyRes_rangeMin = pset.getParameter<double>("dxyRes_rangeMin");
  dxyRes_rangeMax = pset.getParameter<double>("dxyRes_rangeMax");
  dxyRes_nbin = pset.getParameter<int>("dxyRes_nbin");
  
  dzRes_rangeMin = pset.getParameter<double>("dzRes_rangeMin");
  dzRes_rangeMax = pset.getParameter<double>("dzRes_rangeMax");
  dzRes_nbin = pset.getParameter<int>("dzRes_nbin");


}
