#include "RecoTracker/FinalTrackSelectors/src/MultiTrackSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h" 

//#include "RecoTracker/DebugTools/interface/FixTrackHitPattern.h"
#include <Math/DistFunc.h>
#include "TMath.h"
#include "TFile.h"

using reco::modules::MultiTrackSelector;

MultiTrackSelector::MultiTrackSelector()
{
  useForestFromDB_ = true;
  forest_ = 0;
  gbrVals_ = 0;
}

MultiTrackSelector::MultiTrackSelector( const edm::ParameterSet & cfg ) :
  src_( cfg.getParameter<edm::InputTag>( "src" ) ),
  beamspot_( cfg.getParameter<edm::InputTag>( "beamspot" ) ),
  useVertices_( cfg.getParameter<bool>( "useVertices" ) ),
  useVtxError_( cfg.getParameter<bool>( "useVtxError" ) ),
  vertices_( useVertices_ ? cfg.getParameter<edm::InputTag>( "vertices" ) : edm::InputTag("NONE"))
  // now get the pset for each selector
{

  useAnyMVA_ = false;
  forestLabel_ = "MVASelectorIter0";
  std::string type = "BDTG";
  useForestFromDB_ = true;
  dbFileName_ = "";

  forest_ = 0;
  gbrVals_ = 0;

  if(cfg.exists("useAnyMVA")) useAnyMVA_ = cfg.getParameter<bool>("useAnyMVA");

  if(useAnyMVA_){
    gbrVals_ = new float[11];
    if(cfg.exists("mvaType"))type = cfg.getParameter<std::string>("mvaType");
    if(cfg.exists("GBRForestLabel"))forestLabel_ = cfg.getParameter<std::string>("GBRForestLabel");
    if(cfg.exists("GBRForestFileName")){
      dbFileName_ = cfg.getParameter<std::string>("GBRForestFileName");
      useForestFromDB_ = false;
    }

     mvaType_ = type;
  }
  std::vector<edm::ParameterSet> trkSelectors( cfg.getParameter<std::vector< edm::ParameterSet> >("trackSelectors") );
  qualityToSet_.reserve(trkSelectors.size());
  vtxNumber_.reserve(trkSelectors.size());
  vertexCut_.reserve(trkSelectors.size());
  res_par_.reserve(trkSelectors.size());
  chi2n_par_.reserve(trkSelectors.size());
  chi2n_no1Dmod_par_.reserve(trkSelectors.size());
  d0_par1_.reserve(trkSelectors.size());
  dz_par1_.reserve(trkSelectors.size());
  d0_par2_.reserve(trkSelectors.size());
  dz_par2_.reserve(trkSelectors.size());
  applyAdaptedPVCuts_.reserve(trkSelectors.size());
  max_d0_.reserve(trkSelectors.size());
  max_z0_.reserve(trkSelectors.size());
  nSigmaZ_.reserve(trkSelectors.size());
  min_layers_.reserve(trkSelectors.size());
  min_3Dlayers_.reserve(trkSelectors.size());
  max_lostLayers_.reserve(trkSelectors.size());
  min_hits_bypass_.reserve(trkSelectors.size());
  applyAbsCutsIfNoPV_.reserve(trkSelectors.size());
  max_d0NoPV_.reserve(trkSelectors.size());
  max_z0NoPV_.reserve(trkSelectors.size());
  preFilter_.reserve(trkSelectors.size());
  max_relpterr_.reserve(trkSelectors.size());
  min_nhits_.reserve(trkSelectors.size());
  max_minMissHitOutOrIn_.reserve(trkSelectors.size());
  max_lostHitFraction_.reserve(trkSelectors.size());
  min_eta_.reserve(trkSelectors.size());
  max_eta_.reserve(trkSelectors.size());
  useMVA_.reserve(trkSelectors.size());
  //mvaReaders_.reserve(trkSelectors.size());
  min_MVA_.reserve(trkSelectors.size());
  //mvaType_.reserve(trkSelectors.size());

  produces<edm::ValueMap<float> >("MVAVals");

  for ( unsigned int i=0; i<trkSelectors.size(); i++) {

    qualityToSet_.push_back( TrackBase::undefQuality );
    // parameters for vertex selection
    vtxNumber_.push_back( useVertices_ ? trkSelectors[i].getParameter<int32_t>("vtxNumber") : 0 );
    vertexCut_.push_back( useVertices_ ? trkSelectors[i].getParameter<std::string>("vertexCut") : 0);
    //  parameters for adapted optimal cuts on chi2 and primary vertex compatibility
    res_par_.push_back(trkSelectors[i].getParameter< std::vector<double> >("res_par") );
    chi2n_par_.push_back( trkSelectors[i].getParameter<double>("chi2n_par") );
    chi2n_no1Dmod_par_.push_back( trkSelectors[i].getParameter<double>("chi2n_no1Dmod_par") );
    d0_par1_.push_back(trkSelectors[i].getParameter< std::vector<double> >("d0_par1"));
    dz_par1_.push_back(trkSelectors[i].getParameter< std::vector<double> >("dz_par1"));
    d0_par2_.push_back(trkSelectors[i].getParameter< std::vector<double> >("d0_par2"));
    dz_par2_.push_back(trkSelectors[i].getParameter< std::vector<double> >("dz_par2"));
    // Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
    applyAdaptedPVCuts_.push_back(trkSelectors[i].getParameter<bool>("applyAdaptedPVCuts"));
    // Impact parameter absolute cuts.
    max_d0_.push_back(trkSelectors[i].getParameter<double>("max_d0"));
    max_z0_.push_back(trkSelectors[i].getParameter<double>("max_z0"));
    nSigmaZ_.push_back(trkSelectors[i].getParameter<double>("nSigmaZ"));
    // Cuts on numbers of layers with hits/3D hits/lost hits.
    min_layers_.push_back(trkSelectors[i].getParameter<uint32_t>("minNumberLayers") );
    min_3Dlayers_.push_back(trkSelectors[i].getParameter<uint32_t>("minNumber3DLayers") );
    max_lostLayers_.push_back(trkSelectors[i].getParameter<uint32_t>("maxNumberLostLayers"));
    min_hits_bypass_.push_back(trkSelectors[i].getParameter<uint32_t>("minHitsToBypassChecks"));
    // Flag to apply absolute cuts if no PV passes the selection
    applyAbsCutsIfNoPV_.push_back(trkSelectors[i].getParameter<bool>("applyAbsCutsIfNoPV"));
    keepAllTracks_.push_back( trkSelectors[i].getParameter<bool>("keepAllTracks")); 
    max_relpterr_.push_back(trkSelectors[i].getParameter<double>("max_relpterr"));
    min_nhits_.push_back(trkSelectors[i].getParameter<uint32_t>("min_nhits"));
    max_minMissHitOutOrIn_.push_back(
	trkSelectors[i].existsAs<int32_t>("max_minMissHitOutOrIn") ? 
	trkSelectors[i].getParameter<int32_t>("max_minMissHitOutOrIn") : 99);
    max_lostHitFraction_.push_back(
	trkSelectors[i].existsAs<double>("max_lostHitFraction") ?
	trkSelectors[i].getParameter<double>("max_lostHitFraction") : 1.0);
    min_eta_.push_back(trkSelectors[i].existsAs<double>("min_eta") ?
	trkSelectors[i].getParameter<double>("min_eta"):-9999);
    max_eta_.push_back(trkSelectors[i].existsAs<double>("max_eta") ?
	trkSelectors[i].getParameter<double>("max_eta"):9999);
  
    setQualityBit_.push_back( false );
    std::string qualityStr = trkSelectors[i].getParameter<std::string>("qualityBit");
    if (qualityStr != "") {
      setQualityBit_[i] = true;
      qualityToSet_[i]  = TrackBase::qualityByName(trkSelectors[i].getParameter<std::string>("qualityBit"));
    }
  
    if (setQualityBit_[i] && (qualityToSet_[i] == TrackBase::undefQuality)) throw cms::Exception("Configuration") <<
    "You can't set the quality bit " << trkSelectors[i].getParameter<std::string>("qualityBit") << " as it is 'undefQuality' or unknown.\n";

    if (applyAbsCutsIfNoPV_[i]) {
      max_d0NoPV_.push_back(trkSelectors[i].getParameter<double>("max_d0NoPV"));
      max_z0NoPV_.push_back(trkSelectors[i].getParameter<double>("max_z0NoPV"));
    }
    else{//dummy values
      max_d0NoPV_.push_back(0.);
      max_z0NoPV_.push_back(0.);
    }
  
    name_.push_back( trkSelectors[i].getParameter<std::string>("name") );

    preFilter_[i]=trkSelectors.size(); // no prefilter

    std::string pfName=trkSelectors[i].getParameter<std::string>("preFilterName");
    if (pfName!="") {
      bool foundPF=false;
      for ( unsigned int j=0; j<i; j++) 
	if (name_[j]==pfName ) {
	  foundPF=true;
	  preFilter_[i]=j;
	}
      if ( !foundPF)
	throw cms::Exception("Configuration") << "Invalid prefilter name in MultiTrackSelector " 
					      << trkSelectors[i].getParameter<std::string>("preFilterName");
	  
    }

    //    produces<std::vector<int> >(name_[i]).setBranchAlias( name_[i] + "TrackQuals");
    produces<edm::ValueMap<int> >(name_[i]).setBranchAlias( name_[i] + "TrackQuals");
    if(useAnyMVA_){
      bool thisMVA = false;
      if(trkSelectors[i].exists("useMVA"))thisMVA = trkSelectors[i].getParameter<bool>("useMVA");
      useMVA_.push_back(thisMVA);
      if(thisMVA){
	double minVal = -1;
	if(trkSelectors[i].exists("minMVA"))minVal = trkSelectors[i].getParameter<double>("minMVA");
	min_MVA_.push_back(minVal);

      }else{
	min_MVA_.push_back(-9999.0);
      }
    }else{
      min_MVA_.push_back(-9999.0);
    }

  }
}

MultiTrackSelector::~MultiTrackSelector() {
  if(gbrVals_)delete [] gbrVals_;
  if(!useForestFromDB_ && forest_)delete forest_;
}

void MultiTrackSelector::produce( edm::Event& evt, const edm::EventSetup& es ) 
{
  using namespace std; 
  using namespace edm;
  using namespace reco;

  // Get tracks 
  Handle<TrackCollection> hSrcTrack;
  evt.getByLabel( src_, hSrcTrack );
  const TrackCollection& srcTracks(*hSrcTrack);

  // looking for the beam spot
  edm::Handle<reco::BeamSpot> hBsp;
  evt.getByLabel(beamspot_, hBsp);
  const reco::BeamSpot& vertexBeamSpot(*hBsp);

	
  // Select good primary vertices for use in subsequent track selection
  edm::Handle<reco::VertexCollection> hVtx;
  if (useVertices_) evt.getByLabel(vertices_, hVtx);

  unsigned int trkSize=srcTracks.size();
  std::vector<int> selTracksSave( qualityToSet_.size()*trkSize,0);

  processMVA(evt,es);

  for (unsigned int i=0; i<qualityToSet_.size(); i++) {  
    std::vector<int> selTracks(trkSize,0);
    auto_ptr<edm::ValueMap<int> > selTracksValueMap = auto_ptr<edm::ValueMap<int> >(new edm::ValueMap<int>);
    edm::ValueMap<int>::Filler filler(*selTracksValueMap);

    std::vector<Point> points;
    std::vector<float> vterr, vzerr;
    if (useVertices_) selectVertices(i,*hVtx, points, vterr, vzerr);

    // Loop over tracks
    size_t current = 0;
    for (TrackCollection::const_iterator it = srcTracks.begin(), ed = srcTracks.end(); it != ed; ++it, ++current) {
      const Track & trk = * it;
      // Check if this track passes cuts

      LogTrace("TrackSelection") << "ready to check track with pt="<< trk.pt() ;

      //already removed
      bool ok=true;
      if (preFilter_[i]<i && selTracksSave[preFilter_[i]*trkSize+current] < 0) {
	selTracks[current]=-1;
	ok=false;
	if ( !keepAllTracks_[i]) 
	  continue;
      }
      else {
	double mvaVal = 0;
	if(useAnyMVA_)mvaVal = mvaVals_[current];
	ok = select(i,vertexBeamSpot, trk, points, vterr, vzerr,mvaVal);
	if (!ok) { 
	  LogTrace("TrackSelection") << "track with pt="<< trk.pt() << " NOT selected";
	  if (!keepAllTracks_[i]) { 
	    selTracks[current]=-1;
	    continue;
	  }
	}
	else
	  LogTrace("TrackSelection") << "track with pt="<< trk.pt() << " selected";
      }

      if (preFilter_[i]<i ) {
	selTracks[current]=selTracksSave[preFilter_[i]*trkSize+current];
      }
      else {
	selTracks[current]=trk.qualityMask();
      }
      if ( ok && setQualityBit_[i]) {
	selTracks[current]= (selTracks[current] | (1<<qualityToSet_[i]));
	if (!points.empty()) {
	  if (qualityToSet_[i]==TrackBase::loose) {
	    selTracks[current]=(selTracks[current] | (1<<TrackBase::looseSetWithPV));
	  }
	  else if (qualityToSet_[i]==TrackBase::highPurity) {
	    selTracks[current]=(selTracks[current] | (1<<TrackBase::highPuritySetWithPV));
	  }
	}
      }
    }
    for ( unsigned int j=0; j< trkSize; j++ ) selTracksSave[j+i*trkSize]=selTracks[j];
    filler.insert(hSrcTrack, selTracks.begin(),selTracks.end());
    filler.fill();

    //    evt.put(selTracks,name_[i]);
    evt.put(selTracksValueMap,name_[i]);
  }
}


 bool MultiTrackSelector::select(unsigned int tsNum, 
				 const reco::BeamSpot &vertexBeamSpot, 
				 const reco::Track &tk, 
				 const std::vector<Point> &points,
				 std::vector<float> &vterr,
				 std::vector<float> &vzerr,
				 double mvaVal) {
  // Decide if the given track passes selection cuts.

  using namespace std; 
  
  if(tk.found()>=min_hits_bypass_[tsNum]) return true;
  if ( tk.ndof() < 1E-5 ) return false;

  // Cuts on numbers of layers with hits/3D hits/lost hits.
  uint32_t nlayers     = tk.hitPattern().trackerLayersWithMeasurement();
  uint32_t nlayers3D   = tk.hitPattern().pixelLayersWithMeasurement() +
    tk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
  uint32_t nlayersLost = tk.hitPattern().trackerLayersWithoutMeasurement();
  LogDebug("TrackSelection") << "cuts on nlayers: " << nlayers << " " << nlayers3D << " " << nlayersLost << " vs " 
			     << min_layers_[tsNum] << " " << min_3Dlayers_[tsNum] << " " << max_lostLayers_[tsNum];
  if (nlayers < min_layers_[tsNum]) return false;
  if (nlayers3D < min_3Dlayers_[tsNum]) return false;
  if (nlayersLost > max_lostLayers_[tsNum]) return false;
  LogTrace("TrackSelection") << "cuts on nlayers passed";

  float chi2n =  tk.normalizedChi2();
  float chi2n_no1Dmod = chi2n;

  int count1dhits = 0;
  for (trackingRecHit_iterator ith = tk.recHitsBegin(), edh = tk.recHitsEnd(); ith != edh; ++ith) {
    const TrackingRecHit * hit = ith->get();
    if (hit->isValid()) {
      if (typeid(*hit) == typeid(SiStripRecHit1D)) ++count1dhits;
    }
  }
  if (count1dhits > 0) {
    float chi2 = tk.chi2();
    float ndof = tk.ndof();
    chi2n = (chi2+count1dhits)/float(ndof+count1dhits);
  }
  // For each 1D rechit, the chi^2 and ndof is increased by one.  This is a way of retaining approximately
  // the same normalized chi^2 distribution as with 2D rechits.
  if (chi2n > chi2n_par_[tsNum]*nlayers) return false;

  if (chi2n_no1Dmod > chi2n_no1Dmod_par_[tsNum]*nlayers) return false;

  // Get track parameters
  float pt = std::max(float(tk.pt()),0.000001f);
  float eta = tk.eta();
  if (eta<min_eta_[tsNum] || eta>max_eta_[tsNum]) return false;

  //cuts on relative error on pt and number of valid hits
  float relpterr = float(tk.ptError())/pt;
  uint32_t nhits = tk.numberOfValidHits();
  if(relpterr > max_relpterr_[tsNum]) return false;
  if(nhits < min_nhits_[tsNum]) return false;

  int lostIn = tk.trackerExpectedHitsInner().numberOfLostTrackerHits();
  int lostOut = tk.trackerExpectedHitsOuter().numberOfLostTrackerHits();
  int minLost = std::min(lostIn,lostOut);
  if (minLost > max_minMissHitOutOrIn_[tsNum]) return false;
  float lostMidFrac = tk.numberOfLostHits() / (tk.numberOfValidHits() + tk.numberOfLostHits());
  if (lostMidFrac > max_lostHitFraction_[tsNum]) return false;



  ///////////////////////////////////////////////
  //Adding the MVA selection before vertex cuts//
  ///////////////////////////////////////////////

  if(useAnyMVA_ && useMVA_[tsNum]){
    if(mvaVal < min_MVA_[tsNum])return false;
  }

  ////////////////////////////////
  //End of MVA selection section//
  ////////////////////////////////

  //other track parameters
  float d0 = -tk.dxy(vertexBeamSpot.position()), d0E =  tk.d0Error(),
    dz = tk.dz(vertexBeamSpot.position()), dzE =  tk.dzError();

  // parametrized d0 resolution for the track pt
  float nomd0E = sqrt(res_par_[tsNum][0]*res_par_[tsNum][0]+(res_par_[tsNum][1]/pt)*(res_par_[tsNum][1]/pt));
  // parametrized z0 resolution for the track pt and eta
  float nomdzE = nomd0E*(std::cosh(eta));

  float dzCut = min( pow(dz_par1_[tsNum][0]*nlayers,dz_par1_[tsNum][1])*nomdzE, 
		      pow(dz_par2_[tsNum][0]*nlayers,dz_par2_[tsNum][1])*dzE );
  float d0Cut = min( pow(d0_par1_[tsNum][0]*nlayers,d0_par1_[tsNum][1])*nomd0E, 
		      pow(d0_par2_[tsNum][0]*nlayers,d0_par2_[tsNum][1])*d0E );


  // ---- PrimaryVertex compatibility cut
  bool primaryVertexZCompatibility(false);   
  bool primaryVertexD0Compatibility(false);   

  if (points.empty()) { //If not primaryVertices are reconstructed, check just the compatibility with the BS
    //z0 within (n sigma + dzCut) of the beam spot z, if no good vertex is found
    if ( abs(dz) < hypot(vertexBeamSpot.sigmaZ()*nSigmaZ_[tsNum],dzCut) ) primaryVertexZCompatibility = true;  
    // d0 compatibility with beam line
    if (abs(d0) < d0Cut) primaryVertexD0Compatibility = true;     
  }

  int iv=0;
  for (std::vector<Point>::const_iterator point = points.begin(), end = points.end(); point != end; ++point) {
    LogTrace("TrackSelection") << "Test track w.r.t. vertex with z position " << point->z();
    if(primaryVertexZCompatibility && primaryVertexD0Compatibility) break;
    float dzPV = tk.dz(*point); //re-evaluate the dz with respect to the vertex position
    float d0PV = tk.dxy(*point); //re-evaluate the dxy with respect to the vertex position
    if(useVtxError_){
       float dzErrPV = sqrt(dzE*dzE+vzerr[iv]*vzerr[iv]); // include vertex error in z
       float d0ErrPV = sqrt(d0E*d0E+vterr[iv]*vterr[iv]); // include vertex error in xy
       iv++;
       if (abs(dzPV) < dz_par1_[tsNum][0]*pow(nlayers,dz_par1_[tsNum][1])*nomdzE &&
	   abs(dzPV) < dz_par2_[tsNum][0]*pow(nlayers,dz_par2_[tsNum][1])*dzErrPV &&
	   abs(dzPV) < max_z0_[tsNum])  primaryVertexZCompatibility = true;
       if (abs(d0PV) < d0_par1_[tsNum][0]*pow(nlayers,d0_par1_[tsNum][1])*nomd0E &&
	   abs(d0PV) < d0_par2_[tsNum][0]*pow(nlayers,d0_par2_[tsNum][1])*d0ErrPV &&
	   abs(d0PV) < max_d0_[tsNum]) primaryVertexD0Compatibility = true; 
    }else{
       if (abs(dzPV) < dzCut)  primaryVertexZCompatibility = true;
       if (abs(d0PV) < d0Cut) primaryVertexD0Compatibility = true;     
    }
    LogTrace("TrackSelection") << "distances " << dzPV << " " << d0PV << " vs " << dzCut << " " << d0Cut;
  }

  if (points.empty() && applyAbsCutsIfNoPV_[tsNum]) {
    if ( abs(dz) > max_z0NoPV_[tsNum] || abs(d0) > max_d0NoPV_[tsNum]) return false;
  }  else {
    // Absolute cuts on all tracks impact parameters with respect to beam-spot.
    // If BS is not compatible, verify if at least the reco-vertex is compatible (useful for incorrect BS settings)
    if (abs(d0) > max_d0_[tsNum] && !primaryVertexD0Compatibility) return false;
    LogTrace("TrackSelection") << "absolute cuts on d0 passed";
    if (abs(dz) > max_z0_[tsNum] && !primaryVertexZCompatibility) return false;
    LogTrace("TrackSelection") << "absolute cuts on dz passed";
  }

  LogTrace("TrackSelection") << "cuts on PV: apply adapted PV cuts? " << applyAdaptedPVCuts_[tsNum] 
			     << " d0 compatibility? " << primaryVertexD0Compatibility  
			     << " z compatibility? " << primaryVertexZCompatibility ;

  if (applyAdaptedPVCuts_[tsNum]) {
    return (primaryVertexD0Compatibility && primaryVertexZCompatibility);
  } else {
    return true;     
  }

}

 void MultiTrackSelector::selectVertices(unsigned int tsNum, 
					 const reco::VertexCollection &vtxs, 
					 std::vector<Point> &points,
					 std::vector<float> &vterr, 
					 std::vector<float> &vzerr) {
  // Select good primary vertices
  using namespace reco;
  int32_t toTake = vtxNumber_[tsNum]; 
  for (VertexCollection::const_iterator it = vtxs.begin(), ed = vtxs.end(); it != ed; ++it) {

    LogDebug("SelectVertex") << " select vertex with z position " << it->z() << " " 
			     << it->chi2() << " " << it->ndof() << " " << TMath::Prob(it->chi2(), static_cast<int32_t>(it->ndof()));
    Vertex vtx = *it;
    bool pass = vertexCut_[tsNum]( vtx );
    if( pass ) { 
      points.push_back(it->position()); 
      vterr.push_back(sqrt(it->yError()*it->xError()));
      vzerr.push_back(it->zError());
      LogTrace("SelectVertex") << " SELECTED vertex with z position " << it->z();
      toTake--; if (toTake == 0) break;
    }
  }
}

void MultiTrackSelector::processMVA(edm::Event& evt, const edm::EventSetup& es)
{

  using namespace std; 
  using namespace edm;
  using namespace reco;

  // Get tracks 
  Handle<TrackCollection> hSrcTrack;
  evt.getByLabel( src_, hSrcTrack );
  const TrackCollection& srcTracks(*hSrcTrack);

  auto_ptr<edm::ValueMap<float> >mvaValValueMap = auto_ptr<edm::ValueMap<float> >(new edm::ValueMap<float>);
  edm::ValueMap<float>::Filler mvaFiller(*mvaValValueMap);

  mvaVals_.clear();

  if(!useAnyMVA_){
    size_t current = 0;
    for (TrackCollection::const_iterator it = srcTracks.begin(), ed = srcTracks.end(); it != ed; ++it, ++current) {
      mvaVals_.push_back(-99.0);
    }
    mvaFiller.insert(hSrcTrack,mvaVals_.begin(),mvaVals_.end());
    mvaFiller.fill();
    evt.put(mvaValValueMap,"MVAVals");
    return;
  }

  if(!forest_){
    if(useForestFromDB_){
      edm::ESHandle<GBRForest> forestHandle;
      es.get<GBRWrapperRcd>().get(forestLabel_,forestHandle);
      forest_ = (GBRForest*)forestHandle.product();
    }else{
      TFile gbrfile(dbFileName_.c_str());
      forest_ = (GBRForest*)gbrfile.Get(forestLabel_.c_str());
    }
  }
    


  size_t current = 0;
  for (TrackCollection::const_iterator it = srcTracks.begin(), ed = srcTracks.end(); it != ed; ++it, ++current) {
    const Track & trk = * it;
    tmva_ndof_ = trk.ndof();
    tmva_nlayers_ = trk.hitPattern().trackerLayersWithMeasurement();
    tmva_nlayers3D_ = trk.hitPattern().pixelLayersWithMeasurement() + trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
    tmva_nlayerslost_ = trk.hitPattern().trackerLayersWithoutMeasurement();
    float chi2n =  trk.normalizedChi2();
    float chi2n_no1Dmod = chi2n;
    
    int count1dhits = 0;
    for (trackingRecHit_iterator ith = trk.recHitsBegin(), edh = trk.recHitsEnd(); ith != edh; ++ith) {
      const TrackingRecHit * hit = ith->get();
      if (hit->isValid()) {
	if (typeid(*hit) == typeid(SiStripRecHit1D)) ++count1dhits;
      }
    }
    if (count1dhits > 0) {
      float chi2 = trk.chi2();
      float ndof = trk.ndof();
      chi2n = (chi2+count1dhits)/float(ndof+count1dhits);
    }
    tmva_chi2n_ = chi2n;
    tmva_chi2n_no1dmod_ = chi2n_no1Dmod;
    tmva_eta_ = trk.eta();
    tmva_relpterr_ = float(trk.ptError())/std::max(float(trk.pt()),0.000001f);
    tmva_nhits_ = trk.numberOfValidHits();
    int lostIn = trk.trackerExpectedHitsInner().numberOfLostTrackerHits();
    int lostOut = trk.trackerExpectedHitsOuter().numberOfLostTrackerHits();
    int minLost = std::min(lostIn,lostOut);      tmva_minlost_ = minLost;
    tmva_lostmidfrac_ = trk.numberOfLostHits() / (trk.numberOfValidHits() + trk.numberOfLostHits());

    gbrVals_[0] = tmva_lostmidfrac_;
    gbrVals_[1] = tmva_minlost_;
    gbrVals_[2] = tmva_nhits_;
    gbrVals_[3] = tmva_relpterr_;
    gbrVals_[4] = tmva_eta_;
    gbrVals_[5] = tmva_chi2n_no1dmod_;
    gbrVals_[6] = tmva_chi2n_;
    gbrVals_[7] = tmva_nlayerslost_;
    gbrVals_[8] = tmva_nlayers3D_;
    gbrVals_[9] = tmva_nlayers_;
    gbrVals_[10] = tmva_ndof_;

    double gbrVal = forest_->GetClassifier(gbrVals_);
    mvaVals_.push_back((float)gbrVal);
  }
  mvaFiller.insert(hSrcTrack,mvaVals_.begin(),mvaVals_.end());
  mvaFiller.fill();
  evt.put(mvaValValueMap,"MVAVals");

}
