#ifndef RecoAlgos_HIMultiTrackSelector_h
#define RecoAlgos_HIMultiTrackSelector_h
/** \class HIMultiTrackSelector
 *
 * selects a subset of a track collection, copying extra information on demand
 * 
 * \author David Lange
 *
 *
 *
 */

#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"

enum MVAVARIABLES {chi2perdofperlayer = 0,
		     dxyperdxyerror = 1,
		     dzperdzerror = 2,
		     relpterr = 3,
		     lostmidfrac = 4,
		     minlost = 5,
		     nhits = 6,
		     eta = 7,
		     chi2n_no1dmod = 8,
		     chi2n = 9,
		     nlayerslost = 10,
		     nlayers3d = 11,
		     nlayers = 12,
		     ndof = 13,
		     etaerror = 14 };


    class dso_hidden HIMultiTrackSelector : public edm::stream::EDProducer<> {
        private:
        public:
            /// constructor 
	    explicit HIMultiTrackSelector();
            explicit HIMultiTrackSelector( const edm::ParameterSet & cfg ) ;
            /// destructor
            virtual ~HIMultiTrackSelector() ;

        protected:
            void beginStream(edm::StreamID) override final;
 
            // void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const final {
            //  init();
            //}
            //void beginRun(edm::Run const&, edm::EventSetup const&) final { init(); }
            // void init(edm::EventSetup const& es) const;

            typedef math::XYZPoint Point;
            /// process one event
            void produce(edm::Event& evt, const edm::EventSetup& es ) override final {
               run(evt,es);
            }
            virtual void run( edm::Event& evt, const edm::EventSetup& es ) const;

            /// return class, or -1 if rejected
            bool select (unsigned tsNum,
			 const reco::BeamSpot &vertexBeamSpot,
                         const TrackingRecHitCollection & recHits,
			 const reco::Track &tk, 
			 const std::vector<Point> &points,
			 std::vector<float> &vterr,
			 std::vector<float> &vzerr,
			 double mvaVal) const;
            void selectVertices ( unsigned int tsNum,
				  const reco::VertexCollection &vtxs, 
				  std::vector<Point> &points,
				  std::vector<float> &vterr,
				  std::vector<float> &vzerr) const;

	    void processMVA(edm::Event& evt, const edm::EventSetup& es, std::vector<float> & mvaVals_,const reco::VertexCollection &hVtx) const;


	    void ParseForestVars();
            /// source collection label
            edm::EDGetTokenT<reco::TrackCollection> src_;
            edm::EDGetTokenT<TrackingRecHitCollection> hSrc_;
            edm::EDGetTokenT<reco::BeamSpot> beamspot_;
            bool          useVertices_;
            bool          useVtxError_;
	    bool          useAnyMVA_;
            edm::EDGetTokenT<reco::VertexCollection> vertices_;


            // Boolean indicating if pixel track merging related cuts are to be applied
            bool applyPixelMergingCuts_;
            
            /// do I have to set a quality bit?
	    std::vector<bool> setQualityBit_;
	    std::vector<reco::TrackBase::TrackQuality> qualityToSet_;

            /// vertex cuts
	    std::vector<int32_t> vtxNumber_;
	    //StringCutObjectSelector is not const thread safe
	    std::vector<StringCutObjectSelector<reco::Vertex> > vertexCut_;

	    //  parameters for adapted optimal cuts on chi2 and primary vertex compatibility
	    std::vector< std::vector<double> > res_par_;
	    std::vector< double > chi2n_par_;
	    std::vector< double > chi2n_no1Dmod_par_;
	    std::vector< std::vector<double> > d0_par1_;
	    std::vector< std::vector<double> > dz_par1_;
	    std::vector< std::vector<double> > d0_par2_;
	    std::vector< std::vector<double> > dz_par2_;
	    // Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
            std::vector<bool> applyAdaptedPVCuts_;
			
            /// Impact parameter absolute cuts
            std::vector<double> max_d0_;
            std::vector<double> max_z0_;
            std::vector<double> nSigmaZ_;


            // parameters for pixel track merging pT dependent chi2 cuts
            std::vector<std::vector<double> > pixel_pTMinCut_;
            std::vector<std::vector<double> > pixel_pTMaxCut_;


            /// Cuts on numbers of layers with hits/3D hits/lost hits. 
	    std::vector<uint32_t> min_layers_;
	    std::vector<uint32_t> min_3Dlayers_;
	    std::vector<uint32_t> max_lostLayers_;
	    std::vector<uint32_t> min_hits_bypass_;

	    // pterror and nvalid hits cuts
	    std::vector<double> max_relpterr_;
	    std::vector<uint32_t> min_nhits_;

	    std::vector<int32_t> max_minMissHitOutOrIn_;
	    std::vector<int32_t> max_lostHitFraction_;

	    std::vector<double> min_eta_;
	    std::vector<double> max_eta_;

	    // Flag and absolute cuts if no PV passes the selection
	    std::vector<double> max_d0NoPV_;
	    std::vector<double> max_z0NoPV_;
	    std::vector<bool> applyAbsCutsIfNoPV_;
	    //if true, selector flags but does not select 
	    std::vector<bool> keepAllTracks_;

	    // allow one of the previous psets to be used as a prefilter
	    std::vector<unsigned int> preFilter_;
	    std::vector<std::string> name_;

	    //setup mva selector
	    std::vector<bool> useMVA_;
	    //std::vector<TMVA::Reader*> mvaReaders_;

	    std::vector<int>  mvavars_indices;

	    std::vector<double> min_MVA_;

	    //std::vector<std::string> mvaType_;
	    std::string mvaType_;
	    std::string forestLabel_;
	    std::vector<std::string> forestVars_;
	    GBRForest * forest_;
	    bool useForestFromDB_;
	    std::string dbFileName_;


    };

#endif
