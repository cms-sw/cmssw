#include <functional>
#include <algorithm>
#include <iterator>
#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <boost/iterator/transform_iterator.hpp>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackVertexFinder.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrack.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/CandidatePtrTransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "RecoBTag/SecondaryVertex/interface/VertexFilter.h"
#include "RecoBTag/SecondaryVertex/interface/VertexSorting.h"

#include "DataFormats/GeometryVector/interface/VectorUtil.h"

using namespace reco;

namespace {
	template<typename T>
	struct RefToBaseLess : public std::binary_function<edm::RefToBase<T>,
							   edm::RefToBase<T>,
							   bool> {
		inline bool operator()(const edm::RefToBase<T> &r1,
				       const edm::RefToBase<T> &r2) const
		{
			return r1.id() < r2.id() ||
			       (r1.id() == r2.id() && r1.key() < r2.key());
		}
	};
}

GlobalVector flightDirection(const reco::Vertex & pv, const reco::Vertex & sv) {
return  GlobalVector(sv.x() - pv.x(), sv.y() - pv.y(),sv.z() - pv.z());
}
GlobalVector flightDirection(const reco::Vertex & pv, const reco::VertexCompositePtrCandidate & sv) {
return  GlobalVector(sv.vertex().x() - pv.x(), sv.vertex().y() - pv.y(),sv.vertex().z() - pv.z());
}
const math::XYZPoint & position(const reco::Vertex & sv)
{return sv.position();}
const math::XYZPoint & position(const reco::VertexCompositePtrCandidate & sv)
{return sv.vertex();}


template <class IPTI,class VTX>
class TemplatedSecondaryVertexProducer : public edm::stream::EDProducer<> {
    public:
	explicit TemplatedSecondaryVertexProducer(const edm::ParameterSet &params);
	~TemplatedSecondaryVertexProducer();
	typedef std::vector<TemplatedSecondaryVertexTagInfo<IPTI,VTX> > Product;
	typedef TemplatedSecondaryVertex<VTX> SecondaryVertex;
	typedef typename IPTI::input_container input_container;
	typedef typename IPTI::input_container::value_type input_item;
	typedef typename std::vector<reco::btag::IndexedTrackData> TrackDataVector;
	virtual void produce(edm::Event &event, const edm::EventSetup &es) override;

    private:
	enum ConstraintType {
		CONSTRAINT_NONE	= 0,
		CONSTRAINT_BEAMSPOT,
		CONSTRAINT_PV_BEAMSPOT_SIZE,
		CONSTRAINT_PV_BS_Z_ERRORS_SCALED,
		CONSTRAINT_PV_ERROR_SCALED,
		CONSTRAINT_PV_PRIMARIES_IN_FIT
	};
	static ConstraintType getConstraintType(const std::string &name);

        edm::EDGetTokenT<reco::BeamSpot> token_BeamSpot;
        edm::EDGetTokenT<std::vector<IPTI> > token_trackIPTagInfo;
	reco::btag::SortCriteria	sortCriterium;
	TrackSelector			trackSelector;
	ConstraintType			constraint;
	double				constraintScaling;
	edm::ParameterSet		vtxRecoPSet;
	bool				useGhostTrack;
	bool				withPVError;
	double				minTrackWeight;
	VertexFilter			vertexFilter;
	VertexSorting<SecondaryVertex>	vertexSorting;
        bool                            useExternalSV;  
        double                          extSVDeltaRToJet;      
        edm::EDGetTokenT<edm::View<VTX> > token_extSVCollection;

	void markUsedTracks(TrackDataVector & trackData, const input_container & trackRefs, const SecondaryVertex & sv,size_t idx);

	struct SVBuilder :
		public std::unary_function<const VTX&, SecondaryVertex> {

		SVBuilder(const reco::Vertex &pv,
		          const GlobalVector &direction,
		          bool withPVError,
			  double minTrackWeight) :
			pv(pv), direction(direction),
			withPVError(withPVError),
			minTrackWeight(minTrackWeight) {}
		SecondaryVertex operator () (const TransientVertex &sv) const;

		SecondaryVertex operator () (const VTX &sv) const
		{ return SecondaryVertex(pv, sv, direction, withPVError); }


		const Vertex		&pv;
		const GlobalVector	&direction;
		bool			withPVError;
		double 			minTrackWeight;
	};

	struct SVFilter :
		public std::unary_function<const SecondaryVertex&, bool> {

		SVFilter(const VertexFilter &filter, const Vertex &pv,
		         const GlobalVector &direction) :
			filter(filter), pv(pv), direction(direction) {}

		inline bool operator () (const SecondaryVertex &sv) const
		{ return !filter(pv, sv, direction); }

		const VertexFilter	&filter;
		const Vertex		&pv;
		const GlobalVector	&direction;
	};
};
template <class IPTI,class VTX>
typename TemplatedSecondaryVertexProducer<IPTI,VTX>::ConstraintType
TemplatedSecondaryVertexProducer<IPTI,VTX>::getConstraintType(const std::string &name)
{
	if (name == "None")
		return CONSTRAINT_NONE;
	else if (name == "BeamSpot")
		return CONSTRAINT_BEAMSPOT;
	else if (name == "BeamSpot+PVPosition")
		return CONSTRAINT_PV_BEAMSPOT_SIZE;
	else if (name == "BeamSpotZ+PVErrorScaledXY")
		return CONSTRAINT_PV_BS_Z_ERRORS_SCALED;
	else if (name == "PVErrorScaled")
		return CONSTRAINT_PV_ERROR_SCALED;
	else if (name == "BeamSpot+PVTracksInFit")
		return CONSTRAINT_PV_PRIMARIES_IN_FIT;
	else
		throw cms::Exception("InvalidArgument")
			<< "TemplatedSecondaryVertexProducer: ``constraint'' parameter "
			   "value \"" << name << "\" not understood."
			<< std::endl;
}

static GhostTrackVertexFinder::FitType
getGhostTrackFitType(const std::string &name)
{
	if (name == "AlwaysWithGhostTrack")
		return GhostTrackVertexFinder::kAlwaysWithGhostTrack;
	else if (name == "SingleTracksWithGhostTrack")
		return GhostTrackVertexFinder::kSingleTracksWithGhostTrack;
	else if (name == "RefitGhostTrackWithVertices")
		return GhostTrackVertexFinder::kRefitGhostTrackWithVertices;
	else
		throw cms::Exception("InvalidArgument")
			<< "TemplatedSecondaryVertexProducer: ``fitType'' "
			   "parameter value \"" << name << "\" for "
			   "GhostTrackVertexFinder settings not "
			   "understood." << std::endl;
}

template <class IPTI,class VTX>
TemplatedSecondaryVertexProducer<IPTI,VTX>::TemplatedSecondaryVertexProducer(
					const edm::ParameterSet &params) :
	sortCriterium(TrackSorting::getCriterium(params.getParameter<std::string>("trackSort"))),
	trackSelector(params.getParameter<edm::ParameterSet>("trackSelection")),
	constraint(getConstraintType(params.getParameter<std::string>("constraint"))),
	constraintScaling(1.0),
	vtxRecoPSet(params.getParameter<edm::ParameterSet>("vertexReco")),
	useGhostTrack(vtxRecoPSet.getParameter<std::string>("finder") == "gtvr"),
	withPVError(params.getParameter<bool>("usePVError")),
	minTrackWeight(params.getParameter<double>("minimumTrackWeight")),
	vertexFilter(params.getParameter<edm::ParameterSet>("vertexCuts")),
	vertexSorting(params.getParameter<edm::ParameterSet>("vertexSelection"))
{
	token_trackIPTagInfo =  consumes<std::vector<IPTI> >(params.getParameter<edm::InputTag>("trackIPTagInfos"));
	if (constraint == CONSTRAINT_PV_ERROR_SCALED ||
	    constraint == CONSTRAINT_PV_BS_Z_ERRORS_SCALED)
		constraintScaling = params.getParameter<double>("pvErrorScaling");

	if (constraint == CONSTRAINT_PV_BEAMSPOT_SIZE ||
	    constraint == CONSTRAINT_PV_BS_Z_ERRORS_SCALED ||
	    constraint == CONSTRAINT_BEAMSPOT ||
	    constraint == CONSTRAINT_PV_PRIMARIES_IN_FIT )
	    token_BeamSpot = consumes<reco::BeamSpot>(params.getParameter<edm::InputTag>("beamSpotTag"));
        useExternalSV = false;
        if(params.existsAs<bool>("useExternalSV")) useExternalSV = params.getParameter<bool> ("useExternalSV");
        if(useExternalSV) {
	   token_extSVCollection =  consumes<edm::View<VTX> >(params.getParameter<edm::InputTag>("extSVCollection"));
       	   extSVDeltaRToJet = params.getParameter<double>("extSVDeltaRToJet");
        }
	produces<Product>();
}
template <class IPTI,class VTX>
TemplatedSecondaryVertexProducer<IPTI,VTX>::~TemplatedSecondaryVertexProducer()
{
}

template <class IPTI,class VTX>
void TemplatedSecondaryVertexProducer<IPTI,VTX>::produce(edm::Event &event,
                                      const edm::EventSetup &es)
{
//	typedef std::map<TrackBaseRef, TransientTrack,
//	                 RefToBaseLess<Track> > TransientTrackMap;
	//How about good old pointers?
	typedef std::map<const Track *, TransientTrack> TransientTrackMap;
	

	edm::ESHandle<TransientTrackBuilder> trackBuilder;
	es.get<TransientTrackRecord>().get("TransientTrackBuilder",
	                                   trackBuilder);

	edm::Handle<std::vector<IPTI> > trackIPTagInfos;
	event.getByToken(token_trackIPTagInfo, trackIPTagInfos);

        // External Sec Vertex collection (e.g. for IVF usage)
        edm::Handle<edm::View<VTX> > extSecVertex;          
        if(useExternalSV) event.getByToken(token_extSVCollection,extSecVertex);
                                                             

	edm::Handle<BeamSpot> beamSpot;
	unsigned int bsCovSrc[7] = { 0, };
	double sigmaZ = 0.0, beamWidth = 0.0;
	switch(constraint) {
	    case CONSTRAINT_PV_BEAMSPOT_SIZE:
		event.getByToken(token_BeamSpot,beamSpot);
		bsCovSrc[3] = bsCovSrc[4] = bsCovSrc[5] = bsCovSrc[6] = 1;
		sigmaZ = beamSpot->sigmaZ();
		beamWidth = beamSpot->BeamWidthX();
		break;

	    case CONSTRAINT_PV_BS_Z_ERRORS_SCALED:
		event.getByToken(token_BeamSpot,beamSpot);
		bsCovSrc[0] = bsCovSrc[1] = 2;
		bsCovSrc[3] = bsCovSrc[4] = bsCovSrc[5] = 1;
		sigmaZ = beamSpot->sigmaZ();
		break;

	    case CONSTRAINT_PV_ERROR_SCALED:
		bsCovSrc[0] = bsCovSrc[1] = bsCovSrc[2] = 2;
		break;

	    case CONSTRAINT_BEAMSPOT:
	    case CONSTRAINT_PV_PRIMARIES_IN_FIT:
		event.getByToken(token_BeamSpot,beamSpot);
		break;

	    default:
		/* nothing */;
	}

	std::auto_ptr<ConfigurableVertexReconstructor> vertexReco;
	std::auto_ptr<GhostTrackVertexFinder> vertexRecoGT;
	if (useGhostTrack)
		vertexRecoGT.reset(new GhostTrackVertexFinder(
			vtxRecoPSet.getParameter<double>("maxFitChi2"),
			vtxRecoPSet.getParameter<double>("mergeThreshold"),
			vtxRecoPSet.getParameter<double>("primcut"),
			vtxRecoPSet.getParameter<double>("seccut"),
			getGhostTrackFitType(vtxRecoPSet.getParameter<std::string>("fitType"))));
	else
		vertexReco.reset(
			new ConfigurableVertexReconstructor(vtxRecoPSet));

	TransientTrackMap primariesMap;

	// result secondary vertices

	std::auto_ptr<Product> tagInfos(new Product);

	for(typename std::vector<IPTI>::const_iterator iterJets =
		trackIPTagInfos->begin(); iterJets != trackIPTagInfos->end();
		++iterJets) {
		TrackDataVector trackData;
//		      std::cout << "Jet " << iterJets-trackIPTagInfos->begin() << std::endl; 

		const Vertex &pv = *iterJets->primaryVertex();

		std::set<TransientTrack> primaries;
		if (constraint == CONSTRAINT_PV_PRIMARIES_IN_FIT) {
			for(Vertex::trackRef_iterator iter = pv.tracks_begin();
			    iter != pv.tracks_end(); ++iter) {
				TransientTrackMap::iterator pos =
					primariesMap.lower_bound(iter->get());

				if (pos != primariesMap.end() &&
				    pos->first == iter->get())
					primaries.insert(pos->second);
				else {
					TransientTrack track =
						trackBuilder->build(
							iter->castTo<TrackRef>());
					primariesMap.insert(pos,
						std::make_pair(iter->get(), track));
					primaries.insert(track);
				}
			}
		}

		edm::RefToBase<Jet> jetRef = iterJets->jet();

		GlobalVector jetDir(jetRef->momentum().x(),
		                    jetRef->momentum().y(),
		                    jetRef->momentum().z());

		std::vector<std::size_t> indices =
				iterJets->sortedIndexes(sortCriterium);

		input_container trackRefs = iterJets->sortedTracks(indices);

		const std::vector<reco::btag::TrackIPData> &ipData =
					iterJets->impactParameterData();

		// build transient tracks used for vertex reconstruction

		std::vector<TransientTrack> fitTracks;
		std::vector<GhostTrackState> gtStates;
		std::auto_ptr<GhostTrackPrediction> gtPred;
		if (useGhostTrack)
			gtPred.reset(new GhostTrackPrediction(
						*iterJets->ghostTrack()));

		for(unsigned int i = 0; i < indices.size(); i++) {
			typedef typename TemplatedSecondaryVertexTagInfo<IPTI,VTX>::IndexedTrackData IndexedTrackData;

			const input_item &trackRef = trackRefs[i];

			trackData.push_back(IndexedTrackData());
			trackData.back().first = indices[i];

			// select tracks for SV finder

			if (!trackSelector(*reco::btag::toTrack(trackRef), ipData[indices[i]], *jetRef,
			                   RecoVertex::convertPos(
			                   		pv.position()))) {
				trackData.back().second.svStatus =
					  TemplatedSecondaryVertexTagInfo<IPTI,VTX>::TrackData::trackSelected;
				continue;
			}

			TransientTrackMap::const_iterator pos =
					primariesMap.find(reco::btag::toTrack((trackRef)));
			TransientTrack fitTrack;
			if (pos != primariesMap.end()) {
				primaries.erase(pos->second);
				fitTrack = pos->second;
			} else
				fitTrack = trackBuilder->build(trackRef);
			fitTracks.push_back(fitTrack);

			trackData.back().second.svStatus =
				  TemplatedSecondaryVertexTagInfo<IPTI,VTX>::TrackData::trackUsedForVertexFit;

			if (useGhostTrack) {
				GhostTrackState gtState(fitTrack);
				GlobalPoint pos =
					ipData[indices[i]].closestToGhostTrack;
				gtState.linearize(*gtPred, true,
				                  gtPred->lambda(pos));
				gtState.setWeight(ipData[indices[i]].ghostTrackWeight);
				gtStates.push_back(gtState);
			}
		}

		std::auto_ptr<GhostTrack> ghostTrack;
		if (useGhostTrack)
			ghostTrack.reset(new GhostTrack(
				GhostTrackPrediction(
					RecoVertex::convertPos(pv.position()),
					RecoVertex::convertError(pv.error()),
					GlobalVector(
						iterJets->ghostTrack()->px(),
						iterJets->ghostTrack()->py(),
						iterJets->ghostTrack()->pz()),
					0.05),
				*gtPred, gtStates,
				iterJets->ghostTrack()->chi2(),
				iterJets->ghostTrack()->ndof()));

		// perform actual vertex finding


	 	std::vector<VTX>       extAssoCollection;    
		std::vector<TransientVertex> fittedSVs;
		std::vector<SecondaryVertex> SVs;
		if(!useExternalSV){ 
    		  switch(constraint)   {
		    case CONSTRAINT_NONE:
			if (useGhostTrack)
				fittedSVs = vertexRecoGT->vertices(
						pv, *ghostTrack);
			else
				fittedSVs = vertexReco->vertices(fitTracks);
			break;

		    case CONSTRAINT_BEAMSPOT:
			if (useGhostTrack)
				fittedSVs = vertexRecoGT->vertices(
						pv, *beamSpot, *ghostTrack);
			else
				fittedSVs = vertexReco->vertices(fitTracks,
				                                 *beamSpot);
			break;

		    case CONSTRAINT_PV_BEAMSPOT_SIZE:
		    case CONSTRAINT_PV_BS_Z_ERRORS_SCALED:
		    case CONSTRAINT_PV_ERROR_SCALED: {
			BeamSpot::CovarianceMatrix cov;
			for(unsigned int i = 0; i < 7; i++) {
				unsigned int covSrc = bsCovSrc[i];
				for(unsigned int j = 0; j < 7; j++) {
					double v=0.0;
					if (!covSrc || bsCovSrc[j] != covSrc)
						v = 0.0;
					else if (covSrc == 1)
						v = beamSpot->covariance(i, j);
					else if (j<3 && i<3)
						v = pv.covariance(i, j) *
						    constraintScaling;
					cov(i, j) = v;
				}
			}

			BeamSpot bs(pv.position(), sigmaZ,
			            beamSpot.isValid() ? beamSpot->dxdz() : 0.,
			            beamSpot.isValid() ? beamSpot->dydz() : 0.,
			            beamWidth, cov, BeamSpot::Unknown);

			if (useGhostTrack)
				fittedSVs = vertexRecoGT->vertices(
						pv, bs, *ghostTrack);
			else
				fittedSVs = vertexReco->vertices(fitTracks, bs);
		    }	break;

		    case CONSTRAINT_PV_PRIMARIES_IN_FIT: {
			std::vector<TransientTrack> primaries_(
					primaries.begin(), primaries.end());
			if (useGhostTrack)
				fittedSVs = vertexRecoGT->vertices(
						pv, *beamSpot, primaries_,
						*ghostTrack);
			else
				fittedSVs = vertexReco->vertices(
						primaries_, fitTracks,
						*beamSpot);
		    }	break;
		}

		// build combined SV information and filter

		SVBuilder svBuilder(pv, jetDir, withPVError, minTrackWeight);
		std::remove_copy_if(boost::make_transform_iterator(
		                    	fittedSVs.begin(), svBuilder),
		                    boost::make_transform_iterator(
		                    	fittedSVs.end(), svBuilder),
		                    std::back_inserter(SVs),
		                    SVFilter(vertexFilter, pv, jetDir));

		// clean up now unneeded collections
             }else{
                  for(size_t iExtSv = 0; iExtSv < extSecVertex->size(); iExtSv++){
		      const VTX & extVertex = (*extSecVertex)[iExtSv];
//    	              GlobalVector vtxDir = GlobalVector(extVertex.p4().X(),extVertex.p4().Y(),extVertex.p4().Z());
//                     if(Geom::deltaR(extVertex.position() - pv.position(), vtxDir)>0.2) continue; //pointing angle
//		      std::cout << " dR " << iExtSv << " " << Geom::deltaR( ( extVertex.position() - pv.position() ), jetDir ) << "eta: " << ( extVertex.position() - pv.position()).eta() << " vs " << jetDir.eta() << " phi: "  << ( extVertex.position() - pv.position()).phi() << " vs  " << jetDir.phi() <<  std::endl; 
	              if( Geom::deltaR( ( position(extVertex) - pv.position() ), jetDir ) >  extSVDeltaRToJet || extVertex.p4().M() < 0.3)
                	continue;
//		      std::cout << " SV added " << iExtSv << std::endl; 
 		      extAssoCollection.push_back( extVertex );
                   }
                   SVBuilder svBuilder(pv, jetDir, withPVError, minTrackWeight);
	           std::remove_copy_if(boost::make_transform_iterator( extAssoCollection.begin(), svBuilder),
                                boost::make_transform_iterator(extAssoCollection.end(), svBuilder),
                                std::back_inserter(SVs),
                                SVFilter(vertexFilter, pv, jetDir));


                 }
//		std::cout << "size: " << SVs.size() << std::endl; 
		gtPred.reset();
		ghostTrack.reset();
		gtStates.clear();
		fitTracks.clear();
		fittedSVs.clear();
		extAssoCollection.clear();

		// sort SVs by importance
		
		std::vector<unsigned int> vtxIndices = vertexSorting(SVs);

		std::vector<typename TemplatedSecondaryVertexTagInfo<IPTI,VTX>::VertexData> svData;

		svData.resize(vtxIndices.size());
		for(unsigned int idx = 0; idx < vtxIndices.size(); idx++) {
			const SecondaryVertex &sv = SVs[vtxIndices[idx]];

			svData[idx].vertex = sv;
			svData[idx].dist2d = sv.dist2d();
			svData[idx].dist3d = sv.dist3d();
			svData[idx].direction = flightDirection(pv,sv);
			// mark tracks successfully used in vertex fit
			markUsedTracks(trackData,trackRefs,sv,idx);
		}

		// fill result into tag infos

		tagInfos->push_back(
			TemplatedSecondaryVertexTagInfo<IPTI,VTX>(
				trackData, svData, SVs.size(),
				edm::Ref<std::vector<IPTI> >(trackIPTagInfos,
					iterJets - trackIPTagInfos->begin())));
	}

	event.put(tagInfos);
}

//Need specialized template because reco::Vertex iterators are TrackBase and it is a mess to make general
template <>
void  TemplatedSecondaryVertexProducer<TrackIPTagInfo,reco::Vertex>::markUsedTracks(TrackDataVector & trackData, const input_container & trackRefs, const SecondaryVertex & sv,size_t idx)
{
	for(Vertex::trackRef_iterator iter = sv.tracks_begin();	iter != sv.tracks_end(); ++iter) {
		if (sv.trackWeight(*iter) < minTrackWeight)
			continue;

		typename input_container::const_iterator pos =
			std::find(trackRefs.begin(), trackRefs.end(),
					iter->castTo<input_item>());

		if (pos == trackRefs.end() ) {
			if(!useExternalSV)
				throw cms::Exception("TrackNotFound")
					<< "Could not find track from secondary "
					"vertex in original tracks."
					<< std::endl;
		} else {
			unsigned int index = pos - trackRefs.begin();
			trackData[index].second.svStatus =
				(btag::TrackData::Status)
				((unsigned int)btag::TrackData::trackAssociatedToVertex + idx);
		}
	}
}
template <>
void  TemplatedSecondaryVertexProducer<CandIPTagInfo,reco::VertexCompositePtrCandidate>::markUsedTracks(TrackDataVector & trackData, const input_container & trackRefs, const SecondaryVertex & sv,size_t idx)
{
	for(typename input_container::const_iterator iter = sv.daughterPtrVector().begin(); iter != sv.daughterPtrVector().end(); ++iter)
	{
		typename input_container::const_iterator pos =
			std::find(trackRefs.begin(), trackRefs.end(), *iter);

		if (pos != trackRefs.end() )
		{
			unsigned int index = pos - trackRefs.begin();
			trackData[index].second.svStatus =
				(btag::TrackData::Status)
				((unsigned int)btag::TrackData::trackAssociatedToVertex + idx);
		}
	}
}


template <>
typename TemplatedSecondaryVertexProducer<TrackIPTagInfo,reco::Vertex>::SecondaryVertex  
TemplatedSecondaryVertexProducer<TrackIPTagInfo,reco::Vertex>::SVBuilder::operator () (const TransientVertex &sv) const
{
	if(sv.originalTracks().size()>0 && sv.originalTracks()[0].trackBaseRef().isNonnull())
		return SecondaryVertex(pv, sv, direction, withPVError);
	else
	{
		edm::LogError("UnexpectedInputs") << "Building from Candidates, should not happen!";
		return SecondaryVertex(pv, sv, direction, withPVError);
	}
}

template <>
typename TemplatedSecondaryVertexProducer<CandIPTagInfo,reco::VertexCompositePtrCandidate>::SecondaryVertex
TemplatedSecondaryVertexProducer<CandIPTagInfo,reco::VertexCompositePtrCandidate>::SVBuilder::operator () (const TransientVertex &sv) const
{
	if(sv.originalTracks().size()>0 && sv.originalTracks()[0].trackBaseRef().isNonnull())
	{
		edm::LogError("UnexpectedInputs") << "Building from Tracks, should not happen!";
		VertexCompositePtrCandidate vtxCompPtrCand;
		
		vtxCompPtrCand.setCovariance(sv.vertexState().error().matrix_new());
		vtxCompPtrCand.setChi2AndNdof(sv.totalChiSquared(), sv.degreesOfFreedom());
		vtxCompPtrCand.setVertex(Candidate::Point(sv.position().x(),sv.position().y(),sv.position().z()));
		
		return SecondaryVertex(pv, vtxCompPtrCand, direction, withPVError);
	}
	else
	{
		VertexCompositePtrCandidate vtxCompPtrCand;
		
		vtxCompPtrCand.setCovariance(sv.vertexState().error().matrix_new());
		vtxCompPtrCand.setChi2AndNdof(sv.totalChiSquared(), sv.degreesOfFreedom());
		vtxCompPtrCand.setVertex(Candidate::Point(sv.position().x(),sv.position().y(),sv.position().z()));
		
		Candidate::LorentzVector p4;
		for(std::vector<reco::TransientTrack>::const_iterator tt = sv.originalTracks().begin(); tt != sv.originalTracks().end(); ++tt)
		{
			if (sv.trackWeight(*tt) < minTrackWeight)
				continue;
			
			const CandidatePtrTransientTrack* cptt = dynamic_cast<const CandidatePtrTransientTrack*>(tt->basicTransientTrack());
			if ( cptt==0 )
				edm::LogError("DynamicCastingFailed") << "Casting of TransientTrack to CandidatePtrTransientTrack failed!";
			else
			{
				p4 += cptt->candidate()->p4();
				vtxCompPtrCand.addDaughter(cptt->candidate());
			}
		}
		vtxCompPtrCand.setP4(p4);
		
		return SecondaryVertex(pv, vtxCompPtrCand, direction, withPVError);
	}
}



//define this as a plug-in
typedef TemplatedSecondaryVertexProducer<TrackIPTagInfo,reco::Vertex> SecondaryVertexProducer;
typedef TemplatedSecondaryVertexProducer<CandIPTagInfo,reco::VertexCompositePtrCandidate> CandSecondaryVertexProducer;

DEFINE_FWK_MODULE(SecondaryVertexProducer);
DEFINE_FWK_MODULE(CandSecondaryVertexProducer);

