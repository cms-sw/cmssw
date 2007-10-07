#include <cstddef>
#include <cmath>
#include <vector>

#include <Math/VectorUtil.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"  
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/VertexTypes.h"

#include "RecoBTag/SecondaryVertex/interface/ParticleMasses.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"

#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"

using namespace reco;

CombinedSVComputer::CombinedSVComputer(const edm::ParameterSet &params) :
	charmCut(params.getParameter<double>("charmCut")),
	sortCriterium(TrackSorting::getCriterium(params.getParameter<std::string>("trackSort"))),
	trackPseudoSelector(params.getParameter<edm::ParameterSet>("trackPseudoSelection")),
	pseudoMultiplicityMin(params.getParameter<unsigned int>("pseudoMultiplicityMin")),
	useTrackWeights(params.getParameter<bool>("useTrackWeights"))
{
}

const TrackIPTagInfo::TrackIPData &
CombinedSVComputer::threshTrack(const TrackIPTagInfo &trackIPTagInfo,
                                const TrackIPTagInfo::SortCriteria sort) const
{
	const edm::RefVector<TrackCollection> &tracks =
					trackIPTagInfo.selectedTracks();
	const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
					trackIPTagInfo.impactParameterData();
	std::vector<std::size_t> indices = trackIPTagInfo.sortedIndexes(sort);

	TrackKinematics kin;
	for(std::vector<std::size_t>::const_iterator iter = indices.begin();
	    iter != indices.end(); iter++) {
		kin.add(*tracks[*iter]);

		if (kin.vectorSum().M() > charmCut) 
			return ipData[*iter];
	}

	static TrackIPTagInfo::TrackIPData dummy = {
		  Measurement1D(-999.0, 1.0),
		  Measurement1D(-999.0, 1.0),
	};
	return dummy;
}

static double etaRel(const math::XYZVector &dir, const math::XYZVector &track)
{
	double momPar = dir.Dot(track);
	double energy = std::sqrt(track.Mag2() +
	                          ROOT::Math::Square(ParticleMasses::piPlus));

	return 0.5 * std::log((energy + momPar) / (energy - momPar));
}

TaggingVariableList
CombinedSVComputer::operator () (const TrackIPTagInfo &ipInfo,
                                 const SecondaryVertexTagInfo &svInfo) const
{
	using namespace ROOT::Math;

	edm::RefToBase<Jet> jet = ipInfo.jet();
	math::XYZVector jetDir = jet->momentum().Unit();
	bool havePv = ipInfo.primaryVertex().isNonnull();
	GlobalPoint pv;
	if (havePv)
		pv = GlobalPoint(ipInfo.primaryVertex()->x(),
		                 ipInfo.primaryVertex()->y(),
		                 ipInfo.primaryVertex()->z());

	btag::Vertices::VertexType vtxType = btag::Vertices::NoVertex;

	TaggingVariableList vars; // = ipInfo.taggingVariables();

	vars.insert(btau::jetPt, jet->pt(), true);
	vars.insert(btau::jetEta, jet->eta(), true);

	TrackKinematics allKinematics;
	TrackKinematics vertexKinematics;

	if (svInfo.nVertices() > 0) {
		vtxType = btag::Vertices::RecoVertex;
		vars.insert(svInfo.taggingVariables());
		vertexKinematics = TrackKinematics(svInfo.secondaryVertex(0));
	}

	std::vector<std::size_t> indices = ipInfo.sortedIndexes(sortCriterium);
	const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
						ipInfo.impactParameterData();
	const edm::RefVector<TrackCollection> &tracks =
						ipInfo.selectedTracks();

	for(std::vector<size_t>::const_iterator iter = indices.begin();
	    iter != indices.end(); ++iter) {
		const TrackIPTagInfo::TrackIPData &data = ipData[*iter];
		const Track &track = *tracks[*iter];;
		math::XYZVector trackMom = track.momentum();
		double trackMag = std::sqrt(trackMom.Mag2());

		vars.insert(btau::trackSip3dVal, data.ip3d.value(), true);
		vars.insert(btau::trackSip3dSig, data.ip3d.significance(), true);
		vars.insert(btau::trackSip2dVal, data.ip2d.value(), true);
		vars.insert(btau::trackSip2dSig, data.ip2d.significance(), true);
		vars.insert(btau::trackJetDist, data.distanceToJetAxis, true);
		vars.insert(btau::trackFirstTrackDist, data.distanceToFirstTrack, true);
		vars.insert(btau::trackDecayLenVal, havePv ? (data.closestToJetAxis - pv).mag() : -1.0, true);

		vars.insert(btau::trackMomentum, trackMag, true);
		vars.insert(btau::trackEta, trackMom.Eta(), true);

		vars.insert(btau::trackEtaRel, etaRel(jetDir, trackMom), true);
		vars.insert(btau::trackPtRel, VectorUtil::Perp(trackMom, jetDir), true);
		vars.insert(btau::trackPPar, jetDir.Dot(trackMom), true);
		vars.insert(btau::trackDeltaR, VectorUtil::DeltaR(trackMom, jetDir), true);
		vars.insert(btau::trackPtRatio, VectorUtil::Perp(trackMom, jetDir) / trackMag, true);
		vars.insert(btau::trackPParRatio, jetDir.Dot(trackMom) / trackMag, true);

		allKinematics.add(track);

		if (vtxType == btag::Vertices::NoVertex &&
		    trackPseudoSelector(track, data))
			vertexKinematics.add(track);
	} 

	if (vtxType == btag::Vertices::NoVertex &&
	    vertexKinematics.numberOfTracks() >= pseudoMultiplicityMin)
		vtxType = btag::Vertices::PseudoVertex;

	vars.insert(btau::vertexCategory, vtxType, true);
	vars.insert(btau::trackSumJetDeltaR,
	            VectorUtil::DeltaR(allKinematics.vectorSum(), jetDir), true);
	vars.insert(btau::trackSumJetEtRatio,
	            allKinematics.vectorSum().Et() / ipInfo.jet()->et(), true);
	vars.insert(btau::trackSip3dSigAboveCharm, threshTrack(ipInfo,
			TrackIPTagInfo::IP3DSig).ip3d.significance(), true);
	vars.insert(btau::trackSip2dSigAboveCharm, threshTrack(ipInfo,
			TrackIPTagInfo::IP2DSig).ip2d.significance(), true);

	if (vtxType != btag::Vertices::NoVertex) {
		math::XYZTLorentzVector allSum = useTrackWeights
			? allKinematics.weightedVectorSum()
			: allKinematics.vectorSum();
		math::XYZTLorentzVector vertexSum = useTrackWeights
			? vertexKinematics.weightedVectorSum()
			: vertexKinematics.vectorSum();

		if (vtxType != btag::Vertices::RecoVertex) {
			vars.insert(btau::vertexNTracks,
			            vertexKinematics.numberOfTracks(), true);
			vars.insert(btau::vertexJetDeltaR,
			            VectorUtil::DeltaR(vertexSum, jetDir), true);
		}

		vars.insert(btau::vertexMass, vertexSum.M(), true);
		vars.insert(btau::vertexEnergyRatio, vertexSum.E() / allSum.E(), true);
	}

	vars.finalize();

	return vars;
}
