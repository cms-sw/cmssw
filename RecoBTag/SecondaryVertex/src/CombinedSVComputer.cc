#include <iostream>
#include <cstddef>
#include <string>
#include <cmath>
#include <vector>

#include <Math/VectorUtil.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"  
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/VertexTypes.h"

#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"

#include "RecoBTag/SecondaryVertex/interface/ParticleMasses.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"

#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"

using namespace reco;

struct CombinedSVComputer::IterationRange {
	int begin, end, increment;
};

#define range_for(i, x) \
	for(int i = (x).begin; i != (x).end; i += (x).increment)

static edm::ParameterSet dropDeltaR(const edm::ParameterSet &pset)
{
	edm::ParameterSet psetCopy(pset);
	psetCopy.addParameter<double>("jetDeltaRMax", 99999.0);
	return psetCopy;
}

CombinedSVComputer::CombinedSVComputer(const edm::ParameterSet &params) :
	trackFlip(params.getParameter<bool>("trackFlip")),
	vertexFlip(params.getParameter<bool>("vertexFlip")),
	charmCut(params.getParameter<double>("charmCut")),
	sortCriterium(TrackSorting::getCriterium(params.getParameter<std::string>("trackSort"))),
	trackSelector(params.getParameter<edm::ParameterSet>("trackSelection")),
	trackNoDeltaRSelector(dropDeltaR(params.getParameter<edm::ParameterSet>("trackSelection"))),
	trackPseudoSelector(params.getParameter<edm::ParameterSet>("trackPseudoSelection")),
	pseudoMultiplicityMin(params.getParameter<unsigned int>("pseudoMultiplicityMin")),
	trackMultiplicityMin(params.getParameter<unsigned int>("trackMultiplicityMin")),
	minTrackWeight(params.getParameter<double>("minimumTrackWeight")),
	useTrackWeights(params.getParameter<bool>("useTrackWeights")),
	vertexMassCorrection(params.getParameter<bool>("correctVertexMass")),
	pseudoVertexV0Filter(params.getParameter<edm::ParameterSet>("pseudoVertexV0Filter")),
	trackPairV0Filter(params.getParameter<edm::ParameterSet>("trackPairV0Filter"))
{
}

inline double CombinedSVComputer::flipValue(double value, bool vertex) const
{
	return (vertex ? vertexFlip : trackFlip) ? -value : value;
}

inline CombinedSVComputer::IterationRange CombinedSVComputer::flipIterate(
						int size, bool vertex) const
{
	IterationRange range;
	if (vertex ? vertexFlip : trackFlip) {
		range.begin = size - 1;
		range.end = -1;
		range.increment = -1;
	} else {
		range.begin = 0;
		range.end = size;
		range.increment = +1;
	}

	return range;
}

const TrackIPTagInfo::TrackIPData &
CombinedSVComputer::threshTrack(const TrackIPTagInfo &trackIPTagInfo,
                                const TrackIPTagInfo::SortCriteria sort,
                                const reco::Jet &jet,
                                const GlobalPoint &pv) const
{
	const edm::RefVector<TrackCollection> &tracks =
					trackIPTagInfo.selectedTracks();
	const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
					trackIPTagInfo.impactParameterData();
	std::vector<std::size_t> indices = trackIPTagInfo.sortedIndexes(sort);

	IterationRange range = flipIterate(indices.size(), false);
	TrackKinematics kin;
	range_for(i, range) {
		std::size_t idx = indices[i];
		const TrackIPTagInfo::TrackIPData &data = ipData[idx];
		const Track &track = *tracks[idx];

		if (!trackNoDeltaRSelector(track, data, jet, pv))
			continue;

		kin.add(track);
		if (kin.vectorSum().M() > charmCut) 
			return data;
	}

	static const TrackIPTagInfo::TrackIPData dummy = {
 		GlobalPoint(),
		GlobalPoint(),
		Measurement1D(-1.0, 1.0),
		Measurement1D(-1.0, 1.0),
		Measurement1D(-1.0, 1.0),
		Measurement1D(-1.0, 1.0),
		0.
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

	if (ipInfo.selectedTracks().size() < trackMultiplicityMin)
		return vars;

	TrackKinematics allKinematics;
	TrackKinematics vertexKinematics;

	int vtx = -1;
	unsigned int numberofvertextracks = 0;

		IterationRange range = flipIterate(svInfo.nVertices(), true);
	range_for(i, range) {
		if (vtx < 0)
			vtx = i;
	
		numberofvertextracks = numberofvertextracks + (svInfo.secondaryVertex(i)).nTracks();

				const Vertex &vertex = svInfo.secondaryVertex(i);
		bool hasRefittedTracks = vertex.hasRefittedTracks();
		TrackRefVector tracks = svInfo.vertexTracks(i);
		for(TrackRefVector::const_iterator track = tracks.begin();
		    track != tracks.end(); track++) {
			double w = svInfo.trackWeight(i, *track);
			if (w < minTrackWeight)
				continue;
			if (hasRefittedTracks) {
				Track actualTrack =
						vertex.refittedTrack(*track);
				vertexKinematics.add(actualTrack, w);
				vars.insert(btau::trackEtaRel, etaRel(jetDir,
						actualTrack.momentum()), true);
			} else {
				vertexKinematics.add(**track, w);
				vars.insert(btau::trackEtaRel, etaRel(jetDir,
						(*track)->momentum()), true);
			}
		}
	}

	if (vtx >= 0) {
		vtxType = btag::Vertices::RecoVertex;

		vars.insert(btau::flightDistance2dVal,
		            flipValue(
		            	svInfo.flightDistance(vtx, true).value(),
		            	true),
		            true);
		vars.insert(btau::flightDistance2dSig,
		            flipValue(
		            	svInfo.flightDistance(vtx, true).significance(),
		            	true),
		            true);
		vars.insert(btau::flightDistance3dVal,
		            flipValue(
		            	svInfo.flightDistance(vtx, false).value(),
		            	true),
		            true);
		vars.insert(btau::flightDistance3dSig,
		            flipValue(
		            	svInfo.flightDistance(vtx, false).significance(),
		            	true),
		            true);
		vars.insert(btau::vertexJetDeltaR,
		            Geom::deltaR(svInfo.flightDirection(vtx), jetDir),true);
		vars.insert(btau::jetNSecondaryVertices, svInfo.nVertices(), true);
		vars.insert(btau::vertexNTracks, numberofvertextracks, true);
	}

	std::vector<std::size_t> indices = ipInfo.sortedIndexes(sortCriterium);
	const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
						ipInfo.impactParameterData();
	const edm::RefVector<TrackCollection> &tracks =
						ipInfo.selectedTracks();
	std::vector<TrackRef> pseudoVertexTracks;

	TrackRef trackPairV0Test[2];
	range = flipIterate(indices.size(), false);
	range_for(i, range) {
		std::size_t idx = indices[i];
		const TrackIPTagInfo::TrackIPData &data = ipData[idx];
		const TrackRef &trackRef = tracks[idx];
		const Track &track = *trackRef;

		// filter track

		if (!trackSelector(track, data, *jet, pv))
			continue;

		// add track to kinematics for all tracks in jet

		allKinematics.add(track);

		// if no vertex was reconstructed, attempt pseudo vertex

		if (vtxType == btag::Vertices::NoVertex &&
		    trackPseudoSelector(track, data, *jet, pv)) {
			pseudoVertexTracks.push_back(trackRef);
			vertexKinematics.add(track);
		}

		// check against all other tracks for V0 track pairs

		trackPairV0Test[0] = tracks[idx];
		bool ok = true;
		range_for(j, range) {
			if (i == j)
				continue;

			std::size_t pairIdx = indices[j];
			const TrackIPTagInfo::TrackIPData &pairTrackData =
							ipData[pairIdx];
			const TrackRef &pairTrackRef = tracks[pairIdx];
			const Track &pairTrack = *pairTrackRef;

			if (!trackSelector(pairTrack, pairTrackData, *jet, pv))
				continue;

			trackPairV0Test[1] = pairTrackRef;
			if (!trackPairV0Filter(trackPairV0Test, 2)) {
				ok = false;
				break;
			}
		}
		if (!ok)
			continue;

		// add track variables

		math::XYZVector trackMom = track.momentum();
		double trackMag = std::sqrt(trackMom.Mag2());

		vars.insert(btau::trackSip3dVal,
		            flipValue(data.ip3d.value(), false), true);
		vars.insert(btau::trackSip3dSig,
		            flipValue(data.ip3d.significance(), false), true);
		vars.insert(btau::trackSip2dVal,
		            flipValue(data.ip2d.value(), false), true);
		vars.insert(btau::trackSip2dSig,
		            flipValue(data.ip2d.significance(), false), true);
		vars.insert(btau::trackJetDistVal, data.distanceToJetAxis.value(), true);
//		vars.insert(btau::trackJetDistSig, data.distanceToJetAxis.significance(), true);
//		vars.insert(btau::trackFirstTrackDist, data.distanceToFirstTrack, true);
//		vars.insert(btau::trackGhostTrackVal, data.distanceToGhostTrack.value(), true);
//		vars.insert(btau::trackGhostTrackSig, data.distanceToGhostTrack.significance(), true);
		vars.insert(btau::trackDecayLenVal, havePv ? (data.closestToJetAxis - pv).mag() : -1.0, true);

		vars.insert(btau::trackMomentum, trackMag, true);
		vars.insert(btau::trackEta, trackMom.Eta(), true);

		vars.insert(btau::trackPtRel, VectorUtil::Perp(trackMom, jetDir), true);
		vars.insert(btau::trackPPar, jetDir.Dot(trackMom), true);
		vars.insert(btau::trackDeltaR, VectorUtil::DeltaR(trackMom, jetDir), true);
		vars.insert(btau::trackPtRatio, VectorUtil::Perp(trackMom, jetDir) / trackMag, true);
		vars.insert(btau::trackPParRatio, jetDir.Dot(trackMom) / trackMag, true);
	} 

	if (vtxType == btag::Vertices::NoVertex &&
	    vertexKinematics.numberOfTracks() >= pseudoMultiplicityMin &&
	    pseudoVertexV0Filter(pseudoVertexTracks)) {
		vtxType = btag::Vertices::PseudoVertex;
		for(std::vector<TrackRef>::const_iterator track =
						pseudoVertexTracks.begin();
		    track != pseudoVertexTracks.end(); ++track)
			vars.insert(btau::trackEtaRel, etaRel(jetDir,
						(*track)->momentum()), true);
	}

	vars.insert(btau::vertexCategory, vtxType, true);
	vars.insert(btau::trackSumJetDeltaR,
	            VectorUtil::DeltaR(allKinematics.vectorSum(), jetDir), true);
	vars.insert(btau::trackSumJetEtRatio,
	            allKinematics.vectorSum().Et() / ipInfo.jet()->et(), true);
	vars.insert(btau::trackSip3dSigAboveCharm,
	            flipValue(
	            	threshTrack(ipInfo, TrackIPTagInfo::IP3DSig, *jet, pv)
	            					.ip3d.significance(),
	            	false),
	            true);
	vars.insert(btau::trackSip2dSigAboveCharm,
	            flipValue(
	            	threshTrack(ipInfo, TrackIPTagInfo::IP2DSig, *jet, pv)
	            					.ip2d.significance(),
	            	false),
	            true);

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

		double vertexMass = vertexSum.M();
		if (vtxType == btag::Vertices::RecoVertex &&
		    vertexMassCorrection) {
			GlobalVector dir = svInfo.flightDirection(vtx);
			double vertexPt2 =
				math::XYZVector(dir.x(), dir.y(), dir.z()).
					Cross(vertexSum).Mag2() / dir.mag2();
			vertexMass = std::sqrt(vertexMass * vertexMass +
			                       vertexPt2) + std::sqrt(vertexPt2);
		}
		vars.insert(btau::vertexMass, vertexMass, true);
		if (allKinematics.numberOfTracks())
			vars.insert(btau::vertexEnergyRatio,
			            vertexSum.E() / allSum.E(), true);
		else
			vars.insert(btau::vertexEnergyRatio, 1, true);
	}

	vars.finalize();

	return vars;
}
