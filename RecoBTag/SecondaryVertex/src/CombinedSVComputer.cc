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
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
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
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"

#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"

using namespace reco;

CombinedSVComputer::CombinedSVComputer(const edm::ParameterSet &params) :
	charmCut(params.getParameter<double>("charmCut")),
	sortCriterium(TrackSorting::getCriterium(params.getParameter<std::string>("trackSort"))),
	trackSelector(params.getParameter<edm::ParameterSet>("trackSelection")),
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

	static const TrackIPTagInfo::TrackIPData dummy = {
		  Measurement1D(-1.0, 1.0),
		  Measurement1D(-1.0, 1.0),
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

	if (ipInfo.tracks().size() < trackMultiplicityMin)
		return vars;

	TrackKinematics allKinematics;
	TrackKinematics vertexKinematics;

	if (svInfo.nVertices() > 0) {
		vtxType = btag::Vertices::RecoVertex;
		// vars.insert(svInfo.taggingVariables());
		for(unsigned int i = 0; i < svInfo.nVertices(); i++) {
			TrackRefVector tracks = svInfo.vertexTracks(i);
			for(TrackRefVector::const_iterator track = tracks.begin();
			    track != tracks.end(); track++) {
				double w = svInfo.trackWeight(i, *track);
				if (w >= minTrackWeight)
					vertexKinematics.add(**track, w);
			}
		}

		vars.insert(btau::flightDistance2dVal,
		            svInfo.flightDistance(0, true).value(), true);
		vars.insert(btau::flightDistance2dSig,
		            svInfo.flightDistance(0, true).significance(), true);
		vars.insert(btau::flightDistance3dVal,
		            svInfo.flightDistance(0, false).value(), true);
		vars.insert(btau::flightDistance3dSig,
		            svInfo.flightDistance(0, false).significance(), true);
		vars.insert(btau::vertexJetDeltaR,
		            Geom::deltaR(svInfo.flightDirection(0), jetDir), true);
		vars.insert(btau::jetNSecondaryVertices, svInfo.nVertices(), true);
		vars.insert(btau::vertexNTracks, svInfo.nVertexTracks(), true);
	}

	std::vector<std::size_t> indices = ipInfo.sortedIndexes(sortCriterium);
	const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
						ipInfo.impactParameterData();
	const edm::RefVector<TrackCollection> &tracks =
						ipInfo.selectedTracks();
	std::vector<TrackRef> pseudoVertexTracks;

	TrackRef trackPairV0Test[2];
	for(std::vector<size_t>::const_iterator iter = indices.begin();
	    iter != indices.end(); ++iter) {
		const TrackIPTagInfo::TrackIPData &data = ipData[*iter];
		const TrackRef &trackRef = tracks[*iter];
		const Track &track = *trackRef;

		// filter track

		if (!trackSelector(track, data, *jet))
			continue;

		// add track to kinematics for all tracks in jet

		allKinematics.add(track);

		// if no vertex was reconstructed, attempt pseudo vertex

		if (vtxType == btag::Vertices::NoVertex &&
		    trackPseudoSelector(track, data, *jet)) {
			pseudoVertexTracks.push_back(trackRef);
			vertexKinematics.add(track);
		}

		// check against all other tracks for V0 track pairs

		trackPairV0Test[0] = tracks[*iter];
		bool ok = true;
		for(std::vector<size_t>::const_iterator pairIter =
							indices.begin();
		    pairIter != indices.end(); ++pairIter) {
			if (pairIter == iter)
				continue;

			const TrackIPTagInfo::TrackIPData &pairTrackData =
							ipData[*pairIter];
			const TrackRef &pairTrackRef = tracks[*pairIter];
			const Track &pairTrack = *pairTrackRef;

			if (!trackSelector(pairTrack, pairTrackData, *jet))
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
	} 

	if (vtxType == btag::Vertices::NoVertex &&
	    vertexKinematics.numberOfTracks() >= pseudoMultiplicityMin &&
	    pseudoVertexV0Filter(pseudoVertexTracks))
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

		double vertexMass = vertexSum.M();
		if (vtxType == btag::Vertices::RecoVertex &&
		    vertexMassCorrection) {
			GlobalVector dir = svInfo.flightDirection(0);
			double vertexPt2 =
				math::XYZVector(dir.x(), dir.y(), dir.z()).
					Cross(vertexSum).Mag2() / dir.mag2();
			vertexMass = std::sqrt(vertexMass * vertexMass +
			                       vertexPt2) + std::sqrt(vertexPt2);
		}
		vars.insert(btau::vertexMass, vertexMass, true);
		vars.insert(btau::vertexEnergyRatio, vertexSum.E() / allSum.E(), true);
	}

	vars.finalize();

	return vars;
}
