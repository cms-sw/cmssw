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

#include "RecoBTag/SecondaryVertex/interface/ParticleMasses.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"

#include "RecoBTag/SecondaryVertex/interface/GhostTrackComputer.h"

using namespace reco;

static edm::ParameterSet dropDeltaR(const edm::ParameterSet &pset)
{
	edm::ParameterSet psetCopy(pset);
	psetCopy.addParameter<double>("jetDeltaRMax", 99999.0);
	return psetCopy;
}

GhostTrackComputer::GhostTrackComputer(const edm::ParameterSet &params) :
	charmCut(params.getParameter<double>("charmCut")),
	sortCriterium(TrackSorting::getCriterium(params.getParameter<std::string>("trackSort"))),
	trackSelector(params.getParameter<edm::ParameterSet>("trackSelection")),
	trackNoDeltaRSelector(dropDeltaR(params.getParameter<edm::ParameterSet>("trackSelection"))),
	minTrackWeight(params.getParameter<double>("minimumTrackWeight")),
	trackPairV0Filter(params.getParameter<edm::ParameterSet>("trackPairV0Filter"))
{
}

const TrackIPTagInfo::TrackIPData &
GhostTrackComputer::threshTrack(const TrackIPTagInfo &trackIPTagInfo,
                                const TrackIPTagInfo::SortCriteria sort,
                                const reco::Jet &jet,
                                const GlobalPoint &pv) const
{
	const edm::RefVector<TrackCollection> &tracks =
					trackIPTagInfo.selectedTracks();
	const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
					trackIPTagInfo.impactParameterData();
	std::vector<std::size_t> indices = trackIPTagInfo.sortedIndexes(sort);

	TrackKinematics kin;
	for(std::vector<std::size_t>::const_iterator iter = indices.begin();
	    iter != indices.end(); ++iter) {
		const TrackIPTagInfo::TrackIPData &data = ipData[*iter];
		const Track &track = *tracks[*iter];

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

static void addMeas(std::pair<double, double> &sum, Measurement1D meas)
{
	double weight = 1. / meas.error();
	weight *= weight;
	sum.first += weight * meas.value();
	sum.second += weight;
}

TaggingVariableList
GhostTrackComputer::operator () (const TrackIPTagInfo &ipInfo,
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

	TaggingVariableList vars;

	vars.insert(btau::jetPt, jet->pt(), true);
	vars.insert(btau::jetEta, jet->eta(), true);

	TrackKinematics allKinematics;
	TrackKinematics vertexKinematics;
	TrackKinematics trackKinematics;

	std::pair<double, double> vertexDist2D, vertexDist3D;
	std::pair<double, double> tracksDist2D, tracksDist3D;

	unsigned int nVertices = 0;
	unsigned int nVertexTracks = 0;
	unsigned int nTracks = 0;
	for(unsigned int i = 0; i < svInfo.nVertices(); i++) {
		const Vertex &vertex = svInfo.secondaryVertex(i);
		bool hasRefittedTracks = vertex.hasRefittedTracks();
		TrackRefVector tracks = svInfo.vertexTracks(i);
		unsigned int n = 0;
		for(TrackRefVector::const_iterator track = tracks.begin();
		    track != tracks.end(); track++)
			if (svInfo.trackWeight(i, *track) >= minTrackWeight)
				n++;

		if (n < 1)
			continue;
		bool isTrackVertex = (n == 1);
		++*(isTrackVertex ? &nTracks : &nVertices);

		addMeas(*(isTrackVertex ? &tracksDist2D : &vertexDist2D),
					svInfo.flightDistance(i, true));
		addMeas(*(isTrackVertex ? &tracksDist3D : &vertexDist3D),
					svInfo.flightDistance(i, false));

		TrackKinematics &kin = isTrackVertex ? trackKinematics
		                                     : vertexKinematics;
		for(TrackRefVector::const_iterator track = tracks.begin();
		    track != tracks.end(); track++) {
			float w = svInfo.trackWeight(i, *track);
			if (w < minTrackWeight)
				continue;
			if (hasRefittedTracks) {
				Track actualTrack =
						vertex.refittedTrack(*track);
				kin.add(actualTrack, w);
				vars.insert(btau::trackEtaRel, etaRel(jetDir,
						actualTrack.momentum()), true);
			} else {
				kin.add(**track, w);
				vars.insert(btau::trackEtaRel, etaRel(jetDir,
						(*track)->momentum()), true);
			}
			if (!isTrackVertex)
				nVertexTracks++;
		}
	}

	Measurement1D dist2D, dist3D;
	if (nVertices) {
		vtxType = btag::Vertices::RecoVertex;

		if (nVertices == 1 && nTracks) {
			vertexDist2D.first += tracksDist2D.first;
			vertexDist2D.second += tracksDist2D.second;
			vertexDist3D.first += tracksDist3D.first;
			vertexDist3D.second += tracksDist3D.second;
			vertexKinematics += trackKinematics;
		}

		dist2D = Measurement1D(
				vertexDist2D.first / vertexDist2D.second,
				std::sqrt(1. / vertexDist2D.second));
		dist3D = Measurement1D(
				vertexDist3D.first / vertexDist3D.second,
				std::sqrt(1. / vertexDist3D.second));

		vars.insert(btau::jetNSecondaryVertices, nVertices, true);
		vars.insert(btau::vertexNTracks, nVertexTracks, true);
	} else if (nTracks) {
		vtxType = btag::Vertices::PseudoVertex;
		vertexKinematics = trackKinematics;

		dist2D = Measurement1D(
				tracksDist2D.first / tracksDist2D.second,
				std::sqrt(1. / tracksDist2D.second));
		dist3D = Measurement1D(
				tracksDist3D.first / tracksDist3D.second,
				std::sqrt(1. / tracksDist3D.second));
	}

	if (nVertices || nTracks) {
		vars.insert(btau::flightDistance2dVal, dist2D.value(), true);
		vars.insert(btau::flightDistance2dSig, dist2D.significance(), true);
		vars.insert(btau::flightDistance3dVal, dist3D.value(), true);
		vars.insert(btau::flightDistance3dSig, dist3D.significance(), true);
		vars.insert(btau::vertexJetDeltaR,
		            Geom::deltaR(svInfo.flightDirection(0), jetDir),true);
		vars.insert(btau::jetNSingleTrackVertices, nTracks, true);
	}

	std::vector<std::size_t> indices = ipInfo.sortedIndexes(sortCriterium);
	const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
						ipInfo.impactParameterData();
	const edm::RefVector<TrackCollection> &tracks =
						ipInfo.selectedTracks();

	TrackRef trackPairV0Test[2];
	for(unsigned int i = 0; i < indices.size(); i++) {
		std::size_t idx = indices[i];
		const TrackIPTagInfo::TrackIPData &data = ipData[idx];
		const TrackRef &trackRef = tracks[idx];
		const Track &track = *trackRef;

		// filter track

		if (!trackSelector(track, data, *jet, pv))
			continue;

		// add track to kinematics for all tracks in jet

		allKinematics.add(track);

		// check against all other tracks for V0 track pairs

		trackPairV0Test[0] = tracks[idx];
		bool ok = true;
		for(unsigned int j = 0; j < indices.size(); j++) {
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

		vars.insert(btau::trackSip3dVal, data.ip3d.value(), true);
		vars.insert(btau::trackSip3dSig, data.ip3d.significance(), true);
		vars.insert(btau::trackSip2dVal, data.ip2d.value(), true);
		vars.insert(btau::trackSip2dSig, data.ip2d.significance(), true);
		vars.insert(btau::trackJetDistVal, data.distanceToJetAxis.value(), true);
		vars.insert(btau::trackGhostTrackDistVal, data.distanceToGhostTrack.value(), true);
		vars.insert(btau::trackGhostTrackDistSig, data.distanceToGhostTrack.significance(), true);
		vars.insert(btau::trackGhostTrackWeight, data.ghostTrackWeight, true);
		vars.insert(btau::trackDecayLenVal, havePv ? (data.closestToGhostTrack - pv).mag() : -1.0, true);

		vars.insert(btau::trackMomentum, trackMag, true);
		vars.insert(btau::trackEta, trackMom.Eta(), true);

		vars.insert(btau::trackChi2, track.normalizedChi2(), true);
		vars.insert(btau::trackNPixelHits, track.hitPattern().pixelLayersWithMeasurement(), true);
		vars.insert(btau::trackNTotalHits, track.hitPattern().trackerLayersWithMeasurement(), true);
		vars.insert(btau::trackPtRel, VectorUtil::Perp(trackMom, jetDir), true);
		vars.insert(btau::trackDeltaR, VectorUtil::DeltaR(trackMom, jetDir), true);
	} 

	vars.insert(btau::vertexCategory, vtxType, true);
	vars.insert(btau::trackSumJetDeltaR,
	            VectorUtil::DeltaR(allKinematics.vectorSum(), jetDir), true);
	vars.insert(btau::trackSumJetEtRatio,
	            allKinematics.vectorSum().Et() / ipInfo.jet()->et(), true);
	vars.insert(btau::trackSip3dSigAboveCharm,
	            threshTrack(ipInfo, TrackIPTagInfo::IP3DSig, *jet, pv)
	            					.ip3d.significance(),
	            true);
	vars.insert(btau::trackSip2dSigAboveCharm,
	            threshTrack(ipInfo, TrackIPTagInfo::IP2DSig, *jet, pv)
	            					.ip2d.significance(),
	            true);

	if (vtxType != btag::Vertices::NoVertex) {
		math::XYZTLorentzVector allSum = allKinematics.vectorSum();
		math::XYZTLorentzVector vertexSum = vertexKinematics.vectorSum();

		vars.insert(btau::vertexMass, vertexSum.M(), true);
		if (allKinematics.numberOfTracks())
			vars.insert(btau::vertexEnergyRatio,
			            vertexSum.E() / allSum.E(), true);
		else
			vars.insert(btau::vertexEnergyRatio, 1, true);
	}

	vars.finalize();

	return vars;
}
