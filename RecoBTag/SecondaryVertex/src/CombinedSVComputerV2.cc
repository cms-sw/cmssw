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
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"

#include "RecoBTag/SecondaryVertex/interface/ParticleMasses.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"

#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputerV2.h"


using namespace reco;

struct CombinedSVComputerV2::IterationRange {
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

CombinedSVComputerV2::CombinedSVComputerV2(const edm::ParameterSet &params) :
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

inline double CombinedSVComputerV2::flipValue(double value, bool vertex) const
{
	return (vertex ? vertexFlip : trackFlip) ? -value : value;
}

inline CombinedSVComputerV2::IterationRange CombinedSVComputerV2::flipIterate(
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
CombinedSVComputerV2::threshTrack(const TrackIPTagInfo &trackIPTagInfo,
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
	double energy = std::sqrt(track.Mag2() + ROOT::Math::Square(ParticleMasses::piPlus));

	return 0.5 * std::log((energy + momPar) / (energy - momPar));
}

TaggingVariableList
CombinedSVComputerV2::operator () (const TrackIPTagInfo &ipInfo,
                                 const SecondaryVertexTagInfo &svInfo) const
{
	using namespace ROOT::Math;

	edm::RefToBase<Jet> jet = ipInfo.jet();

	math::XYZVector jetDir = jet->momentum().Unit();
	bool havePv = ipInfo.primaryVertex().isNonnull();
	GlobalPoint pv;
	if (havePv)
		pv = GlobalPoint(ipInfo.primaryVertex()->x(), ipInfo.primaryVertex()->y(), ipInfo.primaryVertex()->z());

	btag::Vertices::VertexType vtxType = btag::Vertices::NoVertex;

	TaggingVariableList vars; 

	vars.insert(btau::jetPt, jet->pt(), true);
	vars.insert(btau::jetEta, jet->eta(), true);

	if (ipInfo.tracks().size() < trackMultiplicityMin)
		return vars;
	
	vars.insert(btau::jetNTracks, ipInfo.tracks().size(), true);

	TrackKinematics allKinematics;
	TrackKinematics vertexKinematics;

	double vtx_track_ptSum= 0.; 
	double vtx_track_ESum= 0.; 
	double jet_track_ESum= 0.;

	int vtx = -1;
	unsigned int numberofvertextracks = 0;

	//IF THERE ARE SECONDARY VERTICES THE JET FALLS IN THE RECOVERTEX CATEGORY
	IterationRange range = flipIterate(svInfo.nVertices(), true);
	range_for(i, range) {
		if (vtx < 0) vtx = i; //RecoVertex category (vtx=0) if we enter at least one time in this loop!

		numberofvertextracks = numberofvertextracks + (svInfo.secondaryVertex(i)).nTracks();
			
		const Vertex &vertex = svInfo.secondaryVertex(i);
		bool hasRefittedTracks = vertex.hasRefittedTracks();
		TrackRefVector tracks = svInfo.vertexTracks(i);
		for(TrackRefVector::const_iterator track = tracks.begin(); track != tracks.end(); track++) {
			double w = svInfo.trackWeight(i, *track);
			if (w < minTrackWeight)
				continue;
			if (hasRefittedTracks) {
				Track actualTrack = vertex.refittedTrack(*track);
				vars.insert(btau::trackEtaRel, etaRel(jetDir,actualTrack.momentum()), true);
				vertexKinematics.add(actualTrack, w);
			if(i==0)
				{
					math::XYZVector vtx_trackMom = actualTrack.momentum();
					vtx_track_ptSum += std::sqrt(std::pow(vtx_trackMom.X(),2) + std::pow(vtx_trackMom.Y(),2)); 
					vtx_track_ESum += std::sqrt(vtx_trackMom.Mag2() + ROOT::Math::Square(ParticleMasses::piPlus)); 	
				}
			} else { //THIS ONE IS TAKEN...
				vars.insert(btau::trackEtaRel, etaRel(jetDir,(*track)->momentum()), true);
				vertexKinematics.add(**track, w);
				if(i==0) // calculate this only for the first vertex
				{
					math::XYZVector vtx_trackMom = (*track)->momentum();
					vtx_track_ptSum += std::sqrt(std::pow(vtx_trackMom.X(),2) + std::pow(vtx_trackMom.Y(),2)); 
					vtx_track_ESum += std::sqrt(vtx_trackMom.Mag2() + std::pow(ParticleMasses::piPlus,2)); 
					}
			}
		}
	}

	if (vtx >= 0) {
		vtxType = btag::Vertices::RecoVertex;
		vars.insert(btau::flightDistance2dVal,flipValue(svInfo.flightDistance(vtx, true).value(),true),true);
		vars.insert(btau::flightDistance2dSig,flipValue(svInfo.flightDistance(vtx, true).significance(),true),true);
		vars.insert(btau::flightDistance3dVal,flipValue(svInfo.flightDistance(vtx, false).value(),true),true);
		vars.insert(btau::flightDistance3dSig,flipValue(svInfo.flightDistance(vtx, false).significance(),true),true);
		vars.insert(btau::vertexJetDeltaR,Geom::deltaR(svInfo.flightDirection(vtx), jetDir),true);
		vars.insert(btau::jetNSecondaryVertices, svInfo.nVertices(), true);
//		vars.insert(btau::vertexNTracks, svInfo.nVertexTracks(), true);
		vars.insert(btau::vertexNTracks, numberofvertextracks, true);	
		vars.insert(btau::vertexFitProb,(svInfo.secondaryVertex(vtx)).normalizedChi2(), true);
	}


	//NOW ATTEMPT TO RECONSTRUCT PSEUDOVERTEX!!!
	std::vector<std::size_t> indices = ipInfo.sortedIndexes(sortCriterium);
	const std::vector<TrackIPTagInfo::TrackIPData> &ipData = ipInfo.impactParameterData();
	const edm::RefVector<TrackCollection> &tracks = ipInfo.selectedTracks();
  
	std::vector<TrackRef> pseudoVertexTracks;

	range = flipIterate(indices.size(), false);
	range_for(i, range) {
		std::size_t idx = indices[i];
		const TrackIPTagInfo::TrackIPData &data = ipData[idx];
		const TrackRef &trackRef = tracks[idx];
		const Track &track = *trackRef;

		// if no vertex was reconstructed, attempt pseudo vertex
		if (vtxType == btag::Vertices::NoVertex && trackPseudoSelector(track, data, *jet, pv)) {
			pseudoVertexTracks.push_back(trackRef);
			vertexKinematics.add(track);
		}
	}

	if (vtxType == btag::Vertices::NoVertex && vertexKinematics.numberOfTracks() >= pseudoMultiplicityMin && pseudoVertexV0Filter(pseudoVertexTracks)) 
	{ 
		vtxType = btag::Vertices::PseudoVertex;
		for(std::vector<TrackRef>::const_iterator track = pseudoVertexTracks.begin(); track != pseudoVertexTracks.end(); ++track)
		{
			vars.insert(btau::trackEtaRel, etaRel(jetDir, (*track)->momentum()), true);
			math::XYZVector vtx_trackMom = (*track)->momentum();
			vtx_track_ptSum += std::sqrt(std::pow(vtx_trackMom.X(),2) + std::pow(vtx_trackMom.Y(),2)); 
			vtx_track_ESum += std::sqrt(vtx_trackMom.Mag2() + std::pow(ParticleMasses::piPlus,2)); 
		}
	}

	vars.insert(btau::vertexCategory, vtxType, true);



	// do a tighter track selection to fill the variable plots...
	TrackRef trackPairV0Test[2];
	range = flipIterate(indices.size(), false);
	range_for(i, range) {

		std::size_t idx = indices[i];
		const TrackIPTagInfo::TrackIPData &data = ipData[idx];
		const TrackRef &trackRef = tracks[idx];
		const Track &track = *trackRef;

 		jet_track_ESum += std::sqrt((track.momentum()).Mag2() + std::pow(ParticleMasses::piPlus,2)); 
		// add track to kinematics for all tracks in jet
		//allKinematics.add(track); //would make more sense for some variables, e.g. vertexEnergyRatio nicely between 0 and 1, but not necessarily the best option for the discriminating power...
					
		// filter tracks -> this track selection can be more tight (used to fill the track related variables...)
		if (!trackSelector(track, data, *jet, pv))
			continue;

		// add track to kinematics for all tracks in jet
		allKinematics.add(track);

		// check against all other tracks for K0 track pairs setting the track mass to pi+
		trackPairV0Test[0] = tracks[idx];
		bool ok = true;
		range_for(j, range) {
			if (i == j)
				continue;

			std::size_t pairIdx = indices[j];
			const TrackIPTagInfo::TrackIPData &pairTrackData = ipData[pairIdx];
			const TrackRef &pairTrackRef = tracks[pairIdx];
			const Track &pairTrack = *pairTrackRef;

			if (!trackSelector(pairTrack, pairTrackData, *jet, pv))
				continue;

			trackPairV0Test[1] = pairTrackRef;
			if (!trackPairV0Filter(trackPairV0Test, 2)) { //V0 filter is more tight (0.03) than the one used for the RecoVertex and PseudoVertex tracks (0.05)
			ok = false;
			break;
			}
		}
		
		if (!ok)
			continue;

		// add track variables
		math::XYZVector trackMom = track.momentum();
		double trackMag = std::sqrt(trackMom.Mag2());

		vars.insert(btau::trackSip3dVal,  flipValue(data.ip3d.value(), false), true);
		vars.insert(btau::trackSip3dSig, flipValue(data.ip3d.significance(), false), true);
		vars.insert(btau::trackSip2dVal, flipValue(data.ip2d.value(), false), true);
		vars.insert(btau::trackSip2dSig, flipValue(data.ip2d.significance(), false), true);
		vars.insert(btau::trackJetDistVal, data.distanceToJetAxis.value(), true);
		vars.insert(btau::trackDecayLenVal, havePv ? (data.closestToJetAxis - pv).mag() : -1.0, true);

		vars.insert(btau::trackPtRel, VectorUtil::Perp(trackMom, jetDir), true);
		vars.insert(btau::trackPPar, jetDir.Dot(trackMom), true);
		vars.insert(btau::trackDeltaR, VectorUtil::DeltaR(trackMom, jetDir), true);
		vars.insert(btau::trackPtRatio, VectorUtil::Perp(trackMom, jetDir) / trackMag, true);
		vars.insert(btau::trackPParRatio, jetDir.Dot(trackMom) / trackMag, true);
	} 

	vars.insert(btau::trackSumJetDeltaR,VectorUtil::DeltaR(allKinematics.vectorSum(), jetDir), true);
	vars.insert(btau::trackSumJetEtRatio,allKinematics.vectorSum().Et() / ipInfo.jet()->et(), true);
	vars.insert(btau::trackSip3dSigAboveCharm, flipValue(threshTrack(ipInfo, TrackIPTagInfo::IP3DSig, *jet, pv).ip3d.significance(),false),true);
	vars.insert(btau::trackSip3dValAboveCharm, flipValue(threshTrack(ipInfo, TrackIPTagInfo::IP3DSig, *jet, pv).ip3d.value(),false),true);
	vars.insert(btau::trackSip2dSigAboveCharm, flipValue(threshTrack(ipInfo, TrackIPTagInfo::IP2DSig, *jet, pv).ip2d.significance(),false),true);
	vars.insert(btau::trackSip2dValAboveCharm, flipValue(threshTrack(ipInfo, TrackIPTagInfo::IP2DSig, *jet, pv).ip2d.value(),false),true);

	if (vtxType != btag::Vertices::NoVertex) {
		math::XYZTLorentzVector allSum = useTrackWeights ? allKinematics.weightedVectorSum() : allKinematics.vectorSum();
		math::XYZTLorentzVector vertexSum = useTrackWeights ? vertexKinematics.weightedVectorSum() : vertexKinematics.vectorSum();

		if (vtxType != btag::Vertices::RecoVertex) {
			vars.insert(btau::vertexNTracks,vertexKinematics.numberOfTracks(), true);
			vars.insert(btau::vertexJetDeltaR,VectorUtil::DeltaR(vertexSum, jetDir), true);
		}

		double vertexMass = vertexSum.M();
		double varPi = 0;
		double varB = 0;
		if (vtxType == btag::Vertices::RecoVertex) {
			if(vertexMassCorrection)
			{
				GlobalVector dir = svInfo.flightDirection(vtx);
				double vertexPt2 = math::XYZVector(dir.x(), dir.y(), dir.z()).Cross(vertexSum).Mag2() / dir.mag2();
				vertexMass = std::sqrt(vertexMass * vertexMass + vertexPt2) + std::sqrt(vertexPt2);
			}
		}
		vars.insert(btau::vertexMass, vertexMass, true);
		varPi = (vertexMass/5.2794) * (vtx_track_ESum /jet_track_ESum); //5.2794 should be average B meson mass of PDG! CHECK!!!
		vars.insert(btau::massVertexEnergyFraction, varPi, true);
		varB  = (std::sqrt(5.2794) * vtx_track_ptSum) / ( vertexMass * std::sqrt(jet->pt())); 
		vars.insert(btau::vertexBoostOverSqrtJetPt,varB*varB/(varB*varB + 10.), true);
		
		if (allKinematics.numberOfTracks())
			vars.insert(btau::vertexEnergyRatio, vertexSum.E() / allSum.E(), true);
		else
			vars.insert(btau::vertexEnergyRatio, 1, true);
	}

 	reco::PFJet const * pfJet = dynamic_cast<reco::PFJet const *>( &* jet ) ;
	pat::Jet const * patJet = dynamic_cast<pat::Jet const *>( &* jet ) ;
	if ( pfJet != 0 ) {
		vars.insert(btau::chargedHadronEnergyFraction,pfJet->chargedHadronEnergyFraction(), true);
		vars.insert(btau::neutralHadronEnergyFraction,pfJet->neutralHadronEnergyFraction(), true);
		vars.insert(btau::photonEnergyFraction,pfJet->photonEnergyFraction(), true);
		vars.insert(btau::electronEnergyFraction,pfJet->electronEnergyFraction(), true);
		vars.insert(btau::muonEnergyFraction,pfJet->muonEnergyFraction(), true);
		vars.insert(btau::chargedHadronMultiplicity,pfJet->chargedHadronMultiplicity(), true);
		vars.insert(btau::neutralHadronMultiplicity,pfJet->neutralHadronMultiplicity(), true);
		vars.insert(btau::photonMultiplicity,pfJet->photonMultiplicity(), true);
		vars.insert(btau::electronMultiplicity,pfJet->electronMultiplicity(), true);
		vars.insert(btau::muonMultiplicity,pfJet->muonMultiplicity(), true);
		vars.insert(btau::hadronMultiplicity,pfJet->chargedHadronMultiplicity()+pfJet->neutralHadronMultiplicity(), true);
		vars.insert(btau::hadronPhotonMultiplicity,pfJet->chargedHadronMultiplicity()+pfJet->neutralHadronMultiplicity()+pfJet->photonMultiplicity(), true);
			vars.insert(btau::totalMultiplicity,pfJet->chargedHadronMultiplicity()+pfJet->neutralHadronMultiplicity()+pfJet->photonMultiplicity()+pfJet->electronMultiplicity()+pfJet->muonMultiplicity(), true);

	}
	else if( patJet != 0)
	{
		vars.insert(btau::chargedHadronEnergyFraction,patJet->chargedHadronEnergyFraction(), true);
		vars.insert(btau::neutralHadronEnergyFraction,patJet->neutralHadronEnergyFraction(), true);
		vars.insert(btau::photonEnergyFraction,patJet->photonEnergyFraction(), true);
		vars.insert(btau::electronEnergyFraction,patJet->electronEnergyFraction(), true);
		vars.insert(btau::muonEnergyFraction,patJet->muonEnergyFraction(), true);
		vars.insert(btau::chargedHadronMultiplicity,patJet->chargedHadronMultiplicity(), true);
		vars.insert(btau::neutralHadronMultiplicity,patJet->neutralHadronMultiplicity(), true);
		vars.insert(btau::photonMultiplicity,patJet->photonMultiplicity(), true);
		vars.insert(btau::electronMultiplicity,patJet->electronMultiplicity(), true);
		vars.insert(btau::muonMultiplicity,patJet->muonMultiplicity(), true);
		vars.insert(btau::hadronMultiplicity,patJet->chargedHadronMultiplicity()+patJet->neutralHadronMultiplicity(), true);
		vars.insert(btau::hadronPhotonMultiplicity,patJet->chargedHadronMultiplicity()+patJet->neutralHadronMultiplicity()+patJet->photonMultiplicity(), true);
			vars.insert(btau::totalMultiplicity,patJet->chargedHadronMultiplicity()+patJet->neutralHadronMultiplicity()+patJet->photonMultiplicity()+patJet->electronMultiplicity()+patJet->muonMultiplicity(), true);
	
	}
	else
	{
		throw cms::Exception("InvalidConfiguration") << "From CombinedSVComputerV2::operator: reco::PFJet OR pat::Jet are required by this module" << std::endl;	
	}
 
	vars.finalize();

	return vars;
}
