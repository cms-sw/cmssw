#ifndef RecoBTag_SecondaryVertex_CombinedSVComputer_h
#define RecoBTag_SecondaryVertex_CombinedSVComputer_h

#include <iostream>
#include <cstddef>
#include <string>
#include <cmath>
#include <vector>

#include <Math/VectorUtil.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/VertexTypes.h"
#include "DataFormats/BTauReco/interface/ParticleMasses.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"
#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"


#define range_for(i, x) \
        for(int i = (x).begin; i != (x).end; i += (x).increment)


class CombinedSVComputer {
    public:
	explicit CombinedSVComputer(const edm::ParameterSet &params);

	virtual reco::TaggingVariableList
	operator () (const reco::TrackIPTagInfo &ipInfo,
	             const reco::SecondaryVertexTagInfo &svInfo) const;
	virtual reco::TaggingVariableList
	operator () (const reco::CandIPTagInfo &ipInfo,
	             const reco::CandSecondaryVertexTagInfo &svInfo) const;
	
	struct IterationRange {
	        int begin, end, increment;
	};
	double flipValue(double value, bool vertex) const;
	IterationRange flipIterate(int size, bool vertex) const;
	
	edm::ParameterSet dropDeltaR(const edm::ParameterSet &pset) const;
	
	const reco::btag::TrackIPData &
	threshTrack(const reco::CandIPTagInfo &trackIPTagInfo,
	            const reco::btag::SortCriteria sort,
	            const reco::Jet &jet,
	            const GlobalPoint &pv) const;
	const reco::btag::TrackIPData &
	threshTrack(const reco::TrackIPTagInfo &trackIPTagInfo,
	            const reco::btag::SortCriteria sort,
	            const reco::Jet &jet,
	            const GlobalPoint &pv) const;
	template <class SVTI,class IPTI>
	void fillCommonVariables(reco::TaggingVariableList & vars, reco::TrackKinematics & vertexKinematics,
				 const IPTI & ipInfo,const SVTI & svInfo,
				 double & vtx_track_ptSum, double & vtx_track_ESum) const;

    private:
	bool					trackFlip;
	bool					vertexFlip;
	double					charmCut;
	reco::btag::SortCriteria		sortCriterium;
	reco::TrackSelector			trackSelector;
	reco::TrackSelector			trackNoDeltaRSelector;
	reco::TrackSelector			trackPseudoSelector;
	unsigned int				pseudoMultiplicityMin;
	unsigned int				trackMultiplicityMin;
	double					minTrackWeight;
	bool					useTrackWeights;
	bool					vertexMassCorrection;
	reco::V0Filter				pseudoVertexV0Filter;
	reco::V0Filter				trackPairV0Filter;
	std::vector<reco::btau::TaggingVariableName>	taggingVariables;
};

template <class SVTI,class IPTI>
void CombinedSVComputer::fillCommonVariables(reco::TaggingVariableList & vars, reco::TrackKinematics & vertexKinematics,
					     const IPTI & ipInfo, const SVTI & svInfo,
					     double & vtx_track_ptSum, double & vtx_track_ESum) const
{
        using namespace ROOT::Math;
        using namespace reco;

        edm::RefToBase<Jet> jet = ipInfo.jet();
        math::XYZVector jetDir = jet->momentum().Unit();
        bool havePv = ipInfo.primaryVertex().isNonnull();
        GlobalPoint pv;
        if (havePv)
                pv = GlobalPoint(ipInfo.primaryVertex()->x(),
                                 ipInfo.primaryVertex()->y(),
                                 ipInfo.primaryVertex()->z());

        btag::Vertices::VertexType vtxType = btag::Vertices::NoVertex;


        vars.insert(btau::jetPt, jet->pt(), true);
        vars.insert(btau::jetEta, jet->eta(), true);

        if (ipInfo.selectedTracks().size() < trackMultiplicityMin)
                return;

	vars.insert(btau::jetNTracks, ipInfo.selectedTracks().size(), true);
	
        TrackKinematics allKinematics;
	TrackKinematics trackJetKinematics;
	
	double jet_track_ESum= 0.;
	
	int vtx = -1;
	
	IterationRange range = flipIterate(svInfo.nVertices(), true);
	range_for(i , range) {
		if (vtx < 0) vtx = i;
	}

	if (vtx >= 0) {
		vtxType = btag::Vertices::RecoVertex;
		
		vars.insert(btau::flightDistance2dVal,flipValue(svInfo.flightDistance(vtx, true).value(),true),true);
		vars.insert(btau::flightDistance2dSig,flipValue(svInfo.flightDistance(vtx, true).significance(),true),true);
		vars.insert(btau::flightDistance3dVal,flipValue(svInfo.flightDistance(vtx, false).value(),true),true);
		vars.insert(btau::flightDistance3dSig,flipValue(svInfo.flightDistance(vtx, false).significance(),true),true);
		vars.insert(btau::vertexJetDeltaR,Geom::deltaR(svInfo.flightDirection(vtx), jetDir),true);
		vars.insert(btau::jetNSecondaryVertices, svInfo.nVertices(), true);
	}

	std::vector<std::size_t> indices = ipInfo.sortedIndexes(sortCriterium);
	const std::vector<reco::btag::TrackIPData> &ipData = ipInfo.impactParameterData();

	const typename IPTI::input_container &tracks = ipInfo.selectedTracks();
	std::vector<const Track *> pseudoVertexTracks;

        const Track * trackPairV0Test[2];
        range = flipIterate(indices.size(), false);
        range_for(i, range) {
                std::size_t idx = indices[i];
                const reco::btag::TrackIPData &data = ipData[idx];
                const Track * trackPtr = reco::btag::toTrack(tracks[idx]);
                const Track &track = *trackPtr;

                jet_track_ESum += std::sqrt(track.momentum().Mag2() + ROOT::Math::Square(ParticleMasses::piPlus));

                // add track to kinematics for all tracks in jet
                //allKinematics.add(track); // would make more sense for some variables, e.g. vertexEnergyRatio nicely between 0 and 1, but not necessarily the best option for the discriminating power...

                // filter track -> this track selection can be tighter than the vertex track selection (used to fill the track related variables...)
                if (!trackSelector(track, data, *jet, pv))
                        continue;

                // add track to kinematics for all tracks in jet
                allKinematics.add(track);

                // if no vertex was reconstructed, attempt pseudo vertex
                if (vtxType == btag::Vertices::NoVertex && trackPseudoSelector(track, data, *jet, pv)) {
                        pseudoVertexTracks.push_back(trackPtr);
                        vertexKinematics.add(track);
                }

                // check against all other tracks for V0 track pairs
                trackPairV0Test[0] = reco::btag::toTrack(tracks[idx]);
                bool ok = true;
                range_for(j, range) {
                        if (i == j)
                                continue;

                        std::size_t pairIdx = indices[j];
                        const reco::btag::TrackIPData &pairTrackData = ipData[pairIdx];
                        const Track * pairTrackPtr = reco::btag::toTrack(tracks[pairIdx]);
                        const Track &pairTrack = *pairTrackPtr;

                        if (!trackSelector(pairTrack, pairTrackData, *jet, pv))
                                continue;

                        trackPairV0Test[1] = pairTrackPtr;
                        if (!trackPairV0Filter(trackPairV0Test, 2)) {
                                ok = false;
                                break;
                        }
                }
                if (!ok)
                        continue;

                trackJetKinematics.add(track);

                // add track variables
                math::XYZVector trackMom = track.momentum();
                double trackMag = std::sqrt(trackMom.Mag2());

                vars.insert(btau::trackSip3dVal, flipValue(data.ip3d.value(), false), true);
                vars.insert(btau::trackSip3dSig, flipValue(data.ip3d.significance(), false), true);
                vars.insert(btau::trackSip2dVal, flipValue(data.ip2d.value(), false), true);
                vars.insert(btau::trackSip2dSig, flipValue(data.ip2d.significance(), false), true);
                vars.insert(btau::trackJetDistVal, data.distanceToJetAxis.value(), true);
//              vars.insert(btau::trackJetDistSig, data.distanceToJetAxis.significance(), true);
//              vars.insert(btau::trackFirstTrackDist, data.distanceToFirstTrack, true);
//              vars.insert(btau::trackGhostTrackVal, data.distanceToGhostTrack.value(), true);
//              vars.insert(btau::trackGhostTrackSig, data.distanceToGhostTrack.significance(), true);
                vars.insert(btau::trackDecayLenVal, havePv ? (data.closestToJetAxis - pv).mag() : -1.0, true);

                vars.insert(btau::trackMomentum, trackMag, true);
                vars.insert(btau::trackEta, trackMom.Eta(), true);

                vars.insert(btau::trackPtRel, VectorUtil::Perp(trackMom, jetDir), true);
                vars.insert(btau::trackPPar, jetDir.Dot(trackMom), true);
                vars.insert(btau::trackDeltaR, VectorUtil::DeltaR(trackMom, jetDir), true);
                vars.insert(btau::trackPtRatio, VectorUtil::Perp(trackMom, jetDir) / trackMag, true);
                vars.insert(btau::trackPParRatio, jetDir.Dot(trackMom) / trackMag, true);
        }

        if (vtxType == btag::Vertices::NoVertex && vertexKinematics.numberOfTracks() >= pseudoMultiplicityMin && pseudoVertexV0Filter(pseudoVertexTracks))
        {
                vtxType = btag::Vertices::PseudoVertex;
                for(std::vector<const Track *>::const_iterator track = pseudoVertexTracks.begin(); track != pseudoVertexTracks.end(); ++track)
                {
                        vars.insert(btau::trackEtaRel, reco::btau::etaRel(jetDir,(*track)->momentum()), true);
                        vtx_track_ptSum += std::sqrt((*track)->momentum().Perp2());
                        vtx_track_ESum  += std::sqrt((*track)->momentum().Mag2() + ROOT::Math::Square(ParticleMasses::piPlus));
                }
        }

	vars.insert(btau::vertexCategory, vtxType, true);
	
	vars.insert(btau::trackJetPt, trackJetKinematics.vectorSum().Pt(), true);
	vars.insert(btau::trackSumJetDeltaR,VectorUtil::DeltaR(allKinematics.vectorSum(), jetDir), true);
	vars.insert(btau::trackSumJetEtRatio,allKinematics.vectorSum().Et() / ipInfo.jet()->et(), true);
	
	vars.insert(btau::trackSip3dSigAboveCharm, flipValue(threshTrack(ipInfo, reco::btag::IP3DSig, *jet, pv).ip3d.significance(),false),true);
	vars.insert(btau::trackSip3dValAboveCharm, flipValue(threshTrack(ipInfo, reco::btag::IP3DSig, *jet, pv).ip3d.value(),false),true);
	vars.insert(btau::trackSip2dSigAboveCharm, flipValue(threshTrack(ipInfo, reco::btag::IP2DSig, *jet, pv).ip2d.significance(),false),true);
	vars.insert(btau::trackSip2dValAboveCharm, flipValue(threshTrack(ipInfo, reco::btag::IP2DSig, *jet, pv).ip2d.value(),false),true);

        if (vtxType != btag::Vertices::NoVertex) {
                math::XYZTLorentzVector allSum = useTrackWeights
                        ? allKinematics.weightedVectorSum()
                        : allKinematics.vectorSum();
                math::XYZTLorentzVector vertexSum = useTrackWeights
                        ? vertexKinematics.weightedVectorSum()
                        : vertexKinematics.vectorSum();

                if (vtxType != btag::Vertices::RecoVertex) {
                        vars.insert(btau::vertexNTracks,vertexKinematics.numberOfTracks(), true);
                        vars.insert(btau::vertexJetDeltaR,VectorUtil::DeltaR(vertexSum, jetDir), true);
                }

                double vertexMass = vertexSum.M();
                if (vtxType == btag::Vertices::RecoVertex &&
                    vertexMassCorrection) {
                        GlobalVector dir = svInfo.flightDirection(vtx);
                        double vertexPt2 = math::XYZVector(dir.x(), dir.y(), dir.z()).Cross(vertexSum).Mag2() / dir.mag2();
                        vertexMass = std::sqrt(vertexMass * vertexMass + vertexPt2) + std::sqrt(vertexPt2);
                }
                vars.insert(btau::vertexMass, vertexMass, true);

                double varPi = (vertexMass/5.2794) * (vtx_track_ESum /jet_track_ESum); // 5.2794 should be the average B meson mass of PDG! CHECK!!!
                vars.insert(btau::massVertexEnergyFraction, varPi / (varPi + 0.04), true);
                double varB  = (std::sqrt(5.2794) * vtx_track_ptSum) / ( vertexMass * std::sqrt(jet->pt()));
                vars.insert(btau::vertexBoostOverSqrtJetPt,varB*varB/(varB*varB + 10.), true);

                if (allKinematics.numberOfTracks()) {
                        vars.insert(btau::vertexEnergyRatio, vertexSum.E() / allSum.E(), true);
                }
                else {
                        vars.insert(btau::vertexEnergyRatio, 1, true);
                }
        }

	reco::PFJet const * pfJet = dynamic_cast<reco::PFJet const *>( &* jet ) ;
	pat::Jet const * patJet = dynamic_cast<pat::Jet const *>( &* jet ) ;
	if ( pfJet != 0 ) 
	{
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
	else if( patJet != 0 && patJet->isPFJet() )
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
		vars.insert(btau::chargedHadronEnergyFraction,0., true);
		vars.insert(btau::neutralHadronEnergyFraction,0., true);
		vars.insert(btau::photonEnergyFraction,0., true);
		vars.insert(btau::electronEnergyFraction,0., true);
		vars.insert(btau::muonEnergyFraction,0., true);
		vars.insert(btau::chargedHadronMultiplicity,0, true);
		vars.insert(btau::neutralHadronMultiplicity,0, true);
		vars.insert(btau::photonMultiplicity,0, true);
		vars.insert(btau::electronMultiplicity,0, true);
		vars.insert(btau::muonMultiplicity,0, true);
		vars.insert(btau::hadronMultiplicity,0, true);
		vars.insert(btau::hadronPhotonMultiplicity,0, true);
		vars.insert(btau::totalMultiplicity,0, true);
	}
}


#endif // RecoBTag_SecondaryVertex_CombinedSVComputer_h
