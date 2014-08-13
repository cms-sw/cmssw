#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputerV2.h"

using namespace reco;


CombinedSVComputerV2::CombinedSVComputerV2(const edm::ParameterSet &params) :
	CombinedSVComputer(params)
{
	clearTaggingVariables();

	// define used TaggingVariables
	useTaggingVariable(btau::jetPt);
	useTaggingVariable(btau::jetEta);
	
	useTaggingVariable(btau::jetNTracks);
	useTaggingVariable(btau::trackSip3dVal);
	useTaggingVariable(btau::trackSip3dSig);
	useTaggingVariable(btau::trackSip2dVal);
	useTaggingVariable(btau::trackSip2dSig);
	useTaggingVariable(btau::trackJetDistVal);
	useTaggingVariable(btau::trackDecayLenVal);
	useTaggingVariable(btau::trackPtRel);
	useTaggingVariable(btau::trackPPar);
	useTaggingVariable(btau::trackDeltaR);
	useTaggingVariable(btau::trackPtRatio);
	useTaggingVariable(btau::trackPParRatio);
	useTaggingVariable(btau::trackSumJetDeltaR);
	useTaggingVariable(btau::trackSumJetEtRatio);
	useTaggingVariable(btau::trackSip3dSigAboveCharm);
	useTaggingVariable(btau::trackSip3dValAboveCharm);
	useTaggingVariable(btau::trackSip2dSigAboveCharm);
	useTaggingVariable(btau::trackSip2dValAboveCharm);
	useTaggingVariable(btau::trackJetPt);
	
	useTaggingVariable(btau::vertexCategory);
	useTaggingVariable(btau::trackEtaRel);
	useTaggingVariable(btau::vertexJetDeltaR);
	useTaggingVariable(btau::jetNSecondaryVertices);
	useTaggingVariable(btau::vertexNTracks);
	useTaggingVariable(btau::vertexMass);
	useTaggingVariable(btau::vertexEnergyRatio);
	useTaggingVariable(btau::flightDistance2dVal);
	useTaggingVariable(btau::flightDistance2dSig);
	useTaggingVariable(btau::flightDistance3dVal);
	useTaggingVariable(btau::flightDistance3dSig);
	useTaggingVariable(btau::vertexFitProb);
	useTaggingVariable(btau::massVertexEnergyFraction);
	useTaggingVariable(btau::vertexBoostOverSqrtJetPt);
	
	// sort TaggingVariables for faster lookup later
	sortTaggingVariables();
}
