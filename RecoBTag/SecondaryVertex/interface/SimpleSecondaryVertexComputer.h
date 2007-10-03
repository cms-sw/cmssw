#ifndef RecoBTag_SecondaryVertex_SimpleSecondaryVertexComputer_h
#define RecoBTag_SecondaryVertex_SimpleSecondaryVertexComputer_h

#include <cmath>

#include "Math/GenVector/VectorUtil.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

#include "RecoBTag/SecondaryVertex/interface/TrackKinematics.h"

class SimpleSecondaryVertexComputer : public JetTagComputer {
    public:
	SimpleSecondaryVertexComputer(const edm::ParameterSet &parameters) :
		use2d(!parameters.getParameter<bool>("use3d")),
		useSig(parameters.getParameter<bool>("useSignificance")),
		unBoost(parameters.getParameter<bool>("unBoost"))
	{}

	float discriminator(const reco::BaseTagInfo &baseInfo) const
	{
		const reco::SecondaryVertexTagInfo *info =
			dynamic_cast<const reco::SecondaryVertexTagInfo*>(&baseInfo);
		if (!info)
			return -1.0; // FIXME: report some error?
		if (info->nVertices() == 0)
			return -1.0;

		double gamma;
		if (unBoost) {
			TrackKinematics kinematics(info->secondaryVertex(0));
			gamma = kinematics.vectorSum().M()
			      / kinematics.vectorSum().mag();
		} else
			gamma = 1.0;

		double value;
		if (useSig) {
			value = info->flightDistance(0, use2d).significance();
			value = (value > 0) ? +std::log(1 + value)
			                    : -std::log(1 - value);
		} else
			value = info->flightDistance(0, use2d).value();

		return value / gamma;
	}

    private:
	bool	use2d;
	bool	useSig;
	bool	unBoost;
};

#endif // RecoBTag_SecondaryVertex_SimpleSecondaryVertexComputer_h
