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
	{ uses("svTagInfos"); }

	float discriminator(const TagInfoHelper &tagInfos) const
	{
		const reco::SecondaryVertexTagInfo &info =
				tagInfos.get<reco::SecondaryVertexTagInfo>();
		if (info.nVertices() == 0)
			return -1.0;

		double gamma;
		if (unBoost) {
			reco::TrackKinematics kinematics(
						info.secondaryVertex(0));
			gamma = kinematics.vectorSum().Gamma();
		} else
			gamma = 1.0;

		double value;
		if (useSig)
			value = info.flightDistance(0, use2d).significance();
		else
			value = info.flightDistance(0, use2d).value();

		value /= gamma;

		if (useSig)
			value = (value > 0) ? +std::log(1 + value)
			                    : -std::log(1 - value);

		return value;
	}

    private:
	bool	use2d;
	bool	useSig;
	bool	unBoost;
};

#endif // RecoBTag_SecondaryVertex_SimpleSecondaryVertexComputer_h
