/*
 * =====================================================================================
 *       Filename:  EvtPlaneFilter.cc
 *    Description:  Event plane Q2 filter
 *        Created:  05/11/15 14:37:29
 *         Author:  Quan Wang
 * =====================================================================================
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

class EvtPlaneFilter : public edm::stream::EDFilter<> {
public:
	explicit EvtPlaneFilter(const edm::ParameterSet&);
	~EvtPlaneFilter();
private:
	virtual bool filter(edm::Event&, const edm::EventSetup&) override;

	const double vnlow_;
	const double vnhigh_;
	const int epidx_;
	const int eplvl_;
	edm::EDGetTokenT<reco::EvtPlaneCollection> tag_;
};

EvtPlaneFilter::EvtPlaneFilter(const edm::ParameterSet& ps):
	vnlow_(ps.getParameter<double>("Vnlow")),
	vnhigh_(ps.getParameter<double>("Vnhigh")),
	epidx_(ps.getParameter<int>("EPidx")),
	eplvl_(ps.getParameter<int>("EPlvl"))
{
	tag_ = consumes<reco::EvtPlaneCollection>( ps.getParameter<edm::InputTag>("EPlabel") );
	return;
}

EvtPlaneFilter::~EvtPlaneFilter()
{
	return;
}

bool EvtPlaneFilter::filter(edm::Event& evt, const edm::EventSetup& es)
{
	edm::Handle<reco::EvtPlaneCollection> ep_;
	evt.getByToken(tag_, ep_);
	double qn = (*ep_)[epidx_].vn(eplvl_);
	if ( qn < vnlow_ || qn > vnhigh_ ) return false;
	return true;
}

DEFINE_FWK_MODULE(EvtPlaneFilter);
