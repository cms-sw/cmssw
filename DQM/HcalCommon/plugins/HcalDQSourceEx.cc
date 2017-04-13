// cmssw includes
#include "DQM/HcalCommon/interface/HcalDQSourceEx.h"

//	system includes
#include <iostream>


HcalDQSourceEx::HcalDQSourceEx(edm::ParameterSet const& ps) :
	hcaldqm::HcalDQSource(ps)
{}

/* virtual */ HcalDQSourceEx::~HcalDQSourceEx()
{}

/* virtual */ void HcalDQSourceEx::doWork(edm::Event const& e,
		edm::EventSetup const& es)
{
	std::cout << "Running Module: " << _mi.name << std::endl;
}

DEFINE_FWK_MODULE(HcalDQSourceEx);
