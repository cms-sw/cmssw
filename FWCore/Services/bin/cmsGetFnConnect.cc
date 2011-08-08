// -*- C++ -*-
//
// Package:     Utilities
// Class  :     cmsGetFnConnect
// 
// Implementation:
//     Looks up a frontier connect string 
//
// Original Author:  Dave Dykstra
//         Created:  Tue Feb 22 16:54:06 CST 2011
// $Id: cmsGetFnConnect.cc,v 1.2 2008/10/31 20:37:39 wmtan Exp $
//

#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include <iostream>
#include <string.h>

int
main(int argc, char* argv[])
{
    if ((argc != 2) || (strncmp(argv[1], "frontier://", 11) != 0))
    {
	std::cerr << "Usage: cmsGetFnConnect frontier://shortname" << std::endl;
	return 2;
    }

    std::auto_ptr<edm::SiteLocalConfig> slcptr(new edm::service::SiteLocalConfigService(edm::ParameterSet()));
    boost::shared_ptr<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> > slc(new edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig>(slcptr));
    edm::ServiceToken slcToken = edm::ServiceRegistry::createContaining(slc);
    edm::ServiceRegistry::Operate operate(slcToken);

    edm::Service<edm::SiteLocalConfig> localconfservice;

    std::cout << localconfservice->lookupCalibConnect(argv[1]) << std::endl;

    return 0;
}
