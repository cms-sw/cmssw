// -*- C++ -*-
//
// Package:    CaloTowerTopologyBuilder
// Class:      CaloTowerTopologyBuilder
// 
/**\class CaloTowerTopologyBuilder CaloTowerTopologyBuilder.h tmp/CaloTowerTopologyBuilder/interface/CaloTowerTopologyBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
// $Id: CaloTowerTopologyBuilder.cc,v 1.4 2005/11/02 07:55:24 meridian Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"

//
// class decleration
//

class CaloTowerTopologyBuilder : public edm::ESProducer {
   public:
  CaloTowerTopologyBuilder(const edm::ParameterSet&);
  ~CaloTowerTopologyBuilder();

  typedef std::auto_ptr<CaloTowerTopology> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);
private:
      // ----------member data ---------------------------
};

