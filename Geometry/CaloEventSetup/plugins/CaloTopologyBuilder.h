// -*- C++ -*-
//
// Package:    CaloTopologyBuilder
// Class:      CaloTopologyBuilder
// 
/**\class CaloTopologyBuilder CaloTopologyBuilder.h 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Paolo Meridiani
// $Id: CaloTopologyBuilder.h,v 1.1 2006/03/30 14:48:47 meridian Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

//
// class decleration
//

class CaloTopologyBuilder : public edm::ESProducer {
   public:
  CaloTopologyBuilder(const edm::ParameterSet&);
  ~CaloTopologyBuilder();

  typedef std::auto_ptr<CaloTopology> ReturnType;

  ReturnType produce(const CaloTopologyRecord&);
private:
      // ----------member data ---------------------------
};

