// -*- C++ -*-
//
// Package:    CaloEventSetup
// Class:      ShashlikTopologyBuilder
// 
/**\class ShashlikTopologyBuilder ShashlikTopologyBuilder.h 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
// $Id: ShashlikTopologyBuilder.h,v 1.3 2014/03/26 17:56:34 sunanda Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/ShashlikNumberingRecord.h"
#include "Geometry/CaloTopology/interface/ShashlikTopology.h"

//
// class decleration
//

class ShashlikTopologyBuilder : public edm::ESProducer {

public:
  ShashlikTopologyBuilder( const edm::ParameterSet& iP );
  ~ShashlikTopologyBuilder() ;

  typedef boost::shared_ptr< ShashlikTopology > ReturnType;

  ReturnType produce(const ShashlikNumberingRecord&);

private:
  // ----------member data ---------------------------
};

