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
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

//
// class decleration
//

class CaloTopologyBuilder : public edm::ESProducer 
{
   public:
      CaloTopologyBuilder( const edm::ParameterSet& iP );
      ~CaloTopologyBuilder() ;

      typedef boost::shared_ptr< CaloTopology > ReturnType;

      ReturnType produceCalo(  const CaloTopologyRecord&  );

   private:
      // ----------member data ---------------------------
};

