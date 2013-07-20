// -*- C++ -*-
//
// Package:    L1GeometryProducers
// Class:      L1CaloGeometryProd
// 
/**\class L1CaloGeometryProd L1CaloGeometryProd.h L1TriggerConfig/L1GeometryProducers/interface/L1CaloGeometryProd.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Sun
//         Created:  Tue Oct 24 00:00:00 EDT 2006
// $Id: L1CaloGeometryProd.cc,v 1.4 2009/09/28 23:02:20 wsun Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "L1TriggerConfig/L1GeometryProducers/interface/L1CaloGeometryProd.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1CaloGeometryProd::L1CaloGeometryProd(const edm::ParameterSet& ps)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed

   // This producer should never make more than one version of L1Geometry,
   // so we can initialize it in the ctor.
   m_geom =
     L1CaloGeometry( ps.getParameter<unsigned int>("numberGctEmJetPhiBins"),
		     ps.getParameter<double>("gctEmJetPhiBinOffset"),
		     ps.getParameter<unsigned int>("numberGctEtSumPhiBins"),
		     ps.getParameter<double>("gctEtSumPhiBinOffset"),
		     ps.getParameter<unsigned int>("numberGctHtSumPhiBins"),
		     ps.getParameter<double>("gctHtSumPhiBinOffset"),
		     ps.getParameter<unsigned int>("numberGctCentralEtaBinsPerHalf"),
		     ps.getParameter<unsigned int>("numberGctForwardEtaBinsPerHalf"),
		     ps.getParameter<unsigned int>("etaSignBitOffset"),
		     ps.getParameter< std::vector<double> >("gctEtaBinBoundaries") ) ;
}


L1CaloGeometryProd::~L1CaloGeometryProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1CaloGeometryProd::ReturnType
L1CaloGeometryProd::produce(const L1CaloGeometryRecord& iRecord)
{
   using namespace edm::es;
   std::auto_ptr<L1CaloGeometry> pL1CaloGeometry ;

   pL1CaloGeometry = std::auto_ptr< L1CaloGeometry >(
      new L1CaloGeometry( m_geom ) ) ;

   return pL1CaloGeometry ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1CaloGeometryProd) ;
