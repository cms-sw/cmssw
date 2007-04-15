// -*- C++ -*-
//
// Package:    MuonNumberingInitialization
// Class:      MuonNumberingInitialization
// 
/**\class MuonNumberingInitialization MuonNumberingInitialization.h Geometry/MuonNumberingInitialization/interface/MuonNumberingInitialization.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Case
//         Created:  Thu Sep 28 16:40:29 PDT 2006
// $Id: MuonNumberingInitialization.cc,v 1.3 2006/10/27 01:35:29 wmtan Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>
#include <Geometry/MuonNumbering/interface/MuonDDDConstants.h>
#include <Geometry/Records/interface/MuonNumberingRecord.h>


//
// class decleration
//

class MuonNumberingInitialization : public edm::ESProducer {
   public:
      MuonNumberingInitialization(const edm::ParameterSet&);
      ~MuonNumberingInitialization();

      typedef std::auto_ptr<MuonDDDConstants> ReturnType;

      ReturnType produce(const MuonNumberingRecord&);

      void initializeMuonDDDConstants( const IdealGeometryRecord& igr);

   private:
      std::string label_;
      MuonDDDConstants* muonDDDConst_;
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonNumberingInitialization::MuonNumberingInitialization(const edm::ParameterSet& iConfig) : muonDDDConst_(0)
{
   //the following line is needed to tell the framework what
   // data is being produced
  //   std::cout <<"constructing MuonNumberingInitialization" << std::endl;
   setWhatProduced(this, dependsOn(&MuonNumberingInitialization::initializeMuonDDDConstants));
   //   setWhatProduced(this, );
   //now do what ever other initialization is needed
}


MuonNumberingInitialization::~MuonNumberingInitialization()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
MuonNumberingInitialization::ReturnType
MuonNumberingInitialization::produce(const MuonNumberingRecord& iRecord)
{
  //  std::cout << "in MuonNumberingInitialization::produce" << std::endl;
   using namespace edm::es;
   if ( muonDDDConst_ == 0 ) {
     std::cerr << "MuonNumberingInitialization::produceMuonDDDConstants has NOT been initialized!" << std::endl;
     throw;
   }
   //   std::cout << "about to return the auto_pointer?" << std::endl;
   //   *pMuonDDDConstants = *muonDDDConst_;
   return std::auto_ptr<MuonDDDConstants> (muonDDDConst_) ;
}

void MuonNumberingInitialization::initializeMuonDDDConstants( const IdealGeometryRecord& igr ) {

   edm::ESHandle<DDCompactView> pDD;
   igr.get(label_, pDD );
   //   std::cout << "in MuonNumberingInitialization::initializeMuonDDDConstants" << std::endl;
  try {
    std::string attribute = "OnlyForMuonNumbering"; 
    std::string value     = "any";
    DDValue val(attribute, value, 0.0);

    DDSpecificsFilter filter;
    filter.setCriteria(val,
		       DDSpecificsFilter::not_equals,
		       DDSpecificsFilter::AND, 
		       true, // compare strings otherwise doubles
		       true  // use merged-specifics or simple-specifics
		       );
    DDFilteredView fview(*pDD);
    fview.addFilter(filter);

    DDValue val2("level");
    const DDsvalues_type params(fview.mergedSpecifics());
    //    std::string level;
    //    if (DDfetch(&params,val2)) level = val.strings()[0];   
    //    std::cout << "level " << level << std::endl;
    //    std::cout << fview.logicalPart().name() << std::endl;
    fview.firstChild();
    //    std::cout << "after next name is..." << fview.logicalPart().name() << std::endl;
    if ( muonDDDConst_ != 0 ) {
      delete muonDDDConst_;
    }
    //    std::cout << "about to make my new muonDDDConst_" << std::endl;
    muonDDDConst_ = new MuonDDDConstants();

//     std::string toName("level");
//     std::vector<const DDsvalues_type * > mySpecs = fview.specifics();
//     std::vector<const DDsvalues_type * >::const_iterator bit = mySpecs.begin();
//     std::vector<const DDsvalues_type * >::const_iterator eit = mySpecs.end();
    const DDsvalues_type mySpecs (fview.mergedSpecifics());
    DDsvalues_type::const_iterator bit = mySpecs.begin();
    DDsvalues_type::const_iterator eit = mySpecs.end();
    for ( ; bit != eit; ++bit ) {
      if ( bit->second.isEvaluated() ) {
	muonDDDConst_->addValue( bit->second.name(), int(bit->second.doubles()[0]) );
	//	std::cout << "adding DDConstant of " << bit->second.name() << " = " << int(bit->second.doubles()[0]) << std::endl;
      }
    }
//     std::cout << "loading it with one value" << std::endl;
//     muonDDDConst_->addValue(toName, 3);
//    std::cout << "ok, done with initialize" << std::endl;
  }
  catch (const DDException & e ) {
    std::cerr << "MuonNumberingInitialization::initializeMuonDDDConstants caught a DDD Exception: " << std::endl
	      << "  Message: " << e << std::endl
	      << "  Terminating execution ... " << std::endl;
    throw;
  }
  catch (const std::exception & e) {
    std::cerr << "MuonNumberingInitialization::initializeMuonDDDConstants : an std::exception occured: " << e.what() << std::endl; 
    throw;
  }
  catch (...) {
    std::cerr << "MuonNumberingInitialization::initializeMuonDDDConstants : An unexpected exception occured!" << std::endl
	      << "  Terminating execution ... " << std::endl;
    std::unexpected();           
  }
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(MuonNumberingInitialization);
