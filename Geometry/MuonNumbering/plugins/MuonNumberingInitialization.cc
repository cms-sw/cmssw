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
// $Id: MuonNumberingInitialization.cc,v 1.4 2012/10/18 12:47:41 sunanda Exp $
//
//


// system include files
#include <memory>
#include <boost/shared_ptr.hpp>

// user include files
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/Framework/interface/ESTransientHandle.h>
//#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>
#include <Geometry/MuonNumbering/interface/MuonDDDConstants.h>
#include <Geometry/Records/interface/MuonNumberingRecord.h>

#define LOCAL_DEBUG

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
};

MuonNumberingInitialization::MuonNumberingInitialization(const edm::ParameterSet& iConfig) : muonDDDConst_(0)
{
  //  std::cout <<"constructing MuonNumberingInitialization" << std::endl;
  setWhatProduced(this, dependsOn(&MuonNumberingInitialization::initializeMuonDDDConstants));
}


MuonNumberingInitialization::~MuonNumberingInitialization()
{  }


// ------------ method called to produce the data  ------------
MuonNumberingInitialization::ReturnType
MuonNumberingInitialization::produce(const MuonNumberingRecord& iRecord)
{
#ifdef LOCAL_DEBUG
  std::cout << "in MuonNumberingInitialization::produce" << std::endl;
#endif
   using namespace edm::es;
   if ( muonDDDConst_ == 0 ) {
     std::cerr << "MuonNumberingInitialization::produceMuonDDDConstants has NOT been initialized!" << std::endl;
     throw;
   }
   return std::auto_ptr<MuonDDDConstants> (muonDDDConst_) ;
}

void MuonNumberingInitialization::initializeMuonDDDConstants( const IdealGeometryRecord& igr ) {

   edm::ESTransientHandle<DDCompactView> pDD;
   igr.get(label_, pDD );
#ifdef LOCAL_DEBUG
   std::cout << "in MuonNumberingInitialization::initializeMuonDDDConstants" << std::endl;
#endif
   if ( muonDDDConst_ != 0 ) {
     delete muonDDDConst_;
   }
#ifdef LOCAL_DEBUG
   std::cout << "about to make my new muonDDDConst_" << std::endl;
#endif
   muonDDDConst_ = new MuonDDDConstants( *pDD );
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(MuonNumberingInitialization);
