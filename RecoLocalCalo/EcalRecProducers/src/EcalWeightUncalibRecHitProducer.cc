/** \class EcalWeightUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes
 *
  *  $Id: EcalWeightUncalibRecHitProducer.cc,v 1.12 2006/05/05 08:49:12 meridian Exp $
  *  $Date: 2006/05/05 08:49:12 $
  *  $Revision: 1.12 $
  *  \author Shahram Rahatlou, University of Rome & INFN, Sept 2005
  *
  */
#include "RecoLocalCalo/EcalRecProducers/interface/EcalWeightUncalibRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "FWCore/Framework/interface/Handle.h"

#include <iostream>
#include <iomanip>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "CondFormats/EcalObjects/interface/EcalWeightRecAlgoWeights.h"
#include "CondFormats/DataRecord/interface/EcalWeightRecAlgoWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include <vector>

EcalWeightUncalibRecHitProducer::EcalWeightUncalibRecHitProducer(const edm::ParameterSet& ps) {

   EBdigiCollection_ = ps.getParameter<std::string>("EBdigiCollection");
   EEdigiCollection_ = ps.getParameter<std::string>("EEdigiCollection");
   digiProducer_   = ps.getParameter<std::string>("digiProducer");
   EBhitCollection_  = ps.getParameter<std::string>("EBhitCollection");
   EEhitCollection_  = ps.getParameter<std::string>("EEhitCollection");
   produces< EBUncalibratedRecHitCollection >(EBhitCollection_);
   produces< EEUncalibratedRecHitCollection >(EEhitCollection_);
}

EcalWeightUncalibRecHitProducer::~EcalWeightUncalibRecHitProducer() {
}

void
EcalWeightUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

   using namespace edm;

   Handle< EBDigiCollection > pEBDigis;
   Handle< EEDigiCollection > pEEDigis;

   const EBDigiCollection* EBdigis =0;
   const EEDigiCollection* EEdigis =0;

   try {
     //     evt.getByLabel( digiProducer_, EBdigiCollection_, pEBDigis);
     evt.getByLabel( digiProducer_, pEBDigis);
     EBdigis = pEBDigis.product(); // get a ptr to the produc
     edm::LogInfo("EcalUncalibRecHitInfo") << "total # EBdigis: " << EBdigis->size() ;
   } catch ( std::exception& ex ) {
     // edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EBdigiCollection_.c_str() ;
   }

   try {
     //     evt.getByLabel( digiProducer_, EEdigiCollection_, pEEDigis);
     evt.getByLabel( digiProducer_, pEEDigis);
     EEdigis = pEEDigis.product(); // get a ptr to the product
     edm::LogInfo("EcalUncalibRecHitInfo") << "total # EEdigis: " << EEdigis->size() ;
   } catch ( std::exception& ex ) {
     //edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EEdigiCollection_.c_str() ;
   }

    // fetch map of groups of xtals
    edm::ESHandle<EcalWeightXtalGroups> pGrp;
    es.get<EcalWeightXtalGroupsRcd>().get(pGrp);
    const EcalWeightXtalGroups* grp = pGrp.product();

   // Gain Ratios
   //edm::ESHandle<EcalGainRatios> pRatio;
   //es.get<EcalGainRatiosRcd>().get(pRatio);
   //const EcalGainRatios* gr = pRatio.product();

   // fetch TB weights
   LogDebug("EcalUncalibRecHitDebug") <<"Fetching EcalTBWeights from DB " ;
   edm::ESHandle<EcalTBWeights> pWgts;
   es.get<EcalTBWeightsRcd>().get(pWgts);
   const EcalTBWeights* wgts = pWgts.product();
   LogDebug("EcalUncalibRecHitDebug") << "EcalTBWeightMap.size(): " << std::setprecision(3) << wgts->getMap().size() ;


   // fetch the pedestals from the cond DB via EventSetup
   LogDebug("EcalUncalibRecHitDebug") << "fetching pedestals....";
   edm::ESHandle<EcalPedestals> pedHandle;
   es.get<EcalPedestalsRcd>().get( pedHandle );
   const EcalPedestalsMap& pedMap = pedHandle.product()->m_pedestals; // map of pedestals
   LogDebug("EcalUncalibRecHitDebug") << "done." ;

   // collection of reco'ed ampltudes to put in the event

   std::auto_ptr< EBUncalibratedRecHitCollection > EBuncalibRechits( new EBUncalibratedRecHitCollection );
   std::auto_ptr< EEUncalibratedRecHitCollection > EEuncalibRechits( new EEUncalibratedRecHitCollection );

   EcalPedestalsMapIterator pedIter; // pedestal iterator
   EcalPedestals::Item aped; // pedestal object for a single xtal

   // loop over EB digis
   if (EBdigis)
     {
       for(EBDigiCollection::const_iterator itdg = EBdigis->begin(); itdg != EBdigis->end(); ++itdg) {

	 //     counter_++; // verbosity counter

	 // find pedestals for this channel
	 LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << itdg->id() ;
	 pedIter = pedMap.find(itdg->id().rawId());
	 if( pedIter != pedMap.end() ) {
	   aped = pedIter->second;
	 } else {
	   edm::LogError("EcalUncalibRecHitError") << "error!! could not find pedestals for channel: " << itdg->id() 
						   << "\n  no uncalib rechit will be made for this digi!"
	     ;
	   continue;
	 }
	 std::vector<double> pedVec;
	 pedVec.push_back(aped.mean_x1);pedVec.push_back(aped.mean_x6);pedVec.push_back(aped.mean_x12);

	 // lookup group ID for this channel
	 EcalWeightXtalGroups::EcalXtalGroupsMap::const_iterator git = grp->getMap().find( itdg->id().rawId() );
	 EcalXtalGroupId gid;
	 if( git != grp->getMap().end() ) {
	   gid = git->second;
	 } else {
	   edm::LogError("EcalUncalibRecHitError") << "No group id found for this crystal. something wrong with EcalWeightXtalGroups in your DB?"
						   << "\n  no uncalib rechit will be made for digi with id: " << itdg->id()
	     ;
	   continue;
	 }

	 // use a fake TDC iD for now until it become available in raw data
	 EcalTBWeights::EcalTDCId tdcid(1);

	 // now lookup the correct weights in the map
	 EcalTBWeights::EcalTBWeightMap::const_iterator wit = wgts->getMap().find( std::make_pair(gid,tdcid) );
	 if( wit == wgts->getMap().end() ) {  // no weights found for this group ID
	   edm::LogError("EcalUncalibRecHitError") << "No weights found for EcalGroupId: " << gid.id() << " and  EcalTDCId: " << tdcid
						   << "\n  skipping digi with id: " << itdg->id()
	     ;
	   continue;
	 }

	 EcalWeightSet  wset = wit->second; // this is the EcalWeightSet

	 // EcalWeightMatrix is vec<vec:double>>
	 LogDebug("EcalUncalibRecHitDebug") << "accessing matrices of weights...";
	 const EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
	 const EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
	 const EcalWeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
	 const EcalWeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();
	 LogDebug("EcalUncalibRecHitDebug") << "done." ;

	 // build CLHEP weight matrices
	 std::vector<HepMatrix> weights;
	 HepMatrix  clmat1 = makeMatrixFromVectors(mat1);
	 HepMatrix  clmat2 = makeMatrixFromVectors(mat2);
	 weights.push_back(clmat1);
	 weights.push_back(clmat2);
	 LogDebug("EcalUncalibRecHitDebug") << "weights before switch:\n" << clmat1 ;
	 LogDebug("EcalUncalibRecHitDebug") << "weights after switch:\n" << clmat2 ;


	 // build CLHEP chi2  matrices
	 std::vector<HepSymMatrix> chi2mat;
	 HepSymMatrix  clmat3(10);
	 clmat3.assign(makeMatrixFromVectors(mat3));
	 HepSymMatrix  clmat4(10);
	 clmat4.assign(makeMatrixFromVectors(mat4));
	 chi2mat.push_back(clmat3);
	 chi2mat.push_back(clmat4);
	 //if(!counterExceeded()) LogDebug("EcalUncalibRecHitDebug") << "chi2 matrix before switch:\n" << clmat3 ;
	 //if(!counterExceeded()) LogDebug("EcalUncalibRecHitDebug") << "chi2 matrix after switch:\n" << clmat4 ;

	 EcalUncalibratedRecHit aHit =
	   EBalgo_.makeRecHit(*itdg, pedVec, weights, chi2mat);
	 EBuncalibRechits->push_back( aHit );


	 if(aHit.amplitude()>0.) {
	   LogDebug("EcalUncalibRecHitDebug") << "processed EBDataFrame with id: "
					      << itdg->id() << "\n"
					      << "uncalib rechit amplitude: " << aHit.amplitude()
	     ;
	 }
       }
     }

   // loop over EE digis
   if (EEdigis)
     {
       for(EEDigiCollection::const_iterator itdg = EEdigis->begin(); itdg != EEdigis->end(); ++itdg) {

	 //     counter_++; // verbosity counter

	 // find pedestals for this channel
	 LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << itdg->id() ;
	 pedIter = pedMap.find(itdg->id().rawId());
	 if( pedIter != pedMap.end() ) {
	   aped = pedIter->second;
	 } else {
	   edm::LogError("EcalUncalibRecHitError") << "error!! could not find pedestals for channel: " << itdg->id() 
						   << "\n  no uncalib rechit will be made for this digi!"
	     ;
	   continue;
	 }
	 std::vector<double> pedVec;
	 pedVec.push_back(aped.mean_x1);pedVec.push_back(aped.mean_x6);pedVec.push_back(aped.mean_x12);

	 // lookup group ID for this channel
	 EcalWeightXtalGroups::EcalXtalGroupsMap::const_iterator git = grp->getMap().find( itdg->id().rawId() );
	 EcalXtalGroupId gid;
	 if( git != grp->getMap().end() ) {
	   gid = git->second;
	 } else {
	   edm::LogError("EcalUncalibRecHitError") << "No group id found for this crystal. something wrong with EcalWeightXtalGroups in your DB?"
						   << "\n  no uncalib rechit will be made for digi with id: " << itdg->id()
	     ;
	   continue;
	 }

	 // use a fake TDC iD for now until it become available in raw data
	 EcalTBWeights::EcalTDCId tdcid(1);

	 // now lookup the correct weights in the map
	 EcalTBWeights::EcalTBWeightMap::const_iterator wit = wgts->getMap().find( std::make_pair(gid,tdcid) );
	 if( wit == wgts->getMap().end() ) {  // no weights found for this group ID
	   edm::LogError("EcalUncalibRecHitError") << "No weights found for EcalGroupId: " << gid.id() << " and  EcalTDCId: " << tdcid
						   << "\n  skipping digi with id: " << itdg->id()
	     ;
	   continue;
	 }

	 EcalWeightSet  wset = wit->second; // this is the EcalWeightSet

	 // EcalWeightMatrix is vec<vec:double>>
	 LogDebug("EcalUncalibRecHitDebug") << "accessing matrices of weights...";
	 const EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
	 const EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
	 const EcalWeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
	 const EcalWeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();
	 LogDebug("EcalUncalibRecHitDebug") << "done." ;

	 // build CLHEP weight matrices
	 std::vector<HepMatrix> weights;
	 HepMatrix  clmat1 = makeMatrixFromVectors(mat1);
	 HepMatrix  clmat2 = makeMatrixFromVectors(mat2);
	 weights.push_back(clmat1);
	 weights.push_back(clmat2);
	 LogDebug("EcalUncalibRecHitDebug") << "weights before switch:\n" << clmat1 ;
	 LogDebug("EcalUncalibRecHitDebug") << "weights after switch:\n" << clmat2 ;


	 // build CLHEP chi2  matrices
	 std::vector<HepSymMatrix> chi2mat;
	 HepSymMatrix  clmat3(10);
	 clmat3.assign(makeMatrixFromVectors(mat3));
	 HepSymMatrix  clmat4(10);
	 clmat4.assign(makeMatrixFromVectors(mat4));
	 chi2mat.push_back(clmat3);
	 chi2mat.push_back(clmat4);
	 //if(!counterExceeded()) LogDebug("EcalUncalibRecHitDebug") << "chi2 matrix before switch:\n" << clmat3 ;
	 //if(!counterExceeded()) LogDebug("EcalUncalibRecHitDebug") << "chi2 matrix after switch:\n" << clmat4 ;

	 EcalUncalibratedRecHit aHit =
	   EEalgo_.makeRecHit(*itdg, pedVec, weights, chi2mat);
	 EEuncalibRechits->push_back( aHit );


	 if(aHit.amplitude()>0.) {
	   LogDebug("EcalUncalibRecHitDebug") << "processed EEDataFrame with id: "
					      << itdg->id() << "\n"
					      << "uncalib rechit amplitude: " << aHit.amplitude()
	     ;
	 }
       }
     }
   // put the collection of recunstructed hits in the event
   evt.put( EBuncalibRechits, EBhitCollection_ );
   evt.put( EEuncalibRechits, EEhitCollection_ );
}

HepMatrix
EcalWeightUncalibRecHitProducer::makeMatrixFromVectors(const std::vector< std::vector<EcalWeight> >& vecvec) {
  int nrow = vecvec.size();
  int ncol = (vecvec[0]).size();
  HepMatrix clmat(nrow,ncol);
  //LogDebug("EcalUncalibRecHitDebug") << "created HepMatrix(" << nrow << "," << ncol << ")" ;
  for(int irow=0;irow<nrow;++irow) {
    for(int icol=0;icol<ncol;++icol) {
        clmat[irow][icol] = ((vecvec[irow])[icol]).value();
    }
  }
  return clmat;
}

