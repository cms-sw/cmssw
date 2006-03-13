/** \class EcalWeightUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes
 *
  *  $Id: EcalWeightUncalibRecHitProducer.cc,v 1.8 2006/03/13 09:06:31 rahatlou Exp $
  *  $Date: 2006/03/13 09:06:31 $
  *  $Revision: 1.8 $
  *  \author Shahram Rahatlou, University of Rome & INFN, Sept 2005
  *
  */
#include "RecoLocalCalo/EcalRecProducers/interface/EcalWeightUncalibRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "FWCore/Framework/interface/Handle.h"

#include <iostream>
#include <iomanip>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

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

   digiCollection_ = ps.getParameter<std::string>("digiCollection");
   digiProducer_   = ps.getParameter<std::string>("digiProducer");
   hitCollection_  = ps.getParameter<std::string>("hitCollection");
   nMaxPrintout_   = ps.getUntrackedParameter<int>("nMaxPrintout",10);
   produces< EcalUncalibratedRecHitCollection >(hitCollection_);
   counter_ = 0; // reset local event counter
}

EcalWeightUncalibRecHitProducer::~EcalWeightUncalibRecHitProducer() {
}

void
EcalWeightUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

   using namespace edm;

   Handle< EBDigiCollection > pDigis;
   try {
     //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
     evt.getByLabel( digiProducer_, pDigis);
   } catch ( std::exception& ex ) {
     std::cerr << "Error! can't get the product " << digiCollection_.c_str() << std::endl;
   }
   const EBDigiCollection* digis = pDigis.product(); // get a ptr to the product
   if(!counterExceeded()) std::cout << "EcalWeightUncalibRecHitProducer: total # digis: " << digis->size() << std::endl;


    // fetch map of groups of xtals
    edm::ESHandle<EcalWeightXtalGroups> pGrp;
    es.get<EcalWeightXtalGroupsRcd>().get(pGrp);
    const EcalWeightXtalGroups* grp = pGrp.product();

   // Gain Ratios
   //edm::ESHandle<EcalGainRatios> pRatio;
   //es.get<EcalGainRatiosRcd>().get(pRatio);
   //const EcalGainRatios* gr = pRatio.product();

   // fetch TB weights
   if(!counterExceeded()) std::cout <<"Fetching EcalTBWeights from DB " << std::endl;
   edm::ESHandle<EcalTBWeights> pWgts;
   es.get<EcalTBWeightsRcd>().get(pWgts);
   const EcalTBWeights* wgts = pWgts.product();
   if(!counterExceeded()) std::cout << "EcalTBWeightMap.size(): " << std::setprecision(3) << wgts->getMap().size() << std::endl;


   // fetch the pedestals from the cond DB via EventSetup
   if(!counterExceeded()) std::cout << "fetching pedestals....";
   edm::ESHandle<EcalPedestals> pedHandle;
   es.get<EcalPedestalsRcd>().get( pedHandle );
   const EcalPedestalsMap& pedMap = pedHandle.product()->m_pedestals; // map of pedestals
   if(!counterExceeded()) std::cout << "done." << std::endl;

   // collection of reco'ed ampltudes to put in the event
   std::auto_ptr< EcalUncalibratedRecHitCollection > uncalibRechits( new EcalUncalibratedRecHitCollection );

   EcalPedestalsMapIterator pedIter; // pedestal iterator
   EcalPedestals::Item aped; // pedestal object for a single xtal

   // loop over digis
   for(EBDigiCollection::const_iterator itdg = digis->begin(); itdg != digis->end(); ++itdg) {

     counter_++; // verbosity counter

     // find pedestals for this channel
     if(!counterExceeded()) std::cout << "looking up pedestal for crystal: " << itdg->id() << std::endl;
     pedIter = pedMap.find(itdg->id().rawId());
     if( pedIter != pedMap.end() ) {
        aped = pedIter->second;
     } else {
        std::cout << "error!! could not find pedestals for channel: " << itdg->id() 
                  << "\n  no uncalib rechit will be made for this digi!"
                  << std::endl;
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
       std::cout << "No group id found for this crystal. something wrong with EcalWeightXtalGroups in your DB?"
                  << "\n  no uncalib rechit will be made for digi with id: " << itdg->id()
                 << std::endl;
       continue;
    }

    // use a fake TDC iD for now until it become available in raw data
    EcalTBWeights::EcalTDCId tdcid(1);

    // now lookup the correct weights in the map
    EcalTBWeights::EcalTBWeightMap::const_iterator wit = wgts->getMap().find( std::make_pair(gid,tdcid) );
    if( wit == wgts->getMap().end() ) {  // no weights found for this group ID
      std::cout << "No weights found for EcalGroupId: " << gid.id() << " and  EcalTDCId: " << tdcid
                << "\n  skipping digi with id: " << itdg->id()
                << std::endl;
      continue;
    }

    EcalWeightSet  wset = wit->second; // this is the EcalWeightSet

    // EcalWeightMatrix is vec<vec:double>>
    if(!counterExceeded()) std::cout << "accessing matrices of weights...";
    const EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
    const EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
    const EcalWeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
    const EcalWeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();
    if(!counterExceeded()) std::cout << "done." << std::endl;

    // build CLHEP weight matrices
    std::vector<HepMatrix> weights;
    HepMatrix  clmat1 = makeMatrixFromVectors(mat1);
    HepMatrix  clmat2 = makeMatrixFromVectors(mat2);
    weights.push_back(clmat1);
    weights.push_back(clmat2);
    if(!counterExceeded()) std::cout << "weights before switch:\n" << clmat1 << std::endl;
    if(!counterExceeded()) std::cout << "weights after switch:\n" << clmat2 << std::endl;


    // build CLHEP chi2  matrices
    std::vector<HepSymMatrix> chi2mat;
    HepSymMatrix  clmat3(10);
    clmat3.assign(makeMatrixFromVectors(mat3));
    HepSymMatrix  clmat4(10);
    clmat4.assign(makeMatrixFromVectors(mat4));
    chi2mat.push_back(clmat3);
    chi2mat.push_back(clmat4);
    //if(!counterExceeded()) std::cout << "chi2 matrix before switch:\n" << clmat3 << std::endl;
    //if(!counterExceeded()) std::cout << "chi2 matrix after switch:\n" << clmat4 << std::endl;

    EcalUncalibratedRecHit aHit =
      algo_.makeRecHit(*itdg, pedVec, weights, chi2mat);
    uncalibRechits->push_back( aHit );


     if(aHit.amplitude()>0. && !counterExceeded() ) {
        std::cout << "EcalWeightUncalibRecHitProducer: processed EBDataFrame with id: "
                  << itdg->id() << "\n"
                  << "uncalib rechit amplitude: " << aHit.amplitude()
                  << std::endl;
     }
   }

   // put the collection of recunstructed hits in the event
   evt.put( uncalibRechits, hitCollection_ );
}

HepMatrix
EcalWeightUncalibRecHitProducer::makeMatrixFromVectors(const std::vector< std::vector<EcalWeight> >& vecvec) {
  int nrow = vecvec.size();
  int ncol = (vecvec[0]).size();
  HepMatrix clmat(nrow,ncol);
  //std::cout << "created HepMatrix(" << nrow << "," << ncol << ")" << std::endl;
  for(int irow=0;irow<nrow;++irow) {
    for(int icol=0;icol<ncol;++icol) {
        clmat[irow][icol] = ((vecvec[irow])[icol]).value();
    }
  }
  return clmat;
}

