/** \class EcalWeightUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes
 *
  *  $Id: EcalWeightUncalibRecHitProducer.cc,v 1.6 2006/01/10 11:28:51 meridian Exp $
  *  $Date: 2006/01/10 11:28:51 $
  *  $Revision: 1.6 $
  *  \author Shahram Rahatlou, University of Rome & INFN, Sept 2005
  *
  */
#include "RecoLocalCalo/EcalRecProducers/interface/EcalWeightUncalibRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "FWCore/Framework/interface/Handle.h"

#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/EcalObjects/interface/EcalWeightRecAlgoWeights.h"
#include "CondFormats/DataRecord/interface/EcalWeightRecAlgoWeightsRcd.h"

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include <vector>

EcalWeightUncalibRecHitProducer::EcalWeightUncalibRecHitProducer(const edm::ParameterSet& ps) {

   digiCollection_ = ps.getParameter<std::string>("digiCollection");
   digiProducer_   = ps.getParameter<std::string>("digiProducer");
   hitCollection_  = ps.getParameter<std::string>("hitCollection");
   nMaxPrintout_   = ps.getUntrackedParameter<int>("nMaxPrintout",10);
   produces< EcalUncalibratedRecHitCollection >(hitCollection_);
   nEvt_ = 0; // reset local event counter
}

EcalWeightUncalibRecHitProducer::~EcalWeightUncalibRecHitProducer() {
}

void
EcalWeightUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

   using namespace edm;

   nEvt_++;

   Handle< EBDigiCollection > pDigis;
   try {
     //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
     evt.getByLabel( digiProducer_, pDigis);
   } catch ( std::exception& ex ) {
     std::cerr << "Error! can't get the product " << digiCollection_.c_str() << std::endl;
   }
   const EBDigiCollection* digis = pDigis.product(); // get a ptr to the product
   if(!counterExceeded()) std::cout << "EcalWeightUncalibRecHitProducer: total # digis: " << digis->size() << std::endl;

   // fetch the pedestals from the cond DB via EventSetup
   if(!counterExceeded()) std::cout << "fetching pedestals....";
   edm::ESHandle<EcalPedestals> pedHandle;
   es.get<EcalPedestalsRcd>().get( pedHandle );
   const EcalPedestalsMap& pedMap = pedHandle.product()->m_pedestals; // map of pedestals
   if(!counterExceeded()) std::cout << "done." << std::endl;

   // fetch weights from cond DB via EventSetup
   if(!counterExceeded()) std::cout << "fetching weights....";
   edm::ESHandle<EcalWeightRecAlgoWeights> wgtHandle;
   es.get<EcalWeightRecAlgoWeightsRcd>().get( wgtHandle );
   const EcalWeightRecAlgoWeights* ewgt = wgtHandle.product();
   if(!counterExceeded()) std::cout << "done." << std::endl;

   // EcalWeightMatrix is vec<vec:double>>
   if(!counterExceeded()) std::cout << "accessing matrices of weights...";
   const EcalWeightMatrix& mat1 = ewgt->getWeightsBeforeGainSwitch();
   const EcalWeightMatrix& mat2 = ewgt->getWeightsAfterGainSwitch();
   const EcalWeightMatrix& mat3 = ewgt->getChi2WeightsBeforeGainSwitch();
   const EcalWeightMatrix& mat4 = ewgt->getChi2WeightsAfterGainSwitch();
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
   if(!counterExceeded()) std::cout << "chi2 matrix before switch:\n" << clmat3 << std::endl;
   if(!counterExceeded()) std::cout << "chi2 matrix after switch:\n" << clmat4 << std::endl;

   // collection of reco'ed ampltudes to put in the event
   std::auto_ptr< EcalUncalibratedRecHitCollection > uncalibRechits( new EcalUncalibratedRecHitCollection );

   EcalPedestalsMapIterator pedIter; // pedestal iterator
   EcalPedestals::Item aped; // pedestal object for a single xtal
   for(EBDigiCollection::const_iterator itdg = digis->begin(); itdg != digis->end(); ++itdg) {
     // find pedestals for this channel
     if(!counterExceeded()) std::cout << "looking up pedestal for crystal: " << itdg->id() << std::endl;
     pedIter = pedMap.find(itdg->id().rawId());
     if( pedIter != pedMap.end() ) {
        aped = pedIter->second;
     } else {
        std::cout << "error!! could not find pedestals for channel: " << itdg->id() << std::endl;
     }
     std::vector<double> pedVec;
     pedVec.push_back(aped.mean_x1);pedVec.push_back(aped.mean_x6);pedVec.push_back(aped.mean_x12);

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

