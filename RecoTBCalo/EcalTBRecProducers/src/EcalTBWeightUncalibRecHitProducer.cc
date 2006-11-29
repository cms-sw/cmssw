/** \class EcalTBWeightUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes
 *
  *  $Id: EcalTBWeightUncalibRecHitProducer.cc,v 1.6 2006/11/18 10:00:29 meridian Exp $
  *  $Date: 2006/11/18 10:00:29 $
  *  $Revision: 1.6 $
  *
  *  $Alex Zabi$
  *  $Date: 2006/11/18 10:00:29 $
  *  $Revision: 1.6 $
  *  Modification to detect first sample to switch gain.
  *  used for amplitude recontruction at high energy
  *  Add TDC convention option (P. Meridiani)
  *
  */
#include "RecoTBCalo/EcalTBRecProducers/interface/EcalTBWeightUncalibRecHitProducer.h"
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

EcalTBWeightUncalibRecHitProducer::EcalTBWeightUncalibRecHitProducer(const edm::ParameterSet& ps) {

   EBdigiCollection_ = ps.getParameter<std::string>("EBdigiCollection");
   digiProducer_   = ps.getParameter<std::string>("digiProducer");
   tdcRecInfoCollection_ = ps.getParameter<std::string>("tdcRecInfoCollection");
   tdcRecInfoProducer_   = ps.getParameter<std::string>("tdcRecInfoProducer");
   EBhitCollection_  = ps.getParameter<std::string>("EBhitCollection");
   nbTimeBin_  = ps.getParameter<int>("nbTimeBin");
   use2004OffsetConvention_ = ps.getUntrackedParameter< bool >("use2004OffsetConvention",false);
   produces< EBUncalibratedRecHitCollection >(EBhitCollection_);
}

EcalTBWeightUncalibRecHitProducer::~EcalTBWeightUncalibRecHitProducer() {
}

void
EcalTBWeightUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {

   using namespace edm;
   
   Handle< EBDigiCollection > pEBDigis;
   const EBDigiCollection* EBdigis =0;
   
   try {
     //     evt.getByLabel( digiProducer_, EBdigiCollection_, pEBDigis);
     evt.getByLabel( digiProducer_, pEBDigis);
     EBdigis = pEBDigis.product(); // get a ptr to the produc
     LogDebug("EcalUncalibRecHitInfo") << "total # EBdigis: " << EBdigis->size() ;
   } catch ( std::exception& ex ) {
     edm::LogError("EcalUncalibRecHitError") << "Error! can't get the product " << EBdigiCollection_.c_str() ;
   }

   if (!EBdigis)
     return;

   Handle< EcalTBTDCRecInfo > pRecTDC;
   const EcalTBTDCRecInfo* recTDC =0;

   try 
     {
       //     evt.getByLabel( digiProducer_, EBdigiCollection_, pEBDigis);
       evt.getByLabel( tdcRecInfoProducer_, tdcRecInfoCollection_, pRecTDC);
       recTDC = pRecTDC.product(); // get a ptr to the product
     } 
   catch ( std::exception& ex ) 
     {
     }

   // fetch map of groups of xtals
   edm::ESHandle<EcalWeightXtalGroups> pGrp;
   es.get<EcalWeightXtalGroupsRcd>().get(pGrp);
   const EcalWeightXtalGroups* grp = pGrp.product();

   if (!grp)
     return;

   // Gain Ratios
   edm::ESHandle<EcalGainRatios> pRatio;
   es.get<EcalGainRatiosRcd>().get(pRatio);
   const EcalGainRatios::EcalGainRatioMap& gainMap = pRatio.product()->getMap(); // map of gain ratios


   // fetch TB weights
   LogDebug("EcalUncalibRecHitDebug") <<"Fetching EcalTBWeights from DB " ;
   edm::ESHandle<EcalTBWeights> pWgts;
   es.get<EcalTBWeightsRcd>().get(pWgts);
   const EcalTBWeights* wgts = pWgts.product();

   if (!wgts)
     return;

   LogDebug("EcalUncalibRecHitDebug") << "EcalTBWeightMap.size(): " << std::setprecision(3) << wgts->getMap().size() ;
   

   // fetch the pedestals from the cond DB via EventSetup
   LogDebug("EcalUncalibRecHitDebug") << "fetching pedestals....";
   edm::ESHandle<EcalPedestals> pedHandle;
   es.get<EcalPedestalsRcd>().get( pedHandle );
   const EcalPedestalsMap& pedMap = pedHandle.product()->m_pedestals; // map of pedestals
   LogDebug("EcalUncalibRecHitDebug") << "done." ;

   // collection of reco'ed ampltudes to put in the event

   std::auto_ptr< EBUncalibratedRecHitCollection > EBuncalibRechits( new EBUncalibratedRecHitCollection );

   EcalPedestalsMapIterator pedIter; // pedestal iterator
   EcalPedestals::Item aped; // pedestal object for a single xtal

   EcalGainRatios::EcalGainRatioMap::const_iterator gainIter; // gain iterator
   EcalMGPAGainRatio aGain; // gain object for a single xtal
   // loop over EB digis
     //Getting the TDC bin
   EcalTBWeights::EcalTDCId tdcid(int(nbTimeBin_/2)+1);
   
     if (recTDC)
       if (recTDC->offset() == -999.)
	 {
	   edm::LogError("EcalUncalibRecHitError") << "TDC bin completely out of range. Returning" ;
	   return;
	 }
   
   
   for(EBDigiCollection::const_iterator itdg = EBdigis->begin(); itdg != EBdigis->end(); ++itdg) {
     
     //     counter_++; // verbosity counter
     
     // find pedestals for this channel
     LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << EBDetId(itdg->id()) ;
     pedIter = pedMap.find(itdg->id().rawId());
     if( pedIter != pedMap.end() ) {
       aped = pedIter->second;
     } else {
       edm::LogError("EcalUncalibRecHitError") << "error!! could not find pedestals for channel: " << EBDetId(itdg->id()) 
					       << "\n  no uncalib rechit will be made for this digi!"
	 ;
       continue;
     }

     std::vector<double> pedVec;
     pedVec.push_back(aped.mean_x12);pedVec.push_back(aped.mean_x6);pedVec.push_back(aped.mean_x1);

     // lookup group ID for this channel
     EcalWeightXtalGroups::EcalXtalGroupsMap::const_iterator git = grp->getMap().find( itdg->id().rawId() );
     EcalXtalGroupId gid;
     if( git != grp->getMap().end() ) {
       gid = git->second;
     } else {
       edm::LogError("EcalUncalibRecHitError") << "No group id found for this crystal. something wrong with EcalWeightXtalGroups in your DB?"
					       << "\n  no uncalib rechit will be made for digi with id: " << EBDetId(itdg->id())
	 ;
       continue;
     }

     // find gain ratios
     LogDebug("EcalUncalibRecHitDebug") << "looking up gainRatios for crystal: " << EBDetId(itdg->id()) ;
     gainIter = gainMap.find(itdg->id().rawId());
     if( gainIter != gainMap.end() ) {
       aGain = gainIter->second;
     } else {
       edm::LogError("EcalUncalibRecHitError") << "error!! could not find gain ratios for channel: " << EBDetId(itdg->id()) 
					       << "\n  no uncalib rechit will be made for this digi!"
	 ;
       continue;
     }
     
     std::vector<double> gainRatios;
     gainRatios.push_back(1.);gainRatios.push_back(aGain.gain12Over6());gainRatios.push_back(aGain.gain6Over1()*aGain.gain12Over6());

     //GAIN SWITCHING DETECTION ///////////////////////////////////////////////////////////////////////////////////////////////////     
     double sampleGainRef = 1;
     int    sampleSwitch  = 999;
     for (int sample = 0; sample < itdg->size(); ++sample)
       {
	 double gainSample = itdg->sample(sample).gainId();
	 if(gainSample != sampleGainRef) {sampleGainRef = gainSample; sampleSwitch = sample;}
       }//loop sample
     ///////////////////////////////////////////////////////////////////////////////////////////////////
     
     if (recTDC)
     {
       int tdcBin=0;
       if (recTDC->offset() <= 0.)
	 tdcBin = 1;
       if (recTDC->offset() >= 1.)
	 tdcBin = nbTimeBin_;
       else
	 tdcBin = int(recTDC->offset()*float(nbTimeBin_))+1;
       
       if (tdcBin < 1 || tdcBin > nbTimeBin_ )
	 {
	   edm::LogError("EcalUncalibRecHitError") << "TDC bin out of range " << tdcBin << " offset " << recTDC->offset();
	   continue;
	 }

       // In case gain switching happens at the sample 4 (5th sample) 
       // (sample 5 (6th sample) in 2004 TDC convention) an extra
       // set of weights has to be used. This set of weights is assigned to 
       // TDC values going from 25 and up.
       if (use2004OffsetConvention_ && sampleSwitch == 5)
	 tdcid=EcalTBWeights::EcalTDCId(tdcBin+25);
       else if (!use2004OffsetConvention_ && sampleSwitch == 4)
	 tdcid=EcalTBWeights::EcalTDCId(tdcBin+25);
       else 
	 tdcid=EcalTBWeights::EcalTDCId(tdcBin);
     }//check TDC     
     
     // now lookup the correct weights in the map
     EcalTBWeights::EcalTBWeightMap::const_iterator wit = wgts->getMap().find( std::make_pair(gid,tdcid) );
     if( wit == wgts->getMap().end() ) {  // no weights found for this group ID
       edm::LogError("EcalUncalibRecHitError") << "No weights found for EcalGroupId: " << gid.id() << " and  EcalTDCId: " << tdcid
					       << "\n  skipping digi with id: " << EBDetId(itdg->id())
	 ;
       continue;
     }

     EcalWeightSet  wset = wit->second; // this is the EcalWeightSet

     // EcalWeightMatrix is vec<vec:double>>
     LogDebug("EcalUncalibRecHitDebug") << "accessing matrices of weights...";
     const EcalWeightSet::EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
     const EcalWeightSet::EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
     //Using dummy matrices for chi2
     //      const EcalWeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
     //      const EcalWeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();
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
     clmat3.assign(makeDummySymMatrix(10));
     HepSymMatrix  clmat4(10);
     clmat4.assign(makeDummySymMatrix(10));
     chi2mat.push_back(clmat3);
     chi2mat.push_back(clmat4);
     //if(!counterExceeded()) LogDebug("EcalUncalibRecHitDebug") << "chi2 matrix before switch:\n" << clmat3 ;
     //if(!counterExceeded()) LogDebug("EcalUncalibRecHitDebug") << "chi2 matrix after switch:\n" << clmat4 ;

     EcalUncalibratedRecHit aHit =
       EBalgo_.makeRecHit(*itdg, pedVec, gainRatios, weights, chi2mat);
     EBuncalibRechits->push_back( aHit );

      if(aHit.amplitude()>0.) {
       LogDebug("EcalUncalibRecHitDebug") << "processed EBDataFrame with id: "
					  << EBDetId(itdg->id()) << "\n"
					  << "uncalib rechit amplitude: " << aHit.amplitude()
	 ;
      }
   }
   // put the collection of recunstructed hits in the event
   evt.put( EBuncalibRechits, EBhitCollection_ );
}

HepMatrix
EcalTBWeightUncalibRecHitProducer::makeMatrixFromVectors(const std::vector< std::vector<EcalWeight> >& vecvec) {
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

HepMatrix 
EcalTBWeightUncalibRecHitProducer::makeDummySymMatrix(int size)
{
  HepMatrix clmat(10,10);
  //LogDebug("EcalUncalibRecHitDebug") << "created HepMatrix(" << nrow << "," << ncol << ")" ;
  for(int irow=0; irow<size; ++irow) {
    for(int icol=0 ; icol<size; ++icol) {
      clmat[irow][icol] = irow+icol;
    }
  }
  return clmat;
}
