#ifndef RecoEgamma_EgammaPhotonProducers_PhotonCorrectionProducer_h
#define RecoEgamma_EgammaPhotonProducers_PhotonCorrectionProducer_h
/** \class PhotonCorrectionProducer
 **  
 **
 **  $Id: PhotonCorrectionProducer.h,v 1.4 2006/12/17 14:14:00 futyand Exp $ 
 **  $Date: 2006/12/17 14:14:00 $ 
 **  $Revision: 1.4 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgoBase.h"

class PhotonCorrectionProducer : public edm::EDProducer 
{
   public:
      PhotonCorrectionProducer(const edm::ParameterSet& ps);
      virtual ~PhotonCorrectionProducer();

      virtual void produce(edm::Event& evt, const edm::EventSetup& es);      

   private:
      void registorAlgos();
      void clearAlgos();

      std::map<std::string,PhotonCorrectionAlgoBase*> algo_m;
      std::vector<PhotonCorrectionAlgoBase*> algo_v;

      std::string photonProducer_;
      std::string photonCollection_;
      std::string photonCorrCollection_;
      std::string conversionProducer_;
      std::string conversionCollection_;
      std::string barrelClusterShapeMapProducer_;
      std::string barrelClusterShapeMapCollection_;
      std::string endcapClusterShapeMapProducer_;
      std::string endcapClusterShapeMapCollection_;
      std::string algoCollection_;  //correction algorithm collection      


};


#endif
