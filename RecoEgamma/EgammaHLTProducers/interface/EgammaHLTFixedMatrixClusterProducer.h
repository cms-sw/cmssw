#ifndef RecoEcal_EgammaClusterProducers_EgammaHLTFixedMatrixClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_EgammaHLTFixedMatrixClusterProducer_h_

#include <memory>
#include <time.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaClusterAlgos/interface/FixedMatrixClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
//


class EgammaHLTFixedMatrixClusterProducer : public edm::EDProducer 
{
  public:

      EgammaHLTFixedMatrixClusterProducer(const edm::ParameterSet& ps);

      ~EgammaHLTFixedMatrixClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events

      FixedMatrixClusterAlgo::VerbosityLevel verbosity;

      bool doBarrel_;
      bool doEndcaps_;
      bool doIsolated_;

      edm::InputTag barrelHitProducer_;//it was string
      edm::InputTag endcapHitProducer_;//it was string
      std::string barrelHitCollection_;
      std::string endcapHitCollection_;

      std::string barrelClusterCollection_;
      std::string endcapClusterCollection_;

      /*
      std::string clustershapecollectionEB_;//new
      std::string clustershapecollectionEE_;//new
    //BasicClusterShape AssociationMap
      std::string barrelClusterShapeAssociation_;//new
      std::string endcapClusterShapeAssociation_;//new 
      */

      edm::InputTag l1TagIsolated_;
      edm::InputTag l1TagNonIsolated_;
      double l1LowerThr_;
      double l1UpperThr_;
      double l1LowerThrIgnoreIsolation_;

      double regionEtaMargin_;
      double regionPhiMargin_;

      PositionCalc posCalculator_; // position calculation algorithm
      FixedMatrixClusterAlgo * fixexMatrix_p;
      //ClusterShapeAlgo shapeAlgo_;//new

      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0)); }

      const EcalRecHitCollection * getCollection(edm::Event& evt,
                                                 const std::string& hitProducer_,
                                                 const std::string& hitCollection_);


      void clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
                              const std::string& hitProducer,
                              const std::string& hitCollection,
                              const std::string& clusterCollection,
                              const std::vector<EcalEtaPhiRegion>& regions,
                              const FixedMatrixClusterAlgo::EcalPart& ecalPart);
      /*
      void clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
                              const std::string& hitProducer,
                              const std::string& hitCollection,
                              const std::string& clusterCollection,
			      const std::string& clusterShapeAssociation,
                              const FixedMatrixClusterAlgo::EcalPart& ecalPart);
      */
      void outputValidationInfo(reco::BasicClusterRefVector &clusterRefVector);

};


#endif
