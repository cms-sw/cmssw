#ifndef PhotonMIPHaloTagger_H
#define PhotonMIPHaloTagger_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h" 
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <string>

class PhotonMIPHaloTagger {

public:

  PhotonMIPHaloTagger(){};

  virtual ~PhotonMIPHaloTagger(){};

  void setup(const edm::ParameterSet& conf,edm::ConsumesCollector&& iC);

  void MIPcalculate(const reco::Photon*, 
		    const edm::Event&, 
                    const edm::EventSetup& es,
                    reco::Photon::MIPVariables& mipId);


  //get the seed crystal index
 void GetSeedHighestE(const reco::Photon* photon,
                       const edm::Event& iEvent,
                       const edm::EventSetup& iSetup,
                       edm::Handle<EcalRecHitCollection> Brechit,
                       int &seedIEta,
                       int &seedIPhi,
                       double &seedE);


  //get the MIP  Fit Trail results 
 std::vector<double>  GetMipTrailFit(const reco::Photon* photon,
		         	     const edm::Event& iEvent,
			             const edm::EventSetup& iSetup,
                                     edm::Handle<EcalRecHitCollection> ecalhitsCollEB,
                                     double inputRangeY,    
                                     double inputRangeX,    
                                     double inputResWidth,  
                                     double inputHaloDiscCut,
                                     int & NhitCone_,
                                     bool & ismipHalo_ );


 


 
 protected:

  edm::EDGetToken EBecalCollection_;
  edm::EDGetToken EEecalCollection_;



 //used inside main methhod
 double inputRangeY;
 double inputRangeX;
 double inputResWidth;
 double inputHaloDiscCut;
 

  //Isolation parameters variables as input
  double yRangeFit_;
  double xRangeFit_;
  double residualWidthEnergy_;
  double haloDiscThreshold_;

 //Local Vector for results
 std::vector<double> mipFitResults_;
 int  nhitCone_;
 bool ismipHalo_; 


  };

#endif // PhotonMIPHaloTagger_H
