#ifndef SimplePi0DiscAnalyzer_h
#define SimplePi0DiscAnalyzer_h

//
// Package:         RecoEgamma/Examples
// Class:           SimplePi0DiscAnalyzer
//

//
// Original Author:  A. Kyriakis NCSR "Demokritos" Athens
//                    D Maletic, "Vinca" Belgrade
//         Created:   Mar 27 13:22:06 CEST 2009
// $Id: SimplePi0DiscAnalyzer.h,v 1.3 2011/05/20 17:17:28 wmtan Exp $
//
//

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
//#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include "DataFormats/EgammaCandidates/interface/PhotonPi0DiscriminatorAssociation.h"

class MagneticField;
class TFile;
class TH1F;
class TH2F;
class TH1I;
class TProfile;
class TTree;

class SimplePi0DiscAnalyzer : public edm::EDAnalyzer
{
 public:

     explicit SimplePi0DiscAnalyzer(const edm::ParameterSet& conf);
 
     virtual ~SimplePi0DiscAnalyzer();

     virtual void beginJob();
     virtual void endJob();
     virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

 private:

     // ----------member data ---------------------------

     std::string photonCollectionProducer_;
     std::string photonCollection_;

     std::string outputFile_;
     TFile*  rootFile_;

     TH1F* hConv_ntracks_;

     TH1F* hAll_nnout_Assoc_;
     TH1F* hAll_nnout_NoConv_Assoc_;
     TH1F* hBarrel_nnout_Assoc_;
     TH1F* hBarrel_nnout_NoConv_Assoc_;
     TH1F* hEndcNoPresh_nnout_Assoc_;
     TH1F* hEndcNoPresh_nnout_NoConv_Assoc_;
     TH1F* hEndcWithPresh_nnout_Assoc_;
     TH1F* hEndcWithPresh_nnout_NoConv_Assoc_;
     TH1F* hAll_nnout_NoConv_Assoc_R9_;
     TH1F* hBarrel_nnout_NoConv_Assoc_R9_;
     TH1F* hEndcNoPresh_nnout_NoConv_Assoc_R9_;
     TH1F* hEndcWithPresh_nnout_NoConv_Assoc_R9_;
 
};

#endif



