#ifndef RecoParticleFlow_PFClusterProducer_PFClusterComparator_
#define RecoParticleFlow_PFClusterProducer_PFClusterComparator_

// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

/**\class PFClusterAnalyzer 
\brief test analyzer for PFClusters
*/
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"



class PFClusterComparator : public edm::EDAnalyzer {
 public:

  explicit PFClusterComparator(const edm::ParameterSet&);

  ~PFClusterComparator();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(const edm::Run & r, const edm::EventSetup & c);

 private:
  
  void 
    fetchCandidateCollection(edm::Handle<reco::PFClusterCollection>& c, 
			     const edm::InputTag& tag, 
			     const edm::Event& iSetup) const;

/*   void printElementsInBlocks(const reco::PFCluster& cluster, */
/* 			     std::ostream& out=std::cout) const; */


  
  /// PFClusters in which we'll look for pile up particles 
  edm::InputTag   inputTagPFClusters_;
  edm::InputTag   inputTagPFClustersCompare_;
  
  edm::Service<TFileService> fs_;
  TH1F *log10E_old, *log10E_new, *deltaEnergy;
  TH1F *posX_old, *posX_new, *deltaX;
  TH1F *posY_old, *posY_new, *deltaY;
  TH1F *posZ_old, *posZ_new, *deltaZ;
  

  /// verbose ?
  bool   verbose_;

  /// print the blocks associated to a given candidate ?
  bool   printBlocks_;

};

#endif
