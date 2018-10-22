// -*- C++ -*-
//
// Package:    SiStripBaselineAnalyzer
// Class:      SiStripBaselineAnalyzer
// 
/**\class SiStripBaselineAnalyzer SiStripBaselineAnalyzer.cc Validation/SiStripAnalyzer/src/SiStripBaselineAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ivan Amos Cali
//         Created:  Mon Jul 28 14:10:52 CEST 2008
//
//
 

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"


//ROOT inclusion
#include "TH1F.h"
#include "TProfile.h"

//
// class decleration
//

class SiStripBaselineComparator : public edm::one::EDAnalyzer<edm::one::SharedResources> {
  public:
    explicit SiStripBaselineComparator(const edm::ParameterSet&);
    ~SiStripBaselineComparator() override;
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);


  private:
    void beginJob() override ;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override ;
     
    edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > srcClusters_;
    edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > srcClusters2_;

 
    edm::Service<TFileService> fs_;


    TH1F* h1_nOldClusters_;
    TH1F* h1_nMatchedClusters_;
    TH1F* h1_nSplitClusters_;
    TProfile* h1_matchingMult_;
    TProfile* h1_matchedWidth_;
    TProfile* h1_matchedCharges_;
   
    const uint16_t nbins_clusterSize_ = 128;
    const uint16_t min_clusterSize_ = 0;
    const uint16_t max_clusterSize_ = 128; 

};


SiStripBaselineComparator::SiStripBaselineComparator(const edm::ParameterSet& conf){
 
  usesResource(TFileService::kSharedResource);
  
  srcClusters_  =  consumes<edmNew::DetSetVector<SiStripCluster> >(conf.getParameter<edm::InputTag>("srcClusters"));
  srcClusters2_ =  consumes<edmNew::DetSetVector<SiStripCluster> >(conf.getParameter<edm::InputTag>("srcClusters2"));
 
  h1_nOldClusters_ = fs_->make<TH1F>("nOldClusters","nOldClusters;ClusterSize", nbins_clusterSize_, min_clusterSize_, max_clusterSize_);
  h1_nMatchedClusters_ = fs_->make<TH1F>("nMatchedClusters","nMatchedClusters;ClusterSize", nbins_clusterSize_, min_clusterSize_, max_clusterSize_);
  h1_nSplitClusters_ = fs_->make<TH1F>("nSplitClusters","nMatchedClusters;ClusterSize; n Split Clusters", nbins_clusterSize_, min_clusterSize_, max_clusterSize_);
  h1_matchingMult_ = fs_->make<TProfile>("matchingMult","matchingMult;ClusterSize; average number of clusters if split", nbins_clusterSize_, min_clusterSize_, max_clusterSize_);
  h1_matchedWidth_ = fs_->make<TProfile>("matchedWidth","matchedWidth;ClusterSize;average #Delta Width", nbins_clusterSize_, min_clusterSize_, max_clusterSize_);
  h1_matchedCharges_ = fs_->make<TProfile>("matchedCharges","matchedCharges;ClusterSize;cluster Charge ratio", nbins_clusterSize_, min_clusterSize_, max_clusterSize_);
}


SiStripBaselineComparator::~SiStripBaselineComparator()
{
}

void
SiStripBaselineComparator::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcClusters",  edm::InputTag("siStripClusters"));
  desc.add<edm::InputTag>("srcClusters2", edm::InputTag("moddedsiStripClusters"));
  descriptions.add("siStripBaselineComparator", desc);
}

void
SiStripBaselineComparator::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
  e.getByToken(srcClusters_,clusters);
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters2;
  e.getByToken(srcClusters2_,clusters2);

  for (auto const& set : *clusters){
    for (auto const& clus : set){
      h1_nOldClusters_->Fill(clus.amplitudes().size(),1);
      int nMatched = 0;
      const int charge1 = std::accumulate(clus.amplitudes().begin(), clus.amplitudes().end(), 0);
      std::vector< int > matchedWidths;
      std::vector< int > matchedCharges;

      //scan other set of clusters
      for (auto const& set2 : *clusters2) {
        if(set.id() != set2.id()) continue;
        for (auto const& clus2 : set2) {
          const int charge2 = std::accumulate(clus2.amplitudes().begin(), clus2.amplitudes().end(), 0);
          if ( ( clus.firstStrip() <= clus2.firstStrip() ) && ( clus2.firstStrip() < clus.firstStrip()+clus.amplitudes().size() ) ) {
            matchedWidths.push_back(clus2.amplitudes().size());
            matchedCharges.push_back(charge2);
            if ( nMatched == 0 ) {
              h1_nMatchedClusters_->Fill(clus.amplitudes().size(),1);
            } else if ( nMatched == 1 ) {
              h1_nSplitClusters_->Fill(clus.amplitudes().size(),1);
            }
            ++nMatched;
          }
        }
      }
      for(int i = 0; i<nMatched; i++){
        if(matchedWidths.at(i)-clus.amplitudes().size()<1000) 
          h1_matchedWidth_->Fill(clus.amplitudes().size(),matchedWidths.at(i)-clus.amplitudes().size());
        if(charge1 != 0 && matchedCharges.at(i)/(float)charge1<10000000) 
          h1_matchedCharges_->Fill(clus.amplitudes().size(),matchedCharges.at(i)/(float)charge1);
      }
      if(nMatched>1) h1_matchingMult_->Fill(clus.amplitudes().size(),nMatched);      
    } 
  }
}


// ------------ method called once each job just before starting event loop  ------------
void SiStripBaselineComparator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripBaselineComparator::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripBaselineComparator);

