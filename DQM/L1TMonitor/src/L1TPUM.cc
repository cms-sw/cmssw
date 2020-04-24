/*
 * \file L1TPUM.cc
 *
 * N. Smith <nick.smith@cern.ch>
 */


#include "DQM/L1TMonitor/interface/L1TPUM.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

// TODO: move to configuration?
namespace {
  const unsigned int R10BINS = 1024;
  const float R10MIN = -0.5;
  const float R10MAX = 1023.5;

  const unsigned int PUMETABINS = 22;
  const unsigned int PUMNORMALIZE = 22;

  const unsigned int PUMBINS = 18;
  const float PUMMIN = -0.5;
  const float PUMMAX = 17.5;
}

L1TPUM::L1TPUM(const edm::ParameterSet & ps) :
   regionSource_(consumes<L1CaloRegionCollection>(ps.getParameter<edm::InputTag>("regionSource"))),
   histFolder_(ps.getParameter<std::string>("histFolder"))
{
}

L1TPUM::~L1TPUM()
{
}

void L1TPUM::dqmBeginRun(const edm::Run&, const edm::EventSetup&)
{
}

void L1TPUM::analyze(const edm::Event & event, const edm::EventSetup & es)
{
  edm::Handle<L1CaloRegionCollection> regionCollection;
  event.getByToken(regionSource_, regionCollection);

  int nonzeroRegionsBxP2{0};
  int nonzeroRegionsBx0{0};
  int nonzeroRegionsBxM2{0};

  float regionsTotalEtBxP2{0.};
  float regionsTotalEtBx0{0.};
  float regionsTotalEtBxM2{0.};

  for (const auto& region : *regionCollection) {
    if ( region.et() > 0 ) {
      if ( region.bx() == 0 ) {
        nonzeroRegionsBx0++;
        regionsTotalEtBx0 += region.et();
      }
      else if ( region.bx() == 2 ) {
        nonzeroRegionsBxP2++;
        regionsTotalEtBxP2 += region.et();
      }
      else if ( region.bx() == -2 ) {
        nonzeroRegionsBxM2++;
        regionsTotalEtBxM2 += region.et();
      }
    }
  }

  regionsTotalEtBxP2_->Fill(regionsTotalEtBxP2);
  regionsTotalEtBx0_->Fill(regionsTotalEtBx0);
  regionsTotalEtBxM2_->Fill(regionsTotalEtBxM2);

  regionsAvgEtBxP2_->Fill(regionsTotalEtBxP2/396.);
  regionsAvgEtBx0_->Fill(regionsTotalEtBx0/396.);
  regionsAvgEtBxM2_->Fill(regionsTotalEtBxM2/396.);

  regionsAvgNonZeroEtBxP2_->Fill(regionsTotalEtBxP2/nonzeroRegionsBxP2);
  regionsAvgNonZeroEtBx0_->Fill(regionsTotalEtBx0/nonzeroRegionsBx0);
  regionsAvgNonZeroEtBxM2_->Fill(regionsTotalEtBxM2/nonzeroRegionsBxM2);

  nonZeroRegionsBxP2_->Fill(nonzeroRegionsBxP2);
  nonZeroRegionsBx0_->Fill(nonzeroRegionsBx0);
  nonZeroRegionsBxM2_->Fill(nonzeroRegionsBxM2);

  for (const auto& region : *regionCollection) {
    size_t etaBin = region.gctEta();
    regionBxPopulation_->Fill(etaBin*18+region.gctPhi(), region.bx());
    regionBxEtSum_->Fill(etaBin*18+region.gctPhi(), region.bx(), region.et());
    if ( region.bx() == 0 )
      regionsPUMEtaBx0_[etaBin]->Fill(nonzeroRegionsBx0/PUMNORMALIZE, region.et());
    else if ( region.bx() == 2 )
      regionsPUMEtaBxP2_[etaBin]->Fill(nonzeroRegionsBxP2/PUMNORMALIZE, region.et());
    else if ( region.bx() == -2 )
      regionsPUMEtaBxM2_[etaBin]->Fill(nonzeroRegionsBxM2/PUMNORMALIZE, region.et());
  }
}

void L1TPUM::bookHistograms(DQMStore::IBooker &ibooker, const edm::Run& run , const edm::EventSetup& es) 
{
  ibooker.setCurrentFolder(histFolder_+"/BX0");
  regionsPUMEtaBx0_.resize(PUMETABINS);
  for (size_t ieta=0; ieta<PUMETABINS; ++ieta) {
    regionsTotalEtBx0_ = ibooker.book1D("regionsTotalEt", "Total ET distribution;Sum Rank;Counts", 200, 0, 2000);
    regionsAvgEtBx0_ = ibooker.book1D("regionsAvgEt", "Average Rank;Average Rank;Counts", R10BINS, R10MIN, R10MAX);
    regionsAvgNonZeroEtBx0_ = ibooker.book1D("regionsAvgNonZeroEt", "Average Rank >0;Average Rank Regions>0;Counts", R10BINS, R10MIN, R10MAX);
    nonZeroRegionsBx0_ = ibooker.book1D("nonZeroRegions", "Nonzero regions;Number Regions >0;Counts", 397, -0.5, 396.5);
    regionsPUMEtaBx0_[ieta] = ibooker.book2D("regionsPUMEta"+std::to_string(ieta), "PUM Bin rank distribution;PU bin;Rank", PUMBINS, PUMMIN, PUMMAX, R10BINS, R10MIN, R10MAX);
  }

  ibooker.setCurrentFolder(histFolder_+"/BXP2");
  regionsPUMEtaBxP2_.resize(PUMETABINS);
  for (size_t ieta=0; ieta<PUMETABINS; ++ieta) {
    regionsTotalEtBxP2_ = ibooker.book1D("regionsTotalEt", "Total ET distribution;Sum Rank;Counts", 200, 0, 2000);
    regionsAvgEtBxP2_ = ibooker.book1D("regionsAvgEt", "Average Rank;Average Rank;Counts", R10BINS, R10MIN, R10MAX);
    regionsAvgNonZeroEtBxP2_ = ibooker.book1D("regionsAvgNonZeroEt", "Average Rank >0;Average Rank Regions>0;Counts", R10BINS, R10MIN, R10MAX);
    nonZeroRegionsBxP2_ = ibooker.book1D("nonZeroRegions", "Nonzero regions;Number Regions >0;Counts", 397, -0.5, 396.5);
    regionsPUMEtaBxP2_[ieta] = ibooker.book2D("regionsPUMEta"+std::to_string(ieta), "PUM Bin rank distribution;PU bin;Rank", PUMBINS, PUMMIN, PUMMAX, R10BINS, R10MIN, R10MAX);
  }

  ibooker.setCurrentFolder(histFolder_+"/BXM2");
  regionsPUMEtaBxM2_.resize(PUMETABINS);
  for (size_t ieta=0; ieta<PUMETABINS; ++ieta) {
    regionsTotalEtBxM2_ = ibooker.book1D("regionsTotalEt", "Total ET distribution;Sum Rank;Counts", 200, 0, 2000);
    regionsAvgEtBxM2_ = ibooker.book1D("regionsAvgEt", "Average Rank;Average Rank;Counts", R10BINS, R10MIN, R10MAX);
    regionsAvgNonZeroEtBxM2_ = ibooker.book1D("regionsAvgNonZeroEt", "Average Rank >0;Average Rank Regions>0;Counts", R10BINS, R10MIN, R10MAX);
    nonZeroRegionsBxM2_ = ibooker.book1D("nonZeroRegions", "Nonzero regions;Number Regions >0;Counts", 397, -0.5, 396.5);
    regionsPUMEtaBxM2_[ieta] = ibooker.book2D("regionsPUMEta"+std::to_string(ieta), "PUM Bin rank distribution;PU bin;Rank", PUMBINS, PUMMIN, PUMMAX, R10BINS, R10MIN, R10MAX);
  }

  ibooker.setCurrentFolder(histFolder_+"/RegionBxInfo");
  regionBxPopulation_ = ibooker.book2D("regionBxPopulation", "Event counts per region per bunch crossing;Region index (18*eta+phi);BX index;Counts", 396, -0.5, 395.5, 5, -2.5, 2.5);
  regionBxEtSum_ = ibooker.book2D("regionBxEtSum", "Et per region per bunch crossing;Region index (18*eta+phi);BX index;Counts*et", 396, -0.5, 395.5, 5, -2.5, 2.5);
}

void L1TPUM::beginLuminosityBlock(const edm::LuminosityBlock& ls,const edm::EventSetup& es)
{
}

