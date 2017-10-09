#ifndef HiJetBackground_HiFJGridEmptyAreaCalculator_h
#define HiJetBackground_HiFJGridEmptyAreaCalculator_h


// system include files
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

class HiFJGridEmptyAreaCalculator : public edm::stream::EDProducer<> {
   public:
      explicit HiFJGridEmptyAreaCalculator(const edm::ParameterSet&);
      ~HiFJGridEmptyAreaCalculator();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginStream(edm::StreamID) override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endStream() override;
      
  /// @name setting a new event
  //\{
  //----------------------------------------------------------------

  /// tell the background estimator that it has a new event, composed
  /// of the specified particles.


private:

  /// configure the grid
  void setupGrid(double eta_min, double eta_max);
  void setupGridJet(const reco::Jet *jet);

  /// retrieve the grid cell index for a given PseudoJet
  int tileIndexJet(const reco::PFCandidate *pfCand);
  int tileIndexEta(const reco::PFCandidate *pfCand);
  int tileIndexEtaJet(const reco::PFCandidate *pfCand);
  int tileIndexPhi(const reco::PFCandidate *pfCand);
  
  ///number of grid cells that overlap with jet constituents filling in the in between area
  int numJetGridCells( std::vector<std::pair<int, int> >& indices );
  
  /// calculates the area of jets that fall within the eta 
  /// range by scaling kt areas using grid areas
  void calculateAreaFractionOfJets(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void calculateGridRho(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  
  /// information about the grid
  const double twopi_ = 2*M_PI;
  
  ///internal parameters for grid

  //parameters of grid covering the full acceptance
  double ymin_;
  double ymax_;
  double dy_;
  double dphi_;
  double tileArea_;

  //parameters of grid around jets
  double dyJet_;
  double yminJet_;
  double ymaxJet_;
  double totalInboundArea_;

  //leave bands at boundaries
  double etaminJet_;
  double etamaxJet_;

  //for the grid calculation covering the full acceptance
  int ny_;
  int nphi_;
  int ntotal_;

  //for the grid calculation around each jet
  int ntotalJet_;
  int nyJet_;
  
  ///input parameters
  double gridWidth_;
  double band_;
  int hiBinCut_;
  bool doCentrality_;
  bool keepGridInfo_;
  
  std::vector<double> rhoVsEta_;
  std::vector<double> meanRhoVsEta_;
  std::vector<double> etaMaxGrid_;
  std::vector<double> etaMinGrid_;
  
  int n_tiles()  {return ntotal_;}
  
  /// input tokens
  edm::EDGetTokenT<edm::View<reco::Jet>>                 jetsToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection>          pfCandsToken_;
  edm::EDGetTokenT<std::vector<double>>                  mapEtaToken_;
  edm::EDGetTokenT<std::vector<double>>                  mapRhoToken_;
  edm::EDGetTokenT<std::vector<double>>                  mapRhoMToken_;

  edm::EDGetTokenT<int> centralityBinToken_;
  
};

#endif

