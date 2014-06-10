/** \class DAFValidator
  *  An analyzer suitable for multitrack 
  *  fitting algorithm like DAF, MTF and EA
  *
  *  \author tropiano, genta (?)
  *  \review in May 2014 by brondolin 
  */

#ifndef DAFValidator_h
#define DAFValidator_h

#include <memory>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "TH2F.h"
#include "TTree.h"
#include "TFile.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TCanvas.h"

class TrackerHitAssociator;
class TrackerGeometry;
class PSimHit;

class DAFValidator : public edm::EDAnalyzer
{
  public:
  	explicit DAFValidator(const edm::ParameterSet& conf);
        ~DAFValidator();

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

	virtual void beginJob() override;
        virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
        virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
	virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
        virtual void endJob() override;

	void analyzeHits(const TrackingParticle* tpref, const reco::Track* rtref,
                               TrackerHitAssociator& hitassociator,
                               const edm::Ref<std::vector<Trajectory> > traj_iterator,
                               const TrackerGeometry* geom,
                               int event);
        GlobalPoint getGlobalPositionSim(const PSimHit hit, const TrackerGeometry* geom) const;
        GlobalPoint getGlobalPosition(const TrackingRecHit* hit, const TrackerGeometry* geom) const;
	void fillDAFHistos(std::vector<PSimHit>& matched, float weight,
                                 const TrackingRecHit* rechit, const TrackerGeometry* geom);
	float calculatepull(const TrackingRecHit* hit, PSimHit simhit, const TrackerGeometry* geom);
	void fillPHistos(std::vector<PSimHit>& components);
	int fillMergedHisto(const std::vector<SimHitIdpr>& simhitids, const std::vector<PSimHit>& simhits, 
			    const TrackingParticle* tpref, float weight, const TrackerGeometry* geom) const;
	int fillNotMergedHisto(const std::vector<SimHitIdpr>& simhitids, const std::vector<PSimHit>& simhits,
                                     const TrackingParticle* tpref, float weight, const TrackerGeometry* geom) const;
	float getType(const TrackingRecHit* hit)  const;
//	std::pair<float, std::vector<float> > getAnnealingWeight( const TrackingRecHit& aRecHit ) const;
  
  private:
        edm::ParameterSet theConf_;
	edm::InputTag tracksTag_;
	edm::InputTag trackingParticleTag_;
	std::string associatorTag_;
        TH1F* histo_maxweight;

        TH2F* weight_withassociatedsimhit_vs_type;
        TH2F* weight_withassociatedsimhit_vs_r;
        TH2F* weight_withassociatedsimhit_vs_eta;
        TH1F* weight_partially_unmatched;
        TH2F* weight_vs_type_partially_unmatched;
        TH2F* weight_vs_r_partially_unmatched;
        TH2F* weight_vs_eta_partially_unmatched;
        TH1F* weight_totally_unmatched;
        TH2F* weight_vs_type_totally_unmatched;
        TH2F* weight_vs_r_totally_unmatched;
        TH2F* weight_vs_eta_totally_unmatched;
        TH1F* processtype_withassociatedsimhit_merged;
        TH1F* processtype_withassociatedsimhit;
        TH1F* Hit_Histo;
        TH1F* MergedHisto;
        TH1F* NotMergedHisto;
        TH2F* weight_vs_processtype_merged;
        TH2F* weight_vs_processtype_notmerged;
        TH2F* pull_vs_weight;
        TH2F* Merged_vs_weight;
        TH2F* NotMerged_vs_weight;
        TH2F* NotMergedPos;
        TH2F* MergedPos;
        TTree* mrhit;
        //      TTree* mrhit2;
        int mergedtype;
        int notmergedtype;
        float weight;
        uint32_t detId;
        float r;
        float zeta;
        float phi;
        float hittype;
        int nevent;
        float hitlocalx;
        float hitlocaly;
        float hitlocalsigmax;
        float hitlocalsigmay;
        float hitlocalcov;
        float tsoslocalx;
        float tsoslocaly;
        float tsoslocalsigmax;
        float tsoslocalsigmay;
        float tsoslocalcov;
        float SimTracknum;
        float RecoTracknum;

  	//ERICA
        TH2F* annealing_weight;
        int nHitsAnn1, nHitsAnn2, nHitsAnn3;
        int nHitsAnn4, nHitsAnn5, nHitsAnn6;
        TGraph* annealing_weight_tgraph1;	
        TGraph* annealing_weight_tgraph2;	
        TGraph* annealing_weight_tgraph3;	
        TGraph* annealing_weight_tgraph4;	
        TGraph* annealing_weight_tgraph5;	
        TGraph* annealing_weight_tgraph6;
        TMultiGraph *annealing_weight_tot;	

};
#endif

