#include <map>
#include <string>

#include "TH1D.h"
#include "TH2D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

/**
   \class   PatZToMuMuAnalyzer PatZToMuMuAnalyzer.cc "PhysicsTools/PatExamples/plugins/PatZToMuMuAnalyzer.h"

   \brief   Module to analyze the performance of muon reconstruction on the example of Z->mumu events

   Module to analyze the performance of muon reconstruction on the example of Z->mumu events: transverse 
   momentum and eta of the muon candidates and the mass of the Z boson candidate are plotted from inner,
   outer and global tracks. The mass is recalculated by an extra finction. The difference of the outer 
   track and the global track are plotted for the transverse momentum, eta and phi of the two muon candi-
   dates, for global muons as far as available. The only input parameters are: 
   
   _muons_  --> indicating the muon collection of choice.
   _shift_  --> indicating the relative shift of the transverse momentum for the estimate of the effect 
                on the invariant mass.

   The shift is applied to all mass calculations.
*/


class PatZToMuMuAnalyzer : public edm::EDAnalyzer {
  
 public:
  /// typedef's to simplify get functions
  typedef math::XYZVector Vector;
  typedef math::XYZTLorentzVector LorentzVector;

  /// default constructor
  explicit PatZToMuMuAnalyzer(const edm::ParameterSet& cfg);
  /// default destructor
  ~PatZToMuMuAnalyzer(){};
  
 private:
  /// everything that needs to be done during the event loop
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);

  /// calculate the mass of the Z boson from the tracker momenta by hand
  double mass(const math::XYZVector& t1, const math::XYZVector& t2) const;
  /// check if histogram was booked
  bool booked(const std::string histName) const { return hists_.find(histName.c_str())!=hists_.end(); };
  /// fill histogram if it had been booked before
  void fill(const std::string histName, double value) const { if(booked(histName.c_str())) hists_.find(histName.c_str())->second->Fill(value); };
  /// fill a predefined set of histograms from inner outer or global tracks for first and second mu candidate
  void fill(std::string hists, const reco::TrackRef& t1, const reco::TrackRef& t2) const;

  /// input for muons
  edm::InputTag muons_;
  /// shift in transverse momentum to determine a
  /// rough uncertainty on the Z mass estimation
  double shift_;
  /// management of 1d histograms
  std::map< std::string, TH1D* > hists_;
};

inline double 
PatZToMuMuAnalyzer::mass(const Vector& t1,  const Vector& t2) const
{
  return (LorentzVector(shift_*t1.x(), shift_*t1.y(), t1.z(), sqrt((0.1057*0.1057)+t1.mag2())) + LorentzVector(shift_*t2.x(), shift_*t2.y(), t2.z(), sqrt((0.1057*0.1057)+t2.mag2()))).mass();
}

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

PatZToMuMuAnalyzer::PatZToMuMuAnalyzer(const edm::ParameterSet& cfg):
  muons_(cfg.getParameter< edm::InputTag >("muons")),
  shift_(cfg.getParameter< double >("shift"))
{
  edm::Service< TFileService > fileService;
  
  // mass plot around Z peak from global tracks
  hists_[ "globalMass"] = fileService->make< TH1D >( "globalMass" , "Mass_{Z} (global) (GeV)",   90,    30.,   120.);
  // eta from global tracks
  hists_[ "globalEta" ] = fileService->make< TH1D >( "globalEta"  , "#eta (global)"          ,   48,   -2.4,    2.4);
  // pt from global tracks
  hists_[ "globalPt"  ] = fileService->make< TH1D >( "globalPt"   , "p_{T} (global) (GeV)"   ,  100,     0.,   100.);
  // mass plot around Z peak from inner tracks
  hists_[ "innerMass" ] = fileService->make< TH1D >( "innerMass"  , "Mass_{Z} (inner) (GeV)" ,   90,    30.,   120.);
  // eta from inner tracks
  hists_[ "innerEta"  ] = fileService->make< TH1D >( "innerEta"   , "#eta (inner)"           ,   48,   -2.4,    2.4);
  // pt from inner tracks
  hists_[ "innerPt"   ] = fileService->make< TH1D >( "innerPt"    , "p_{T} (inner) (GeV)"    ,  100,     0.,   100.);
  // mass plot around Z peak from outer tracks
  hists_[ "outerMass" ] = fileService->make< TH1D >( "outerMass"  , "Mass_{Z} (outer) (GeV)" ,   90,    30.,   120.);
  // eta from outer tracks
  hists_[ "outerEta"  ] = fileService->make< TH1D >( "outerEta"   , "#eta (outer)"           ,   48,   -2.4,    2.4);
  // pt from outer tracks
  hists_[ "outerPt"   ] = fileService->make< TH1D >( "outerPt"    , "p_{T} (outer) (GeV)"    ,  100,     0.,   100.);
  // delta pt between global and outer track
  hists_[ "deltaPt"   ] = fileService->make< TH1D >( "deltaPt"    , "#Delta p_{T} (GeV)"     ,  100,   -20.,    20.);
  // delta eta between global and outer track
  hists_[ "deltaEta"  ] = fileService->make< TH1D >( "deltaEta"   , "#Delta #eta"            ,  100,   -0.2,    0.2);
  // delta phi between global and outer track
  hists_[ "deltaPhi"  ] = fileService->make< TH1D >( "deltaPhi"   , "#Delta #phi"            ,  100,   -0.2,    0.2);
}

void PatZToMuMuAnalyzer::fill(std::string hists, const reco::TrackRef& t1, const reco::TrackRef& t2) const 
{
  if( t1.isAvailable() ){
    // fill pt from global track for first muon
    fill( std::string(hists).append("Pt") , t1->pt() );
    // fill pt from global track for second muon
    fill( std::string(hists).append("Eta"), t1->eta() );
  }
  if( t2.isAvailable() ){
    // fill eta from global track for first muon
    fill( std::string(hists).append("Pt") , t2->pt() );
    // fill eta from global track for second muon
    fill( std::string(hists).append("Eta"), t2->eta() );
  }
  if( t1.isAvailable() && t2.isAvailable() ){
    // fill invariant mass of the Z boson candidate
    fill( std::string(hists).append("Mass"), mass(t1->momentum(), t2->momentum()));
  }
}

void PatZToMuMuAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  // pat candidate collection
  edm::Handle< edm::View<pat::Muon> > muons;
  event.getByLabel(muons_, muons);

  // Fill some basic muon quantities as 
  // reconstructed from inner and outer 
  // tack 
  for(edm::View<pat::Muon>::const_iterator mu1=muons->begin(); mu1!=muons->end(); ++mu1){
    for(edm::View<pat::Muon>::const_iterator mu2=muons->begin(); mu2!=muons->end(); ++mu2){
      if(mu2>mu1){ // prevent double conting
	if( mu1->charge()*mu2->charge()<0 ){ // check only muon pairs of unequal charge 
	  fill(std::string("inner" ), mu1->innerTrack (), mu2->innerTrack ());
	  fill(std::string("outer" ), mu1->outerTrack (), mu2->outerTrack ());
	  fill(std::string("global"), mu1->globalTrack(), mu2->globalTrack());
	  
	  if(mu1->isGlobalMuon()){
	    fill("deltaPt" , mu1->outerTrack()->pt ()-mu1->globalTrack()->pt ());
	    fill("deltaEta", mu1->outerTrack()->eta()-mu1->globalTrack()->eta());
	    fill("deltaPhi", mu1->outerTrack()->phi()-mu1->globalTrack()->phi());
	  }
	  if(mu2->isGlobalMuon()){
	    fill("deltaPt" , mu2->outerTrack()->pt ()-mu2->globalTrack()->pt ());
	    fill("deltaEta", mu2->outerTrack()->eta()-mu2->globalTrack()->eta());
	    fill("deltaPhi", mu2->outerTrack()->phi()-mu2->globalTrack()->phi());
	  }
	}
      }
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PatZToMuMuAnalyzer );
