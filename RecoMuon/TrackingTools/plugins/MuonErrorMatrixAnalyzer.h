#ifndef RecoMuon_TrackingTools_MuonErrorMatrixAnalyzer_H
#define RecoMuon_TrackingTools_MuonErrorMatrixAnalyzer_H

/** \class MuonErrorMatrixAnalyzer
 * 
 * EDAalyzer which compare reconstructed tracks to simulated tracks parameter in bins of pt, eta, (phi)
 * to give an empirical parametrization of the track parameters errors.
 *
 * $Dates: 2007/09/04 13:28 $
 * $Revision: 1.1 $
 *
 * \author Jean-Roch Vlimant  UCSB
 * \author Finn Rebassoo      UCSB
*/

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/TrackingTools/plugins/MuonErrorMatrix.h"

#include <FWCore/Framework/interface/ESHandle.h>

class MagneticField;
class TrackAssociatorBase;
class TH1;
class TH2;

//
// class decleration
//

class MuonErrorMatrixAnalyzer : public edm::EDAnalyzer {
 public:

  /// constructor
  explicit MuonErrorMatrixAnalyzer(const edm::ParameterSet&);
   
  ///destructor
  ~MuonErrorMatrixAnalyzer();
     
     
 private:
  /// framework methods
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
     
  /// produce error parametrization from reported errors
  void analyze_from_errormatrix(const edm::Event&, const edm::EventSetup&);
  /// produces error parametrization from the pull of track parameters
  void analyze_from_pull(const edm::Event&, const edm::EventSetup&);


  // ----------member data ---------------------------
  /// log category: "MuonErrorMatrixAnalyzer"
  std::string theCategory;

  /// input tags for reco::Track and TrackingParticle
  edm::InputTag theTrackLabel;
  edm::InputTag trackingParticleLabel;

  /// The associator used for reco/gen association (configurable)
  std::string theAssocLabel;
  edm::ESHandle<TrackAssociatorBase> theAssociator;

  /// hold on the magnetic field
  edm::ESHandle<MagneticField> theField;

  /// class holder for the reported error parametrization
  MuonErrorMatrix * theErrorMatrixStore_Reported;
  edm::ParameterSet theErrorMatrixStore_Reported_pset;

  /// class holder for the empirical error parametrization from residual
  MuonErrorMatrix * theErrorMatrixStore_Residual;
  edm::ParameterSet theErrorMatrixStore_Residual_pset;

  /// class holder for the empirical error scale factor parametrization from pull
  MuonErrorMatrix * theErrorMatrixStore_Pull;
  edm::ParameterSet theErrorMatrixStore_Pull_pset;

  /// control plot root file (auxiliary, configurable)
  TFile * thePlotFile;
  std::string thePlotFileName;

  /// arrays of plots for the empirical error parametrization
  typedef TH1* TH1ptr;
  TH1ptr* theHist_array_residual[15];
  TH1ptr* theHist_array_pull[15];

  /// index whithin the array of plots
  inline uint index(TProfile3D * pf, uint i ,uint j,uint k)   {return (((i*pf->GetNbinsY())+j) * pf->GetNbinsZ())+k;}
  uint maxIndex(TProfile3D * pf)  {return pf->GetNbinsX()*pf->GetNbinsY()*pf->GetNbinsZ();}

  struct extractRes{
    double corr;
    double x;
    double y;
  };
  /// fit procedure to extract sigma_x sigma_y and correlation factor from 2D residual histogram
  extractRes extract(TH2 * h2);
};
#endif
