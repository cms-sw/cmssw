#ifndef RecoMuon_TrackingTools_MuonErrorMatrixAnalyzer_H
#define RecoMuon_TrackingTools_MuonErrorMatrixAnalyzer_H

/** \class MuonErrorMatrixAnalyzer
 * 
 * EDAalyzer which compare reconstructed tracks to simulated tracks parameter in bins of pt, eta, (phi)
 * to give an empirical parametrization of the track parameters errors.
 *
 *
 * \author Jean-Roch Vlimant  UCSB
 * \author Finn Rebassoo      UCSB
*/

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/TrackingTools/interface/MuonErrorMatrix.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

class MagneticField;
class IdealMagneticFieldRecord;
class TrackingComponentsRecord;
class TH1;
class TH2;
class Propagator;
//
// class decleration
//

class MuonErrorMatrixAnalyzer : public edm::one::EDAnalyzer<> {
public:
  /// constructor
  explicit MuonErrorMatrixAnalyzer(const edm::ParameterSet&);

  ///destructor
  ~MuonErrorMatrixAnalyzer();

private:
  /// framework methods
  virtual void beginJob();
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

  /// hold on the magnetic field
  edm::ESHandle<MagneticField> theField;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;

  /// class holder for the reported error parametrization
  MuonErrorMatrix* theErrorMatrixStore_Reported;
  edm::ParameterSet theErrorMatrixStore_Reported_pset;

  /// class holder for the empirical error parametrization from residual
  MuonErrorMatrix* theErrorMatrixStore_Residual;
  edm::ParameterSet theErrorMatrixStore_Residual_pset;

  /// class holder for the empirical error scale factor parametrization from pull
  MuonErrorMatrix* theErrorMatrixStore_Pull;
  edm::ParameterSet theErrorMatrixStore_Pull_pset;

  /// the range of the pull fit is [-theGaussianPullFitRange, theGaussianPullFitRange] [-2,2] by default
  double theGaussianPullFitRange;

  /// radius at which the comparison is made: =0 is using TSCPBuilderNoMaterial, !=0 is using the propagator
  double theRadius;

  /// reference to the cylinder of radius theRadius
  Cylinder::CylinderPointer refRSurface;

  /// z at which the comparison is made: =0 is using TSCPBuilderNoMaterial, !=0 is using the propagator
  double theZ;

  ///reference to a plane at -z [0] and +z [1]:  [(z>0)]
  Plane::PlanePointer refZSurface[2];

  /// propagator used to go to the cylinder surface, ALONG momentum
  std::string thePropagatorName;
  edm::ESHandle<Propagator> thePropagator;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> thePropagatorToken;

  /// put the free trajectory state to the TSCPBuilderNoMaterial or the cylinder surface
  FreeTrajectoryState refLocusState(const FreeTrajectoryState& fts);

  /// control plot root file (auxiliary, configurable)
  TFile* thePlotFile;
  TFileDirectory* thePlotDir;
  TList* theBookKeeping;
  std::string thePlotFileName;

  /// arrays of plots for the empirical error parametrization
  typedef TH1* TH1ptr;
  TH1ptr* theHist_array_residual[15];
  TH1ptr* theHist_array_pull[15];

  /// index whithin the array of plots
  inline unsigned int index(TProfile3D* pf, unsigned int i, unsigned int j, unsigned int k) {
    return (((i * pf->GetNbinsY()) + j) * pf->GetNbinsZ()) + k;
  }
  unsigned int maxIndex(TProfile3D* pf) { return pf->GetNbinsX() * pf->GetNbinsY() * pf->GetNbinsZ(); }

  struct extractRes {
    double corr;
    double x;
    double y;
  };
  /// fit procedure to extract sigma_x sigma_y and correlation factor from 2D residual histogram
  extractRes extract(TH2* h2);
};
#endif
