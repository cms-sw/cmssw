#ifndef AlcaBeamSpotHarvester_H
#define AlcaBeamSpotHarvester_H

/** \class AlcaBeamSpotHarvester
 *  No description available.
 *
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */
#include "Calibration/TkAlCaRecoProducers/interface/AlcaBeamSpotManager.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotWrite2Txt.h"

// #include "FWCore/ParameterSet/interface/ParameterSet.h"

class AlcaBeamSpotHarvester : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  AlcaBeamSpotHarvester(const edm::ParameterSet &);

  /// Destructor
  ~AlcaBeamSpotHarvester() override = default;

  // Operations
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginRun(const edm::Run &, const edm::EventSetup &);
  void endRun(const edm::Run &, const edm::EventSetup &);
  void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);
  void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);

protected:
private:
  // Parameters
  std::string beamSpotOutputBase_;
  std::string outputrecordName_;
  double sigmaZValue_;
  double sigmaZCut_;
  bool dumpTxt_;
  std::string outTxtFileName_;
  // Member Variables
  AlcaBeamSpotManager theAlcaBeamSpotManager_;

  //   edm::ParameterSet metadataForOfflineDropBox_;
};
#endif
