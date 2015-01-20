#ifndef DTnoiseDBValidation_H
#define DTnoiseDBValidation_H

/** \class DTnoiseDBValidation
 *  Plot the noise from the DB comparaison
 *
 *  \author G. Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTChamberId;
class DTStatusFlag;
class TFile;

class DTnoiseDBValidation : public DQMEDAnalyzer {
public:
  /// Constructor
  DTnoiseDBValidation(const edm::ParameterSet& pset);

    /// Destructor
  virtual ~DTnoiseDBValidation();

  /// Operations
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endStream(void) override;
 
protected:

private:
  void bookHisto(DQMStore::IBooker &, const DTChamberId&);

  // The DB label
  std::string labelDBRef_;
  std::string labelDB_;
  std::string diffTestName_,wheelTestName_,stationTestName_,
              sectorTestName_,layerTestName_;

  // The DTGeometry
  edm::ESHandle<DTGeometry> dtGeom_;

  // The noise map
  const DTStatusFlag *noiseMap_;
  const DTStatusFlag *noiseRefMap_;
 
  //the total number of noisy cell
  int noisyCellsRef_;
  int noisyCellsValid_;
  // the histos
  MonitorElement * diffHisto_;
  MonitorElement * wheelHisto_;
  MonitorElement * stationHisto_;
  MonitorElement * sectorHisto_;
  MonitorElement * layerHisto_;
  std::map<DTChamberId, MonitorElement*> noiseHistoMap_;

};
#endif

