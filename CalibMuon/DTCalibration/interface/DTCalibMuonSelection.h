//
// Original Author:  Mario Pelliccioni, Gianluca Cerminara
//         Created:  Tue Sep  9 15:56:24 CEST 2008


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include <vector>
#include <string>

class DTCalibMuonSelection : public edm::EDFilter {
public:

  explicit DTCalibMuonSelection(const edm::ParameterSet&);

  ~DTCalibMuonSelection() override;
  
private:
  void beginJob() override ;

  bool filter(edm::Event&, const edm::EventSetup&) override;

  void endJob() override ;
  
  edm::EDGetTokenT<reco::MuonCollection> muonList;

  double etaMin;
  double etaMax;
  double ptMin;

};
