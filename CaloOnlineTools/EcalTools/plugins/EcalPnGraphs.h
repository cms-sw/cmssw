#ifndef ECALPNGRAPHS_h
#define ECALPNGRAPHS_h

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>

#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>

#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"
#include <iostream>
#include <vector>

#include "TFile.h"
#include "TGraph.h"

class EcalPnGraphs : public edm::EDAnalyzer {
public:
  EcalPnGraphs(const edm::ParameterSet& ps);
  ~EcalPnGraphs() override;

private:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void beginJob() override;
  void endJob() override;

  //  void pnGraphs (edm::Handle<EcalPnDiodeDigiCollection> PNs );

  std::string intToString(int num);

  EcalFedMap* fedMap;

protected:
  //  std::string ebDigiCollection_;
  //std::string eeDigiCollection_;
  std::string digiProducer_;

  std::vector<int> feds_;
  std::vector<std::string> ebs_;

  int verbosity;
  int eventCounter;

  //  std::vector<int ieb_id;
  int first_Pn;

  bool inputIsOk;

  std::string fileName;

  std::vector<int> listChannels;
  std::vector<int> listAllChannels;
  std::vector<int> listPns;
  std::vector<int> listAllPns;

  int numPn;

  int abscissa[50];
  int ordinate[50];

  std::vector<TGraph> graphs;

  TFile* root_file;
};

#endif
