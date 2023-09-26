//
// Original Author:  Yetkin Yilmaz
//         Created:  Wed May  2 21:41:30 EDT 2007
//
//

// system include files
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"

using namespace std;

//
// class decleration
//

class HiTrivialConditionRetriever : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  HiTrivialConditionRetriever(const edm::ParameterSet&);

protected:
  //overriding from ContextRecordIntervalFinder
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  virtual std::unique_ptr<CentralityTable> produceTable(const HeavyIonRcd&);
  void printBin(const CentralityTable::CBin*);

  // ----------member data ---------------------------

  int verbose_;
  string inputFileName_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HiTrivialConditionRetriever::HiTrivialConditionRetriever(const edm::ParameterSet& iConfig) {
  setWhatProduced(this, &HiTrivialConditionRetriever::produceTable);
  findingRecord<HeavyIonRcd>();

  //now do what ever initialization is needed
  verbose_ = iConfig.getUntrackedParameter<int>("verbosity", 1);
  inputFileName_ = iConfig.getParameter<string>("inputFile");
}

std::unique_ptr<CentralityTable> HiTrivialConditionRetriever::produceTable(const HeavyIonRcd&) {
  auto CT = std::make_unique<CentralityTable>();

  // Get values from text file
  ifstream in(edm::FileInPath(inputFileName_).fullPath().c_str());
  string line;

  int i = 0;
  while (getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    CentralityTable::CBin thisBin;
    istringstream ss(line);
    ss >> thisBin.bin_edge >> thisBin.n_part.mean >> thisBin.n_part.var >> thisBin.n_coll.mean >> thisBin.n_coll.var >>
        thisBin.n_hard.mean >> thisBin.n_hard.var >> thisBin.b.mean >> thisBin.b.var;
    CT->m_table.push_back(thisBin);
    i++;
  }

  return CT;
}

void HiTrivialConditionRetriever::printBin(const CentralityTable::CBin* thisBin) {
  cout << "HF Cut = " << thisBin->bin_edge << endl;
  cout << "Npart = " << thisBin->n_part.mean << endl;
  cout << "sigma = " << thisBin->n_part.var << endl;
  cout << "Ncoll = " << thisBin->n_coll.mean << endl;
  cout << "sigma = " << thisBin->n_coll.var << endl;
  cout << "B     = " << thisBin->b.mean << endl;
  cout << "sigma = " << thisBin->b.var << endl;
  cout << "__________________________________________________" << endl;
}

void HiTrivialConditionRetriever::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& rk,
                                                 const edm::IOVSyncValue& iTime,
                                                 edm::ValidityInterval& oValidity) {
  if (verbose_ >= 1)
    std::cout << "HiTrivialConditionRetriever::setIntervalFor(): record key = " << rk.name()
              << "\ttime: " << iTime.time().value() << std::endl;
  //For right now, we will just use an infinite interval of validity
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(HiTrivialConditionRetriever);
