/*
 * PatternsPtAssignment.cc
 *
 *  Created on: Mar 9, 2020
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Tools/PatternsPtAssignment.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

PatternsPtAssignment::PatternsPtAssignment(const edm::ParameterSet& edmCfg,
                                           const OMTFConfiguration* omtfConfig,
                                           const std::vector<std::shared_ptr<GoldenPattern> >& gps,
                                           std::string rootFileName)
    : PatternOptimizerBase(edmCfg, omtfConfig), gps(gps) {
  gpResultsToPt = new GpResultsToPt(gps, omtfConfig, 1024);  //TODO move to processor
}

PatternsPtAssignment::~PatternsPtAssignment() {
  // TODO Auto-generated destructor stub
}

void PatternsPtAssignment::observeEventEnd(const edm::Event& iEvent,
                                           std::unique_ptr<l1t::RegionalMuonCandBxCollection>& finalCandidates) {
  int muonCharge = 0;
  if (simMuon) {
    if (abs(simMuon->momentum().eta()) < 0.8 || abs(simMuon->momentum().eta()) > 1.24)
      return;

    muonCharge = (abs(simMuon->type()) == 13) ? simMuon->type() / -13 : 0;
  }

  if (simMuon == nullptr || !omtfCand->isValid())  //no sim muon or empty candidate
    return;

  /*  for(auto algoCandidate: algoCandidates) {
    if(algoCandidate->isValid())
      gpResultsToPt->updateStat(algoCandidate, candProcIndx, simMuon->momentum().pt(), muonCharge);
  }*/

  gpResultsToPt->updateStat(omtfCand, candProcIndx, simMuon->momentum().pt(), muonCharge);
}

void PatternsPtAssignment::endJob() {
  gpResultsToPt->caluateLutValues();

  std::string fileName = edmCfg.getParameter<std::string>("gpResultsToPtFile");
  std::ofstream ofs(fileName);

  boost::archive::text_oarchive outArch(ofs);
  //boost::archive::text_oarchive txtOutArch(ofs);

  //const PdfModule* pdfModuleImpl = dynamic_cast<const PdfModule*>(pdfModule);
  // write class instance to archive
  edm::LogImportant("l1tOmtfEventPrint") << __FUNCTION__ << ": " << __LINE__ << " writing gpResultsToPt to file "
                                         << fileName << std::endl;
  outArch << *gpResultsToPt;
  //outArch << gpResultsToPt->getGpResultsToPtLuts();

  //txtOutArch << (*pdfModuleImpl);
  // archive and stream closed when destructors are called
}
