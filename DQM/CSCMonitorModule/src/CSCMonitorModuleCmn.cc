/*
 * =====================================================================================
 *
 *       Filename:  CSCMonitorModuleCmn.cc
 *
 *    Description:  Common CSC Monitor Module
 *
 *        Version:  1.0
 *        Created:  11/13/2008 02:31:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCMonitorModuleCmn.h"

CSCMonitorModuleCmn::CSCMonitorModuleCmn(const edm::ParameterSet& ps) {

  edm::FileInPath bookFile = ps.getParameter<edm::FileInPath>(PARAM_BOOKING_FILE);
  collection.load(bookFile.fullPath());

}

CSCMonitorModuleCmn::~CSCMonitorModuleCmn() {
}

void CSCMonitorModuleCmn::beginJob(const edm::EventSetup& c) {
}

void CSCMonitorModuleCmn::beginRun(const edm::Run& r, const edm::EventSetup& c) {
}

void CSCMonitorModuleCmn::setup() {
}

void CSCMonitorModuleCmn::analyze(const edm::Event& e, const edm::EventSetup& c) {
}

void CSCMonitorModuleCmn::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {
}

void CSCMonitorModuleCmn::endRun(const edm::Run& r, const edm::EventSetup& c) {
}

void CSCMonitorModuleCmn::endJob() {
}

