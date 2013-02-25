#include "SimHitValidator.h"
#include "DigiValidator.h"

int main(int argc, char* argv[])
{
  SimHitValidator* val = new SimHitValidator();    
  val->setInFileName("gem_sh_ana.test.root");
  val->makeValidationPlots(SimHitValidator::Muon);
  val->makeValidationPlots(SimHitValidator::NonMuon);
  val->makeValidationPlots(SimHitValidator::All);
  val->makeTrackValidationPlots();
  
  /*    
	DigiValidator* val = new DigiValidator();
	val->setInFileName("gem_digi_ana.test.root");
	val->makeValidationPlots();
	val->makeGEMCSCPadDigiValidationPlots("GEMCSCPadDigiTree");  
	val->makeGEMCSCPadDigiValidationPlots("GEMCSCCoPadDigiTree");  
	val->makeTrackValidationPlots();
  */

  val->setOutFileName("productionReport.tex");
  val->setTitle("DIGI+L1CSC level, MuonGun, Pt20, 1M events, PU0");
  val->setPriority("Very high priority");
  val->setDateOfRequest("February 8 2013");
  val->setDescription("DIGI+L1CSC level, MuonGun, Pt20, 1M events shooting into only a couple of chambers, PU0, SimMuon/GEMDigitizer V00-02-17, submitted to condor");
  val->setLinkToTwiki("https://twiki.cern.ch/twiki/bin/view/MPGD/GEMSimulationsValidationSamples");
  val->setProductionStartDate("9 February 2013");
  val->setResponsible("Sven Dildick");
  val->setDataSetPath("/pT20_1M_v1/dildick-DigiL1CSC-MuonGunPt20_1M-82325e40d6202e6fec2dd983c477f3ca/USER");
  val->setDataSetSize("12GB");
  val->setProductionEndDate("February 9 2013");
  val->setTimeToComplete("5s");
  val->setNumberOfEvents(1000);
  val->setNumberOfJobs(1000);
  val->setCrabConfiguration("crab.cfg"); // link to crab config
  val->setObsolete(false);

  val->makeValidationReport();
  
  system(("pdflatex " + val->getOutFileName()).c_str());
  
  return 0;
}
