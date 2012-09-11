#include "JetMETCorrections/Type1MET/interface/SysShiftMETcorrExtractor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

SysShiftMETcorrExtractor::SysShiftMETcorrExtractor(const edm::ParameterSet& cfg)
  : name_(cfg.getParameter<std::string>("name"))
{
  typedef std::vector<edm::ParameterSet> vParameterSet;
  vParameterSet cfgCorrections = cfg.getParameter<vParameterSet>("parameter");
  for ( vParameterSet::const_iterator cfgCorrection = cfgCorrections.begin();
	cfgCorrection != cfgCorrections.end(); ++cfgCorrection ) {
    corrections_.push_back(new metCorrEntryType(name_, *cfgCorrection));
  }
}

SysShiftMETcorrExtractor::~SysShiftMETcorrExtractor()
{
  for ( std::vector<metCorrEntryType*>::iterator it = corrections_.begin();
	it != corrections_.end(); ++it ) {
    delete (*it);
  }
}

CorrMETData SysShiftMETcorrExtractor::operator()(double sumEt, int Nvtx, int numJets) const
{
  for ( std::vector<metCorrEntryType*>::const_iterator correction = corrections_.begin();
	correction != corrections_.end(); ++correction ) {
    if ( ((*correction)->numJetsMin_ == -1 || numJets >= (*correction)->numJetsMin_) &&
	 ((*correction)->numJetsMax_ == -1 || numJets <= (*correction)->numJetsMax_) ) return (**correction)(sumEt, Nvtx);
  }

  edm::LogWarning ("SysShiftMETcorrExtractor::operator()") 
    << "No MET sys. shift correction defined for numJets = " << numJets << " --> skipping !!";

  CorrMETData retVal;
  retVal.mex = 0.;
  retVal.mey = 0.;
  return retVal;
}
