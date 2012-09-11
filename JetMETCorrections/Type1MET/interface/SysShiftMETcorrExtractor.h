#ifndef JetMETCorrections_Type1MET_SysShiftMETcorrExtractor_h
#define JetMETCorrections_Type1MET_SysShiftMETcorrExtractor_h

/** \class SysShiftMETcorrExtractor
 *
 * Compute MET correction to compensate systematic shift of MET in x/y-direction
 * (cf. https://indico.cern.ch/getFile.py/access?contribId=1&resId=0&materialId=slides&confId=174318 )
 *
 * \authors Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: SysShiftMETcorrExtractor.h,v 1.2 2012/04/09 14:19:01 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/METReco/interface/CorrMETData.h"

#include <TFormula.h>
#include <TString.h>

#include <string>

class SysShiftMETcorrExtractor
{
 public:

  explicit SysShiftMETcorrExtractor(const edm::ParameterSet&);
  ~SysShiftMETcorrExtractor();
    
  CorrMETData operator()(double sumEt, int Nvtx, int numJets) const;

 private:

  std::string name_;

  struct metCorrEntryType
  {
    metCorrEntryType(const std::string& name, const edm::ParameterSet& cfg)
      : px_(0),
	py_(0)
    {
      numJetsMin_ = cfg.getParameter<int>("numJetsMin");
      numJetsMax_ = cfg.getParameter<int>("numJetsMax");

      TString pxFormula = cfg.getParameter<std::string>("px").data();
      pxFormula.ReplaceAll("sumEt", "x");
      pxFormula.ReplaceAll("Nvtx", "y");
      std::string pxName = std::string(name).append("_px");
      px_ = new TFormula(pxName.data(), pxFormula.Data());

      TString pyFormula = cfg.getParameter<std::string>("py").data();
      pyFormula.ReplaceAll("sumEt", "x");
      pyFormula.ReplaceAll("Nvtx", "y");
      std::string pyName = std::string(name).append("_py");
      py_ = new TFormula(pyName.data(), pyFormula.Data());
    }
    ~metCorrEntryType()
    {
      delete px_;
      delete py_;
    }
    CorrMETData operator()(double sumEt, int Nvtx) const
    {
      CorrMETData retVal;
      retVal.mex = -px_->Eval(sumEt, Nvtx);
      retVal.mey = -py_->Eval(sumEt, Nvtx);
      return retVal;
    }
    int numJetsMin_;
    int numJetsMax_;
    TFormula* px_;
    TFormula* py_;
  };
  
  std::vector<metCorrEntryType*> corrections_;
};

#endif


 

