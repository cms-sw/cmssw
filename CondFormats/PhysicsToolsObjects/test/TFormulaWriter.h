#ifndef RecoTauTag_TauTagTools_TFormulaWriter_h
#define RecoTauTag_TauTagTools_TFormulaWriter_h

/** \class TgraphWriter
 *
 * Read TFormula objects from ROOT file input
 * and store it in SQL-lite output file
 *
 * \author Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>

class TFormulaWriter : public edm::EDAnalyzer 
{
 public:
  TFormulaWriter(const edm::ParameterSet&);
  ~TFormulaWriter();
  
 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  bool hasRun_;

  struct jobEntryType
  {
    jobEntryType(const edm::ParameterSet& cfg)
    {
      if ( cfg.existsAs<edm::FileInPath>("inputFileName") ) {
	edm::FileInPath inputFileName_fip = cfg.getParameter<edm::FileInPath>("inputFileName");
	if ( inputFileName_fip.location() == edm::FileInPath::Unknown ) 
	  throw cms::Exception("TFormulaWriter") 
	    << " Failed to find File = " << inputFileName_fip << " !!\n";
	inputFileName_ = inputFileName_fip.fullPath();
      } else if ( cfg.existsAs<std::string>("inputFileName") ) {
	inputFileName_ = cfg.getParameter<std::string>("inputFileName");
      } else throw cms::Exception("TFormulaWriter") 
	       << " Undefined Configuration Parameter 'inputFileName !!\n";
      formulaName_ = cfg.getParameter<std::string>("formulaName");
      outputRecord_ = cfg.getParameter<std::string>("outputRecord");
    }
    ~jobEntryType() {}
    std::string inputFileName_;
    std::string formulaName_;
    std::string outputRecord_;
  };
  std::vector<jobEntryType*> jobs_;
};

#endif
