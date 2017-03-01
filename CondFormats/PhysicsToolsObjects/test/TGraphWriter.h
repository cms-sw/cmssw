#ifndef RecoTauTag_TauTagTools_TGraphWriter_h
#define RecoTauTag_TauTagTools_TGraphWriter_h

/** \class TGraphWriter
 *
 * Read TGraph objects from ROOT file input
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

class TGraphWriter : public edm::EDAnalyzer 
{
 public:
  TGraphWriter(const edm::ParameterSet&);
  ~TGraphWriter();
  
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
	if ( inputFileName_fip.location() == edm::FileInPath::Unknown) 
	  throw cms::Exception("TGraphWriter") 
	    << " Failed to find File = " << inputFileName_fip << " !!\n";
	inputFileName_ = inputFileName_fip.fullPath();
      } else if ( cfg.existsAs<std::string>("inputFileName") ) {
	inputFileName_ = cfg.getParameter<std::string>("inputFileName");
      } else throw cms::Exception("TGraphWriter") 
	       << " Undefined Configuration Parameter 'inputFileName !!\n";
      graphName_ = cfg.getParameter<std::string>("graphName");
      outputRecord_ = cfg.getParameter<std::string>("outputRecord");
    }
    ~jobEntryType() {}
    std::string inputFileName_;
    std::string graphName_;
    std::string outputRecord_;
  };
  std::vector<jobEntryType*> jobs_;
};

#endif
