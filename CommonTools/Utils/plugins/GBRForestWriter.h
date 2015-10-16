#ifndef RecoMET_METPUSubtraction_GBRForestWriter_h
#define RecoMET_METPUSubtraction_GBRForestWriter_h

/** \class GBRForestWriter
 *
 * Read GBRForest objects from ROOT file input
 * and store it in SQL-lite output file
 *
 * \authors Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>

class GBRForestWriter : public edm::EDAnalyzer 
{
 public:
  GBRForestWriter(const edm::ParameterSet&);
  ~GBRForestWriter();
  
 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  std::string moduleLabel_;

  bool hasRun_;

  typedef std::vector<std::string> vstring;

  struct categoryEntryType
  {
    categoryEntryType(const edm::ParameterSet& cfg)
    {
      if ( cfg.existsAs<edm::FileInPath>("inputFileName") ) {
	edm::FileInPath inputFileName_fip = cfg.getParameter<edm::FileInPath>("inputFileName");
	inputFileName_ = inputFileName_fip.fullPath();
      } else if ( cfg.existsAs<std::string>("inputFileName") ) {
	inputFileName_ = cfg.getParameter<std::string>("inputFileName");
      } else throw cms::Exception("GBRForestWriter") 
	  << " Undefined Configuration Parameter 'inputFileName !!\n";
      std::string inputFileType_string = cfg.getParameter<std::string>("inputFileType");
      if      ( inputFileType_string == "XML"       ) inputFileType_ = kXML;
      else if ( inputFileType_string == "GBRForest" ) inputFileType_ = kGBRForest;
      else throw cms::Exception("GBRForestWriter") 
	<< " Invalid Configuration Parameter 'inputFileType' = " << inputFileType_string << " !!\n";
      if ( inputFileType_ == kXML ) {
	inputVariables_ = cfg.getParameter<vstring>("inputVariables");
	spectatorVariables_ = cfg.getParameter<vstring>("spectatorVariables");
        methodName_ = cfg.getParameter<std::string>("methodName");
        gbrForestName_ = ( cfg.existsAs<std::string>("gbrForestName") ? cfg.getParameter<std::string>("gbrForestName") : methodName_ );
      }
      else
        gbrForestName_ = cfg.getParameter<std::string>("gbrForestName");
    }
    ~categoryEntryType() {}
    std::string inputFileName_;
    enum { kXML, kGBRForest };
    int inputFileType_;
    vstring inputVariables_;
    vstring spectatorVariables_;
    std::string gbrForestName_;
    std::string methodName_;
  };
  struct jobEntryType
  {
    jobEntryType(const edm::ParameterSet& cfg)
    {
      if ( cfg.exists("categories") ) {
	edm::VParameterSet cfgCategories = cfg.getParameter<edm::VParameterSet>("categories");
	for ( edm::VParameterSet::const_iterator cfgCategory = cfgCategories.begin();
	      cfgCategory != cfgCategories.end(); ++cfgCategory ) {
	  categoryEntryType* category = new categoryEntryType(*cfgCategory);
	  categories_.push_back(category);
	} 
      } else {
	categoryEntryType* category = new categoryEntryType(cfg);
	categories_.push_back(category);
      }
      std::string outputFileType_string = cfg.getParameter<std::string>("outputFileType");
      if      ( outputFileType_string == "GBRForest" ) outputFileType_ = kGBRForest;
      else if ( outputFileType_string == "SQLLite"   ) outputFileType_ = kSQLLite;
      else throw cms::Exception("GBRForestWriter") 
	<< " Invalid Configuration Parameter 'outputFileType' = " << outputFileType_string << " !!\n";
      if ( outputFileType_ == kGBRForest ) {
	outputFileName_ = cfg.getParameter<std::string>("outputFileName");
      } 
      if ( outputFileType_ == kSQLLite ) {
	outputRecord_ = cfg.getParameter<std::string>("outputRecord");
      } 
    }
    ~jobEntryType() 
    {
      for ( std::vector<categoryEntryType*>::iterator it = categories_.begin();
	    it != categories_.end(); ++it ) {
	delete (*it);
      }
    }
    std::vector<categoryEntryType*> categories_;
    enum { kGBRForest, kSQLLite };
    int outputFileType_;
    std::string outputFileName_;
    std::string outputRecord_;
  };
  std::vector<jobEntryType*> jobs_;
};

#endif
