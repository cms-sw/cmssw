#ifndef Validation_RecoMuon_PostProcessor_H
#define Validation_RecoMuon_PostProcessor_H

/*
 *  Class:PostProcessor 
 *
 *  DQM histogram post processor
 *
 *  $Date: 2008/07/25 10:34:52 $
 *  $Revision: 1.1 $
 *
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <vector>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace edm;

class DQMStore;

typedef boost::escaped_list_separator<char> elsc;

class PostProcessor : public EDAnalyzer
{
 public:
  PostProcessor(const ParameterSet& pset);
  ~PostProcessor() {};

  void analyze(const Event& event, const EventSetup& eventSetup) {};
  void endJob();

  void computeEfficiency(const string &, const string& efficMEName, const string& efficMETitle,
			 const string& recoMEName, const string& simMEName,
			 const string& xTitle, const string& yTitle );
  void computeResolution(const string &, 
			 const string& fitMEPrefix, const string& fitMETItlePrefix, 
			 const string& srcMEName);

 private:
  void processLoop( const string& dir, vector<boost::tokenizer<elsc>::value_type> args) ;

 private:
  DQMStore* theDQM;
  string subDir_;
  string outputFileName_;
  vector<string> commands_;
};

#endif
