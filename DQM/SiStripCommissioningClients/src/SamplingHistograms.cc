#include "DQM/SiStripCommissioningClients/interface/SamplingHistograms.h"
#include "CondFormats/SiStripObjects/interface/SamplingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DQM/SiStripCommissioningAnalysis/interface/SamplingAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include "TProfile.h"
 
using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SamplingHistograms::SamplingHistograms( const edm::ParameterSet& pset,
                                        DQMStore* bei,
                                        const sistrip::RunType& task )
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("SamplingParameters"),
                             bei,
                             task ),
    sOnCut_(3)
{
  LogTrace(mlDqmClient_) 
       << "[SamplingHistograms::" << __func__ << "]"
       << " Constructing object...";
  factory_ = unique_ptr<SamplingSummaryFactory>( new SamplingSummaryFactory );
  // retreive the latency code from the root file
  std::string dataPath = std::string(sistrip::collate_) + "/" + sistrip::root_ + "/latencyCode";
  MonitorElement* codeElement = bei->get(dataPath);
  if(codeElement) latencyCode_ = codeElement->getIntValue() ;
  else latencyCode_ = 0;
}

// -----------------------------------------------------------------------------
/** */
SamplingHistograms::~SamplingHistograms() {
  LogTrace(mlDqmClient_) 
       << "[SamplingHistograms::" << __func__ << "]"
       << " Deleting object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void SamplingHistograms::histoAnalysis( bool debug ) {

  // Clear map holding analysis objects
  Analyses::iterator ianal;
  for ( ianal = data().begin(); ianal != data().end(); ianal++ ) {
    if ( ianal->second ) { delete ianal->second; }
  }
  data().clear();
  
  // Iterate through map containing vectors of profile histograms
  HistosMap::const_iterator iter = histos().begin();
  for ( ; iter != histos().end(); iter++ ) {
    // Check vector of histos is not empty (should be 1 histo)
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_)
 	   << "[SamplingHistograms::" << __func__ << "]"
 	   << " Zero collation histograms found!";
      continue;
    }
    
    // Retrieve pointers to profile histos for this FED channel 
    vector<TH1*> profs;
    Histos::const_iterator ihis = iter->second.begin();
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TProfile* prof = ExtractTObject<TProfile>().extract( (*ihis)->me_ );
      if ( prof ) { profs.push_back(prof); }
    } 
    
    // Perform histo analysis
    SamplingAnalysis* anal = new SamplingAnalysis( iter->first );
    anal->setSoNcut(sOnCut_);
    SamplingAlgorithm algo( this->pset(), anal, latencyCode_ );
    algo.analysis( profs );
    data()[iter->first] = anal; 
    
  }
  
}

void SamplingHistograms::configure( const edm::ParameterSet& pset, const edm::EventSetup& )
{
 //TODO: should use the parameter set. Why is this crashing ???
//  sOnCut_ = pset.getParameter<double>("SignalToNoiseCut");
   sOnCut_ = 3.;
}

