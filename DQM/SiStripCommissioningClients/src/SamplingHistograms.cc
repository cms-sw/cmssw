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
SamplingHistograms::SamplingHistograms( DQMStore* bei,const sistrip::RunType& task ) 
  : CommissioningHistograms( bei, task ),sOnCut_(3)
{
  LogTrace(mlDqmClient_) 
       << "[SamplingHistograms::" << __func__ << "]"
       << " Constructing object...";
  factory_ = auto_ptr<SamplingSummaryFactory>( new SamplingSummaryFactory );
}

// -----------------------------------------------------------------------------
/** */
SamplingHistograms::SamplingHistograms( DQMOldReceiver* mui,const sistrip::RunType& task ) 
  : CommissioningHistograms( mui, task ),sOnCut_(3)
{
  LogTrace(mlDqmClient_) 
       << "[SamplingHistograms::" << __func__ << "]"
       << " Constructing object...";
  factory_ = auto_ptr<SamplingSummaryFactory>( new SamplingSummaryFactory );
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
    SamplingAlgorithm algo( anal );
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

