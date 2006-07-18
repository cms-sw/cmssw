#ifndef IORawData_SiStripInputSources_CommissioningInputSource_h
#define IORawData_SiStripInputSources_CommissioningInputSource_h

#include <map>
#include <string>
#include <vector>
//FWCore
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ExternalInputSource.h"
//Data Formats
#include "DataFormats/SiStripDigi/interface/Profile.h"
#include "DataFormats/SiStripDigi/interface/Histo.h"
#include "DataFormats/Common/interface/DetSetVector.h"
//common
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
//CondFormats
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

//root
#include "TFile.h"
#include "TDirectory.h"

/** 
    @class : CommissioningInputSource
    @author : M.Wingham

    @brief : An InputSource that reads in Commissioning
    source-histogram file(s) and attaches all TH1Fs to the
    Event. Currently, due to the absense of a commissioning "client"
    to interpret the 3 "source" histograms per device, a temporary
    measure is in place to combine them appropriately. The combined
    "commissioning" histograms are written to a separate file.

    The DetSetVector container is indexed with a key representing the
    module control path. The key is described in :
    DataFormats/SiStripDetId/interface/SiStripControlKey.h.
*/

class CommissioningInputSource : public edm::ExternalInputSource {

 public:

  /** Constructor */
  explicit CommissioningInputSource( const edm::ParameterSet & pset, 
			    edm::InputSourceDescription const& desc );
  /** Destructor */
  virtual ~CommissioningInputSource();

 protected:

 /** Reads Commissioning Histogram (Source) file(s); extracts TH1Fs; combines them to produce "Commissioning Histograms" (as will be seen at the Client) and adds them to the Event.*/
  virtual bool produce( edm::Event& );

  /** Sets run number based on file name(s). Multiple files must be of same run. */
  virtual void setRunAndEventInfo();
  
 private:

  /** Finds all TH1Fs, iterating through sub-directories and fills the DetSetVector. Also updates output file to have identical TDirectory structure to input file. */
  void findHistos(TDirectory*, edm::DetSetVector< Histo >*);

  /** Finds all TH1Fs, ignoring sub-directories and fills the DetSetVector. */
  void dirHistos(TDirectory*, std::vector< TDirectory* >*, edm::DetSetVector< Histo >*);

  /** Finds input file directory "mother" in output file and adds a copy of the child directory, "child". */
  void addOutputDir(TDirectory* mother, TDirectory* child);

  /** Opens file if it exists. Takes the filename as the argument. */
  void openFile( const string&);

  /** Combines 3 "source" histograms into 1 "commissioning" histogram*/
  void combine(const TH1& sum, const TH1& sum2, const TH1& entries, TProfile& commHist);

  /** Set values for a TProfile bin*/
  void setBinStats(TProfile& prof, Int_t bin, Int_t entries, Double_t content, Double_t error);
  
  /** Input file */
  TFile* m_inputFile;

  /** Output file */
  TFile* m_outputFile;

  /** Output file name */
  string m_outputFilename;

  /** Run number */
  unsigned int m_run;
};

#endif // IORawData_SiStripInputSources_CommissioningInputSource_h
