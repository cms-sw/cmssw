#ifndef IORawData_SiStripInputSources_TBMonitorInputSource_h
#define IORawData_SiStripInputSources_TBMonitorInputSource_h

#include <map>
#include <string>
#include <vector>
//FWCore
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ExternalInputSource.h"
//Data Formats
#include "DataFormats/SiStripDigi/interface/Profile.h"
#include "DataFormats/Common/interface/DetSetVector.h"
//common
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"

//root
#include "TFile.h"
#include "TDirectory.h"

/** 
    @class : TBMonitorInputSource
    @author : M.Wingham

    @brief : An InputSource that reads in TBMonitor file(s), renames the TProfiles of interest (in accordance with the scheme in DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h) and attaches them to the event. 

The DetSetVector container is indexed with a key representing the module control path. The key is described in : DQM/SiStripSources/interface/SiStripGenerateKey.h.
*/

class TBMonitorInputSource : public edm::ExternalInputSource {

 public:

  /** Constructor */
  explicit TBMonitorInputSource( const edm::ParameterSet & pset, 
			    edm::InputSourceDescription const& desc );
  /** Destructor */
  virtual ~TBMonitorInputSource();

 protected:

  /** Reads TBMonitor file(s), extracts TProfiles that match the commissioning task of interest and adds them to the Event.*/
  virtual bool produce( edm::Event& e );

  /** Sets run number based on file name(s). Multiple files must be of same run. */
  virtual void setRunAndEventInfo();

 private:

  /** Finds TProfiles for the commissioning task of interest, iterating through sub-directories, and fills the DetSetVector. The histogram names are converted to the preferred format as defined in DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h */
  void findProfiles(TDirectory*, edm::DetSetVector< Profile >*);

 /** Finds TProfiles for the commissioning task of interest, ignoring sub-directories, and fills the DetSetVector */
  void dirProfiles(TDirectory*, std::vector< TDirectory* >*, edm::DetSetVector< Profile >*);

  /** Opens file if it exists. Takes the filename as the argument.*/
  void openFile( const std::string&);

  /** Takes task name as the argument and updates the "task id" : a common sub-string of the names of histograms for storage.*/
  static std::string taskId(SiStripHistoNamingScheme::Task);

  /** Unpacks TBMonitor histogram name into the module's control path */
  SiStripHistoNamingScheme::HistoTitle histoTitle(const string&);

  /** Set values for a TProfile bin*/
  void setBinStats(TProfile& prof, Int_t bin, Int_t entries, Double_t content, Double_t error);

 /** Splits Pedestals Profile on the module-level into 2 or 3 Profiles on the LLD-channel-level. Renames accordingly.*/
  void lldPedestals(Profile& module, vector<Profile>& llds);

  /** Input file. */
  TFile* m_file;

  /** Commissioning task. */
  SiStripHistoNamingScheme::Task m_task;

  /** Task id found in TBMonitor histogram title */
  string m_taskId;
 
};

#endif // IORawData_SiStripInputSources_TBMonitorInputSource_h
