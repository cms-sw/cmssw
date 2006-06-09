#ifndef DQM_SiStripCommissioningClients_CommissioningHistograms_H
#define DQM_SiStripCommissioningClients_CommissioningHistograms_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
#include <boost/cstdint.hpp>
#include "TProfile.h"
#include "TH1F.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>

class MonitorUserInterface;

class CommissioningHistograms {

 public:
  
  struct HistoSet {
    SiStripHistoNamingScheme::HistoTitle title_;
    MonitorElement* combined_;
    MonitorElement* sumOfSquares_;
    MonitorElement* sumOfContents_;
    MonitorElement* numOfEntries_;
    MonitorElement* profile_;
  };

  /** */
  CommissioningHistograms( MonitorUserInterface* );
  /** */
  virtual ~CommissioningHistograms();

  // ----- "Actions" -----
  
  /** */
  void createCollateMEs();
  /** */
  void createProfileHistos();
  /** */
  virtual void createSummaryHistos();
  /** */
  virtual void createTrackerMap();
  /** */
  virtual void uploadToConfigDb();
  
 protected:
  
  inline MonitorUserInterface* const mui() const;
  
  inline void task( const sistrip::Task& );
  inline const sistrip::Task& task() const;
  
  void initHistoSet( const SiStripHistoNamingScheme::HistoTitle&, 
		     HistoSet&, 
		     MonitorElement* );
  void updateHistoSet( HistoSet& );
  
 private: // ----- private methods -----
  
  void getListOfDirs( std::vector<std::string>& dir_list );
  void cdIntoDir( const std::string& pwd, 
		  std::vector<std::string>& dir_list );

  /** */
  virtual void book( const std::vector<std::string>& me_list );
  /** */
  virtual void update();
  
 private: // ----- private data members -----
  
  /** */
  MonitorUserInterface* mui_;
  /** */
  std::vector<std::string> cme_;
  /** */
  sistrip::Task task_;

};

// inline methods
MonitorUserInterface* const CommissioningHistograms::mui() const { return mui_; }
void CommissioningHistograms::task( const sistrip::Task& task ) { task_ = task; }
const sistrip::Task& CommissioningHistograms::task() const { return task_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistograms_H


