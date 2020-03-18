#ifndef SRCONDACCESS_H
#define SRCONDACCESS_H

/*
 * $Id: EcalLaserCondTools.h,v 1.2 2010/06/14 10:45:16 pgras Exp $
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>
/**
 */
class EcalLaserCondTools : public edm::EDAnalyzer {
  //methods
public:
  /** Constructor
   * @param ps analyser configuration
   */
  EcalLaserCondTools(const edm::ParameterSet&);

  /** Destructor
   */
  ~EcalLaserCondTools();


  /** Called by CMSSW event loop
   * @param evt the event
   * @param es events setup
   */
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& es);

private:
  static std::string toNth(int n);
  static std::string timeToString(time_t t);  
  class CorrReader{
  public:
    CorrReader(): verb_(0){
    }
    virtual bool readTime(int& t1, int t2[92], int& t3) {return false;}
    virtual bool readPs(DetId& rawdetid, EcalLaserAPDPNRatios::EcalLaserAPDPNpair& corr) {return false;}
    virtual ~CorrReader(){}
    void setVerbosity(int verb){ verb_ = verb; }
  protected:
    int verb_;
  };
  
  class FileReader: public EcalLaserCondTools::CorrReader{
  public:
    FileReader(const std::vector<std::string>& fnames): f_(0), fnames_(fnames), ifile_(-1), iline_(0){
    }
    virtual bool readTime(int& t1, int t2[92], int& t3);
    virtual bool readPs(DetId& rawdetid, EcalLaserAPDPNRatios::EcalLaserAPDPNpair& corr);
    virtual ~FileReader(){ }
    
  private:
    bool nextFile();
    void trim();
    FILE* f_;
    std::vector<std::string> fnames_;
    unsigned ifile_;
    int iline_;
  };

private:
  void fillDb(CorrReader& r);
  void dbToAscii(const edm::EventSetup& es);
  void processIov(CorrReader& r, int t1, int t2[92], int t3);
  
  //fields
private:
  FILE* fout_;
  FILE* eventList_;
  std::string eventListFileName_;
  int verb_;
  std::string mode_;
  std::vector<std::string> fnames_;
  edm::Service<cond::service::PoolDBOutputService> db_;
  int skipIov_;
  int nIovs_;
  int fromTime_;
  int toTime_;
  FILE* ferr_;
};

#endif //SRCONDACCESS_H not defined
