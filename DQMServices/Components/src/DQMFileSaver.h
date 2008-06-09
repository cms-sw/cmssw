#ifndef DQMSERVICES_COMPONEntS_DQMFILESAVER_H
# define DQMSERVICES_COMPONEntS_DQMFILESAVER_H

# include "FWCore/Framework/interface/EDAnalyzer.h"
# include <sys/time.h>
# include <string>

class DQMStore;
class DQMFileSaver : public edm::EDAnalyzer
{
public:
  DQMFileSaver(const edm::ParameterSet &ps);

protected:
  virtual void beginJob(const edm::EventSetup &);
  virtual void beginRun(const edm::Run &, const edm::EventSetup &);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);
  virtual void analyze(const edm::Event &e, const edm::EventSetup &);
  virtual void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);
  virtual void endRun(const edm::Run &, const edm::EventSetup &);
  virtual void endJob(void);

private:
  enum Convention
  {
    Online,
    Offline,
    RelVal
  };

  Convention	convention_;
  std::string	workflow_;
  std::string	producer_;
  std::string	dirName_;

  int		saveByLumiSection_;
  int		saveByEvent_;
  int		saveByMinute_;
  int		saveByRun_;
  bool		saveAtJobEnd_;
  int		forceRunNumber_;

  std::string	fileBaseName_;
  DQMStore	*dbe_;
  int		irun_;
  int		ilumi_;
  int		ilumiprev_;
  int		ievent_;
  int		nrun_;
  int		nlumi_;
  int		nevent_;
  timeval	start_;
  timeval	saved_;
};

#endif // DQMSERVICES_COMPONEntS_DQMFILESAVER_H
