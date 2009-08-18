#ifndef HDQMInspector_h
#define HDQMInspector_h

//---------------------------------------------------------//
//
//-- extract summary informations from historic DB --
//-- plot summary informations vs run number or vs detID --
//
//---------------------------------------------------------//
//---------------------------------------------------------//
// 
//  12-08-2008 - domenico.giordano@cern.ch 
//  12-06-2008 - anne-catherine.le.bihan@cern.ch 
//
//---------------------------------------------------------//

#include "vector"
#include "string"
#include "iostream"
#include "cmath"
#include "CondCore/Utilities/interface/CondCachedIter.h"
#include "CondFormats/DQMObjects/interface/HDQMSummary.h"
#include "DQMServices/Diagnostic/interface/HDQMInspectorConfigBase.h"
#include "TFile.h" 

class HDQMInspector
{
 public:  
  HDQMInspector():
    DBName_(""),
    DBTag_(""),
    DBuser_(""),
    DBpasswd_(""),
    DBblob_(""),
    Iterator(0),
    iDebug(0),
    iDoStat(0)
    {
      fHDQMInspectorConfig = 0x0;
    };
  
  HDQMInspector(const HDQMInspectorConfigBase* InConfig):
    DBName_(""),
    DBTag_(""),
    DBuser_(""),
    DBpasswd_(""),
    DBblob_(""),
    Iterator(0),
    iDebug(0),
    iDoStat(0)
    {
      fHDQMInspectorConfig = InConfig;
    };
  
  virtual ~HDQMInspector(){
    delete Iterator;
  };
  struct DetIdItemList {
    unsigned int detid;
    std::vector<std::string> items;
    std::vector<float> values;
  };

  void setDB(std::string DBName, std::string DBTag, std::string DBuser="", std::string DBpasswd="", std::string DBblob="");
  void createTrend(const std::string ListItems, const std::string CanvasName="", const int logy=0, const std::string Conditions="",
                   const unsigned int firstRun=1, const unsigned int lastRun=0xFFFFFFFE);
  void createTrendLastRuns(const std::string ListItems, const std::string CanvasName="",
                           const int logy=0, const std::string Conditions="", const unsigned int nRuns=10); 
  void setDebug(int i){iDebug=i;}
  void setDoStat(int i){iDoStat=i;}
  void setBlackList(std::string const& ListItems);
  void closeFile ()
  { 
    if( fOutFile ) {
      fOutFile->Close();
    }
  }
  
  inline std::vector<unsigned int> getRuns() { return vRun_;}
  inline std::vector<float> getSummary()     { return vSummary_;}
 
  inline std::vector<std::string>  getListItems() { return vlistItems_;}
  inline std::vector<unsigned int> getvDetId()    { return vdetId_;}
 
private:

  void style();
  void plot(size_t& nPads, std::string CanvasName, int logy=0);
  void accessDB();
  void InitializeIOVList();
  bool setRange(unsigned int& firstRun, unsigned int& lastRun);
  void setItems(std::string);
  size_t unpackItems(std::string& );
  void unpackConditions(std::string&, std::vector<DetIdItemList>&);
  bool ApplyConditions(std::string&, std::vector<DetIdItemList>&);
  bool isListed(unsigned int run, std::vector<unsigned int>& vList);


  std::string DBName_, DBTag_, DBuser_, DBpasswd_, DBblob_;
  
  CondCachedIter<HDQMSummary>* Iterator; 
  
  std::vector<unsigned int> iovList;
  std::vector<unsigned int> blackList;
  
  std::vector<unsigned int> vRun_;
  std::vector<float> vSummary_;
  std::vector<DetIdItemList> vDetIdItemList_;
  std::vector<std::string> vlistItems_;
  std::vector<unsigned int> vdetId_;   

  int iDebug;
  int iDoStat;

  const HDQMInspectorConfigBase* fHDQMInspectorConfig;

public:
  TFile *fOutFile;
  
};

#endif
