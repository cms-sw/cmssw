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
#include <fstream>
#include "cmath"
#include "CondCore/Utilities/interface/CondCachedIter.h"
#include "CondFormats/DQMObjects/interface/HDQMSummary.h"
#include "DQMServices/Diagnostic/interface/HDQMInspectorConfigBase.h"
#include "TFile.h"
#include "TGraphErrors.h"

class HDQMInspector
{
 public:  
  HDQMInspector():
    DBName_(""),
    DBTag_(""),
    DBauth_(""),
    Iterator(0),
    iDebug(0),
    iDoStat(0),
    fSkip99s(false),
    fSkip0s(false),
    fHDQMInspectorConfig(0x0),
    fSep("@")
    {
    };
  
  HDQMInspector(const HDQMInspectorConfigBase* InConfig):
    DBName_(""),
    DBTag_(""),
    DBauth_(""),
    Iterator(0),
    iDebug(0),
    iDoStat(0),
    fSkip99s(false),
    fSkip0s(false),
    fHDQMInspectorConfig(InConfig),
    fSep("@")
    {
    };
  
  virtual ~HDQMInspector() {
    delete Iterator;
  };
  struct DetIdItemList {
    unsigned int detid;
    std::vector<std::string> items;
    std::vector<float> values;
  };

  void setDB(const std::string & DBName, const std::string & DBTag, const std::string & DBauth = "");
  void createTrend(const std::string ListItems, const std::string CanvasName="", const int logy=0, const std::string Conditions="",
                   std::string const& Labels="", const unsigned int firstRun=1, const unsigned int lastRun=0xFFFFFFFE, int const UseYRange = 0, double const& YMin = 999999, double const& YMax = -999999);
  void createTrendLastRuns(const std::string ListItems, const std::string CanvasName="",
                           const int logy=0, const std::string Conditions="", std::string const& Labels="", const unsigned int nRuns=10, int const UseYRange = 0, double const& YMin = 999999, double const& YMax = -999999); 
  void setDebug(int i){iDebug=i;}
  void setDoStat(int i){iDoStat=i;}
  void setBlackList(std::string const& ListItems);
  void setWhiteList(std::string const& ListItems);
  std::string readListFromFile(const std::string & listFileName);
  void setSkip99s (bool const in) {
    fSkip99s = in;
    return;
  }
  void setSkip0s (bool const in) {
    fSkip0s = in;
    return;
  }
  void closeFile ()
  { 
    if( fOutFile ) {
      fOutFile->Close();
    }
  }
  double findGraphMax(TGraphErrors*);
  double findGraphMin(TGraphErrors*);
  void setSeparator(std::string const in) {
    fSep = in;
    return;
  }

  
  inline std::vector<unsigned int> getRuns() { return vRun_;}
  inline std::vector<float> getSummary()     { return vSummary_;}
 
  inline std::vector<std::string>  getListItems() { return vlistItems_;}
  inline std::vector<unsigned int> getvDetId()    { return vdetId_;}
 
private:

  void style();
  void plot(size_t& nPads, std::string CanvasName, int logy=0, std::string const& Labels = "", int const UseYRange = 0, double const XMin = 999999, double const YMin = -999999);
  void accessDB();
  void InitializeIOVList();
  bool setRange(unsigned int& firstRun, unsigned int& lastRun);
  void setItems(std::string);
  size_t unpackItems(std::string& );
  void unpackConditions(std::string&, std::vector<DetIdItemList>&);
  bool ApplyConditions(std::string&, std::vector<DetIdItemList>&);
  bool isListed(unsigned int run, std::vector<unsigned int>& vList);


  std::string DBName_, DBTag_, DBauth_;
  
  CondCachedIter<HDQMSummary>* Iterator; 
  
  std::vector<unsigned int> iovList;
  std::vector<unsigned int> blackList;
  std::vector<unsigned int> whiteList;
  
  std::vector<unsigned int> vRun_;
  std::vector<float> vSummary_;
  std::vector<DetIdItemList> vDetIdItemList_;
  std::vector<std::string> vlistItems_;
  std::vector<unsigned int> vdetId_;   

  int iDebug;
  int iDoStat;
  bool fSkip99s;
  bool fSkip0s;

  const HDQMInspectorConfigBase* fHDQMInspectorConfig;

  std::string fSep;

public:
  TFile *fOutFile;
  
};

#endif
