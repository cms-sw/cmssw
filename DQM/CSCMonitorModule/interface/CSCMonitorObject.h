#ifndef CSCMonitorObject_h
#define CSCMonitorObject_h


#include <iostream>
#include <string>
#include <map>
#include <ext/hash_map>
#include <string>
#include <iomanip>
#include <set>
#include <sstream>

// #include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNodeList.hpp>

/** BEGIN FIX **/
namespace gnu=__gnu_cxx;
namespace __gnu_cxx
{
  template<> struct hash< std::string >
  {
    size_t operator()( const std::string& x ) const
    {
      return hash< const char* >()( x.c_str() );
    }
  };
}
/** END FIX **/


using namespace XERCES_CPP_NAMESPACE;

// ==  ROOT Section
#include <TROOT.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TString.h>
#include <TFile.h>
#include <TRandom.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TPaveStats.h>
#include <TColor.h>
#include <TPaletteAxis.h>
#include <TCollection.h>

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"



// class MonitorElement: public TH1 {};
typedef MonitorElement CSCMonitorElement;
// typedef TH1 MonitorElement;
// typedef TH1 CSCMonitorElement;

class CSCMonitorObject;

typedef gnu::hash_map<std::string, CSCMonitorObject*> ME_List;
typedef ME_List::iterator ME_List_iterator;
typedef ME_List::const_iterator ME_List_const_iterator;

#define DEF_HISTO_COLOR 48


class CSCMonitorObject
{
	
 public:
  CSCMonitorObject();
  // == Copy constructor
  CSCMonitorObject(const CSCMonitorObject&);

  //	CSCMonitorObject();	
  CSCMonitorObject(DOMNode *info);
  ~CSCMonitorObject();
  CSCMonitorObject& operator=(const CSCMonitorObject&);
  bool operator<( const CSCMonitorObject& s1)
    {return (getFullName()<s1.getFullName());};
  bool operator>(const CSCMonitorObject& s1)
    {return (getFullName()>s1.getFullName());};
  bool operator==(const CSCMonitorObject& s1)
    {return (getFullName()==s1.getFullName());};


  int Book(DaqMonitorBEInterface* dbe);
  int Book(DOMNode *info, DaqMonitorBEInterface* dbe);
  int Fill(double);
  // can be used with 2D (x,y) or 1D (x, w) histograms
  int Fill(double, double);
  // can be used with 3D (x, y, z) or 2D (x, y, w) histograms
  int Fill(double, double, double);
  // can be used with 3D (x, y, z, w) histograms
  int Fill(double, double, double, double);

  CSCMonitorElement* getObject() {return object;}
  void setPrefix(std::string);
  std::string getPrefix() const {return prefix;}
  std::string getName() const {return name;}
  void setName(std::string);
  std::string getTitle() const {return title;}
  std::string getFolder() const {return folder;}
  void setFolder(std::string newfolder) {folder = newfolder;}
  void setTitle(std::string);
  int setParameter(std::string, std::string);	
  std::string getParameter(std::string);
  int setParameters(std::map<std::string, std::string>, bool resetParams = true);
  std::map<std::string, std::string>getParameters() const { return params;}
  std::string getFullName() const { return type+"_"+prefix+"_"+name;}

  void SetEntries(double);
  void SetBinContent(int, double);
  void SetBinContent(int, int, double);
  double GetBinContent(int);
  double GetBinContent(int, int);
  void SetAxisRange(double, double, std::string);
  void SetAxisRange(double, double, int);	
  int GetMaximumBin();
  double GetEntries();
  void SetNormFactor(double);
  void SetBinError(int, double);
  double GetBinError(int);
  void SetBinLabel(int, std::string, int);
	
  void Reset();
  void Write();
  void Draw();
	
  int getQTestResult() const {return QTest_result;}
  int doQTest();


 private:

  int parseDOMNode(DOMNode* info);
 protected:
  CSCMonitorElement* object;
  std::map<std::string, std::string>params;
  std::string type;
  std::string prefix;
  std::string name;
  std::string title;
  std::string folder;
  int QTest_result;
};
/*
  bool operator<(const CSCMonitorObject& s1, const CSCMonitorObject& s2) 
  {return (s1.getFullName()<s2.getFullName());};
  bool operator>(const CSCMonitorObject& s1, const CSCMonitorObject& s2)
  {return (s1.getFullName()>s2.getFullName());};
  bool operator==(const CSCMonitorObject& s1, const CSCMonitorObject& s2)
  {return (s1.getFullName()==s2.getFullName());};
*/


#endif
