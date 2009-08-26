//////////////////////////////////////////////////////////
// L1/HLT Menu Class
//////////////////////////////////////////////////////////

#ifndef OHltMenu_h
#define OHltMenu_h

#include <iostream>
#include <vector>
#include <map>
#include <TROOT.h>

#include "HLTDatasets.h"    //SAK

class OHltMenu {
 public:
  OHltMenu();
  
  OHltMenu(bool isL1) { isL1Menu = isL1; }
  virtual ~OHltMenu() {};

  inline bool 		                DoL1preLoop()  {return doL1preloop;}
  inline bool 		                IsL1Menu()  {return isL1Menu;}
  inline bool 		                IsHltMenu()  {return !isL1Menu;}
  inline bool                           IsRealData() {return isRealData;}

  inline unsigned int 		        GetTriggerSize()  {return names.size();}
  inline std::vector<TString>& 		GetTriggerNames() {return names;}
  inline TString  		        GetTriggerName(int i) {return names[i];}
  inline std::map<TString,int>& 	GetPrescaleMap()  {return prescales;}
  inline int 		                GetPrescale(int i)  {return prescales[names[i]];}
  inline int 		                GetPrescale(TString s)  {return prescales[s];}
  inline std::map<TString,float>&       GetEventsizeMap() {return eventSizes;}
  inline float                          GetEventsize(int i) {return eventSizes[names[i]];}
  inline TString  		        GetSeedCondition(int i) {return seedcondition[names[i]];}
  inline TString  		        GetSeedCondition(TString s) {return seedcondition[s];}

  void SetMapL1SeedsOfStandardHLTPath(std::map<TString, std::vector<TString> >);
  std::map<TString, std::vector<TString> >&
    GetL1SeedsOfHLTPathMap() { return map_L1SeedsOfStandardHLTPath; }; // mapping to all seeds

  void AddTrigger(TString trigname, int prescale, float eventSize);
  void AddTrigger(TString trigname, TString seedcond, int prescale, float eventSize);
  void SetIsL1Menu(bool isL1) {isL1Menu=isL1;};
  void SetDoL1preloop(bool doL1prel) {doL1preloop=doL1prel;};
  void SetIsRealData(bool isRealDataLS) {isRealData=isRealDataLS;}
  void print();

  // For L1 prescale preloop to be used in HLT mode only
  void AddL1forPreLoop(TString trigname, int prescale); 
  inline unsigned int 		        GetL1TriggerSize()  {return L1names.size();}
  inline std::vector<TString>& 		GetL1Names() {return L1names;}
  inline TString  		        GetL1TriggerName(int i) {return L1names[i];}
  inline std::map<TString,int>& 	GetL1PrescaleMap() {return L1prescales;}
  inline int                		GetL1Prescale(int i) {return L1prescales[L1names[i]];}
  inline int                		GetL1Prescale(TString s) {return L1prescales[s];}

 private:
  bool isL1Menu;  // if false: is HLTMenu
  bool doL1preloop;  // if false: is HLTMenu
  bool isRealData; // if true: count lumi sections for real data
  std::vector<TString> 		names;
  std::map<TString,TString> 	seedcondition;
  std::map<TString,float>       eventSizes;
  std::map<TString,int> 	prescales;	

  // For L1 prescale preloop to be used in HLT mode only
  std::vector<TString> 		L1names;
  std::map<TString,int> 	L1prescales;	

  std::map<TString, std::vector<TString> > map_L1SeedsOfStandardHLTPath; // mapping to all seeds

};
#endif
