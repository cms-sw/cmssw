//////////////////////////////////////////////////////////
// Class to hold Open HLT Menu
// in form of maps b/w path names and L1Bits, Thresholds, 
// Descriptions, Prescales.
// Author:  Vladimir Rekovic (UMN)   Date: Mar, 2008
//////////////////////////////////////////////////////////

#ifndef OHltMenu_h
#define OHltMenu_h

#include <vector>
#include <map>
#include <TROOT.h>

class OHltMenu {
 public:

  std::vector<TString> 		hlts;
  std::map<TString,TString> 	hltL1Bit;
  std::map<TString,TString> 	hltThreshold;
  std::map<TString,TString> 	hltDescription;
  std::map<TString,TString> 	l1Prescale;	
  std::map<TString,int> 	hltPrescale;	
  std::map<TString,int> 	totalPrescale;	
  std::map<TString,double>      eventSize;
  std::map<TString,int> 	hltmulele;	
  std::map<TString,int> 	hltmulpho;	
  std::map<TString,int> 	hltmulmu;	
  std::map<TString,int> 	hltmuljets;	
  std::map<TString,int> 	hltmulmet;	
  std::vector<TString>          levelones;
  std::map<TString,int>         levelonePrescale;

  OHltMenu();
  virtual ~OHltMenu() { };

  inline std::vector<TString> 			GetHlts() {return hlts;}
  inline std::map<TString,TString> 		GetHltL1BitMap() {return hltL1Bit;}
  inline std::map<TString,TString> 		GetHltThresholdMap() {return hltThreshold;}
  inline std::map<TString,TString> 		GetHltDescriptionMap() {return hltDescription;}
  inline std::map<TString,TString>		GetL1PrescaleMap() {return l1Prescale;}
  inline std::map<TString,int>		      	GetHltPrescaleMap() {return hltPrescale;}
  inline std::map<TString,int>		      	GetTotalPrescaleMap() {return totalPrescale;}
  inline std::map<TString,double>               GetEventsizeMap() {return eventSize;}
  inline std::map<TString,int>                  GetMultEleMap() {return hltmulele;}
  inline std::map<TString,int>                  GetMultPhoMap() {return hltmulpho;}
  inline std::map<TString,int>                  GetMultMuMap() {return hltmulmu;}
  inline std::map<TString,int>                  GetMultJetsMap() {return hltmuljets;}
  inline std::map<TString,int>                  GetMultMETMap() {return hltmulmet;}
  inline std::vector<TString>                   GetAllL1s() {return levelones;} 
  inline std::map<TString,int>                  GetAllL1PrescaleMap() {return levelonePrescale;}

  void AddHlt(TString trig, TString l1Bit, int prescale, TString threshold, TString desc);
  void AddHlt(TString trig, TString l1Bit, int l1prescale, int hltprescale, TString threshold, TString desc);
  void AddHlt(TString trig, TString l1Bit, int l1prescale, int hltprescale, TString threshold, TString desc, double eventsize, int multele, int multpho, int multmu, int multjet, int multmet);
  void AddHlt(TString trig, TString l1Bit, TString l1prescale, int hltprescale, TString threshold, TString desc, double eventsize, int multele, int multpho, int multmu, int multjets, int multmet); 
  void AddL1(TString trig, int l1prescale);
};
#endif

#ifdef OHltMenu_cxx
OHltMenu::OHltMenu() {

}

#endif

