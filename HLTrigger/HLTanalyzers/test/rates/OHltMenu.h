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
  std::map<TString,int> 	l1Prescale;	
  std::map<TString,int> 	hltPrescale;	
  std::map<TString,int> 	totalPrescale;	

  OHltMenu();
  virtual ~OHltMenu() { };

  inline std::vector<TString> 			GetHlts() {return hlts;}
  inline std::map<TString,TString> 		GetHltL1BitMap() {return hltL1Bit;}
  inline std::map<TString,TString> 		GetHltThresholdMap() {return hltThreshold;}
  inline std::map<TString,TString> 		GetHltDescriptionMap() {return hltDescription;}
  inline std::map<TString,int>		      	GetL1PrescaleMap() {return l1Prescale;}
  inline std::map<TString,int>		      	GetHltPrescaleMap() {return hltPrescale;}
  inline std::map<TString,int>		      	GetTotalPrescaleMap() {return totalPrescale;}

  void AddHlt(TString trig, TString l1Bit, int prescale, TString threshold, TString desc);
  void AddHlt(TString trig, TString l1Bit, int l1prescale, int hltprescale, TString threshold, TString desc);

};
#endif

#ifdef OHltMenu_cxx
OHltMenu::OHltMenu() {

}

#endif

