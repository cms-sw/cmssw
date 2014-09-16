#ifndef BTagDifferentialPlot_H
#define BTagDifferentialPlot_H

// #include "BTagPlotPrintC.h"

#include "TH1F.h"
#include "TCanvas.h"

#include <vector>
#include <string>

#include "DQMOffline/RecoB/interface/EtaPtBin.h"
#include "DQMOffline/RecoB/interface/JetTagPlotter.h"
#include "DQMServices/Core/interface/DQMStore.h"

class BTagDifferentialPlot {
  
 public:

  enum ConstVarType {constPT, constETA };

  BTagDifferentialPlot (const double& bEff, const ConstVarType& constVariable, const std::string & tagName, DQMStore::IBooker & ibook) ;

  ~BTagDifferentialPlot () ;


  void addBinPlotter ( JetTagPlotter * aPlotter ) { theBinPlotters.push_back ( aPlotter ) ; }

  void process () ;


  void epsPlot(const std::string & name);

  void psPlot(const std::string & name);

  void plot (TCanvas & theCanvas) ;

  void plot(const std::string & name, const std::string & ext);


// 
//   void print () const ;

  TH1F * getDifferentialHistoB_d    () { return theDifferentialHistoB_d ->getTH1F()   ; }
  TH1F * getDifferentialHistoB_u    () { return theDifferentialHistoB_u ->getTH1F()   ; }
  TH1F * getDifferentialHistoB_s    () { return theDifferentialHistoB_s ->getTH1F()   ; }
  TH1F * getDifferentialHistoB_c    () { return theDifferentialHistoB_c ->getTH1F()   ; }
  TH1F * getDifferentialHistoB_b    () { return theDifferentialHistoB_b ->getTH1F()   ; }
  TH1F * getDifferentialHistoB_g    () { return theDifferentialHistoB_g ->getTH1F()   ; }
  TH1F * getDifferentialHistoB_ni   () { return theDifferentialHistoB_ni->getTH1F()   ; }
  TH1F * getDifferentialHistoB_dus  () { return theDifferentialHistoB_dus->getTH1F()  ; }
  TH1F * getDifferentialHistoB_dusg () { return theDifferentialHistoB_dusg->getTH1F() ; }
  TH1F * getDifferentialHistoB_pu   () { return theDifferentialHistoB_pu->getTH1F()   ; }

  
 private:
  
  void setVariableName () ;
  
  void bookHisto () ;


  void fillHisto () ;
  std::pair<double, double> getMistag(const double& fixedBEfficiency, TH1F * effPurHist);


  // the fixed b-efficiency (later: allow more than one) for which the misids have to be plotted
  double fixedBEfficiency ;
  
  // flag if processing should be skipped
  bool noProcessing ;
  bool processed;

  ConstVarType constVar;
  // the name for the variable with constant value
  std::string constVariableName ;
  // the name of the variable to be plotted on the x-axis (e.g. "eta", "pt")
  std::string diffVariableName ;

  // value of the constant variable (lower/upper edge of interval)
  std::pair<double,double> constVariableValue ;

  // the common name to describe histograms
  std::string commonName ;
  DQMStore::IBooker & ibook_;

  // the input
  std::vector<JetTagPlotter *> theBinPlotters ;

  // the histo to create/fill
  MonitorElement * theDifferentialHistoB_d    ;
  MonitorElement * theDifferentialHistoB_u    ;
  MonitorElement * theDifferentialHistoB_s    ;
  MonitorElement * theDifferentialHistoB_c    ;
  MonitorElement * theDifferentialHistoB_b    ;
  MonitorElement * theDifferentialHistoB_g    ;
  MonitorElement * theDifferentialHistoB_ni   ;
  MonitorElement * theDifferentialHistoB_dus  ;
  MonitorElement * theDifferentialHistoB_dusg ;
  MonitorElement * theDifferentialHistoB_pu   ;

  // the plot Canvas
//   TCanvas * thePlotCanvas ;

  
} ;


#endif
