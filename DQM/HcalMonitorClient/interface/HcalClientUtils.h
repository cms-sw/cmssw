#ifndef DQM_HCALCLIENTUTILS_H
#define DQM_HCALCLIENTUTILS_H

#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include <string>
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/QualityTests/interface/QCriterionRoot.h"
#include "DQMServices/Core/interface/QReport.h"

#include "TROOT.h"
#include "TGaxis.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

TH2F* getHisto2(string name, string process, MonitorUserInterface* mui_, bool verb, bool clone);
TH1F* getHisto(string name, string process, MonitorUserInterface* mui_, bool verb, bool clone);

TH2F* getHisto2(MonitorElement* me, bool verb, bool clone);
TH1F* getHisto(MonitorElement* me, bool verb, bool clone);

string getPNG(TH1F* hist, int size, string htmlDir, const char* xlab, const char* ylab,bool save=true);
string getPNG2(TH2F* hist, int size, string htmlDir, const char* xlab, const char* ylab,bool save = true);
  
void histoHTML(TH1F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, string htmlDir);
void histoHTML2(TH2F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, string htmlDir);

void htmlErrors(string htmlDir, string client, string process, MonitorUserInterface* mui, map<string, vector<QReport*> > mapE, map<string, vector<QReport*> > mapW, map<string, vector<QReport*> > mapO);

void createXRangeTest(MonitorUserInterface* mui, vector<string>& params);
void createYRangeTest(MonitorUserInterface* mui, vector<string>& params);
void createMeanValueTest(MonitorUserInterface* mui, vector<string>& params);
void createH2CompTest(MonitorUserInterface* mui, vector<string>& params, TH2F* ref);
void createH2ContentTest(MonitorUserInterface* mui, vector<string>& params);

#endif
