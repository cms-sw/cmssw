#ifndef DQM_HCALCLIENTUTILS_H
#define DQM_HCALCLIENTUTILS_H

#include "TH1F.h"
#include "TH1.h"
#include "TH2F.h"
#include "TCanvas.h"
#include <string>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMOldReceiver.h"
#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/interface/QReport.h"

#include "TROOT.h"
#include "TGaxis.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

void resetME(const char* name, DQMStore* dbe);

bool isValidGeom(int subdet, int iEta, int iPhi, int depth);

TH2F* getHisto2(string name, string process, DQMStore* dbe_, bool verb=false, bool clone=false);
TH1F* getHisto(string name, string process, DQMStore* dbe_, bool verb=false, bool clone=false);

TH2F* getHisto2(const MonitorElement* me, bool verb=false, bool clone=false);
TH1F* getHisto(const MonitorElement* me, bool verb=false, bool clone=false);

string getIMG(int runNo,TH1F* hist, int size, string htmlDir, const char* xlab, const char* ylab);
string getIMG2(int runNo,TH2F* hist, int size, string htmlDir, const char* xlab, const char* ylab, bool color=false);
  
void histoHTML(int runNo,TH1F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, string htmlDir);
void histoHTML2(int runNo,TH2F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, string htmlDir, bool color=false);

void htmlErrors(int runNo,string htmlDir, string client, string process, DQMStore* dbe, map<string, vector<QReport*> > mapE, map<string, vector<QReport*> > mapW, map<string, vector<QReport*> > mapO);

void createXRangeTest(DQMStore* dbe, vector<string>& params);
void createYRangeTest(DQMStore* dbe, vector<string>& params);
void createMeanValueTest(DQMStore* dbe, vector<string>& params);
void createH2CompTest(DQMStore* dbe, vector<string>& params, TH2F* ref);
void createH2ContentTest(DQMStore* dbe, vector<string>& params);

void dumpHisto(TH1F* hist, vector<string> &names, 
	       vector<double> &meanX, vector<double> &meanY, 
	       vector<double> &rmsX, vector<double> &rmsY);
void dumpHisto2(TH2F* hist, vector<string> &names, 
	       vector<double> &meanX, vector<double> &meanY, 
	       vector<double> &rmsX, vector<double> &rmsY);


// There's got to be a better way than just creating a separate method for each histogram type!
// But so far, template functions haven't worked!
TProfile* getHistoTProfile(string name, string process, DQMStore* dbe_, bool verb=false, bool clone=false);
TProfile* getHistoTProfile(const MonitorElement* me, bool verb=false, bool clone=false);
//getIMG, histoHTML are now deprecated by tools in HcalHistoUtils.h
// Eventually, the getHisto algorithms will become deprecated as well
string getIMGTProfile(int runNo,TProfile* hist, int size, string htmlDir, const char* xlab, const char* ylab,string opts="NONE");
void histoHTMLTProfile(int runNo,TProfile* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, string htmlDir, string opts="NONE");



TProfile2D* getHistoTProfile2D(string name, string process, DQMStore* dbe_, bool verb=false, bool clone=false);
TProfile2D* getHistoTProfile2D(const MonitorElement* me, bool verb=false, bool clone=false);
TH3F* getHistoTH3F(string name, string process, DQMStore* dbe_, bool verb=false, bool clone=false);
TH3F* getHistoTH3F(const MonitorElement* me, bool verb=false, bool clone=false);




#endif

