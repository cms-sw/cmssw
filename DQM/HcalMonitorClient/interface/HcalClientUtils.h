#ifndef DQM_HCALCLIENTUTILS_H
#define DQM_HCALCLIENTUTILS_H

#include "TH1F.h"
#include "TH1.h"
#include "TH2F.h"
#include "TCanvas.h"
#include <string>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/interface/QReport.h"

#include "TROOT.h"
#include "TGaxis.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>


class HcalUtilsClient 
{
  // Stolen directly from EcalCommon/interface/UtilsClient.h
 public:
  
  /*! \fn template<class T> static T getHisto( const MonitorElement* me, bool clone = false, T ret = 0 )
    \brief Returns the histogram contained by the Monitor Element
    \param me Monitor Element
    \param clone (boolean) if true clone the histogram 
    \param ret in case of clonation delete the histogram first
    \param debug  dump out debugging info
  */
  template<class T> static T getHisto( const MonitorElement* me, bool clone = false, T ret = 0,int debug=0) {
    if( me ) {
      if (debug>0) std::cout << "Found '" << me->getName() <<"'" << std::endl;
      TObject* ob = const_cast<MonitorElement*>(me)->getRootObject();
      if( ob ) { 
        if( clone ) {
          if( ret ) {
            delete ret;
          }
          std::string s = "ME " + me->getName();
          ret = dynamic_cast<T>(ob->Clone(s.c_str())); 
          if( ret ) {
            ret->SetDirectory(0);
          }
        } else {
          ret = dynamic_cast<T>(ob); 
        }
      } else {
        ret = 0;
      }
    } else {
      if( !clone ) {
        ret = 0;
      }
    }
    return ret;
  }

}; // class HcalUtilsClient




void resetME(const char* name, DQMStore* dbe);

bool isValidGeom(std::string type, int depth);
bool isValidGeom(int subdet, int iEta, int iPhi, int depth);

TH2F* getHisto2(std::string name, std::string process, DQMStore* dbe_, bool verb=false, bool clone=false);
TH1F* getHisto(std::string name, std::string process, DQMStore* dbe_, bool verb=false, bool clone=false);

TH2F* getHisto2(const MonitorElement* me, bool verb=false, bool clone=false);
TH1F* getHisto(const MonitorElement* me, bool verb=false, bool clone=false);

std::string getIMG(int runNo,TH1F* hist, int size, std::string htmlDir, const char* xlab, const char* ylab);
std::string getIMG2(int runNo,TH2F* hist, int size, std::string htmlDir, const char* xlab, const char* ylab, bool color=false);
  
void histoHTML(int runNo,TH1F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, std::string htmlDir);
void histoHTML2(int runNo,TH2F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, std::string htmlDir, bool color=false);

void htmlErrors(int runNo,std::string htmlDir, std::string client, std::string process, DQMStore* dbe, std::map<std::string, std::vector<QReport*> > mapE, std::map<std::string, std::vector<QReport*> > mapW, std::map<std::string, std::vector<QReport*> > mapO);

void createXRangeTest(DQMStore* dbe, std::vector<std::string>& params);
void createYRangeTest(DQMStore* dbe, std::vector<std::string>& params);
void createMeanValueTest(DQMStore* dbe, std::vector<std::string>& params);
void createH2CompTest(DQMStore* dbe, std::vector<std::string>& params, TH2F* ref);
void createH2ContentTest(DQMStore* dbe, std::vector<std::string>& params);

void dumpHisto(TH1F* hist, std::vector<std::string> &names, 
	       std::vector<double> &meanX, std::vector<double> &meanY, 
	       std::vector<double> &rmsX, std::vector<double> &rmsY);
void dumpHisto2(TH2F* hist, std::vector<std::string> &names, 
	       std::vector<double> &meanX, std::vector<double> &meanY, 
	       std::vector<double> &rmsX, std::vector<double> &rmsY);


// There's got to be a better way than just creating a separate method for each histogram type!
// But so far, template functions haven't worked!
TProfile* getHistoTProfile(std::string name, std::string process, DQMStore* dbe_, bool verb=false, bool clone=false);
TProfile* getHistoTProfile(const MonitorElement* me, bool verb=false, bool clone=false);
//getIMG, histoHTML are now deprecated by tools in HcalHistoUtils.h
// Eventually, the getHisto algorithms will become deprecated as well
std::string getIMGTProfile(int runNo,TProfile* hist, int size, std::string htmlDir, const char* xlab, const char* ylab,std::string opts="NONE");
void histoHTMLTProfile(int runNo,TProfile* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, std::string htmlDir, std::string opts="NONE");



TProfile2D* getHistoTProfile2D(std::string name, std::string process, DQMStore* dbe_, bool verb=false, bool clone=false);
TProfile2D* getHistoTProfile2D(const MonitorElement* me, bool verb=false, bool clone=false);
TH3F* getHistoTH3F(std::string name, std::string process, DQMStore* dbe_, bool verb=false, bool clone=false);
TH3F* getHistoTH3F(const MonitorElement* me, bool verb=false, bool clone=false);




#endif

