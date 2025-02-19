#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <cstdlib>


void resetME(const char* name, DQMStore* dbe){
  if(dbe==NULL) return;
  MonitorElement* me= dbe->get(name);
  if(me) dbe->softReset(me);
  return;
}

bool isValidGeom(std::string type, int depth)
{
  if (type=="HB" && (depth==1 || depth==2))
    return true;
  if (type=="HE" && (depth==1 || depth==2 || depth==3))
    return true;
  if (type=="HO" && (depth==4))
    return true;
  if (type=="HF" && (depth==1 || depth==2))
    return true;
  return false;
}

bool isValidGeom(int subdet, int iEta, int iPhi, int depth){
  
  if(subdet<0 || subdet>3) return false;
  
  int EtaMin[4]; int EtaMax[4];
  int PhiMin[4]; int PhiMax[4];
  int DepMin[4]; int DepMax[4];
  
  //HB ieta/iphi/depths
  EtaMin[0]=1; EtaMax[0]=16;
  PhiMin[0]=1; PhiMax[0]=72;
  DepMin[0]=1; DepMax[0]=2;
  
  //HE ieta/iPhi/Depths
  EtaMin[1]=16; EtaMax[1]=29;
  PhiMin[1]=1; PhiMax[1]=72;
  DepMin[1]=1; DepMax[1]=3;
  
  //HF ieta/iphi/depths
  EtaMin[2]=29; EtaMax[2]=41;
  PhiMin[2]=1; PhiMax[2]=72;
  DepMin[2]=1; DepMax[2]=2;
  
  //HO ieta/iphi/depths
  EtaMin[3]=1; EtaMax[3]=15;
  PhiMin[3]=1; PhiMax[3]=72;
  DepMin[3]=4; DepMax[3]=4;
  
  if(iEta!=0) if(abs(iEta)<EtaMin[subdet] || abs(iEta)>EtaMax[subdet]) return false;
  if(iPhi!=0) if(abs(iPhi)<PhiMin[subdet] || abs(iPhi)>PhiMax[subdet]) return false;
  if(depth!=0) if(abs(depth)<DepMin[subdet] || abs(depth)>DepMax[subdet]) return false;
  
  
  if(subdet==0 && abs(depth)==2 && abs(iEta)<15) return false;
  else if(subdet==1){
    if(abs(iEta)>20 && (iPhi%2)==0) return false;    
    if(abs(iEta)>39 && (iPhi%4)!=3) return false;    
    if(abs(iEta)==16 && abs(depth)!=3) return false;
    if(abs(iEta)==17 && abs(depth)!=1) return false;
    if(abs(depth)==3){
      if(abs(iEta)!=27 && abs(iEta)!=28 &&abs(iEta)!=16) return false;
    }
  }
  else if(subdet==2 && (iPhi%2)==0) return false;   
  
  return true;
}

void dumpHisto(TH1F* hist, std::vector<std::string> &names, 
	       std::vector<double> &meanX, std::vector<double> &meanY, 
	       std::vector<double> &rmsX, std::vector<double> &rmsY){
  
  names.push_back((std::string)hist->GetName());
  meanX.push_back(hist->GetMean(1));
  meanY.push_back(-123e10);
  rmsX.push_back(hist->GetRMS(1));
  rmsY.push_back(-123e10);  
  return;
}
void dumpHisto2(TH2F* hist, std::vector<std::string> &names, 
	       std::vector<double> &meanX, std::vector<double> &meanY, 
	       std::vector<double> &rmsX, std::vector<double> &rmsY){
  
  names.push_back((std::string)hist->GetName());
  meanX.push_back(hist->GetMean(1));
  meanY.push_back(hist->GetMean(2));
  rmsX.push_back(hist->GetRMS(1));
  rmsY.push_back(hist->GetRMS(2));
  return;
}

void cleanString(std::string& title){

  for ( unsigned int i = 0; i < title.size(); i++ ) {
    if ( title.substr(i, 6) == " - Run" ){
      title.replace(i, title.size()-i, "");
    }
    if ( title.substr(i, 4) == "_Run" ){
      title.replace(i, title.size()-i, "");
    }
    if ( title.substr(i, 5) == "__Run" ){
      title.replace(i, title.size()-i, "");
    }
  }
}

void parseString(std::string& title){
  
  for ( unsigned int i = 0; i < title.size(); i++ ) {
    if ( title.substr(i, 1) == " " ){
      title.replace(i, 1, "_");
    }
    if ( title.substr(i, 1) == "#" ){
      title.replace(i, 1, "N");
    }
    if ( title.substr(i, 1) == "-" ){
      title.replace(i, 1, "_");
    }    
    if ( title.substr(i, 1) == "&" ){
      title.replace(i, 1, "_and_");
    }
    if ( title.substr(i, 1) == "(" 
	 || title.substr(i, 1) == ")" 
	 )  {
      title.replace(i, 1, "_");
    }
  }
  
  return;
}

std::string getIMG2(int runNo,TH2F* hist, int size, std::string htmlDir, const char* xlab, const char* ylab,bool color){

  if(hist==NULL) {
    printf("getIMG2:  This histo is NULL, %s, %s\n",xlab,ylab);
    return "";
  }

  std::string name = hist->GetTitle();
  cleanString(name);
  char dest[512];
  if(runNo>-1) sprintf(dest,"%s - Run %d",name.c_str(),runNo);
  else sprintf(dest,"%s",name.c_str());
  hist->SetTitle(dest);
  std::string title = dest;

  int xwid = 900; int ywid =540;
  if(size==1){
    title = title+"_tmb";
    xwid = 600; ywid = 360;
  }
  TCanvas* can = new TCanvas(dest,dest, xwid, ywid);

  // Put grids on all 2-D histograms
  can->SetGridx();
  can->SetGridy();
  

  parseString(title);
  std::string outName = title + ".gif";
  std::string saveName = htmlDir + outName;
  hist->SetXTitle(xlab);
  hist->SetYTitle(ylab);
  if(color) hist->Draw();
  else{
    hist->SetStats(false);
    hist->Draw("COLZ");
  }
  can->SaveAs(saveName.c_str());  
  delete can;

  return outName;
}

std::string getIMG(int runNo,TH1F* hist, int size, std::string htmlDir, const char* xlab, const char* ylab){

  if(hist==NULL) {
    printf("getIMG:  This histo is NULL, %s, %s\n",xlab,ylab);
    return "";
  }

  std::string name = hist->GetTitle();
  cleanString(name);
  char dest[512];
  if(runNo>-1) sprintf(dest,"%s - Run %d",name.c_str(),runNo);
  else sprintf(dest,"%s",name.c_str());
  hist->SetTitle(dest);
  std::string title = dest;

  int xwid = 900; int ywid =540;
  if(size==1){
    title = title+"_tmb";
    xwid = 600; ywid = 360;
  }
  TCanvas* can = new TCanvas(dest,dest, xwid, ywid);

  parseString(title);
  std::string outName = title + ".gif";
  std::string saveName = htmlDir + outName;
  hist->SetXTitle(xlab);
  hist->SetYTitle(ylab);
  hist->Draw();

  can->SaveAs(saveName.c_str());  
  delete can;

  return outName;
}

TH2F* getHisto2(std::string name, std::string process, DQMStore* dbe_, bool verb, bool clone){

  if(!dbe_) return NULL;

  TH2F* out = NULL;
  char title[150];  
  sprintf(title, "%sHcal/%s",process.c_str(),name.c_str());

  MonitorElement* me = dbe_->get(title);

  if ( me ) {      
    if ( verb) std::cout << "Found '" << title << "'" << std::endl;
    if ( clone) {
      char histo[150];
      sprintf(histo, "ME %s",name.c_str());
      out = dynamic_cast<TH2F*> (me->getTH2F()->Clone(histo));
    } else {
      out = me->getTH2F();
    }
  }
  return out;
}

TH1F* getHisto(std::string name, std::string process, DQMStore* dbe_, bool verb, bool clone){
  if(!dbe_) return NULL;
  
  char title[150];  
  sprintf(title, "%sHcal/%s",process.c_str(),name.c_str());
  TH1F* out = NULL;

  const MonitorElement* me = dbe_->get(title);
  if (me){      
    if ( verb ) std::cout << "Found '" << title << "'" << std::endl;
    if ( clone ) {
      char histo[150];
      sprintf(histo, "ME %s",name.c_str());
      out = dynamic_cast<TH1F*> (me->getTH1F()->Clone(histo));
    } else {
      out = me->getTH1F();
    }
  }

  return out;
}


TH2F* getHisto2(const MonitorElement* me, bool verb,bool clone){
  
  TH2F* out = NULL;

  if ( me ) {      
    if ( verb) std::cout << "Found '" << me->getName() << "'" << std::endl;
    //    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
    if ( clone ) {
      char histo[150];
      sprintf(histo, "ME %s", ((std::string)(me->getName())).c_str());
      out = dynamic_cast<TH2F*> (me->getTH2F()->Clone(histo));
    } else {
      out = me->getTH2F();
    }
  }
  return out;
}

TH1F* getHisto(const MonitorElement* me, bool verb,bool clone){
  TH1F* out = NULL;

  if ( me ) {      
    if ( verb ) std::cout << "Found '" << me->getName() << "'" << std::endl;
    if ( clone ) {
      char histo[150];
      sprintf(histo, "ME %s",((std::string)(me->getName())).c_str());
      out = dynamic_cast<TH1F*> (me->getTH1F()->Clone(histo));
    } else {
      out = me->getTH1F();
    }
 }
  return out;
}


void histoHTML(int runNo, TH1F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, std::string htmlDir){
  
  if(hist!=NULL){    
    std::string imgNameTMB = "";   
    imgNameTMB = getIMG(runNo,hist,1,htmlDir,xlab,ylab); 
    std::string imgName = "";   
    imgName = getIMG(runNo,hist,2,htmlDir,xlab,ylab);  

    if (imgName.size() != 0 )
      htmlFile << "<td><a href=\"" <<  imgName << "\"><img src=\"" <<  imgNameTMB << "\"></a></td>" << std::endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << std::endl;
  }
  else htmlFile << "<td><img src=\"" << " " << "\"></td>" << std::endl;
  return;
}

void histoHTML2(int runNo, TH2F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, std::string htmlDir, bool color){
  if(hist!=NULL){
    std::string imgNameTMB = "";
    imgNameTMB = getIMG2(runNo,hist,1,htmlDir,xlab,ylab,color);  
    std::string imgName = "";
    imgName = getIMG2(runNo,hist,2,htmlDir,xlab,ylab,color);  
    if (imgName.size() != 0 )
      htmlFile << "<td><a href=\"" <<  imgName << "\"><img src=\"" <<  imgNameTMB << "\"></a></td>" << std::endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << std::endl;
  }
  else htmlFile << "<td><img src=\"" << " " << "\"></td>" << std::endl;
  return;
}

void createXRangeTest(DQMStore* dbe, std::vector<std::string>& params){
  if (params.size() < 6) return;
  if(!dbe) return;

  QCriterion* qc = dbe->getQCriterion(params[1]);
  if(qc == NULL){
    qc = dbe->createQTest(ContentsXRange::getAlgoName(),params[1]);
    // Contents within [Xmin, Xmax]
    ContentsXRange* me_qc = (ContentsXRange*) qc;
    //set probability limit for test warning 
    me_qc->setWarningProb(atof(params[2].c_str()));
    //set probability limit for test error 
    me_qc->setErrorProb(atof(params[3].c_str()));
    // set allowed range in X-axis (default values: histogram's X-range)
    me_qc->setAllowedXRange(atof(params[4].c_str()), atof(params[5].c_str()));
  }
  // link it to the monitor element
  dbe->useQTest(params[0], params[1]);
  return;
}

void createYRangeTest(DQMStore* dbe, std::vector<std::string>& params){
  if (params.size() < 6) return;
  if(!dbe) return;

  QCriterion* qc = dbe->getQCriterion(params[1]);
  if(qc == NULL){
    qc = dbe->createQTest(ContentsYRange::getAlgoName(),params[1]);
    // Contents within [Xmin, Xmax]
    ContentsYRange* me_qc = (ContentsYRange*) qc;
    //set probability limit for test warning 
    me_qc->setWarningProb(atof(params[2].c_str()));
    //set probability limit for test error 
    me_qc->setErrorProb(atof(params[3].c_str()));
    // set allowed range in Y-axis (default values: histogram's Y-range)
    me_qc->setAllowedYRange(atof(params[4].c_str()), atof(params[5].c_str()));
  }
  // link it to the monitor element
  dbe->useQTest(params[0], params[1]);
  return;
}

void createMeanValueTest(DQMStore* dbe, std::vector<std::string>& params){
  if (params.size() < 7 ) return;
  if(!dbe) return;

  QCriterion* qc = dbe->getQCriterion(params[1]);
  if(qc == NULL){
    qc = dbe->createQTest("MeanWithinExpected",params[1]);
    // Contents within a mean value
    MeanWithinExpected* me_qc = (MeanWithinExpected*) qc;
    //set probability limit for test warning
    me_qc->setWarningProb(atof(params[2].c_str()));
    //set probability limit for test error
    me_qc->setErrorProb(atof(params[3].c_str()));
    // set Expected Mean
    me_qc->setExpectedMean(atof(params[4].c_str()));
    // set Test Type
    if (params[6] == "useRMS") me_qc->useRMS();
    else if (params[6] == "useSigma") me_qc->useSigma(atof(params[5].c_str()));
  }
  // link it to the monitor element
  dbe->useQTest(params[0], params[1]);
  return;
}

void createH2ContentTest(DQMStore* dbe, std::vector<std::string>& params){
  if (params.size() < 2 ) return;
  if(!dbe) return;

  QCriterion* qc = dbe->getQCriterion(params[1]);
  MonitorElement* me =  dbe->get(params[0]);
  if(me!=NULL && qc == NULL){
    /*  Time to get with 
  DQMServices/Core V03-00-12-qtest
  DQMServices/ClientConfig V03-00-04-qtest
    qc = dbe->createQTest(ContentsTH2FWithinRange::getAlgoName(),params[1]);
    // Contents within a mean value     
    ContentsTH2FWithinRangeROOT* me_qc = dynamic_cast<ContentsTH2FWithinRangeROOT*> (qc);
    me_qc->setMeanRange(0,1e-10);
    me_qc->setRMSRange(0,1e-10);
    // link it to the monitor element
    dbe->useQTest(params[0], params[1]);
    */
  }
  
  return;
}

void createH2CompTest(DQMStore* dbe, std::vector<std::string>& params, TH2F* ref){
  if (params.size() < 2 ) return;
  if(ref==NULL) return;
  if(!dbe) return;

  QCriterion* qc = dbe->getQCriterion(params[1]);
  MonitorElement* me =  dbe->get(params[0]);
  if(me!=NULL && qc == NULL){
    printf("\n\nDon't have this QC, but have the me!\n\n");
    const QReport* qr = me->getQReport(params[1]);
    if(qr) return;
    printf("\n\nThe ME doesn't have the QC!!\n\n");
    qc = dbe->createQTest("Comp2RefEqualH2",params[1]);
    /*  Time to get with 
  DQMServices/Core V03-00-12-qtest
  DQMServices/ClientConfig V03-00-04-qtest
    // Contents within a mean value     
    Comp2RefEqualH2ROOT* me_qc = dynamic_cast<Comp2RefEqualH2ROOT*> (qc);
    //set reference histogram
    me_qc->setReference(ref);
    // link it to the monitor element
    printf("\n\nGonna run it...\n\n");
    dbe->useQTest(params[0], params[1]);
    */
  }
  else printf("\n\nAlready had the QC or didn't have the ME!\n\n");

  return;
}

void htmlErrors(int runNo, std::string htmlDir, std::string client, std::string process, DQMStore* dbe, std::map<std::string, std::vector<QReport*> > mapE, std::map<std::string, std::vector<QReport*> > mapW, std::map<std::string, std::vector<QReport*> > mapO){
  if(!dbe) return;

  std::map<std::string, std::vector<QReport*> >::iterator mapIter;

  ofstream errorFile;
  errorFile.open((htmlDir + client+ "Errors.html").c_str());
  errorFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  errorFile << "<html>  " << std::endl;
  errorFile << "<head>  " << std::endl;
  errorFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  errorFile << " http-equiv=\"content-type\">  " << std::endl;
  errorFile << "  <title>Monitor: Hcal " << client <<" Task Error Output</title> " << std::endl;
  errorFile << "</head>  " << std::endl;
  errorFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  errorFile << "<body>  " << std::endl;
  errorFile << "<br>  " << std::endl;
  errorFile << "<h2>" << client <<" Errors</h2> " << std::endl;

  for (mapIter=mapE.begin(); mapIter!=mapE.end();mapIter++){
    std::string meName = mapIter->first;
    std::vector<QReport*> errors = mapIter->second;
    errorFile << "<br>" << std::endl;     
    errorFile << "<hr>" << std::endl;
    errorFile << "Monitorable '" << meName << "' has the following errors: <br>" << std::endl;
    for(std::vector<QReport*>::iterator report=errors.begin(); report!=errors.end(); report++){
      errorFile << "     "<< (*report)->getQRName() << ": "<< (*report)->getMessage() << std::endl;
    }
    MonitorElement* me = dbe->get(meName);
    errorFile << "<br>" << std::endl;
    errorFile << "<br>" << std::endl;
    const char * substr = strstr(meName.c_str(), client.c_str());
    if(me->getMeanError(2)==0){
      TH1F* obj1f = getHisto(substr, process.c_str(), dbe);
      std::string save = getIMG(runNo,obj1f,1,htmlDir,"X1a","Y1a");
      errorFile << "<img src=\"" <<  save << "\">" << std::endl;
    }
    else{
      TH2F* obj2f = getHisto2(substr, process.c_str(), dbe);
      std::string save = getIMG2(runNo,obj2f,1,htmlDir,"X2a","Y2a");
      errorFile << "<img src=\"" <<  save << "\">" << std::endl;
    }
    errorFile << "<br>" << std::endl;
    errorFile << std::endl;
  }
  errorFile << "<hr>" << std::endl;
  errorFile.close();


  errorFile.open((htmlDir + client+ "Warnings.html").c_str());
  errorFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  errorFile << "<html>  " << std::endl;
  errorFile << "<head>  " << std::endl;
  errorFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  errorFile << " http-equiv=\"content-type\">  " << std::endl;
  errorFile << "  <title>Monitor: Hcal " << client <<" Task Warning Output</title> " << std::endl;
  errorFile << "</head>  " << std::endl;
  errorFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  errorFile << "<body>  " << std::endl;
  errorFile << "<br>  " << std::endl;
  errorFile << "<h2>" << client <<" Warnings</h2> " << std::endl;

  for (mapIter=mapW.begin(); mapIter!=mapW.end();mapIter++){
    std::string meName = mapIter->first;
    std::vector<QReport*> errors = mapIter->second;
    errorFile << "<br>" << std::endl;     
    errorFile << "<hr>" << std::endl;
    errorFile << "Monitorable '" << meName << "' has the following warnings: <BR>" << std::endl;
    for(std::vector<QReport*>::iterator report=errors.begin(); report!=errors.end(); report++){
      errorFile << "     "<< (*report)->getQRName() << ": "<< (*report)->getMessage() << std::endl;
    }
    MonitorElement* me = dbe->get(meName);
    errorFile << "<br>" << std::endl;
    errorFile << "<br>" << std::endl;
    const char * substr = strstr(meName.c_str(), client.c_str());
    if(me->getMeanError(2)==0){
      TH1F* obj1f = getHisto(substr, process.c_str(), dbe);
      std::string save = getIMG(runNo,obj1f,1,htmlDir,"X1b","Y1b");
      errorFile << "<img src=\"" <<  save << "\">" << std::endl;
    }
    else{
      TH2F* obj2f = getHisto2(substr, process.c_str(), dbe);
      std::string save = getIMG2(runNo,obj2f,1,htmlDir,"X2b","Y2b");
      errorFile << "<img src=\"" <<  save << "\">" << std::endl;
    }
    errorFile << "<br>" << std::endl;
    errorFile << std::endl;  
  }
  errorFile << "<hr>" << std::endl;
  errorFile.close();
  
  errorFile.open((htmlDir + client+ "Messages.html").c_str());
  errorFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  errorFile << "<html>  " << std::endl;
  errorFile << "<head>  " << std::endl;
  errorFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  errorFile << " http-equiv=\"content-type\">  " << std::endl;
  errorFile << "  <title>Monitor: Hcal " << client <<" Task Message Output</title> " << std::endl;
  errorFile << "</head>  " << std::endl;
  errorFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  errorFile << "<body>  " << std::endl;
  errorFile << "<br>  " << std::endl;
  errorFile << "<h2>" << client <<" Messages</h2> " << std::endl;

  for (mapIter=mapO.begin(); mapIter!=mapO.end();mapIter++){
    std::string meName = mapIter->first;
    std::vector<QReport*> errors = mapIter->second;
    errorFile << "<br>" << std::endl;     
    errorFile << "<hr>" << std::endl;
    errorFile << "Monitorable '" << meName << "' has the following messages: <br>" << std::endl;
    for(std::vector<QReport*>::iterator report=errors.begin(); report!=errors.end(); report++){
      errorFile << "     "<< (*report)->getQRName() << ": "<< (*report)->getMessage() << std::endl;
    }
    errorFile << "<br>" << std::endl;
    errorFile << "<br>" << std::endl;
    MonitorElement* me = dbe->get(meName);
    const char * substr = strstr(meName.c_str(), client.c_str());
    if(me->getMeanError(2)==0){
      TH1F* obj1f = getHisto(substr, process.c_str(), dbe);
      std::string save = getIMG(runNo,obj1f,1,htmlDir,"X1c","Y1c");
      errorFile << "<img src=\"" <<  save << "\">" << std::endl;
    }
    else{
      TH2F* obj2f = getHisto2(substr, process.c_str(), dbe);
      std::string save = getIMG2(runNo,obj2f,1,htmlDir,"X2c","Y2c");
      errorFile << "<img src=\"" <<  save << "\">" << std::endl;
    }
    errorFile << "<br>" << std::endl;
    errorFile << std::endl;
  }
  errorFile << "<hr>" << std::endl;
  errorFile.close();

  return;

}



// TProfile histogram-related stuff

TProfile* getHistoTProfile(std::string name, std::string process, DQMStore* dbe_, bool verb, bool clone){
  if(!dbe_) return NULL;
  
  char title[150];  
  sprintf(title, "%sHcal/%s",process.c_str(),name.c_str());
  TProfile* out = NULL;

  const MonitorElement* me = dbe_->get(title);
  if (me){      
    if ( verb ) std::cout << "Found '" << title << "'" << std::endl;
    if ( clone ) {
      char histo[150];
      sprintf(histo, "ME %s",name.c_str());
      out = dynamic_cast<TProfile*> (me->getTProfile()->Clone(histo));
    } else {
      out = me->getTProfile();
    }
  }

  return out;
}

TProfile* getHistoTProfile(const MonitorElement* me, bool verb,bool clone){
  TProfile* out = NULL;
  
  if ( me ) {      
    if ( verb ) std::cout << "Found '" << me->getName() << "'" << std::endl;
    if ( clone ) {
      char histo[150];
      sprintf(histo, "ME %s",((std::string)(me->getName())).c_str());
      out = dynamic_cast<TProfile*> (me->getTProfile()->Clone(histo));
    } else {
      out = me->getTProfile();
    }
 }
  return out;
}

std::string getIMGTProfile(int runNo,TProfile* hist, int size, std::string htmlDir, const char* xlab, const char* ylab, std::string opts){

  if(hist==NULL) {
    printf("getIMG:  This histo is NULL, %s, %s\n",xlab,ylab);
    return "";
  }

  std::string name = hist->GetTitle();
  cleanString(name);
  char dest[512];
  if(runNo>-1) sprintf(dest,"%s - Run %d",name.c_str(),runNo);
  else sprintf(dest,"%s",name.c_str());
  hist->SetTitle(dest);
  std::string title = dest;

  int xwid = 900; int ywid =540;
  if(size==1){
    title = title+"_tmb";
    xwid = 600; ywid = 360;
  }
  TCanvas* can = new TCanvas(dest,dest, xwid, ywid);

  parseString(title);
  std::string outName = title + ".gif";
  std::string saveName = htmlDir + outName;
  hist->SetXTitle(xlab);
  hist->SetYTitle(ylab);
  if (opts!="NONE")
    hist->SetOption(opts.c_str());
  hist->Draw();

  can->SaveAs(saveName.c_str());  
  delete can;

  return outName;
}

void histoHTMLTProfile(int runNo, TProfile* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, std::string htmlDir, std::string opts){
  
  if(hist!=NULL){    
    std::string imgNameTMB = "";   
    imgNameTMB = getIMGTProfile(runNo,hist,1,htmlDir,xlab,ylab,opts); 
    std::string imgName = "";   
    imgName = getIMGTProfile(runNo,hist,2,htmlDir,xlab,ylab,opts);  

    if (imgName.size() != 0 )
      htmlFile << "<td><a href=\"" <<  imgName << "\"><img src=\"" <<  imgNameTMB << "\"></a></td>" << std::endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << std::endl;
  }
  else htmlFile << "<td><img src=\"" << " " << "\"></td>" << std::endl;
  return;
}


// Get Histograms for TProfile2D

TProfile2D* getHistoTProfile2D(std::string name, std::string process, DQMStore* dbe_, bool verb, bool clone){
  if(!dbe_) return NULL;
  
  char title[150];  
  sprintf(title, "%sHcal/%s",process.c_str(),name.c_str());
  TProfile2D* out = NULL;

  const MonitorElement* me = dbe_->get(title);
  if (me){      
    if ( verb ) std::cout << "Found '" << title << "'" << std::endl;
    if ( clone ) {
      char histo[150];
      sprintf(histo, "ME %s",name.c_str());
      out = dynamic_cast<TProfile2D*> (me->getTProfile2D()->Clone(histo));
    } else {
      out = me->getTProfile2D();
    }
  }

  return out;
}

TProfile2D* getHistoTProfile2D(const MonitorElement* me, bool verb,bool clone){
  TProfile2D* out = NULL;
  
  if ( me ) {      
    if ( verb ) std::cout << "Found '" << me->getName() << "'" << std::endl;
    if ( clone ) {
      char histo[150];
      sprintf(histo, "ME %s",((std::string)(me->getName())).c_str());
      out = dynamic_cast<TProfile2D*> (me->getTProfile2D()->Clone(histo));
    } else {
      out = me->getTProfile2D();
    }
 }
  return out;
}

// Get histograms for TH3F
TH3F* getHistoTH3F(std::string name, std::string process, DQMStore* dbe_, bool verb, bool clone){
  if(!dbe_) return NULL;
  
  char title[150];  
  sprintf(title, "%sHcal/%s",process.c_str(),name.c_str());
  TH3F* out = NULL;

  const MonitorElement* me = dbe_->get(title);
  if (me){      
    if ( verb ) std::cout << "Found '" << title << "'" << std::endl;
    if ( clone ) {
      char histo[150];
      sprintf(histo, "ME %s",name.c_str());
      out = dynamic_cast<TH3F*> (me->getTH3F()->Clone(histo));
    } else {
      out = me->getTH3F();
    }
  }

  return out;
}

TH3F* getHistoTH3F(const MonitorElement* me, bool verb,bool clone){
  TH3F* out = NULL;
  
  if ( me ) {      
    if ( verb ) std::cout << "Found '" << me->getName() << "'" << std::endl;
    if ( clone ) {
      char histo[150];
      sprintf(histo, "ME %s",((std::string)(me->getName())).c_str());
      out = dynamic_cast<TH3F*> (me->getTH3F()->Clone(histo));
    } else {
      out = me->getTH3F();
    }
 }
  return out;
}
