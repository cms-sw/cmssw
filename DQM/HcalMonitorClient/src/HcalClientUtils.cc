#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include <sys/time.h>

string getIMG2(TH2F* hist, int size, string htmlDir, const char* xlab, const char* ylab, bool save){
  //  timeval a,b;
  //  gettimeofday(&a,NULL);

  if(hist==NULL) {
    printf("getIMG2:  This histo is NULL, %s, %s\n",xlab,ylab);
    return "";
  }
  
  string title = hist->GetTitle();
  int xwid = 900; int ywid =540;
  if(size==1){
    title = title+"_tmb";
    xwid = 600; ywid = 360;
  }
  TCanvas* can = new TCanvas(title.c_str(), "tmp can2", xwid, ywid);

  for ( unsigned int i = 0; i < title.size(); i++ ) {
    if ( title.substr(i, 1) == " " )  {
      title.replace(i, 1, "_");
    }
    if ( title.substr(i, 1) == "#" )  {
      title.replace(i, 1, "N");
    }
  }

  string outName = title + ".png";
  if(!save) return outName;

  string saveName = htmlDir + outName;
  hist->SetXTitle(xlab);
  hist->SetYTitle(ylab);
  hist->SetDrawOption("box");
  hist->Draw();
  can->SaveAs(saveName.c_str());  
  delete can;
  /*
  gettimeofday(&b,NULL);
  double t1=a.tv_sec+(a.tv_usec/1000000.0);
  double t2=b.tv_sec+(b.tv_usec/1000000.0);
  printf("\ngetIMG2:  %s\n",title.c_str());
  printf("getIMG2:  %.6f seconds elapsed\n", t2-t1);
  */
  return outName;
}

string getIMG(TH1F* hist, int size, string htmlDir, const char* xlab, const char* ylab, bool save){
  //  timeval a,b;
  //  gettimeofday(&a,NULL);

  if(hist==NULL) {
    printf("getIMG:  This histo is NULL, %s, %s\n",xlab,ylab);
    return "";
  }

  string title = hist->GetTitle();
  int xwid = 900; int ywid =540;
  if(size==1){
    title = title+"_tmb";
    xwid = 600; ywid = 360;
  }
  TCanvas* can = new TCanvas(title.c_str(), "tmp can", xwid, ywid);
  
  for ( unsigned int i = 0; i < title.size(); i++ ) {
    if ( title.substr(i, 1) == " " )  {
      title.replace(i, 1, "_");
    }
    if ( title.substr(i, 1) == "#" )  {
      title.replace(i, 1, "N");
    }
  }
  
  string outName = title + ".png";
  if(!save) return outName;
  
  string saveName = htmlDir + outName;
  hist->SetXTitle(xlab);
  hist->SetYTitle(ylab);
  hist->Draw();
  can->SaveAs(saveName.c_str());  
  delete can;
  /*
  gettimeofday(&b,NULL);
  double t1=a.tv_sec+(a.tv_usec/1000000.0);
  double t2=b.tv_sec+(b.tv_usec/1000000.0);
  printf("\ngetIMG:  %s\n",title.c_str());
  printf("getIMG:  %.6f seconds elapsed\n", t2-t1);
  */
  return outName;
}

TH2F* getHisto2(string name, string process, MonitorUserInterface* mui_, bool verb, bool clone, TH2F* out){

  char title[150];  
  sprintf(title, "Collector/%s/HcalMonitor/%s",process.c_str(),name.c_str());

  const MonitorElement* me = mui_->get(title);

  if ( me ) {      
    if ( verb) cout << "Found '" << title << "'" << endl;
    //    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );

    if ( ob ) {
      if ( clone) {
	char histo[150];
	sprintf(histo, "ME %s",name.c_str());
	out = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
      } else {
	out = dynamic_cast<TH2F*> (ob->operator->());
      }
    }
  }
  return out;
}

TH1F* getHisto(string name, string process, MonitorUserInterface* mui_, bool verb, bool clone, TH1F* out){

  char title[150];  
  sprintf(title, "Collector/%s/HcalMonitor/%s",process.c_str(),name.c_str());

  const MonitorElement* me = mui_->get(title);
  if ( me ) {      
    if ( verb ) cout << "Found '" << title << "'" << endl;
    //    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
    if ( ob ) {
      if ( clone ) {
	char histo[150];
	sprintf(histo, "ME %s",name.c_str());
	out = dynamic_cast<TH1F*> ((ob->operator->())->Clone(histo));
      } else {
	out = dynamic_cast<TH1F*> (ob->operator->());
      }
    }
  }
  return out;
}


TH2F* getHisto2(const MonitorElement* me, bool verb,bool clone, TH2F* out){
  
  if ( me ) {      
    if ( verb) cout << "Found '" << me->getName() << "'" << endl;
    //    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
    if ( ob ) {
      if ( clone ) {
	char histo[150];
	sprintf(histo, "ME %s", ((string)(me->getName())).c_str());
	out = dynamic_cast<TH2F*> ((ob->operator->())->Clone(histo));
      } else {
	out = dynamic_cast<TH2F*> (ob->operator->());
      }
    }
  }
  return out;
}

TH1F* getHisto(const MonitorElement* me, bool verb,bool clone, TH1F* out){
  //  TH1F* out = NULL;

  if ( me ) {      
    if ( verb ) cout << "Found '" << me->getName() << "'" << endl;
    //    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
    if ( ob ) {
      if ( clone ) {
	char histo[150];
	sprintf(histo, "ME %s",((string)(me->getName())).c_str());
	out = dynamic_cast<TH1F*> ((ob->operator->())->Clone(histo));
      } else {
	out = dynamic_cast<TH1F*> (ob->operator->());
      }
    }
 }
  return out;
}


void histoHTML(TH1F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, string htmlDir){
  if(hist!=NULL){    
    string imgNameTMB = "";   
    imgNameTMB = getIMG(hist,1,htmlDir,xlab,ylab); 
    string imgName = "";   
    imgName = getIMG(hist,2,htmlDir,xlab,ylab);  
    if (imgName.size() != 0 )
      htmlFile << "<td><a href=\"" <<  imgName << "\"><img src=\"" <<  imgNameTMB << "\"></a></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }
  else htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  return;
}

void histoHTML2(TH2F* hist, const char* xlab, const char* ylab, int width, ofstream& htmlFile, string htmlDir){
  if(hist!=NULL){
    string imgNameTMB = "";
    imgNameTMB = getIMG2(hist,1,htmlDir,xlab,ylab);  
    string imgName = "";
    imgName = getIMG2(hist,2,htmlDir,xlab,ylab);  
    if (imgName.size() != 0 )
      htmlFile << "<td><a href=\"" <<  imgName << "\"><img src=\"" <<  imgNameTMB << "\"></a></td>" << endl;
    else
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  }
  else htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  return;
}

void createXRangeTest(MonitorUserInterface* mui, vector<string>& params){
  if (params.size() < 6) return;

  QCriterion* qc = mui->getQCriterion(params[1]);
  if(qc == NULL){
    qc = mui->createQTest(ContentsXRangeROOT::getAlgoName(),params[1]);
    // Contents within [Xmin, Xmax]
    MEContentsXRangeROOT* me_qc = (MEContentsXRangeROOT*) qc;
    //set probability limit for test warning 
    me_qc->setWarningProb(atof(params[2].c_str()));
    //set probability limit for test error 
    me_qc->setErrorProb(atof(params[3].c_str()));
    // set allowed range in X-axis (default values: histogram's X-range)
    me_qc->setAllowedXRange(atof(params[4].c_str()), atof(params[5].c_str()));
  }
  // link it to the monitor element
  mui->useQTest(params[0], params[1]);
  return;
}

void createYRangeTest(MonitorUserInterface* mui, vector<string>& params){
  if (params.size() < 6) return;
  
  QCriterion* qc = mui->getQCriterion(params[1]);
  if(qc == NULL){
    qc = mui->createQTest(ContentsYRangeROOT::getAlgoName(),params[1]);
    // Contents within [Xmin, Xmax]
    MEContentsYRangeROOT* me_qc = (MEContentsYRangeROOT*) qc;
    //set probability limit for test warning 
    me_qc->setWarningProb(atof(params[2].c_str()));
    //set probability limit for test error 
    me_qc->setErrorProb(atof(params[3].c_str()));
    // set allowed range in Y-axis (default values: histogram's Y-range)
    me_qc->setAllowedYRange(atof(params[4].c_str()), atof(params[5].c_str()));
  }
  // link it to the monitor element
  mui->useQTest(params[0], params[1]);
  return;
}

void createMeanValueTest(MonitorUserInterface* mui, vector<string>& params){
  if (params.size() < 7 ) return;
  
  QCriterion* qc = mui->getQCriterion(params[1]);
  if(qc == NULL){
    qc = mui->createQTest("MeanWithinExpected",params[1]);
    // Contents within a mean value
    MEMeanWithinExpectedROOT* me_qc = (MEMeanWithinExpectedROOT*) qc;
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
  mui->useQTest(params[0], params[1]);
  return;
}

void createH2ContentTest(MonitorUserInterface* mui, vector<string>& params){
  if (params.size() < 2 ) return;

  QCriterion* qc = mui->getQCriterion(params[1]);
  MonitorElement* me =  mui->get(params[0]);
  if(me!=NULL && qc == NULL){
    qc = mui->createQTest(ContentsTH2FWithinRangeROOT::getAlgoName(),params[1]);
    // Contents within a mean value     
    ContentsTH2FWithinRangeROOT* me_qc = dynamic_cast<ContentsTH2FWithinRangeROOT*> (qc);
    me_qc->setMeanRange(0,1e-10);//(atof(params[2].c_str())
    me_qc->setRMSRange(0,1e-10);
    // link it to the monitor element
    mui->useQTest(params[0], params[1]);
  }
  
  return;
}

void createH2CompTest(MonitorUserInterface* mui, vector<string>& params, TH2F* ref){
  if (params.size() < 2 ) return;
  if(ref==NULL) return;
  
  QCriterion* qc = mui->getQCriterion(params[1]);
  MonitorElement* me =  mui->get(params[0]);
  if(me!=NULL && qc == NULL){
    printf("\n\nDon't have this QC, but have the me!\n\n");
    const QReport* qr = me->getQReport(params[1]);
    if(qr) return;
    printf("\n\nThe ME doesn't have the QC!!\n\n");
    qc = mui->createQTest("Comp2RefEqualH2",params[1]);
    // Contents within a mean value     
    Comp2RefEqualH2ROOT* me_qc = dynamic_cast<Comp2RefEqualH2ROOT*> (qc);
    //set reference histogram
    me_qc->setReference(ref);
    // link it to the monitor element
    printf("\n\nGonna run it...\n\n");
    mui->useQTest(params[0], params[1]);
  }
  else printf("\n\nAlready had the QC or didn't have the ME!\n\n");

  return;
}

void htmlErrors(string htmlDir, string client, string process, MonitorUserInterface* mui, map<string, vector<QReport*> > mapE, map<string, vector<QReport*> > mapW, map<string, vector<QReport*> > mapO){

  map<string, vector<QReport*> >::iterator mapIter;

  ofstream errorFile;
  errorFile.open((htmlDir + client+ "Errors.html").c_str());
  errorFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  errorFile << "<html>  " << endl;
  errorFile << "<head>  " << endl;
  errorFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  errorFile << " http-equiv=\"content-type\">  " << endl;
  errorFile << "  <title>Monitor: Hcal " << client <<" Task Error Output</title> " << endl;
  errorFile << "</head>  " << endl;
  errorFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  errorFile << "<body>  " << endl;
  errorFile << "<br>  " << endl;
  errorFile << "<h2>" << client <<" Errors</h2> " << endl;

  for (mapIter=mapE.begin(); mapIter!=mapE.end();mapIter++){
    string meName = mapIter->first;
    vector<QReport*> errors = mapIter->second;
    errorFile << "<br>" << endl;     
    errorFile << "<hr>" << endl;
    errorFile << "Monitorable " << meName << " has the following errors: <br>" << endl;
    for(vector<QReport*>::iterator report=errors.begin(); report!=errors.end(); report++){
      errorFile << "     "<< (*report)->getQRName() << ": "<< (*report)->getMessage() << endl;
    }
    MonitorElement* me = mui->get(meName);
    errorFile << "<br>" << endl;
    errorFile << "<br>" << endl;
    char* substr = strstr(meName.c_str(), client.c_str());
    if(me->getMeanError(2)==0){
      TH1F* obj1f = getHisto(substr, process.c_str(), mui);
      string save = getIMG(obj1f,1,htmlDir,"X1a","Y1a");
      errorFile << "<img src=\"" <<  save << "\">" << endl;
    }
    else{
      TH2F* obj2f = getHisto2(substr, process.c_str(), mui);
      string save = getIMG2(obj2f,1,htmlDir,"X2a","Y2a");
      errorFile << "<img src=\"" <<  save << "\">" << endl;
    }
    errorFile << "<br>" << endl;
    errorFile << endl;
  }
  errorFile << "<hr>" << endl;
  errorFile.close();


  errorFile.open((htmlDir + client+ "Warnings.html").c_str());
  errorFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  errorFile << "<html>  " << endl;
  errorFile << "<head>  " << endl;
  errorFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  errorFile << " http-equiv=\"content-type\">  " << endl;
  errorFile << "  <title>Monitor: Hcal " << client <<" Task Warning Output</title> " << endl;
  errorFile << "</head>  " << endl;
  errorFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  errorFile << "<body>  " << endl;
  errorFile << "<br>  " << endl;
  errorFile << "<h2>" << client <<" Warnings</h2> " << endl;

  for (mapIter=mapW.begin(); mapIter!=mapW.end();mapIter++){
    string meName = mapIter->first;
    vector<QReport*> errors = mapIter->second;
    errorFile << "<br>" << endl;     
    errorFile << "<hr>" << endl;
    errorFile << "Monitorable " << meName << " has the following warnings: <BR>" << endl;
    for(vector<QReport*>::iterator report=errors.begin(); report!=errors.end(); report++){
      errorFile << "     "<< (*report)->getQRName() << ": "<< (*report)->getMessage() << endl;
    }
    MonitorElement* me = mui->get(meName);
    errorFile << "<br>" << endl;
    errorFile << "<br>" << endl;
    char* substr = strstr(meName.c_str(), client.c_str());
    if(me->getMeanError(2)==0){
      TH1F* obj1f = getHisto(substr, process.c_str(), mui);
      string save = getIMG(obj1f,1,htmlDir,"X1b","Y1b");
      errorFile << "<img src=\"" <<  save << "\">" << endl;
    }
    else{
      TH2F* obj2f = getHisto2(substr, process.c_str(), mui);
      string save = getIMG2(obj2f,1,htmlDir,"X2b","Y2b");
      errorFile << "<img src=\"" <<  save << "\">" << endl;
    }
    errorFile << "<br>" << endl;
    errorFile << endl;  
  }
  errorFile << "<hr>" << endl;
  errorFile.close();
  
  errorFile.open((htmlDir + client+ "Messages.html").c_str());
  errorFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  errorFile << "<html>  " << endl;
  errorFile << "<head>  " << endl;
  errorFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  errorFile << " http-equiv=\"content-type\">  " << endl;
  errorFile << "  <title>Monitor: Hcal " << client <<" Task Message Output</title> " << endl;
  errorFile << "</head>  " << endl;
  errorFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  errorFile << "<body>  " << endl;
  errorFile << "<br>  " << endl;
  errorFile << "<h2>" << client <<" Messages</h2> " << endl;

  for (mapIter=mapO.begin(); mapIter!=mapO.end();mapIter++){
    string meName = mapIter->first;
    vector<QReport*> errors = mapIter->second;
    errorFile << "<br>" << endl;     
    errorFile << "<hr>" << endl;
    errorFile << "Monitorable " << meName << " has the following messages: <br>" << endl;
    for(vector<QReport*>::iterator report=errors.begin(); report!=errors.end(); report++){
      errorFile << "     "<< (*report)->getQRName() << ": "<< (*report)->getMessage() << endl;
    }
    errorFile << "<br>" << endl;
    errorFile << "<br>" << endl;
    MonitorElement* me = mui->get(meName);
    char* substr = strstr(meName.c_str(), client.c_str());
    if(me->getMeanError(2)==0){
      TH1F* obj1f = getHisto(substr, process.c_str(), mui);
      string save = getIMG(obj1f,1,htmlDir,"X1c","Y1c");
      errorFile << "<img src=\"" <<  save << "\">" << endl;
    }
    else{
      TH2F* obj2f = getHisto2(substr, process.c_str(), mui);
      string save = getIMG2(obj2f,1,htmlDir,"X2c","Y2c");
      errorFile << "<img src=\"" <<  save << "\">" << endl;
    }
    errorFile << "<br>" << endl;
    errorFile << endl;
  }
  errorFile << "<hr>" << endl;
  errorFile.close();

  return;

}
