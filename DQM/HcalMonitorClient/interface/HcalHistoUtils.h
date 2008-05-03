#ifndef GUARD_HCALTESTUTILS_h
#define GUARD_HCALTESTUTILS_h

/***********************************************
 *
 * HcalTestUtils.h
 * v1.0
 * 24 April 2008
 * by Jeff Temple (jtemple AT fnal.gov)
 *
 *  Various template methods to make .gif, .html
 *  output for any type of histogram stored as
 *  a MonitorElement.
 *  Still to come:  a "getAnyHisto" routine
 *  that will return get the histogram from the ME
 *  and return an appropriate pointer
 *
 ************************************************
 */

template <class myHist>
myHist* getAnyHisto(string histtype, myHist* hist, 
		    string name, string process, DQMStore* dbe_,
		    bool verb, bool clone)
{
  
  /* Method returns histogram named 'name' from DQMStore dbe_;
     There hould be a more concise way of determining histogram type 
     than by passing both a string indicating type and a pointer to the 
     histogram.
     ( The pointer is needed to set the histogram type myHist.
     However, at this point, the pointer does not point to anything,
     which means that we can't access type info via hist->ClassName(),
     and so the separate histtype variable is needed.)
  */

  using std::cout;
  using std::endl;

  if (!dbe_) return NULL;

  char title[150];  // title of histogram to grab from dbe
  char clonehisto[150];
  sprintf(title, "%sHcal/%s",process.c_str(),name.c_str());

  MonitorElement* me = dbe_->get(title); // get Monitor Element named 'title'

  if (!me) return NULL; // ME not found

  if (verb) 
    cout << "Found '" << title << "'" << endl;

  if (clone)
    sprintf(clonehisto, "ME %s",name.c_str()); // set clone histogram name

  /* As of 25 April 2008, there are 5 histogram types associated with 
     Monitor Elements (TH1F, TH2F, TH3F, TProfile, and TProfile2D).
     Provide a separate getter for each type.  Add others if necessary.
  */

  // return TH1F from ME

  if (histtype=="TH1F")
    {
      TH1F* out;
      if (clone) out = dynamic_cast<TH1F*>(me->getTH1F()->Clone(clonehisto));
      else out = me->getTH1F();
      return dynamic_cast<myHist*>(out);
    }

  // return TH2F from ME
  else if (histtype=="TH2F")
    {
      TH2F* out;
      if (clone) out = dynamic_cast<TH2F*>(me->getTH2F()->Clone(clonehisto));
      else out = me->getTH2F();
      return dynamic_cast<myHist*>(out);
    }

  // return TH3F from ME
  else if (histtype=="TH3F")
    {
      TH3F* out;
      if (clone) out = dynamic_cast<TH3F*>(me->getTH3F()->Clone(clonehisto));
      else out = me->getTH3F();
      return dynamic_cast<myHist*>(out);
    }

  // return TProfile from ME
  else if (histtype=="TProfile")
    {
      TProfile* out;
      if (clone) out = dynamic_cast<TProfile*>(me->getTProfile()->Clone(clonehisto));
      else out = me->getTProfile();
      return dynamic_cast<myHist*>(out);
    }

  // return TProfile2D from ME
  else if (histtype=="TProfile2D")
    {
      TProfile2D* out;
      if (clone) out = dynamic_cast<TProfile2D*>(me->getTProfile2D()->Clone(clonehisto));
      else out = me->getTProfile2D();
      return dynamic_cast<myHist*>(out);
    }

  else
    {
      if (verb) 
	{
	  cout <<"Don't know how to access histogram '"<<title;
	  cout<<"' of type '"<<histtype<<"'"<<endl;
	}
      return NULL;
    }

  // Should never reach this point
  if (verb)
    cout <<"<HcalHistUtils::getAnyHisto>  YOU SHOULD NEVER SEE THIS MESSAGE!"<<endl;
  return NULL;

} // myHist* getAnyHisto(...)




// MAKE GIF FROM HISTOGRAM IMAGE
template <class myHist>
std::string getAnyIMG(int runNo,myHist* hist, int size, std::string htmlDir,
		      const char* xlab, const char* ylab) 
{
  /* template functions draws histogram plot, and saves it as a .gif image.
     If size==2, thumbnail image is made.  Otherwise, full-size image is made
  */

  if(hist==NULL)
    {
      return "";
    }

  // Run cleanString algorithm  -- direct call of cleanString causes a crash 
  std::string name = (std::string)hist->GetTitle();
  for ( unsigned int i = 0; i < name.size(); ++i ) {
    if ( name.substr(i, 6) == " - Run" ){
      name.replace(i, name.size()-i, "");
    }
    if ( name.substr(i, 4) == "_Run" ){
      name.replace(i, name.size()-i, "");
    }
    if ( name.substr(i, 5) == "__Run" ){
      name.replace(i, name.size()-i, "");
    }
  } // for (unsigned int i=0; i< name.size();


  char dest[512]; // stores name of destination .gif file
  if(runNo>-1) sprintf(dest,"%s - Run %d",name.c_str(),runNo);
  else sprintf(dest,"%s",name.c_str());

  hist->SetTitle(dest);
  std::string title = dest;

  int xwid = 900; int ywid =540;
  if(size==1)
    {
      title = title+"_tmb";
      xwid = 600; ywid = 360;
    }

  TCanvas* can = new TCanvas(dest,dest, xwid, ywid);

  // run parseString algorithm -- calling it directly causes a crash
  for ( unsigned int i = 0; i < name.size(); ++i ) {
    if ( name.substr(i, 1) == " " ){
      name.replace(i, 1, "_");
    }
    if ( name.substr(i, 1) == "#" ){
      name.replace(i, 1, "N");
    }
    if ( name.substr(i, 1) == "-" ){
      name.replace(i, 1, "_");
    }    
    if ( name.substr(i, 1) == "&" ){
      name.replace(i, 1, "_and_");
    }
    if ( name.substr(i, 1) == "(" 
	 || name.substr(i, 1) == ")" 
	 )  {
      name.replace(i, 1, "_");
    } 
    if ( name.substr(i,1) == "="){
      name.replace(i,1,"_");
    }
  } // for (unsigned int i=0; i < name.size();...)
  

  //std::string outName = title + ".gif";
  std::string outName = name+".gif";
  std::string saveName = htmlDir + outName;

  hist->SetXTitle(xlab);
  hist->SetYTitle(ylab);

  // Don't draw stat box for color plots

  if (hist->GetOption()=="col" || hist->GetOption()=="colz")
    hist->SetStats(false);

  // Draw with whatever options are set for the particluar histogram

  
  hist->Draw(hist->GetOption());// I think that Draw should automatically use the GetOption() value, but include it here to be sure.
  
  can->SaveAs(saveName.c_str());  
  delete can;

  return outName;
}


// make HTML from histogram
template <class myHist>
void htmlAnyHisto(int runNo, myHist *hist, 
		  const char* xlab, const char* ylab, 
		  int width, ofstream& htmlFile, 
		  std::string htmlDir)
{

  // Generates histogram output for any kind of input histogram

  using std::cout;
  using std::endl;

  if(hist!=NULL)
    {    
      // Set default option for 2D histogram
      if (hist->ClassName()=="TH2F" && hist->GetOption()=="")
	{
	  cout <<"HUZZAH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
	  hist->SetOption("colz");
	}

      std::string imgNameTMB = "";   
      imgNameTMB = getAnyIMG(runNo,hist,1,htmlDir,xlab,ylab); 
      std::string imgName = "";   
      imgName = getAnyIMG(runNo,hist,2,htmlDir,xlab,ylab);  
      
      if (imgName.size() != 0 )
	{
	htmlFile << "<td><a href=\"" <<  imgName << "\"><img src=\"" <<  imgNameTMB << "\"></a></td>" << endl;
	}
      else
	{
	  htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
	}
    } // (hist != NULL)

  else 
    {
      htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    }
  return;
}

#endif
