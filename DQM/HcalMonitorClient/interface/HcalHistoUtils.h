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
std::string getAnyIMG(int runNo,myHist* hist, int size, std::string htmlDir, const char* xlab, const char* ylab, std::string opts="NONE")
{
  /* template functions draws histogram plot, and saves it as a .gif image
     if size ==2, thumbnail image is made.  Otherwise, full-size image is made
  */

  if(hist==NULL)
    {
      std::cout <<"getIMG:  This histo is NULL, "<<xlab<<", "<<ylab<<std::endl;
      return "";
    }

  // Run cleanString algorithm
  std::string name = (std::string)hist->GetTitle();

  for ( unsigned int i = 0; i < name.size(); i++ ) {
    if ( name.substr(i, 6) == " - Run" ){
      name.replace(i, name.size()-i, "");
    }
    if ( name.substr(i, 4) == "_Run" ){
      name.replace(i, name.size()-i, "");
    }
    if ( name.substr(i, 5) == "__Run" ){
      name.replace(i, name.size()-i, "");
    }
  }


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

  // run parseString algorithm
  for ( unsigned int i = 0; i < name.size(); i++ ) {
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
  }

  std::string outName = title + ".gif";
  std::string saveName = htmlDir + outName;
  hist->SetXTitle(xlab);
  hist->SetYTitle(ylab);

  if (opts!="NONE")
    {
      // set default values by looking at hist->ClassName()  (TH1F, etc)?
      if (opts=="col" || opts=="colz")
	hist->SetStats(false);
      hist->SetDrawOption(opts.c_str());// does this actually do anything useful?
      hist->Draw(opts.c_str());
      
    }
  else
    hist->Draw();


  can->SaveAs(saveName.c_str());  
  delete can;

  return outName;
}


template <class myHist>
void htmlAnyHisto(int runNo, myHist *hist, 
		  const char* xlab, const char* ylab, 
		  int width, ofstream& htmlFile, 
		  std::string htmlDir, std::string opts="NONE")
{

  // Generates histogram output for any kind of input histogram

  if(hist!=NULL)
    {    
      std::string imgNameTMB = "";   
      imgNameTMB = getAnyIMG(runNo,hist,1,htmlDir,xlab,ylab,opts); 
      std::string imgName = "";   
      imgName = getAnyIMG(runNo,hist,2,htmlDir,xlab,ylab,opts);  
      
      if (imgName.size() != 0 )
	htmlFile << "<td><a href=\"" <<  imgName << "\"><img src=\"" <<  imgNameTMB << "\"></a></td>" << endl;
      else
	htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    }
  else htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  return;
}


// Sigh.  Specifying a default "opts" argument doesn't seem to work.  Make a copy of the function without "opts" for now, until I get this figured out.
template <class myHist>
void htmlAnyHisto(int runNo, myHist *hist, 
		  const char* xlab, const char* ylab, 
		  int width, ofstream& htmlFile, 
		  std::string htmlDir)
{

  // Generates histogram output for any kind of input histogram

  if(hist!=NULL)
    {    
      std::string histtype=hist->ClassName();
      std::string opts="NONE";
      // Set default option for 2D histogram
      if (histtype=="TH2F" || histtype=="TH2D" || histtype=="TH2I")
	{
	  opts="colz";
	}
      std::string imgNameTMB = "";   
      imgNameTMB = getAnyIMG(runNo,hist,1,htmlDir,xlab,ylab,opts); 
      std::string imgName = "";   
      imgName = getAnyIMG(runNo,hist,2,htmlDir,xlab,ylab,opts);  
      
      if (imgName.size() != 0 )
	htmlFile << "<td><a href=\"" <<  imgName << "\"><img src=\"" <<  imgNameTMB << "\"></a></td>" << endl;
      else
	htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
    }
  else htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;
  return;
}

#endif
