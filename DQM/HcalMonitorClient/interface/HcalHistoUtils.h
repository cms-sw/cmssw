#ifndef GUARD_HCALTESTUTILS_H
#define GUARD_HCALTESTUTILS_H

#define UTILS_ETAMIN -44.5
#define UTILS_ETAMAX 44.5
#define UTILS_PHIMIN -0.5
#define UTILS_PHIMAX 73.5

#include "TROOT.h"
#include "TStyle.h"
#include "TColor.h"

#include "TH1.h"
#include "TCanvas.h"
#include "TGaxis.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TFile.h"

#include <iostream>
#include <fstream>

/******************************************************************************
 *
 * HcalTestUtils.h
 * v1.1
 * 4 May 2008
 * by Jeff Temple (jtemple AT fnal.gov)
 *
 *  Various template methods to grab any type of
 *  histogram stored as a MonitorElement, and 
 *  to return .gif, .html output from the histogram.
 *
 * Methods:
 *
 * myHist* getAnyHisto(myHist* hist, std::string name, std::string process, 
                         DQMStore* dbe_, bool verb, bool clone)  
 *			   
 * std::string getAnyIMG(int runNo,myHist* hist, int size, std::string htmlDir,
    		         const char* xlab, const char* ylab, bool setLogy, bool setLogx) 
 *
 * void htmlAnyHisto(int runNo, myHist *hist, 
		     const char* xlab, const char* ylab, 
		     int width, std::ofstream& htmlFile, 
		     std::string htmlDir, bool setLogy, bool setLogx)
 
 *
 *****************************************************************************
 */

#include "DQMServices/Core/interface/DQMStore.h"

// Template class 'getAnyHisto' functions contain memory leaks somewhere.  Re-introduce getTH1F, getTH2F, getTProfile

inline TH1F* getTH1F(std::string name, std::string process, std::string rootfolder, DQMStore* dbe_, bool verb, bool clone)
{
  
  if (!dbe_) return NULL;
  std::stringstream title;
  title <<process.c_str()<<rootfolder.c_str()<<"/"<<name.c_str();

  MonitorElement* me = dbe_->get(title.str().c_str()); // get Monitor Element named 'title'
  
  if (!me) 
    {
      if (verb) std::cout <<"SORRY, COULD NOT FIND HISTOGRAM NAMED ["<< title.str().c_str()<<"]"<<std::endl;
      return NULL; // ME not found
    } // if (!me)

  if (verb) 
    std::cout << "Found '" << title.str().c_str() << "'" << std::endl;

  std::stringstream clonehisto;
  if (clone)
    {
      clonehisto<<"ME "<<name.c_str();
      TH1F *out = dynamic_cast<TH1F*>(me->getTH1F()->Clone(clonehisto.str().c_str()));
      return out;
    }
  else return (me->getTH1F());
}

inline TH2F* getTH2F(std::string name, std::string process, std::string rootfolder, DQMStore* dbe_, bool verb, bool clone)
{
  if (!dbe_) return NULL;
  std::stringstream title;
  title <<process.c_str()<<rootfolder.c_str()<<"/"<<name.c_str();

  MonitorElement* me = dbe_->get(title.str().c_str()); // get Monitor Element named 'title'
  
  if (!me) 
    {
      if (verb) std::cout <<"SORRY, COULD NOT FIND HISTOGRAM NAMED ["<< title.str().c_str()<<"]"<<std::endl;
      return NULL; // ME not found
    } // if (!me)

  if (verb) 
    std::cout << "Found '" << title.str().c_str() << "'" << std::endl;

  std::stringstream clonehisto;
  if (clone)
    {
      clonehisto<<"ME "<<name.c_str();
      TH2F *out = dynamic_cast<TH2F*>(me->getTH2F()->Clone(clonehisto.str().c_str()));
      return out;
    }
  else return (me->getTH2F());
}


inline TH3F* getTH3F(std::string name, std::string process, std::string rootfolder, DQMStore* dbe_, bool verb, bool clone)
{
  if (!dbe_) return NULL;
  std::stringstream title;
  title <<process.c_str()<<rootfolder.c_str()<<"/"<<name.c_str();

  MonitorElement* me = dbe_->get(title.str().c_str()); // get Monitor Element named 'title'
  
  if (!me) 
    {
      if (verb) std::cout <<"SORRY, COULD NOT FIND HISTOGRAM NAMED ["<< title.str().c_str()<<"]"<<std::endl;
      return NULL; // ME not found
    } // if (!me)

  if (verb) 
    std::cout << "Found '" << title.str().c_str() << "'" << std::endl;

  std::stringstream clonehisto;
  if (clone)
    {
      clonehisto<<"ME "<<name.c_str();
      TH3F *out = dynamic_cast<TH3F*>(me->getTH3F()->Clone(clonehisto.str().c_str()));
      return out;
    }
  else return (me->getTH3F());
}

inline TProfile* getTProfile(std::string name, std::string process, std::string rootfolder, DQMStore* dbe_, bool verb, bool clone)
{
  if (!dbe_) return NULL;
  std::stringstream title;
  title <<process.c_str()<<rootfolder.c_str()<<"/"<<name.c_str();

  MonitorElement* me = dbe_->get(title.str().c_str()); // get Monitor Element named 'title'
  
  if (!me) 
    {
      if (verb) std::cout <<"SORRY, COULD NOT FIND HISTOGRAM NAMED ["<< title.str().c_str()<<"]"<<std::endl;
      return NULL; // ME not found
    } // if (!me)

  if (verb) 
    std::cout << "Found '" << title.str().c_str() << "'" << std::endl;

  std::stringstream clonehisto;
  if (clone)
    {
      clonehisto<<"ME "<<name.c_str();
      TProfile *out = dynamic_cast<TProfile*>(me->getTProfile()->Clone(clonehisto.str().c_str()));
      return out;
    }
  else return (me->getTProfile());
}

inline TProfile2D* getTProfile2D(std::string name, std::string process, std::string rootfolder, DQMStore* dbe_, bool verb, bool clone)
{
  if (!dbe_) return NULL;
  std::stringstream title;
  title <<process.c_str()<<rootfolder.c_str()<<"/"<<name.c_str();

  MonitorElement* me = dbe_->get(title.str().c_str()); // get Monitor Element named 'title'
  
  if (!me) 
    {
      if (verb) std::cout <<"SORRY, COULD NOT FIND HISTOGRAM NAMED ["<< title.str().c_str()<<"]"<<std::endl;
      return NULL; // ME not found
    } // if (!me)

  if (verb) 
    std::cout << "Found '" << title.str().c_str() << "'" << std::endl;

  std::stringstream clonehisto;
  if (clone)
    {
      clonehisto<<"ME "<<name.c_str();
      TProfile2D *out = dynamic_cast<TProfile2D*>(me->getTProfile2D()->Clone(clonehisto.str().c_str()));
      return out;
    }
  else return (me->getTProfile2D());
}



template <class myHist>
myHist* getAnyHisto(myHist* hist,
		    std::string name, std::string process, DQMStore* dbe_,
		    bool verb, bool clone)
{

  // If subsystem folder not specified, assume that it's "Hcal"
  myHist* theHist=getAnyHisto(hist, name, process, "Hcal",dbe_, verb, clone);
  return theHist;
}


template <class myHist>
myHist* getAnyHisto(myHist* hist,
		    std::string name, std::string process, std::string rootfolder, DQMStore* dbe_,
		    bool verb, bool clone)
{
  /* 
     Method returns histogram named 'name' from DQMStore dbe_;
     'hist' pointer must be declared with 'new' (e.g., new TH2F())
     in function call so that the pointer actually points to something.
     Otherwise, the call to hist->ClassName() will fail.
     We might implement a scale functionality at some later point?
  */

  if (!dbe_) return NULL;

  std::stringstream clonehisto;
  std::stringstream title;
  title <<process.c_str()<<rootfolder.c_str()<<"/"<<name.c_str();
  //sprintf(title, "%sHcal/%s",process.c_str(),name.c_str());

  MonitorElement* me = dbe_->get(title.str().c_str()); // get Monitor Element named 'title'
  
  if (!me) 
    {
      if (verb) std::cout <<"SORRY, COULD NOT FIND HISTOGRAM NAMED ["<< title.str()<<"]"<<std::endl;
      return NULL; // ME not found
    } // if (!me)

  if (verb) 
    std::cout << "Found '" << title.str() << "'" << std::endl;

  if (clone)
    clonehisto<<"ME "<<name; // set clone histogram name

  /* As of 25 April 2008, there are 5 histogram types associated with 
     Monitor Elements (TH1F, TH2F, TH3F, TProfile, and TProfile2D).
     Provide a separate getter for each type.  Add others if necessary.
  */

  std::string histtype = hist->ClassName();
  
  // return TH1F from ME
  if (histtype=="TH1F")
    {
      TH1F* out;
      if (clone) out = dynamic_cast<TH1F*>(me->getTH1F()->Clone(clonehisto.str().c_str()));
      else out = me->getTH1F();
      if (verb) std::cout <<"Got histogram!  Max = "<<out->GetMaximum()<<std::endl;
      return dynamic_cast<myHist*>(out);
    }

  // return TH2F from ME
  else if (histtype=="TH2F")
    {
      TH2F* out;
      if (clone) out = dynamic_cast<TH2F*>(me->getTH2F()->Clone(clonehisto.str().c_str()));
      else out = me->getTH2F();

      if (verb) std::cout <<"Got histogram!  Max = "<<out->GetMaximum()<<std::endl;
      return dynamic_cast<myHist*>(out);
    }

  // return TH3F from ME
  else if (histtype=="TH3F")
    {
      TH3F* out;
      if (clone) out = dynamic_cast<TH3F*>(me->getTH3F()->Clone(clonehisto.str().c_str()));
      else out = me->getTH3F();
      return dynamic_cast<myHist*>(out);
    }

  // return TProfile from ME
  else if (histtype=="TProfile")
    {
      TProfile* out;
      if (clone) out = dynamic_cast<TProfile*>(me->getTProfile()->Clone(clonehisto.str().c_str()));
      else out = me->getTProfile();
      return dynamic_cast<myHist*>(out);
    }

  // return TProfile2D from ME
  else if (histtype=="TProfile2D")
    {
      TProfile2D* out;
      if (clone) out = dynamic_cast<TProfile2D*>(me->getTProfile2D()->Clone(clonehisto.str().c_str()));
      else out = me->getTProfile2D();
      return dynamic_cast<myHist*>(out);
    }

  else
    {
      if (verb) 
	{
	  std::cout <<"Don't know how to access histogram '"<<title.str();
	  std::cout<<"' of type '"<<histtype<<"'"<<std::endl;
	}
      return NULL;
    }

  // Should never reach this point
  if (verb)
    std::cout <<"<HcalHistUtils::getAnyHisto>  YOU SHOULD NEVER SEE THIS MESSAGE!"<<std::endl;
  return NULL;

} // myHist* getAnyHisto(...)




// MAKE GIF FROM HISTOGRAM IMAGE
template <class myHist>
std::string getAnyIMG(int runNo,myHist* hist, int size, std::string htmlDir,
		      const char* xlab, const char* ylab, int debug ) 
{
  /* 
     Template function draws histogram plot, and saves it as a .gif image.
     If size==1, thumbnail image is made.  Otherwise, full-size image is made.
  */

  if(hist==NULL)
    {
      return ""; // no histogram provided
    }
  
  // Grab the histogram's title, and convert it to something more palatable for use as a file name
  
  // Run cleanString algorithm  -- direct call of cleanString causes a crash 
  std::string name = (std::string)hist->GetTitle();
  if (debug>9) std::cout <<"NAME = ["<<name<<"]"<<std::endl;
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

    if (name.substr(i,1) == "(" || name.substr(i,1)==")")
      name.replace(i,1,"_");
    else if (name.substr(i,1)==",")
      name.replace(i,1,"_");
    else if (name.substr(i,1)=="<")
      name.replace(i,1,"_lt_");
    else if (name.substr(i,1)==">")
      name.replace(i,1,"_gt_");
    else if (name.substr(i,1)=="+")
      name.replace(i,1,"_plus_");
    else if (name.substr(i,1)=="#")
      name.replace(i,1,"");
    else if (name.substr(i,1)=="/")
      name.replace(i,1,"_div_");
  } // for (unsigned int i=0; i< name.size();
  //std::cout <<"NEWNAME = ["<<name<<"]"<<std::endl;

  char dest[512]; // stores name of destination .gif file
  if(runNo>-1) sprintf(dest,"%s - Run %d",name.c_str(),runNo);
  else sprintf(dest,"%s",name.c_str());

  //hist->SetTitle(dest); // no need to change the histogram title itself, right?
  std::string title = dest;

  int xwid = 900; 
  int ywid =540;

  if(size==1) // thumbnail specified
    {
      title = title+"_tmb";
      xwid = 600; 
      ywid = 360;
    }

  // run parseString algorithm -- calling it directly causes a crash
  for ( unsigned int i = 0; i < title.size(); ++i ) {
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
    if ( title.substr(i,1) == "="){
      title.replace(i,1,"_");
    }
  } // for (unsigned int i=0; i < title.size();...)
  
  std::string outName = title+".gif";
  std::string saveName = htmlDir + outName;


  // Create canvas for histogram
  TCanvas* can = new TCanvas(dest,dest, xwid, ywid);
  TAxis* xaxis=0;
  TAxis* yaxis=0;
  TLine* vert=0;
  TLine* horiz=0;
  hist->SetXTitle(xlab);
  hist->SetYTitle(ylab);
  std::string histtype=hist->ClassName();
  //can->GetFrame()->SetFillColor(21); // change canvas to different default color?   

  // Don't draw stat box for color plots
  if (((std::string)hist->GetOption())=="col" || 
      ((std::string)hist->GetOption())=="colz")
    hist->SetStats(false);

  // Draw with whatever options are set for the particular histogram

  hist->Draw(hist->GetOption());// I think that Draw should automatically use the GetOption() value, but include it here to be sure.

  // Draw Grid Lines

  if (histtype=="TH2F")
    {
      TAxis *xaxis = hist->GetXaxis();
      TAxis *yaxis=hist->GetYaxis();
      // Draw vertical lines
      //for (int xx=int(UTILS_ETAMIN);xx<=int(UTILS_ETAMAX);++xx)
 
      
	if (xaxis->GetXmax()==UTILS_ETAMAX && xaxis->GetXmin()==UTILS_ETAMIN 
	 && yaxis->GetXmax()==UTILS_PHIMAX && yaxis->GetXmin()==UTILS_PHIMIN) // ad hoc method for only drawing grids for eta-phi graphs; need to be more clever later?
	{
	  for (int xx=int(xaxis->GetXmin());
	       xx<=int(xaxis->GetXmax()); ++xx)
	    {
	      if (xx<-42 || xx >= 42) continue;
	      vert = new TLine(xx+0.5,0.5,xx+0.5,72.5);
	      //if (xx%vertlinespace!=0) continue;
	      //TLine *vert = new TLine(xx,yaxis->GetXmin(),xx,yaxis->GetXmax());
	      
	      vert->SetLineStyle(3);
	      vert->Draw("same");
	    }
	  // Draw horizontal lines
	  for (int yy=int(yaxis->GetXmin()); yy<int(yaxis->GetXmax());++yy)
	    {
	      if (yy%4==0)
		horiz = new TLine(-41.5,yy+0.5,41.5,yy+0.5);
	      else if (yy%2==0)
		horiz = new TLine(-39.5,yy+0.5,39.5,yy+0.5);
	      else
		horiz = new TLine(-20.5,yy+0.5,20.5,yy+0.5);
	      //if (yy%horizlinespace!=0) continue;
	      //TLine *horiz = new TLine(xaxis->GetXmin(),yy,xaxis->GetXmax(),yy);
	      horiz->SetLineStyle(3);
	      horiz->Draw("same");
	    }
	} //if (xaxis->GetXmax()==44)
    } // if (histtype=="TH2F")

  can->SaveAs(saveName.c_str());  
  delete can;
  delete vert;
  delete horiz;
  delete xaxis;
  delete yaxis;

  return outName;
} // std::string getAnyIMG(...)




// make HTML from histogram
template <class myHist>
void htmlAnyHisto(int runNo, myHist *hist, 
		  const char* xlab, const char* ylab, 
		  int width, std::ofstream& htmlFile, 
		  std::string htmlDir,
		  int debug=0)
{

  /*
    Generates html output from any kind of input histogram
  */

  if(hist!=NULL)
    {    
      std::string histtype=hist->ClassName();

      // Set 2D histogram default option to "colz"
      if (histtype=="TH2F" && ((std::string)hist->GetOption())=="")
	{
	  hist->SetOption("colz");
	}

      // Form full-sized and thumbnail .gifs from histogram
      //std::string imgNameTMB = "";   
      //std::string imgNameTMB = getAnyIMG(runNo,hist,1,htmlDir,xlab,ylab,debug);
      //std::string imgName = "";   
      std::string imgName = getAnyIMG(runNo,hist,2,htmlDir,xlab,ylab,debug);
      
      // Add thumbnail image to html code, linked to full-sized image
      if (imgName.size() != 0 )
	{
	  // Always make width = 100% ?
	  //htmlFile << "<td align=\"center\"><a href=\"" <<  imgName << "\"><img src=\"" <<  imgName << "\" width = \"100%\"></a><br>"<<hist->GetName()<<"</td>" << std::endl;
	  htmlFile <<"<td align=\"center\"><a href=\"" <<imgName<<"\"><img src=\""<<imgName<<"\" width=600 height=360\"></a><br>"<<hist->GetName()<<"</td>"<<std::endl;
}
      else
	{
	  htmlFile << "<td align=\"center\"><img src=\"" << " " << "\"></td>" << std::endl;
	}
    } // (hist != NULL)

  else  // if no image found, make a blank table entry (maybe improve on this later? -- write an error message?)
    {
       htmlFile<<"<td align=\"center\"><br><br> Histogram does not exist in ROOT file!<br>Diagnostic flag may be off.<br>(This may be normal in online running.)</td>"<<std::endl;
       //htmlFile << "<td><img src=\"" << " " << "\"></td>" << std::endl;
    }
  return;
} //void htmlAnyHisto(...)



#endif
