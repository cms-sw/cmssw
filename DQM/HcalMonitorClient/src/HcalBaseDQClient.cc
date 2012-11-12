#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <fstream>
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "FWCore/Framework/interface/ESHandle.h"


HcalBaseDQClient::HcalBaseDQClient(std::string s, const edm::ParameterSet& ps)
{
  name_=s;
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");

  validHtmlOutput_       = ps.getUntrackedParameter<bool>("validHtmlOutput",true);
  Online_                = ps.getUntrackedParameter<bool>("online",false);

  subdir_="HcalInfo/";
  subdir_=prefixME_+subdir_;

  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);
  badChannelStatusMask_   = 0;
  enoughevents_=true;
  minerrorrate_=0;
  minevents_=0;

  ProblemCells=0;
  ProblemCellsByDepth=0;

  logicalMap_=0;
  needLogicalMap_=false;
}

HcalBaseDQClient::~HcalBaseDQClient()
{}

void HcalBaseDQClient::beginJob()
{
  dqmStore_ = edm::Service<DQMStore>().operator->();
  if (debug_>0) 
    {
      std::cout <<"<HcalBaseDQClient::beginJob()>  Displaying dqmStore directory structure:"<<std::endl;
      dqmStore_->showDirStructure();
    }
}

void HcalBaseDQClient::setStatusMap(std::map<HcalDetId, unsigned int>& map)
  {
    /* Get the list of all bad channels in the status map,
       and combine it with the bad cell requirements for the particular task
       to form a new map
    */

    if (debug_>1) std::cout <<"<HcalBaseDQClient::setStatusMap>  Input map size = "<<map.size()<<std::endl;
    for (std::map<HcalDetId, unsigned int>::const_iterator iter = map.begin(); 
	 iter!=map.end();++iter)
      {
	if ((iter->second & badChannelStatusMask_) == 0 ) continue; // channel not marked as bad by this test
	badstatusmap[iter->first]=iter->second;
      }
    
    if (debug_>1) std::cout <<"<HcalBaseDQClient::setStatusMap>  "<<name_<<" Output map size = "<<badstatusmap.size()<<std::endl;
  } // void HcalBaseDQClient::getStatusMap


bool HcalBaseDQClient::validHtmlOutput()
{
  return validHtmlOutput_;
}

void HcalBaseDQClient::htmlOutput(std::string htmlDir)
{
  if (dqmStore_==0) 
    {
      if (debug_>0) std::cout <<"<HcalBaseDQClient::htmlOutput> dqmStore object does not exist!"<<std::endl;
      return;
    }

  if (debug_>2) std::cout <<"\t<HcalBaseDQClient::htmlOutput>  Preparing html for task: "<<name_<<std::endl;
  int pcol_error[105];
 for( int i=0; i<105; ++i )
    {
      
      TColor* color = gROOT->GetColor( 901+i );
      if( ! color ) color = new TColor( 901+i, 0, 0, 0, "" );
      if (i<5)
	color->SetRGB(i/5.,1.,0);
      else if (i>100)
	color->SetRGB(0,0,0);
      else
	color->SetRGB(1,1-0.01*i,0);
      pcol_error[i]=901+i;
    } // for (int i=0;i<105;++i)

  ofstream htmlFile;
  std::string outfile=htmlDir+name_+".html";
  htmlFile.open(outfile.c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << std::endl;
  htmlFile << "<html>  " << std::endl;
  htmlFile << "<head>  " << std::endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << std::endl;
  htmlFile << " http-equiv=\"content-type\">  " << std::endl;
  htmlFile << "  <title>Monitor: Hcal "<<name_<<" output</title> " << std::endl;
  htmlFile << "</head>  " << std::endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << std::endl;
  htmlFile << "<body>  " << std::endl;
  htmlFile << "<br>  " << std::endl;
  htmlFile << "<hr>" << std::endl;

  gStyle->SetPalette(105,pcol_error);
  gStyle->SetNumberContours(105);
  gROOT->ForceStyle();

  if (debug_>0) std::cout <<"<HcalBaseDQClient::htmlOutput>  Writing html output for client "<<this->name()<<std::endl;
  if (ProblemCells!=0)
    {
      (ProblemCells->getTH2F())->SetMaximum(1.05);
      (ProblemCells->getTH2F())->SetMinimum(0.);
      htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      htmlFile<<"<tr align=\"center\">"<<std::endl;
      htmlAnyHisto(-1,ProblemCells->getTH2F(),"ieta","iphi",92, htmlFile,htmlDir,debug_);
      htmlFile<<"</tr>"<<std::endl;
      htmlFile<<"</table>"<<std::endl;
    }
  if (ProblemCellsByDepth!=0)
    {
      htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << std::endl;
      htmlFile << "cellpadding=\"10\"> " << std::endl;
      for (unsigned int i=0;i<ProblemCellsByDepth->depth.size()/2;++i)
	{
	  if (ProblemCellsByDepth->depth[2*i]==0) continue;
	  if (ProblemCellsByDepth->depth[2*i+1]==0) continue;
	  (ProblemCellsByDepth->depth[2*i]->getTH2F())->SetMaximum(1.05);
	  (ProblemCellsByDepth->depth[2*i]->getTH2F())->SetMinimum(0.);
	  (ProblemCellsByDepth->depth[2*i+1]->getTH2F())->SetMaximum(1.05);
	  (ProblemCellsByDepth->depth[2*i+1]->getTH2F())->SetMinimum(0.);
	  htmlFile<<"<tr align=\"center\">"<<std::endl;
	  htmlAnyHisto(-1,ProblemCellsByDepth->depth[2*i]->getTH2F(),"ieta","iphi",92, htmlFile,htmlDir,debug_);
	  htmlAnyHisto(-1,ProblemCellsByDepth->depth[2*i+1]->getTH2F(),"ieta","iphi",92, htmlFile,htmlDir,debug_);
	  
	  htmlFile<<"</tr>"<<std::endl;
	}
      htmlFile<<"</table>"<<std::endl;
   }
    
  htmlFile << "<table align=\"center\" border=\"1\" cellspacing=\"0\" " << std::endl;
  htmlFile << "cellpadding=\"10\"> " << std::endl;
  

  std::vector<MonitorElement*> hists = dqmStore_->getAllContents(subdir_);
  gStyle->SetPalette(1);
  
  int counter=0;
  for (unsigned int i=0;i<hists.size();++i)
    {
      if (hists[i]->kind()==MonitorElement::DQM_KIND_TH1F)
	{
	  ++counter;
	  if (counter%2==1) 
	    htmlFile << "<tr align=\"center\">" << std::endl;
	  htmlAnyHisto(-1,(hists[i]->getTH1F()),"","", 92, htmlFile, htmlDir,debug_);
	  if (counter%2==2)
	    htmlFile <<"</tr>"<<std::endl;
	}

      else if (hists[i]->kind()==MonitorElement::DQM_KIND_TH2F)
	{
	  std::string histname=hists[i]->getName();
	  bool isproblem=false;
	  for (unsigned int j=0;j<problemnames_.size();++j)
	    {
	      if (problemnames_[j]==histname)
		{
		  isproblem=true;
		  if (debug_>1) std::cout <<"<HcalBaseDQClient::htmlOutput>  Found Problem Histogram '"<<histname<<"' in list of histograms"<<std::endl;
		  break;
		}	
	    }
	  if (isproblem) continue; // don't redraw problem histograms
	  ++counter;
	  if (counter%2==1) 
	    htmlFile << "<tr align=\"center\">" << std::endl;
	  htmlAnyHisto(-1,(hists[i]->getTH2F()),"","", 92, htmlFile, htmlDir,debug_);
	  if (counter%2==2)
	    htmlFile <<"</tr>"<<std::endl;
	}

      else if (hists[i]->kind()==MonitorElement::DQM_KIND_TPROFILE)
	{
	  ++counter;
	  if (counter%2==1) 
	    htmlFile << "<tr align=\"center\">" << std::endl;
	  htmlAnyHisto(-1,(hists[i]->getTProfile()),"","", 92, htmlFile, htmlDir,debug_);
	  if (counter%2==2)
	    htmlFile <<"</tr>"<<std::endl;
	}
    }
  htmlFile<<"</table>"<<std::endl;
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  htmlFile.close();
  return;
}
void HcalBaseDQClient::getLogicalMap(const edm::EventSetup& c) {
  if (needLogicalMap_ && logicalMap_==0) {
    edm::ESHandle<HcalTopology> pT;
    c.get<IdealGeometryRecord>().get(pT);   
    HcalLogicalMapGenerator gen;
    logicalMap_=new HcalLogicalMap(gen.createMap(&(*pT)));
  }
}
