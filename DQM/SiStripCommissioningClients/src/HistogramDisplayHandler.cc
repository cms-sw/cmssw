#include "DQM/SiStripCommissioningClients/interface/HistogramDisplayHandler.h"
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include <SealBase/Callback.h>
#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/CgiUtils.h"
#include "cgicc/HTTPResponseHeader.h"
#include "cgicc/HTMLClasses.h"
#include "TROOT.h"
#include "TPad.h"
#include "TSystem.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TKey.h"
#include "TString.h"
#include "TProfile.h"
#include "TH1.h"
#include "TH2.h"
#include "TImage.h"
#include "TPaveText.h"


// This line is necessary

using namespace std;
using namespace cgicc;

// -----------------------------------------------------------------------------
/** */
HistogramDisplayHandler::HistogramDisplayHandler( MonitorUserInterface* mui, BSem* b )
{
  mui_=mui;
  fCanvas = new TCanvas("TestCanvas", "Test Canvas",1000, 1000);

  //   // Service allows use of MessageLogger
  //   edm::AssertHandler ah;
  //   boost::shared_ptr<edm::Presence> message = boost::shared_ptr<edm::Presence>( edm::PresenceFactory::get()->makePresence("MessageServicePresence").release() );
  fCallBack=b;
}

// -----------------------------------------------------------------------------

void HistogramDisplayHandler::HistogramViewer(xgi::Input * in, xgi::Output * out)  throw (xgi::exception::Exception)
{
  try 
    {
      // Create a new Cgicc object containing all the CGI data
      cgicc::Cgicc cgi(in);
const CgiEnvironment& env = cgi.getEnvironment();
    std::string qString = form_urldecode(env.getQueryString());
     cerr << qString << endl;

    fillMap(qString);
    std::string fCommand="";
    if (hasKey("command")) fCommand = getValue("command");
    cout <<"The command is " <<fCommand<<endl;
    if (fCommand == "filelist") {
	std::ostringstream xmlstr;
  	out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  	*out << "<?xml version=\"1.0\" ?>" << endl;
  	xmlstr << "<fileList>" << endl;
    	xmlstr << "  <name>ONLINE</name>" << endl;
  	xmlstr << "</fileList>" << endl;
	  *out << xmlstr.str();

	  // 	 cout << " Histogram Viewer filelist call " << endl;
    } 
    else if (fCommand == "mod_histo_list") {
	std::vector<std::string> histolist;
 
       std::vector<std::string> contents;
       mui_->getContents( contents ); 
	std::ostringstream xmlstr;
	bool modulefound=false;
	//out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  	//*out << "<?xml version=\"1.0\" ?>" << endl;
	//xmlstr << "Content-Type: text/xml\n Cache-Control: no-store, no-cache, must-revalidate, max-age=0"<< endl; 
	xmlstr << endl;
	xmlstr << "<ModuleAndHistoList>" << endl;
	xmlstr << "<ModuleList>" << endl;
	for (std::vector<std::string>::iterator it = contents.begin(); 
	it != contents.end(); it++) {
	std::string::size_type pos = (*it).find("Fec", 0);
	//	cout << (*it) <<endl;
	//	cout <<"Fec position " <<pos <<endl;
	if (pos<=0 || pos>512) continue;
 	std::string::size_type histopos = (*it).find_first_of(":", 0);
	//	cout <<"Hisoto position " <<pos <<endl;

	//	cout << (*it).substr(pos,histopos-pos)<<endl;
	xmlstr << "  <ModuleNum>" << (*it).substr(pos,histopos-pos)<< "</ModuleNum>" << endl;
	if (!modulefound && (*it).substr(pos,histopos-pos)== getValue("module"))
	  {
		tokenize((*it).substr(histopos+1,(*it).length()-histopos),histolist,",");
		modulefound=true;
	  }
	}
	xmlstr << "</ModuleList>" << endl;
	xmlstr << "<HistoList>" << endl;
	for (std::vector<std::string>::iterator im = histolist.begin(); 
	im != histolist.end(); im++) {
	xmlstr << " <Histo>" << *im << "</Histo>" << endl;
	}
	xmlstr << "</HistoList>" << endl;
	xmlstr << "</ModuleAndHistoList>" << endl;
	//	cerr << xmlstr.str();
	
	*out << xmlstr.str();    
      //readModuleAndHistoList();
    }
    else if (fCommand == "mod_list") {

 
       std::vector<std::string> contents;
       mui_->getContents( contents ); 
	std::ostringstream xmlstr;
	//bool modulefound=false;
	//out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  	//*out << "<?xml version=\"1.0\" ?>" << endl;
	//xmlstr << "Content-Type: text/xml\n Cache-Control: no-store, no-cache, must-revalidate, max-age=0"<< endl; 
	xmlstr << endl;
	xmlstr << "<ModuleAndHistoList>" << endl;
	xmlstr << "<ModuleList>" << endl;
	for (std::vector<std::string>::iterator it = contents.begin(); 
	it != contents.end(); it++) {
	std::string::size_type pos = (*it).find("Fec", 0);
	//	cout << (*it) <<endl;
	//	cout <<"Fec position " <<pos <<endl;
	if (pos<=0 || pos>512) continue;
 	std::string::size_type histopos = (*it).find_first_of(":", 0);
	//	cout <<"Hisoto position " <<pos <<endl;

	//	cout << (*it).substr(pos,histopos-pos)<<endl;
	xmlstr << "  <ModuleNum>" << (*it).substr(pos,histopos-pos)<< "</ModuleNum>" << endl;
	}
	xmlstr << "</ModuleList>" << endl;
	xmlstr << "</ModuleAndHistoList>" << endl;
	//	cerr << xmlstr.str();
	
	*out << xmlstr.str();    
      //readModuleAndHistoList();
    }
    else if (fCommand == "histo_list") {
	std::vector<std::string> histolist;
 
       std::vector<std::string> contents;
       mui_->getContents( contents ); 
	std::ostringstream xmlstr;
	bool modulefound=false;
	//out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  	//*out << "<?xml version=\"1.0\" ?>" << endl;
	//xmlstr << "Content-Type: text/xml\n Cache-Control: no-store, no-cache, must-revalidate, max-age=0"<< endl; 
	xmlstr << endl;
	xmlstr << "<ModuleAndHistoList>" << endl;

	for (std::vector<std::string>::iterator it = contents.begin(); 
	it != contents.end(); it++) {
	std::string::size_type pos = (*it).find("Fec", 0);
	//	cout << (*it) <<endl;
	//	cout <<"Fec position " <<pos <<endl;
	if (pos<=0 || pos>512) continue;
 	std::string::size_type histopos = (*it).find_first_of(":", 0);
	//	cout <<"Hisoto position " <<pos <<endl;

	//	cout << (*it).substr(pos,histopos-pos)<<endl;

	if (!modulefound && (*it).substr(pos,histopos-pos)== getValue("module"))
	  {
		tokenize((*it).substr(histopos+1,(*it).length()-histopos),histolist,",");
		modulefound=true;
		break;
	  }
	}

	xmlstr << "<HistoList>" << endl;
	for (std::vector<std::string>::iterator im = histolist.begin(); 
	im != histolist.end(); im++) {
	xmlstr << " <Histo>" << *im << "</Histo>" << endl;
	}
	xmlstr << "</HistoList>" << endl;
	xmlstr << "</ModuleAndHistoList>" << endl;
	//	cerr << xmlstr.str();
	
	*out << xmlstr.str();    
      //readModuleAndHistoList();
    } 
    else if (fCommand == "plot_as_module") {
      
      std::string mod_id = getValue("module");
      std::vector<std::string> hlist;
      getHistogramList(hlist);
      fCanvas->Clear();
      int ncol, nrow;
 
  float xlow = -1.0;
  float xhigh = -1.0;
  
  if (hlist.size() == 1) {
    if (hasKey("xmin"))  xlow  = atof(getValue("xmin").c_str());
    if (hasKey("xmax"))  xhigh = atof(getValue("xmax").c_str()); 
    ncol = 1;
    nrow = 1;
  } else {
    ncol = atoi(getValue("cols").c_str());
    nrow = atoi(getValue("rows").c_str());
  }
  fCanvas->Divide(ncol, nrow);

  for (unsigned int i = 0; i < hlist.size(); i++) 
	{
	  TProfile* prof=NULL;TH1F* his=NULL;
	  int ntry=0;
	do {
	  mui_->setCurrentFolder( "SiStrip/ControlView/"+mod_id );
	  cout << mui_->pwd()<<endl;

	  std::vector<std::string> tme = mui_->getMEs();
	  for (unsigned int j=0;j<tme.size();j++)
	    cout << tme[j] <<endl;

	  std::string  hname="SiStrip/ControlView/"+mod_id+"/"+hlist[i];
    
	  
	MonitorElement* me = mui_->get( hname ); // path + name
	  cout << mui_->pwd()<<endl;
	  cout << " Monitoring element " << hname  << " gives" << me <<endl;

	 prof = ExtractTObject<TProfile>().extract( me );
	 his = ExtractTObject<TH1F>().extract( me );
	if ( !prof && !his) usleep(10000);
	ntry++;
	if (ntry>100) break;
	} while ( !prof && !his);
	//if ( prof ) { prof->SetErrorOption("s"); } //@@ is this necessary? (until bug fix applied to dqm)...
	if (ntry>100) cout <<" All histos pointers are nulll....!" <<endl;
	
    if ( prof) {
	fCanvas->cd(i+1) ;
        if (xlow != -1 &&  xhigh != -1.0) {
          TAxis* xa = prof->GetXaxis();
          xa->SetRangeUser(xlow, xhigh);
        }

        prof->DrawCopy();

        fCanvas->Update();

      } else if (his) {
	fCanvas->cd(i+1) ;
        if (xlow != -1 && xhigh != -1.0) {
          TAxis* xa = his->GetXaxis();
          xa->SetRangeUser(xlow, xhigh);
        }
        his->DrawCopy();
	// check and set if Log option is choosen
	if (hasKey("logy")) {
	  gPad->SetLogy(1);
	}
        fCanvas->Update();
      }
	else
	{
	fCanvas->Clear();
	}
    }
  
  
  printImage(fCanvas,out);

      //plotSingleModuleHistos();
    } else if (fCommand == "summary_histo_list") {

      
      // createSummaryHistoList();
    } else if (fCommand == "plot_summary") {
      //plotSummaryHistos();
    }

  
    }
 catch(const std::exception& e) 
    {
      cout <<"give"<< endl;fCallBack->give();
      XCEPT_RAISE(xgi::exception::Exception,  e.what());
    }
   cout <<"give"<< endl;fCallBack->give();
}
//
// -- Fill ReqMap 
//
void HistogramDisplayHandler::fillMap(const std::string& urlstr){
  fReqMap.clear();
  std::vector<std::string> tokens;
  HistogramDisplayHandler::tokenize(urlstr, tokens, "&");
  for (std::vector<std::string>::const_iterator it = tokens.begin();
       it != tokens.end(); it++)  {
    std::string item = *it;
    std::string key, value;
    getPair(item, "=", key, value);
    fReqMap.insert(make_pair(key, value));
  }

  // print content
  if (0) {
    std::multimap<std::string,std::string>::iterator pos;
    for (pos = fReqMap.begin(); pos != fReqMap.end(); ++pos) {
      cerr << "'[" << pos->first << "]' => '[" << pos->second << "]'"<< endl;
    }
    cerr << endl;
  }
}
void HistogramDisplayHandler::getPair(const std::string& urlParam, const std::string& pat, std::string& key, std::string& value) {
  int index = urlParam.find(pat);
  if (index != (int) std::string::npos) {
    key   = urlParam.substr(0, index);
    value = urlParam.substr(index+1);
  }
}
std::string HistogramDisplayHandler::getValue(const std::string& key) {
  std::multimap<std::string,std::string>::iterator pos = fReqMap.find(key);
  std::string value = " ";
  if (pos != fReqMap.end()) {
    value = pos->second;
  }
  return value;
}
bool HistogramDisplayHandler::hasKey(const std::string& key) {
  std::multimap<std::string,std::string>::iterator pos = fReqMap.find(key);
  if (pos != fReqMap.end()) return true;
  return false;  
}
void HistogramDisplayHandler::getHistogramList(std::vector<std::string>& hlist) {
  hlist.clear();
  for (std::multimap<std::string,std::string>::iterator it = fReqMap.begin();
       it != fReqMap.end(); it++) {
    if (it->first == "hist") {
      hlist.push_back(it->second);
    }
  }
}
void HistogramDisplayHandler::tokenize(const std::string& str, std::vector<std::string>& tokens,
                               const std::string& delimiters)
{
  // ----------------------------
  // Skip delimiters at beginning.
  // ----------------------------
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);

  // ----------------------------
  // Find first "non-delimiter".
  // ----------------------------
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    // ------------------------------------
    // Found a token, add it to the std::vector.
    // ------------------------------------
      tokens.push_back(str.substr(lastPos, pos - lastPos));

    // ------------------------------------
    // Skip delimiters.  Note the "not_of"
    // ------------------------------------
      lastPos = str.find_first_not_of(delimiters, pos);

    // --------------------------
    // Find next "non-delimiter"
    // --------------------------
      pos = str.find_first_of(delimiters, lastPos);
  }
}
void HistogramDisplayHandler::printImage(TCanvas* c1,xgi::Output * out) {
  // Draw the canvas
  //  c1->UseCurrentStyle();
  c1->Draw();
  c1->SetFixedAspectRatio(kTRUE);
  c1->SetCanvasSize(520, 440);

  // Now extract the image
  TImage *image = TImage::Create();
  image->FromPad(c1);
  char *buf;
  int sz;
  image->GetImageBuffer(&buf, &sz);         /* raw buffer */

  std::ostringstream xmlstr;
  // xmlstr << "Content-Type: image/png\n Cache-Control: no-store, no-cache, must-revalidate, max-age=0" << endl;
  // xmlstr << "Expires: Mon, 26 Jul 1997 05:00:00 GMT" << endl
  //       << endl;
  for (int i = 0; i < sz; i++)
    xmlstr << buf[i];
  *out << xmlstr.str();


  delete [] buf;
}
