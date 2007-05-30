#include "DQM/L1TMonitorClient/interface/TriggerWebInterface.h"

#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"
#include "DQMServices/WebComponents/interface/HTMLLink.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQM/L1TMonitorClient/interface/DisplaySystemME.h"
#include "DQM/L1TMonitorClient/interface/DisplaySystemSummary.h"

#include <cstdio> // perror

bool meVerbose = true;
bool verbose() 
{
  return meVerbose;
}

/*
  Create your widgets in the constructor of your web interface
*/
TriggerWebInterface::TriggerWebInterface(std::string theContextURL, std::string theApplicationURL, std::string _url, MonitorUserInterface ** _mui_p)
  : WebInterface(theContextURL, theApplicationURL, _mui_p)
{
	context_url = theContextURL;
	application_url = theApplicationURL;
	url=_url;
	
	checkQTGlobalStatus=false;
	checkQTDetailedStatus=false;
  	ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "10px");

        HTMLLink *link = new HTMLLink(getApplicationURL(), "380px", "10px","<i>Go To Trigger Monitor WI</i>",url+"/DQMpage");  
	page_p = new WebPage(getApplicationURL());
 
        page_p->add("contentViewer", cont);
        page_p->add("htmlLink", link);
	
}

//****************************************************************************************
/*
  Only implement the handleCustomRequest function if you have widgets that invoke 
  custom-made methods defined in your client. In this example we have created a 
  Button that makes custom requests, therefore we need to implement it.
*/

void TriggerWebInterface::handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  // this is the way to get the string that identifies the request:
  std::multimap<std::string, std::string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);
  std::string requestID = get_from_multimap(request_multimap, "RequestID");
  request=requestID;
  // if you have more than one custom requests, add 'if' statements accordingly:
  if (requestID == "GoToTriggerMonitorWI")       this->GoToTriggerMonitorWI(in, out);
  if (requestID == "RetrieveMeList")             RetrieveMeList(in, out);
  if (requestID == "PlotMeList")                 PlotMeList(in, out);
  if ( requestID == "Summary") Summary(in, out);

}

//****************************************************************************************

void TriggerWebInterface::CheckQTGlobalStatus(xgi::Input * in, xgi::Output * out, bool start ) throw (xgi::exception::Exception)
{
  checkQTGlobalStatus=start;
}

//****************************************************************************************

void TriggerWebInterface::RetrieveMeList(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
  std::multimap<std::string, std::string> nav_map;

  CgiReader reader(in);
  reader.read_form(nav_map);

  std::string to_open = get_from_multimap(nav_map, "Source");
  
  
//  std::cout << "Enter RetrieveMeList" <<std::endl;

  if(to_open=="L1TECALTPG")   printMeListXML(to_open, out); 
  if(to_open=="L1TRCT")       printMeListXML(to_open, out); 
  if(to_open=="GCT")          printMeListXML(to_open, out); 
  if(to_open=="L1TDTTPG")     printMeListXML(to_open, out); 
  if(to_open=="L1TGT")        printMeListXML(to_open, out); 
  if(to_open=="L1TDTTF")      printMeListXML(to_open, out); 
  if(to_open=="L1TGMT")       printMeListXML(to_open, out); 

  if ( to_open == "Summary") Summary(in, out);


}

//****************************************************************************************

void TriggerWebInterface::printMeListXML(std::string source, xgi::Output * out)
{
  std::cout << "Enter printMeListXML" <<std::endl;

  if (!(*mui_p)) 
    {
      cout << "NO MUI!!!" << endl;
      return;
    }
  
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  
  *out << "<?xml version=\"1.0\" ?>" << endl;
   
  *out << "<navigator>" << endl;
  
  std::cout << "printMeListXML: written header" <<std::endl;
   
  
  std::vector<std::string> contents;
  std::vector<std::string> contentVec;
  (*mui_p)->getContents(contentVec);

//  std::cout << "contentVec size = " << contentVec.size() << std::endl;

  for (std::vector<std::string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
                                                                                                             
//       std::cout << "Me string = " << *it << std::endl;

       std::string::size_type dirCharNumber = it->find( ":", 0 );
       std::string dirName=it->substr(0 , dirCharNumber);
       dirName+= "/";

//       std::cout << "dirName = " << dirName << std::endl;
       
       std::string meCollectionName=it->substr(dirCharNumber+1);
       
//       std::cout << "meCollectionName = " << meCollectionName << std::endl;
       
       int CollectionNameSize = meCollectionName.length();
		                                                                                                             
       std::string::size_type SourceCharNumber = it->rfind("/");
       
       std::string sourceName = it->substr(SourceCharNumber+1);
        
       int sourceNameSize = sourceName.length()-CollectionNameSize-1;
      
       sourceName = it->substr(SourceCharNumber+1, sourceNameSize);
       
//       std::cout << "sourceName = " << sourceName << std::endl;
       
       if(source != sourceName) continue; // return only ME belonging to the source calling
                                          // this function


       std::string reminingNames=meCollectionName;
       bool anotherME=true;

//       std::cout << "reminingNames = " << reminingNames << std::endl;

       while(anotherME){
       if(reminingNames.find(",") == std::string::npos) anotherME =false;
       std::string::size_type singleMeNameCharNumber= reminingNames.find( ",", 0 );
       std::string singleMeName=reminingNames.substr(0 , singleMeNameCharNumber );
//       std::cout << "singleMeName = " << singleMeName << std::endl;
////               std::string fullpath=dirName + singleMeName;
////       contents.push_back(fullpath);
       *out << "<subscribe>" << singleMeName << "</subscribe>" << endl;
//       std::cout << "ME name = " << singleMeName << std::endl;
                                                                                                             
       reminingNames=reminingNames.substr(singleMeNameCharNumber+1);
//       std::cout << "reminingNames = " << reminingNames << std::endl;
       }
    }
                                                                                                             

  *out << "</navigator>" << endl;

}

//****************************************************************************************

void TriggerWebInterface::PlotMeList(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
  std::cout << "Enter PlotMeList" <<std::endl;

  std::multimap<std::string, std::string> nav_map;

  CgiReader reader(in);
  reader.read_form(nav_map);
  if ( verbose() ) {
    std::cout << "Navigation map follows:" << std::endl;
    std::multimap<std::string, std::string>::const_iterator i ;
    for ( i = nav_map.begin(); i != nav_map.end(); ++i ) {
      std::cout << i->first << " --> " << i->second << std::endl;
    }
    std::cout << "end Navigation map" << std::endl;
  }


    displayMeXML(in, out); 

}


//****************************************************************************************

void TriggerWebInterface::displayMeXML(xgi::Input * in, xgi::Output * out)
{
   std::multimap<std::string, std::string> view_multimap; 
   ME_map view_map; 

   CgiReader cgi_reader(in);
   cgi_reader.read_form(view_multimap);
   
   std::string source = get_from_multimap(view_multimap, "Source");
   std::string name = get_from_multimap(view_multimap, "View");

   DaqMonitorBEInterface * myBei = (*mui_p)->getBEInterface();
   myBei->lock();
   
   
   if (!(*mui_p)) 
     {
       std::cout << "mui not available!" << std::endl;
       return;
     }

   if ( verbose() ) {
     std::cout << "Requesting Me " << name << " from source " 
	       << source << std::endl;
   }

    name = "Collector/GlobalDQM/L1TMonitor/" + source +"/" + name;
  
   MonitorElement *pointer = (*mui_p)->get(name);

   if (pointer != 0) {
     view_map.add(name, pointer);
     std::cout << "ADDING " << name << " TO view_map!!!" << std::endl;
   
   // Print the ME_map into a file
   std::string id = get_from_multimap(view_multimap, "DisplayFrameName");
   std::cout << "will try to print " << id << std::endl;
   
     seal::Callback 
       action(seal::CreateCallback(this, 
				   &TriggerWebInterface::printMeMap, 
				   view_map, id));
     (*mui_p)->addCallback(action);

     out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
     *out << "<?xml version=\"1.0\" ?>" << std::endl;
     *out << "<fileURL>" << std::endl;
     *out << getContextURL() + "/temporary/" + id + ".gif" << std::endl;
     *out << "</fileURL>" << std::endl;
 
   } 
   else {
     std::cout << "No Reference to: " << name << std::endl;
     out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
     *out << "<?xml version=\"1.0\" ?>" << std::endl;
     *out << "nothing here " << std::endl;

   }
   myBei->unlock();

}
                                                                                                             

//****************************************************************************************

void TriggerWebInterface::printMeMap(ME_map view_map, std::string id)
{
  view_map.print(id);
}


//****************************************************************************************

void TriggerWebInterface::CheckQTDetailedStatus(xgi::Input * in, xgi::Output * out, bool start ) throw (xgi::exception::Exception)
{
  checkQTDetailedStatus=start;
}


//****************************************************************************************

void TriggerWebInterface::GoToTriggerMonitorWI(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{
    *out << cgicc::HTMLDoctype(cgicc::HTMLDoctype::eStrict) << std::endl;
    *out << cgicc::html().set("lang", "en").set("dir","ltr") << std::endl;
    *out << cgicc::title("Simple Web") << std::endl;
    *out << cgicc::a("Visit the XDAQ Web site").set("href","http://xdaq.web.cern.ch") << endl;
}


//****************************************************************************************

void TriggerWebInterface::CreateWI(xgi::Input * in, xgi::Output * out) 
  throw (xgi::exception::Exception)
{

  CgiWriter writer(out, getContextURL());
  
  writer.output_preamble();
  
  writer.output_head();

  *out << "<frameset rows=\"99%,1%\">" << std::endl;
  *out << "<frameset cols=\"10%,20%,70%\">" << std::endl;
  *out << "  <frame name=\"menu\" src=\"" << url << "/menu" << "\">" << std::endl;
  *out << "  <frame name=\"status\" src=\"" << url << "/status" << "\">" << std::endl;
  *out << "  <frame name=\"display\" src=\"" << url << "/display" << "\">" << std::endl;
  *out << "</frameset>" << std::endl;
  *out << "<frame name=\"debug\" src=\"" << url << "/debug" << "\">" 
       << std::endl;
  *out << "</frameset>" << std::endl;

  writer.output_finish();

}


//****************************************************************************************

void TriggerWebInterface::CreateMenu(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{

       WebPage *menu_p = new WebPage(getApplicationURL());
       menu_p->clear();

       // Summary has a different command that's called on click
       DisplaySystemSummary * summary = 
	 new DisplaySystemSummary(getApplicationURL(), "80px", "10px",
				  "Summary", "makeSummary(\'Summary\')" );

       DisplaySystemME * EcalTpg = new DisplaySystemME(getApplicationURL() , "110px", "10px", "L1TECALTPG");
       
       DisplaySystemME * HcalTpg = new DisplaySystemME(getApplicationURL() , "140px", "10px","HCAL_TPGs");
       
       DisplaySystemME * Rct = new DisplaySystemME(getApplicationURL() , "170px", "10px","L1TRCT");
       
       DisplaySystemME * Gct = new DisplaySystemME(getApplicationURL() , "200px", "10px","GCT");
       
       DisplaySystemME * rpc = new DisplaySystemME(getApplicationURL() , "230px", "10px","L1TDTTPG");
       
       DisplaySystemME * csc = new DisplaySystemME(getApplicationURL() , "260px", "10px","L1TGT");
       
       DisplaySystemME * dt = new DisplaySystemME(getApplicationURL() , "290px", "10px","L1TDTTF");
       
       DisplaySystemME * Gmt = new DisplaySystemME(getApplicationURL() , "320px", "10px","L1TGMT");
       
       DisplaySystemME * Gt = new DisplaySystemME(getApplicationURL() , "350px", "10px","RPC");
       
       DisplaySystemME * Emulator = new DisplaySystemME(getApplicationURL(), "380px", "10px","Emulator");

       menu_p->add("summary", summary);
       menu_p->add("EcalTpg", EcalTpg);
       menu_p->add("HcalTpg", HcalTpg);
       menu_p->add("Rct", Rct);
       menu_p->add("Gct", Gct);
       menu_p->add("rpc", rpc);
       menu_p->add("csc", csc);
       menu_p->add("dt", dt);
       menu_p->add("Gmt", Gmt);
       menu_p->add("Gt", Gt);
       menu_p->add("Emulator", Emulator);

       CgiWriter writer(out, getContextURL());
       writer.output_preamble();
       writer.output_head();
       *out << cgicc::h1("Menu").set("style", "font-family: arial") << std::endl;
       menu_p->printHTML(out);
       writer.output_finish();
}


//****************************************************************************************


void TriggerWebInterface::CreateStatus(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{

       WebPage *status_p = new WebPage(getApplicationURL());
       
       CgiWriter writer(out, getContextURL());
       writer.output_preamble();
       writer.output_head();
       *out << cgicc::h1("Elements Status").set("style", "font-family: arial") << std::endl;
       *out << cgicc::p().set("id","formsParId")  << std::endl;
       *out << cgicc::p() << std::endl;
       status_p->printHTML(out);
       writer.output_finish();
}

void TriggerWebInterface::CreateDebug(xgi::Input * in, xgi::Output * out) 
  throw (xgi::exception::Exception)
{
  WebPage *debug_p = new WebPage(getApplicationURL());
  
  CgiWriter writer(out, getContextURL());
  writer.output_preamble();
  writer.output_head();
  *out << cgicc::p().set("id","debug")  << std::endl;
  *out << cgicc::p() << std::endl;
  debug_p->printHTML(out);
  writer.output_finish();
}


//****************************************************************************************

void TriggerWebInterface::CreateDisplay(xgi::Input * in, xgi::Output * out) throw (xgi::exception::Exception)
{

       CgiWriter writer(out, getContextURL());
       writer.output_preamble();
       writer.output_head();
       *out << cgicc::h1("Display").set("style", "font-family: arial") << std::endl;
       writer.output_finish();
       
}


void TriggerWebInterface::Summary(xgi::Input * in, xgi::Output * out) 
  throw (xgi::exception::Exception)
{
  if ( verbose() ) 
    std::cout << "Entering Summary" <<std::endl;

  // get the list of elements. we get this every time for now.
//   std::string localPath = string("DQM/L1TMonitorClient/test/summary_mes.txt");
//   std::string fullPath =  edm::FileInPath(localPath).fullPath();
  std::string fullPath("/afs/cern.ch/user/w/wittich/scratch0/CMSSW_1_4_0_pre4/src/DQM/L1TMonitorClient/test/summary_mes.txt");
  if (verbose() ) {
    std::cout << "full path is " << fullPath << std::endl;
  }
  ifstream infile; 
  infile.open(fullPath.c_str(), ifstream::in);
  if ( ! infile.is_open() ) {
    std::cout << "ALERT: could no open list of ME's" << std::endl;
    std::ostringstream msg;
    msg << "open of " << fullPath;
    perror(msg.str().c_str());
    return;
  }
  std::vector<std::string> meList;
  std::string line;
  while ( infile >> line ) {
    std::cout << "Summary ME: " << line << std::endl; 
    meList.push_back(line);
  }
  infile.close();
  std::cout << "Total of " << meList.size() << " summary plots." <<std::endl;
  if ( meList.empty() ) {
    std::cout << "list is empty?" << std::endl;
    return;
  }
  //now subscribe to them
  
  DaqMonitorBEInterface * myBei = (*mui_p)->getBEInterface();
  myBei->lock();
   
  if (!(*mui_p))     {
    std::cout << "mui not available!" << std::endl;
    // should I unlock here?
    return;
  }
  std::multimap<std::string, std::string> view_multimap; 
  ME_map view_map; 

  CgiReader cgi_reader(in);
  cgi_reader.read_form(view_multimap);
  std::string source = get_from_multimap(view_multimap, "Source");
  std::string name = get_from_multimap(view_multimap, "View");
  if ( meList.empty() )
    return;

  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
  *out << "<meList>" << std::endl;
   
  for(std::vector<std::string>::const_iterator i = meList.begin();
      i != meList.end(); ++i ) {
    MonitorElement *p = (*mui_p)->get(*i);
    if ( p == 0 ) {
      std::cout << "No such ME " << *i << std::endl;
      continue;
    }

    std::string id = i->substr(1+i->rfind("/", i->length()), i->length());
    view_map.add(id, p);
    if ( verbose() ) 
      std::cout << "ADDING " << id << " TO view_map!!!" << std::endl;
    if ( verbose() ) 
      std::cout << "will try to print " << id << std::endl;
     seal::Callback 
       action(seal::CreateCallback(this, 
				   &TriggerWebInterface::printMeMap, 
				   view_map, id));
     (*mui_p)->addCallback(action);

     *out << "<SingleMe>" << std::endl;
     //*out << getContextURL() + "/temporary/" + id + ".gif" << std::endl;
     *out << *i << std::endl;
     *out << "</SingleMe>" << std::endl;

  }
  *out << "</meList>" << std::endl;



  myBei->unlock();

  

}
