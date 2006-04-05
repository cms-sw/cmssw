/** \file
 *
 *  Implementation of RPCWebClient
 *
 *  $Date: 2006/03/14 11:24:20 $
 *  $Revision: 1.6 $
 *  \author Ilaria Segoni
 */
#include "DQM/RPCMonitorClient/interface/RPCWebClient.h"
#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"
#include "DQM/RPCMonitorClient/interface/WebMessage.h"


#include "DQM/RPCMonitorClient/interface/QTestConfigurationParser.h"
#include "DQM/RPCMonitorClient/interface/QTestEnabler.h"

RPCWebClient::RPCWebClient(xdaq::ApplicationStub * s) : DQMWebClient(s)
{
  printout=true;
  testsConfigured=false;
  testsRunning=false;
  taskList.clear();
  
  yCoordinateMessage=350;
  
  

/// Drop down menus  
  ConfigBox * box = new ConfigBox(getApplicationURL(), "50px", "50px");
  Navigator * nav = new Navigator(getApplicationURL(), "1500px", "50px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "1700px", "50px");

///  BUTTONS:  

  ///Button to get list of Monitoring Tasks Available
  Button * butTasks = new Button(getApplicationURL(), "50px", "500px", "GetMonitoringTasks", "Get Monitoring Tasks List"); 
  ///Button to set up Quality tests (from txt configuration file)
  Button * butConfigQT = new Button(getApplicationURL(), "350px", "50px", "ConfigQltyTests", "Configure Quality Tests");
  ///Button to run Quality tests
  butRunQT = new Button(getApplicationURL(), "380px", "50px", "RunQltyTests",       "Run Quality Tests");
  ///Buttin to check Quality tests results
  butCheckQT = new Button(getApplicationURL(), "410px", "50px", "CheckQltyTests",   "Check Global Status Quality Tests");
  ///Buttin to check Quality tests results
  butCheckQTSingle = new Button(getApplicationURL(), "440px", "50px", "CheckQltyTestsSingle",   "Check Detailed Status Quality Tests");
  

///Displays  
  ///Display for basic plots:
  GifDisplay * disBasePlots = new GifDisplay(getApplicationURL(), "1000px","50px", "400px", "600px", "BasePlotsGifDisplay");   
  ///Display for on-line requests
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "1500px","700px", "400px", "600px", "MyGifDisplay");  
  ///Display for reference/blessed plots:
  GifDisplay * disRefPlots = new GifDisplay(getApplicationURL(), "2000px","50px", "400px", "600px", "RefPlotsGifDisplay");   
 
 
  /// Add elementa to the page
  page = new WebPage(getApplicationURL());
  page->add("configBox", box);
  page->add("navigator", nav);
  page->add("contentViewer", cont);
  
  page->add("button_getTasks", butTasks);
  
  page->add("button_setTests", butConfigQT);
  page->add("button_runTests", butRunQT);
  page->add("button_checkTests", butCheckQT);
  page->add("button_checkTestsSingle", butCheckQTSingle);
  
  page->add("gifDisplay", dis);
  page->add("gifDisplayBase", disBasePlots);
  page->add("gifDisplayRef", disRefPlots);

}


void RPCWebClient::Default(xgi::Input * in, xgi::Output * out ) 
  throw (xgi::exception::Exception)
{
  string pagetitle = "RPCWebClient";
  CgiWriter writer(out, getContextURL());
  writer.output_preamble();
  writer.output_head();
  page->printHTML(out);
  writer.output_finish();
}

 

void RPCWebClient::Request(xgi::Input * in, xgi::Output * out ) 
  throw (xgi::exception::Exception)
{
  // put the request information in a multimap...
  std::multimap<string, string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);

  // get the string that identifies the request:
  std::string requestID = this->get_from_multimap(request_multimap, "RequestID");
  
  
  for(std::vector<std::string>::iterator it=taskList.begin(); it!=taskList.end(); ++it){   
    if (requestID == *it)    this->AddTaskButtons(in, out,*it);
  }
  
  
  
  if (requestID == "GetMonitoringTasks")    this->GetAvailableTasks(in, out);
  
  if (requestID == "ConfigQltyTests")       this->ConfigQTestsRequest(in, out);
  if (requestID == "RunQltyTests")          this->RunQTestsRequest(in, out);
  if (requestID == "CheckQltyTests")        this->CheckQTestsRequest(in, out);
  if (requestID == "CheckQltyTestsSingle")  this->CheckQTestsRequestSingle(in, out);
}

void RPCWebClient::GetAvailableTasks(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception){

  mui->subscribe("*/MonitoringTask/*");
  
  usleep(1000000);

  mui->cd("Collector/FU0/MonitoringTask");
  mui->pwd();
  std::vector<std::string> meNames=mui->getMEs(); 
  int ycoord=80;  
  char yValue[20];
  sprintf(yValue,"%dpx",ycoord);
  WebMessage * taskType= new WebMessage(getApplicationURL(), yValue , "500px","Available Monitoring Tasks:","green"  );
  page->add("availableTasks",taskType);
  
  taskList.clear();
  for(std::vector<std::string>::iterator it = meNames.begin(); 
			it != meNames.end();++it)
  {        
	char fullPath[128];
	sprintf(fullPath,"Collector/FU0/MonitoringTask/%s",(*it).c_str());
   	MonitorElement * me =mui->get(fullPath);
	std::string taskName=me->valueString();
	taskList.push_back(taskName);
 	ycoord+=40;
 	sprintf(yValue,"%dpx",ycoord);
        Button * butTask= new Button(getApplicationURL(),yValue , "500px",taskName , taskName);
        char butName[120];
	sprintf(butName,"But_%s",taskName.c_str());
	page->add(butName,butTask);
  }   
   


}
void RPCWebClient::AddTaskButtons(xgi::Input * in, xgi::Output *out,std::string taskType) throw (xgi::exception::Exception) {


 std::cout<<"Adding Buttons for task: "<<taskType<<endl;


}





void RPCWebClient::ConfigQTestsRequest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  if(testsConfigured){   
	if(printout) std::cout<< "Quality tests are already configured!"<<std::endl;
	return;     
  } else{  
  	if(printout) std::cout << "Quality Tests are being configured" << std::endl;
  } 
  
  
  std::string xmlFile="QualityTests.xml";
  QTestConfigurationParser * qtParser=new QTestConfigurationParser(xmlFile);
  std::map<std::string, std::map<std::string, std::string> > testsList=qtParser->testsList();
  QTestEnabler * testsEnabler= new QTestEnabler();
  testsEnabler->enableTests(testsList,mui);
  std::vector<std::string> tests= testsEnabler->testsReady();

  std::vector<std::string>::iterator itr;
  for(itr=tests.begin();  itr!=tests.end();++itr){
        std::cout<<"Tests configured: "<<*itr<<std::endl;
  }
  
  

  if(testsList.size() == 0 ){
        if(printout) std::cout<< "Error Configuring Quality Tests"<<std::endl;
	return;
  }
  
  testsConfigured=true;
 
  return;

}


void RPCWebClient::RunQTestsRequest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{

  if(! testsConfigured){   
	if(printout) std::cout<< "Configure quality tests first!"<<std::endl;
	return;     
  } else{  
	if(printout) std::cout << "Beginning to run quality tests" << std::endl;
  }
  //qualityTests->RunTests(mui);

  testsRunning=true;
  return;

}



void RPCWebClient::CheckQTestsRequest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  if(!testsRunning){   
    if(printout) cout<< "No quality tests are running. Start Quality tests First!"<<endl;
    return;     
  }

  // std::pair<std::string,std::string> globalStatus = qualityTests->CheckTestsGlobal(mui);
  // WebMessage * message= new WebMessage(getApplicationURL(), "350px" , "500px", globalStatus.first,globalStatus.second  );
  // if(printout) cout<< globalStatus.first<<endl;
  

 
 



   return;
}

void RPCWebClient::CheckQTestsRequestSingle(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  if(!testsRunning){   
    if(printout) cout<< "No quality tests are running. Start Quality tests First!"<<endl;
    return;     
  }
/*  
  //std::map< std::string, std::vector<std::string> > messages= qualityTests->CheckTestsSingle(mui);  
///Error messages  
  char alarm[128] ;
  std::vector<std::string> errors = messages["red"];
  sprintf(alarm,"Number of Errors :%d",errors.size());
  if(printout) cout<< alarm <<endl;
  //WebMessage * messageNumberOfErrors= new WebMessage(getApplicationURL(), "380px" , "500px", alarm,"red"  );
  //page->add("numberoferrors",messageNumberOfErrors);
///Warning messages  
  std::vector<std::string> warnings = messages["orange"];
  sprintf(alarm,"Number of Warnings :%d", warnings.size() );
  if(printout) cout<< alarm <<endl;
  //WebMessage * messageNumberOfWarningss= new WebMessage(getApplicationURL(), "410px" , "500px", alarm,"orange"  );
  //page->add("numberofwarnings",messageNumberOfWarningss );
   
  int yCoordinateMessage=440;



  int counter=0;
  for(std::vector<std::string>::iterator itr=errors.begin(); itr!=errors.end(); ++itr){
  	std::cout<<"Messaggi: "<<(*itr)<<std::endl;  
	++counter;
	char messName[20];
  	sprintf(messName,"errorMessage%d",counter);
	
	char yValue[20];
  	string extension="px";
  	sprintf(yValue,"%d%s",yCoordinateMessage,extension.c_str());
  	
	WebMessage * message= new WebMessage(getApplicationURL(), yValue , "500px", (*itr),"red"  );
  	page->add(messName,  message);
  	yCoordinateMessage+=30;
  }
  
  counter=0;
  for(std::vector<std::string>::iterator itr= warnings.begin(); itr!=warnings.end(); ++itr){
  	std::cout<<"Messaggi: "<<(*itr)<<std::endl;  
	++counter;
	char messName[20];
  	sprintf(messName,"warningMessage%d",counter);
	
	char yValue[20];
  	string extension="px";
  	sprintf(yValue,"%d%s",yCoordinateMessage,extension.c_str());
  	
	WebMessage * message= new WebMessage(getApplicationURL(), yValue , "500px", (*itr),"orange"  );
  	page->add(messName,  message);
  	yCoordinateMessage+=30;
  }
 
*/  
   
  return;
}


//
// provides factory method for instantion of SimpleWeb application
//
XDAQ_INSTANTIATOR_IMPL(RPCWebClient)
  
