#include "ExceptionGenerator.h"

#include <iostream>
#include <typeinfo>
#include <map>
#include <sstream>

#include "TRandom3.h"

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/tokenizer.hpp"

#include <stdio.h>
#include <sys/types.h>
#include <signal.h>

using namespace std;

namespace evf{

    const std::string ExceptionGenerator::menu[menu_items] =  
      {"Sleep x ms", "SleepForever", "Cms Exception", "Exit with error", "Abort", "Unknown Exception", "Endless loop", "Generate Error Message", "Segfault", 
       "Burn CPU","HLT timing distribution","HLT timing with memory access","Timed segfault","Invalid free()"};

    ExceptionGenerator::ExceptionGenerator( const edm::ParameterSet& pset) : 
      ModuleWeb("ExceptionGenerator"), 
      actionId_(pset.getUntrackedParameter<int>("defaultAction",-1)),
      intqualifier_(pset.getUntrackedParameter<unsigned int>("defaultQualifier",0)), 
      actionRequired_(actionId_!=-1)
    {
      
    }
  void ExceptionGenerator::beginJob()
  {
    // timing destribution from (https://twiki.cern.ch/twiki/bin/viewauth/CMS/HLTCpuTimingFAQ#2011_Most_Recent_Data)
    // /castor/cern.ch/user/d/dsperka/HLT/triggerSkim_HLTPhysics_run178479_68_188.root
    // Baseline result with CMSSW_4_2_9_HLT3_hltpatch3 and /online/collisions/2011/5e33/v2.1/HLT/V9 :
    // vocms110:/store/timing_178479/outfile-178479-col1.root
    
    timingHisto_ = new TH1D("timingHisto_","Total time for all modules per event",100,0,1000);
    timingHisto_->SetBinContent(1,5016);
    timingHisto_->SetBinContent(2,4563);
    timingHisto_->SetBinContent(3,3298);
    timingHisto_->SetBinContent(4,1995);
    timingHisto_->SetBinContent(5,1708);
    timingHisto_->SetBinContent(6,1167);
    timingHisto_->SetBinContent(7,928);
    timingHisto_->SetBinContent(8,785);
    timingHisto_->SetBinContent(9,643);
    timingHisto_->SetBinContent(10,486);
    timingHisto_->SetBinContent(11,427);
    timingHisto_->SetBinContent(12,335);
    timingHisto_->SetBinContent(13,332);
    timingHisto_->SetBinContent(14,327);
    timingHisto_->SetBinContent(15,258);
    timingHisto_->SetBinContent(16,257);
    timingHisto_->SetBinContent(17,222);
    timingHisto_->SetBinContent(18,253);
    timingHisto_->SetBinContent(19,223);
    timingHisto_->SetBinContent(20,177);
    timingHisto_->SetBinContent(21,148);
    timingHisto_->SetBinContent(22,148);
    timingHisto_->SetBinContent(23,113);
    timingHisto_->SetBinContent(24,83);
    timingHisto_->SetBinContent(25,84);
    timingHisto_->SetBinContent(26,75);
    timingHisto_->SetBinContent(27,61);
    timingHisto_->SetBinContent(28,66);
    timingHisto_->SetBinContent(29,51);
    timingHisto_->SetBinContent(30,43);
    timingHisto_->SetBinContent(31,38);
    timingHisto_->SetBinContent(32,27);
    timingHisto_->SetBinContent(33,34);
    timingHisto_->SetBinContent(34,28);
    timingHisto_->SetBinContent(35,18);
    timingHisto_->SetBinContent(36,26);
    timingHisto_->SetBinContent(37,18);
    timingHisto_->SetBinContent(38,11);
    timingHisto_->SetBinContent(39,11);
    timingHisto_->SetBinContent(40,12);
    timingHisto_->SetBinContent(41,14);
    timingHisto_->SetBinContent(42,11);
    timingHisto_->SetBinContent(43,8);
    timingHisto_->SetBinContent(44,4);
    timingHisto_->SetBinContent(45,2);
    timingHisto_->SetBinContent(46,5);
    timingHisto_->SetBinContent(47,3);
    timingHisto_->SetBinContent(48,4);
    timingHisto_->SetBinContent(49,6);
    timingHisto_->SetBinContent(50,6);
    timingHisto_->SetBinContent(51,3);
    timingHisto_->SetBinContent(52,5);
    timingHisto_->SetBinContent(53,6);
    timingHisto_->SetBinContent(54,6);
    timingHisto_->SetBinContent(55,6);
    timingHisto_->SetBinContent(56,4);
    timingHisto_->SetBinContent(57,5);
    timingHisto_->SetBinContent(58,9);
    timingHisto_->SetBinContent(59,3);
    timingHisto_->SetBinContent(60,3);
    timingHisto_->SetBinContent(61,8);
    timingHisto_->SetBinContent(62,7);
    timingHisto_->SetBinContent(63,5);
    timingHisto_->SetBinContent(64,7);
    timingHisto_->SetBinContent(65,5);
    timingHisto_->SetBinContent(66,5);
    timingHisto_->SetBinContent(67,4);
    timingHisto_->SetBinContent(68,2);
    timingHisto_->SetBinContent(69,2);
    timingHisto_->SetBinContent(70,4);
    timingHisto_->SetBinContent(71,5);
    timingHisto_->SetBinContent(72,4);
    timingHisto_->SetBinContent(73,5);
    timingHisto_->SetBinContent(74,3);
    timingHisto_->SetBinContent(75,5);
    timingHisto_->SetBinContent(76,3);
    timingHisto_->SetBinContent(77,9);
    timingHisto_->SetBinContent(78,2);
    timingHisto_->SetBinContent(79,2);
    timingHisto_->SetBinContent(80,5);
    timingHisto_->SetBinContent(81,5);
    timingHisto_->SetBinContent(82,5);
    timingHisto_->SetBinContent(83,5);
    timingHisto_->SetBinContent(84,4);
    timingHisto_->SetBinContent(85,4);
    timingHisto_->SetBinContent(86,9);
    timingHisto_->SetBinContent(87,5);
    timingHisto_->SetBinContent(88,4);
    timingHisto_->SetBinContent(89,4);
    timingHisto_->SetBinContent(90,5);
    timingHisto_->SetBinContent(91,3);
    timingHisto_->SetBinContent(92,3);
    timingHisto_->SetBinContent(93,3);
    timingHisto_->SetBinContent(94,7);
    timingHisto_->SetBinContent(95,5);
    timingHisto_->SetBinContent(96,6);
    timingHisto_->SetBinContent(97,2);
    timingHisto_->SetBinContent(98,3);
    timingHisto_->SetBinContent(99,5);
    timingHisto_->SetBinContent(101,147);
    timingHisto_->SetEntries(24934);
  }
  void ExceptionGenerator::beginRun(edm::Run& r, const edm::EventSetup& iSetup)
  {
    gettimeofday(&tv_start_,0);
  }

  void ExceptionGenerator::analyze(const edm::Event & e, const edm::EventSetup& c)
    {
      float dummy = 0.;
      unsigned int iterations = 0;
      if(actionRequired_) 
	{
	  int *pi = 0;
	  int ind = 0; 
	  int step = 1; 
	  switch(actionId_)
	    {
	    case 0:
	      ::usleep(intqualifier_*1000);
	      break;
	    case 1:
	      ::sleep(0xFFFFFFF);
	      break;
	    case 2:
	      throw cms::Exception(qualifier_) << "This exception was generated by the ExceptionGenerator";
	      break;
	    case 3:
	      exit(-1);
	      break;
	    case 4:
	      abort();
	      break;
	    case 5:
	      throw qualifier_;
	      break;
	    case 6:
	      while(1){ind+=step; if(ind>1000000) step = -1; if(ind==0) step = 1;}
	      break;
	    case 7:
	      edm::LogError("TestErrorMessage") << qualifier_;
	      break;
	    case 8:
	      *pi=0;
	      break;
	    case 9:
	      for(unsigned int j=0; j<intqualifier_*1000;j++){
		dummy += sqrt(log(float(j+1)))/float(j*j);
	      }
	      break;
            case 10:
              iterations = static_cast<unsigned int>(
                timingHisto_->GetRandom() * intqualifier_*17. + 0.5
              );
	      for(unsigned int j=0; j<iterations;j++){
		dummy += sqrt(log(float(j+1)))/float(j*j);
	      }
              break;
            case 11:
	      {
                iterations = static_cast<unsigned int>(
                  timingHisto_->GetRandom() * intqualifier_*12. + 0.5
                );
                TRandom3 random(iterations);
                const size_t dataSize = 32*500; // 124kB
                std::vector<double> data(dataSize);
                random.RndmArray(dataSize, &data[0]);
              
	        for(unsigned int j=0; j<iterations;j++){
                  const size_t index = static_cast<size_t>(random.Rndm() * dataSize + 0.5);
                  const double value = data[index];
		  dummy += sqrt(log(value+1))/(value*value);
                  if ( random.Rndm() < 0.1 )
                    data[index] = dummy;
	        }
	      }
              break;
	    case 12:
	      {
		timeval tv_now;
	        gettimeofday(&tv_now,0);
		if (static_cast<unsigned long>(tv_now.tv_sec-tv_start_.tv_sec) > intqualifier_)
		  *pi=0;
	      }
	      break;
	    case 13:
	      void *vp = malloc(1024);
	      memset((char *)vp - 32, 0, 1024);
	      free(vp);
	      break;
	    }
	}
    }
    
    void ExceptionGenerator::endLuminosityBlock(edm::LuminosityBlock const &lb, edm::EventSetup const &es)
    {

    }
    
    void ExceptionGenerator::defaultWebPage(xgi::Input *in, xgi::Output *out)
    {
      gettimeofday(&tv_start_,0);
      std::string path;
      std::string urn;
      std::string mname;
      std::string query;
      try 
	{
	  cgicc::Cgicc cgi(in);
	  if ( xgi::Utils::hasFormElement(cgi,"exceptiontype") )
	    {
	      actionId_ = xgi::Utils::getFormElement(cgi, "exceptiontype")->getIntegerValue();
	      try {
	        qualifier_ = xgi::Utils::getFormElement(cgi, "qualifier")->getValue();
	        intqualifier_ =  xgi::Utils::getFormElement(cgi, "qualifier")->getIntegerValue();
	      }
	      catch (...) {
	        //didn't have some parameters
	      }
	      actionRequired_ = true;
	    }
	  if ( xgi::Utils::hasFormElement(cgi,"module") )
	    mname = xgi::Utils::getFormElement(cgi, "module")->getValue();
	  cgicc::CgiEnvironment cgie(in);
	  if(original_referrer_ == "")
	    original_referrer_ = cgie.getReferrer();
	  path = cgie.getPathInfo();
	  query = cgie.getQueryString();
	  if(actionId_>=0)
	    std::cout << " requested action " << actionId_ << " " 
		      << menu[actionId_] << ". Number of cycles " 
		      << intqualifier_ << std::endl;
	}
      catch (const std::exception & e) 
	{
	  // don't care if it did not work
	}

      using std::endl;
      *out << "<html>"                                                   << endl;
      *out << "<head>"                                                   << endl;


      *out << "<STYLE type=\"text/css\"> #T1 {border-width: 2px; border: solid blue; text-align: center} </STYLE> "                                      << endl; 
      *out << "<link type=\"text/css\" rel=\"stylesheet\"";
      *out << " href=\"/" <<  urn
	   << "/styles.css\"/>"                   << endl;

      *out << "<title>" << moduleName_
	   << " MAIN</title>"                                            << endl;

      *out << "</head>"                                                  << endl;
      *out << "<body onload=\"loadXMLDoc()\">"                           << endl;
      *out << "<table border=\"0\" width=\"100%\">"                      << endl;
      *out << "<tr>"                                                     << endl;
      *out << "  <td align=\"left\">"                                    << endl;
      *out << "    <img"                                                 << endl;
      *out << "     align=\"middle\""                                    << endl;
      *out << "     src=\"/evf/images/systemerror.jpg\""	         << endl;
      *out << "     alt=\"main\""                                        << endl;
      *out << "     width=\"90\""                                        << endl;
      *out << "     height=\"64\""                                       << endl;
      *out << "     border=\"\"/>"                                       << endl;
      *out << "    <b>"                                                  << endl;
      *out <<             moduleName_                                    << endl;
      *out << "    </b>"                                                 << endl;
      *out << "  </td>"                                                  << endl;
      *out << "  <td width=\"32\">"                                      << endl;
      *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
      *out << "      <img"                                               << endl;
      *out << "       align=\"middle\""                                  << endl;
      *out << "       src=\"/hyperdaq/images/HyperDAQ.jpg\""             << endl;
      *out << "       alt=\"HyperDAQ\""                                  << endl;
      *out << "       width=\"32\""                                      << endl;
      *out << "       height=\"32\""                                     << endl;
      *out << "       border=\"\"/>"                                     << endl;
      *out << "    </a>"                                                 << endl;
      *out << "  </td>"                                                  << endl;
      *out << "  <td width=\"32\">"                                      << endl;
      *out << "  </td>"                                                  << endl;
      *out << "  <td width=\"32\">"                                      << endl;
      *out << "    <a href=\"" << original_referrer_  << "\">"           << endl;
      *out << "      <img"                                               << endl;
      *out << "       align=\"middle\""                                  << endl;
      *out << "       src=\"/evf/images/spoticon.jpg\""			 << endl;
      *out << "       alt=\"main\""                                      << endl;
      *out << "       width=\"32\""                                      << endl;
      *out << "       height=\"32\""                                     << endl;
      *out << "       border=\"\"/>"                                     << endl;
      *out << "    </a>"                                                 << endl;
      *out << "  </td>"                                                  << endl;
      *out << "</tr>"                                                    << endl;
      *out << "</table>"                                                 << endl;

      *out << "<hr/>"                                                    << endl;
  
      *out << cgicc::form().set("method","GET").set("action", path ) 
	   << std::endl;
      boost::char_separator<char> sep("&");
      boost::tokenizer<boost::char_separator<char> > tokens(query, sep);
      for (boost::tokenizer<boost::char_separator<char> >::iterator tok_iter = tokens.begin();
	   tok_iter != tokens.end(); ++tok_iter){
	size_t pos = (*tok_iter).find_first_of("=");
	if(pos != std::string::npos){
	  std::string first  = (*tok_iter).substr(0    ,                        pos);
	  std::string second = (*tok_iter).substr(pos+1, (*tok_iter).length()-pos-1);
	  *out << cgicc::input().set("type","hidden").set("name",first).set("value", second) 
	       << std::endl;
	}
      }

      *out << "Select   "						 << endl;
      *out << cgicc::select().set("name","exceptiontype")     << std::endl;
      char istring[2];

      for(int i = 0; i < menu_items; i++)
	{
	  sprintf(istring,"%d",i);
	  *out << cgicc::option().set("value",istring) << menu[i] << cgicc::option()       << std::endl;
	}
      *out << cgicc::select() 	     << std::endl;
      *out << "<br>"                                                     << endl;
      *out << "Qualifier"      						 << endl;
      *out << cgicc::input().set("type","text").set("name","qualifier")  	     << std::endl;
      *out << cgicc::input().set("type","submit").set("value","Do It !")  	     << std::endl;
      *out << cgicc::form()						   << std::endl;  

      *out << "</body>"                                                  << endl;
      *out << "</html>"                                                  << endl;
    }
  void ExceptionGenerator::publish(xdata::InfoSpace *is)
  {
  }
  void ExceptionGenerator::publishForkInfo(moduleweb::ForkInfoObj *forkInfoObj)
  {
  }
} // end namespace evf
