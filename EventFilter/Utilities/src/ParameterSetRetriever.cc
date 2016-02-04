#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/CurlUtils.h"
#include "EventFilter/Utilities/interface/SquidNet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "curl/curl.h"
#include <netdb.h>
#include <sys/socket.h>        /* for AF_INET */

#include <fstream>



namespace evf{
  //______________________________________________________________________________


  const std::string ParameterSetRetriever::fileheading="file:";
  const std::string ParameterSetRetriever::dbheading  ="db:";  
  const std::string ParameterSetRetriever::webheading ="http://";  

  ParameterSetRetriever::ParameterSetRetriever(const std::string& in)
  {
    using std::string;
    using std::ostringstream; 
    using std::ifstream; 



    if (fileheading==in.substr(0,fileheading.size()))
      { 
	string filename=in.substr(fileheading.size());
	edm::LogInfo("psetRetriever")<<"filename is --> "<<filename<<" <--";
      
	string   line;
	ifstream configFile(filename.c_str());
	while(std::getline(configFile,line)) {
	  pset+=line;
	  pset+="\n";
	}
      }
    else if (webheading==in.substr(0,webheading.size()))
      {
	string hostname = getHostString(in);
	SquidNet sn(3128,"http://localhost:8000/RELEASE-NOTES.txt");	
	edm::LogInfo("psetRetriever")<<"Using cfg from " << in;
	CURL* han = curl_easy_init();
	if(han==0)
	  {
	    XCEPT_RAISE(evf::Exception,"could not create handle for web ParameterSet");	    
	  }

	struct curl_slist *headers=NULL; /* init to NULL is important */
	headers = curl_slist_append(headers, "Pragma:");
	curl_easy_setopt(han, CURLOPT_HTTPHEADER, headers);
	char error[CURL_ERROR_SIZE];
	if(sn.check())
	  curl_easy_setopt(han, CURLOPT_PROXY, "localhost:3128");

	curl_easy_setopt(han, CURLOPT_URL, hostname.c_str());
	curl_easy_setopt(han, CURLOPT_VERBOSE,"");
	curl_easy_setopt(han, CURLOPT_NOSIGNAL,"");
	//	curl_easy_setopt(han, CURLOPT_TIMEOUT, 60.0L);

	curl_easy_setopt(han, CURLOPT_WRITEFUNCTION, &write_data);
	curl_easy_setopt(han, CURLOPT_WRITEDATA, &pset);
	curl_easy_setopt(han, CURLOPT_ERRORBUFFER, error);
	int success = curl_easy_perform(han);
	curl_slist_free_all(headers); /* free the header list */
	curl_easy_cleanup(han);
	han = 0;
	if(success != 0)
	  {
	    ostringstream msg;
	    msg <<  "could not get config from url " << in << " error #" 
		<< success << " " << error;
	    XCEPT_RAISE(evf::Exception,msg.str().c_str());
	  }

	//now get path-index table 	
	han = curl_easy_init();
	if(han==0)
	  {
	    XCEPT_RAISE(evf::Exception,"could not create handle for web ParameterSet");	    
	  }
	headers = NULL;
	headers = curl_slist_append(headers, "Pragma:");
	curl_easy_setopt(han, CURLOPT_HTTPHEADER, headers);
	if(sn.check())
	  curl_easy_setopt(han, CURLOPT_PROXY, "localhost:3128");
	hostname = getHostString(in,"path");
	curl_easy_setopt(han, CURLOPT_URL, hostname.c_str());
	curl_easy_setopt(han, CURLOPT_VERBOSE,"");
	curl_easy_setopt(han, CURLOPT_NOSIGNAL,"");
	//	curl_easy_setopt(han, CURLOPT_TIMEOUT, 60.0L);

	curl_easy_setopt(han, CURLOPT_WRITEFUNCTION, &write_data);
	curl_easy_setopt(han, CURLOPT_WRITEDATA, &pathIndexTable);
	curl_easy_setopt(han, CURLOPT_ERRORBUFFER, error);
	success = curl_easy_perform(han);
	curl_slist_free_all(headers); /* free the header list */
	curl_easy_cleanup(han);

	if(success != 0)
	  {
	    ostringstream msg;
	    msg <<  "could not get pathindex table from url " << in << " error #" 
		<< success << " " << error;
	    XCEPT_RAISE(evf::Exception,msg.str().c_str());
	  }


      }
    else if (dbheading==in.substr(0,dbheading.size()))
      {
	XCEPT_RAISE(evf::Exception,"db access for ParameterSet not yet implemented");
      } 
    else
      {
	edm::LogInfo("psetRetriever")<<"Using string cfg from RunControl or XML";
	pset = in;
      }
  }


  //______________________________________________________________________________
  std::string ParameterSetRetriever::getAsString() const
  {
    return pset;
  }
  //______________________________________________________________________________
  std::string ParameterSetRetriever::getPathTableAsString() const
  {
    return pathIndexTable;
  }


  std::string ParameterSetRetriever::getHostString(const std::string &in, std::string modifier) const
  {
    using std::string;
    string innohttp = in.substr(webheading.size());
    string::size_type pos = innohttp.find(':',0);
    string host = innohttp.substr(0,pos);
    string path = innohttp.substr(pos);
    if(modifier != "")
      {
	pos = path.find_last_of('.');
	if(pos != string::npos)
	  {
	    string upath = path.substr(0,pos);
	    path = upath + '.' + modifier;
	  }
      }
    struct hostent *heb = gethostbyname(host.c_str());    
    struct hostent *he = gethostbyaddr(heb->h_addr_list[0], heb->h_length, heb->h_addrtype);
    string completehost = "http://";
    completehost += he->h_name;
    completehost += path;
    return completehost;
  }
} //end namespace evf
