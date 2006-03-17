
#include "EventFilter/StorageManager/src/EventStreamHttpReader.h"
#include "IOPool/Streamer/interface/BufferArea.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/Utilities.h"

#include <algorithm>
#include <iterator>
#include "curl/curl.h"
#include <string>

using namespace std;
using namespace edm;

namespace edmtestp
{  
  struct ReadData
  {
    std::string d_;
  };  

  size_t func(void* buf,size_t size, size_t nmemb, void* userp)
  {
    ReadData* rdata = (ReadData*)userp;
    size_t sz = size * nmemb;
    char* cbuf = (char*)buf;
    rdata->d_.insert(rdata->d_.end(),cbuf,cbuf+sz);
    return sz;
  }

  template <class Han, class Opt, class Par>
  int setopt(Han han,Opt opt,Par par)
  {
    if(curl_easy_setopt(han,opt,par)!=0)
      {
        cerr << "could not setopt " << opt << endl;
        abort();
      }
    return 0;
  }

  // ----------------------------------

  EventStreamHttpReader::EventStreamHttpReader(edm::ParameterSet const& ps,
					       edm::InputSourceDescription const& desc):
    edm::InputSource(desc),
    sourceurl_(ps.getParameter<string>("sourceURL")),
    buf_(1000*1000*7)
  {
    eventurl_ = sourceurl_ + "/geteventdata";
    headerurl_ = sourceurl_ + "/getregdata";
    std::auto_ptr<SendJobHeader> p = readHeader();
    edm::mergeWithRegistry(*p,productRegistry());
    prods_ = productRegistry(); // is this the one I want? Or pre-merge?

    // jbk - the next line should not be needed
    edm::declareStreamers(productRegistry());
    edm::buildClassCache(productRegistry());
    loadExtraClasses();
  }

  EventStreamHttpReader::~EventStreamHttpReader()
  {
  }

  std::auto_ptr<edm::EventPrincipal> EventStreamHttpReader::read()
  {
    CURL* han = curl_easy_init();
    ReadData data;

    if(han==0)
      {
        cerr << "could not create handle" << endl;
        return std::auto_ptr<edm::EventPrincipal>();
      }

    setopt(han,CURLOPT_URL,"http://lxplus014.cern.ch:1972/urn:xdaq-application:lid=29/geteventdata");
    //setopt(han,CURLOPT_URL,eventurl_);
    setopt(han,CURLOPT_WRITEFUNCTION,func);
    setopt(han,CURLOPT_WRITEDATA,&data);

    if(curl_easy_perform(han)!=0)
      {
        cerr << "curl perform failed for event" << endl;
        return std::auto_ptr<edm::EventPrincipal>();
      }
    curl_easy_cleanup(han);

    int len = data.d_.length();
    buf_.resize(len);
    for (int i=0; i<len ; i++) buf_[i] = data.d_[i];
    edm::EventMsg msg(&buf_[0],len);
    return decoder_.decodeEvent(msg,prods_);
    // or just use line below?
    //return decoder_.decodeEvent(msg,*productRegistry());
  }

  std::auto_ptr<SendJobHeader> EventStreamHttpReader::readHeader()
  {
    CURL* han = curl_easy_init();
    ReadData data;

    if(han==0)
      {
        cerr << "could not create handle" << endl;
        return std::auto_ptr<SendJobHeader>(); // is this right?
        //return 0; //or use this?
      }

    setopt(han,CURLOPT_URL,"http://lxplus014.cern.ch:1972/urn:xdaq-application:lid=29/getregdata");
    //setopt(han,CURLOPT_URL,headerurl_);
    setopt(han,CURLOPT_WRITEFUNCTION,func);
    setopt(han,CURLOPT_WRITEDATA,&data);

    if(curl_easy_perform(han)!=0)
      {
        cerr << "curl perform failed for header" << endl;
        return std::auto_ptr<SendJobHeader>(); // is this right?
        //return 0; //or use this?
      }
    curl_easy_cleanup(han);

    JobHeaderDecoder hdecoder;
    std::vector<char> regdata(1000*1000);

    int len = data.d_.length();
    regdata.resize(len);
    for (int i=0; i<len ; i++) regdata[0] = data.d_[i];
    edm::InitMsg msg(&regdata[0],len);
    std::auto_ptr<SendJobHeader> p = hdecoder.decodeJobHeader(msg);
    return p;
  }
}
