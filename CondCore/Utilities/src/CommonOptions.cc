#include "CondCore/Utilities/interface/CommonOptions.h"
#include <string>
cond::CommonOptions::CommonOptions( const std::string& commandname):m_name(commandname),m_description(new boost::program_options::options_description("options")),m_visible(new boost::program_options::options_description(std::string("Usage: ")+m_name+std::string(" [options] \n")) ){
  m_visible->add_options()
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  m_description->add(*m_visible);
}

cond::CommonOptions::CommonOptions( const std::string& commandname,
				    const std::string& positionparameter):m_name(commandname),m_description(new boost::program_options::options_description("options")),m_visible(new boost::program_options::options_description(std::string("Usage: ")+m_name+std::string(" [options] ")+positionparameter+std::string(" \n")) ){
  m_visible->add_options()
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  m_description->add(*m_visible);
}

cond::CommonOptions::~CommonOptions(){
  delete m_visible;
  m_visible=0;
  delete m_description;
  m_description=0;
}
void 
cond::CommonOptions::addAuthentication(const bool withEnvironmentAuth){
  m_visible->add_options()
    ("authPath,P",boost::program_options::value<std::string>(),"path to authentication xml(default .)");
  if (withEnvironmentAuth){
    m_visible->add_options()
      ("user,u",boost::program_options::value<std::string>(),"user name (default \"\")")
      ("pass,p",boost::program_options::value<std::string>(),"password (default \"\")");
  }
  m_description->add(*m_visible);
}

void
cond::CommonOptions::addConnect(){
  m_visible->add_options()
    ("connect,c",boost::program_options::value<std::string>(),"connection string(required)");
  m_description->add(*m_visible);
}
void 
cond::CommonOptions::addLogDB(){
   m_visible->add_options()
     ("logDB,l",boost::program_options::value<std::string>(),"logDB(optional");
   m_description->add(*m_visible);
}
void 
cond::CommonOptions::addDictionary(){
  m_visible->add_options()
    ("dictionary,D",boost::program_options::value<std::string>(),"data dictionary(required if no plugin available)");
  m_description->add(*m_visible);
}
void 
cond::CommonOptions::addFileConfig(){
  m_visible->add_options()
    ("configFile,f",boost::program_options::value<std::string>(),"configuration file(optional)");
  m_description->add(*m_visible);
}
void 
cond::CommonOptions::addBlobStreamer(){
  m_visible->add_options()
    ("blobStreamer,B",boost::program_options::value<std::string>(),"BlobStreamerName(default to COND/Services/TBufferBlobStreamingService)");
  m_description->add(*m_visible);
}
    
boost::program_options::options_description & 
cond::CommonOptions::description(){
  return *m_description;
}

boost::program_options::options_description& 
cond::CommonOptions::visibles(){
  return *m_visible;
}

