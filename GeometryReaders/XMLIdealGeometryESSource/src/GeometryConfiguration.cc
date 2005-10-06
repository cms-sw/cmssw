#include "Utilities/General/interface/FileInPath.h"
#include "Utilities/General/interface/envUtil.h"

#include "DetectorDescription/Parser/interface/DDLParser.h"

#include "GeometryReaders/XMLIdealGeometryESSource/interface/GeometryConfiguration.h"

#include <iostream>

GeometryConfiguration::GeometryConfiguration(std::string  fname, DDLParser & parser) : 
  m_config(parser.newConfig()),
  configfile(fname){

  envUtil eutil("Geometry_PATH",".");
  configpath = eutil.getEnv();

  const std::string thisdir=envUtil("PWD","").getEnv();
  std::cout << "Searching DDD geometry files in " << configpath << std::endl;

  FileInPath f1(configpath,configfile);
  std::cout << "Reading DDD configuration from " << f1.name() << std::endl;
  if (f1.name().empty())
      std::cout << "GeometryConfiguration: file not found" <<
      configfile << " not in " << configpath << std::endl;
  m_config->readConfig(f1.name());
  std::vector < std::string >  filenames = m_config->getFileList();
  std::vector < std::string >  urlnames =  m_config->getURLList();

  unsigned int idx=0;
  for (std::vector<std::string>::iterator it = filenames.begin(); it != filenames.end(); it++) 
    {
      // check if there is a real URL, e.g. http://
      if (urlnames[idx].find(":") > urlnames[idx].length()) {
	// we have a normal filename - search for it
	FileInPath f1(configpath,*it);
	if ( f1() == 0) {
	    std::cout << "XML file not found"
	    << *it << " not found in " << configpath << std::endl;
	} else {
	  *it = f1.name();
	  std::string url=f1.name();
	  // replace the local directory (.) by the full pathname
	  if (url.find(".")==0) {
	    url.erase(0,1);
	    url=thisdir+url;
	  }
	  myFilenames.push_back(url);
	  myURLs.push_back("");
	  std::cout << "File is " << url << std::endl;
	}
      } else {
	// keep the URL as is.
	myFilenames.push_back(filenames[idx]);
	myURLs.push_back(urlnames[idx]);
	std::cout << "URL  is " << urlnames[idx]+"/"+filenames[idx] << std::endl;
      }
      idx++; // next filename/urlname
    }
  dumpFileList();
  std::cout << "======== Geometry Configuration read ==========" << std::endl;
}

const std::vector<std::string>  & GeometryConfiguration::getFileList(void) const
{
  return myFilenames;
}

const std::vector<std::string> & GeometryConfiguration::getURLList(void) const
{
  return myURLs;
}

  /// Print out the list of files.
void GeometryConfiguration::dumpFileList(void) const {
  std::cout << "File List:" << std::endl;
  const std::vector<std::string> & vst = this->getFileList();
  std::cout << "  number of files=" << vst.size() << std::endl;
  for (std::vector<std::string>::const_iterator it = vst.begin(); it != vst.end(); it++) {
    std::cout << *it <<std:: endl;
  }
}

/// Return a flag whether to do xml validation or not.
bool GeometryConfiguration::doValidation() const {
  return m_config->doValidation();
}

  /// Return the Schema Location.
std::string GeometryConfiguration::getSchemaLocation() const {
  return m_config->getSchemaLocation();
}



