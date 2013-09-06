#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImplXMLFile.hh"
#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationItemNotFoundException.hh"
#include <zlib.h>

#ifdef HAVE_XDAQ
#include <toolbox/string.h>
#else
#include "CaloOnlineTools/HcalOnlineDb/interface/xdaq_compat.h"  // Replaces toolbox::toString
#endif

DECLARE_PLUGGABLE(hcal::ConfigurationDatabaseImpl,ConfigurationDatabaseImplXMLFile)

ConfigurationDatabaseImplXMLFile::ConfigurationDatabaseImplXMLFile() {
}
ConfigurationDatabaseImplXMLFile::~ConfigurationDatabaseImplXMLFile() {
}
bool ConfigurationDatabaseImplXMLFile::canHandleMethod(const std::string& method) const { return method=="xmlfile"; }

void ConfigurationDatabaseImplXMLFile::connect(const std::string& accessor) throw (hcal::exception::ConfigurationDatabaseException) {
  // open file and copy into a string
  std::string theFile=accessor;
  std::string::size_type i=theFile.find("://");
  if (i!=std::string::npos) theFile.erase(0,i+2); // remove up to the ://
  gzFile f=gzopen(theFile.c_str(),"rb");

  if (f==0) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Unable to open file "+theFile);
  }
  int c;
  while ((c=gzgetc(f))!=EOF) m_buffer+=(unsigned char)c;
  gzclose(f);

  // iterate through the string and extract the CFGBrick boundaries
  std::string::size_type j=0;
  while ((i=m_buffer.find("<CFGBrick>",j))!=std::string::npos) {
    j=m_buffer.find("</CFGBrick>",i)+strlen("</CFGBrick>");
    if (j==std::string::npos) break; 
    // extract all parameters
    std::map<std::string,std::string> params=extractParams(i,j);
    std::string key=createKey(params);
    //  printf(" --> %s\n",key.c_str());
    std::pair<int,int> ptrs(i,j);
    m_lookup.insert(std::pair<std::string, std::pair<int,int> >(key,ptrs));
  }
}

std::string ConfigurationDatabaseImplXMLFile::createKey(const std::map<std::string,std::string>& params) {
  std::string retval;
  if (params.find("PATTERN_SPEC_NAME")!=params.end()) { // HTR pattern
    retval=params.find("TAG")->second+":"+
      params.find("CRATE")->second+":"+
      params.find("SLOT")->second+":"+
      params.find("TOPBOTTOM")->second+":"+
      params.find("FIBER")->second;
  } else if (params.find("LUT_TYPE")!=params.end()) { // HTR LUT
    retval=params.find("TAG")->second+":"+
      params.find("CRATE")->second+":"+
      params.find("SLOT")->second+":"+
      params.find("TOPBOTTOM")->second+":"+
      params.find("LUT_TYPE")->second;
    if (params.find("FIBER")!=params.end()) 
      retval+=":"+params.find("FIBER")->second+":" +
	params.find("FIBERCHAN")->second;
    if (params.find("SLB")!=params.end()) 
      retval+=":"+params.find("SLB")->second+":" +
	params.find("SLBCHAN")->second;
  } else if (params.find("BOARD")!=params.end()) { // firmware!
    int ver=strtol(params.find("VERSION")->second.c_str(),0,0);
    retval=params.find("BOARD")->second+":"+
      ::toolbox::toString("%x",ver);
  } else if (params.find("ZS_TYPE")!=params.end()) { // ZS thresholds
    retval=params.find("TAG")->second+":"+
      params.find("CRATE")->second+":"+
      params.find("SLOT")->second+":"+
      params.find("TOPBOTTOM")->second;
  } else retval="WHAT";
  return retval;
}


std::map<std::string, std::string> ConfigurationDatabaseImplXMLFile::extractParams(int beg, int end) {
  std::map<std::string,std::string> pval;
  std::string::size_type l=beg,i,j;
  std::string name,val;
  
  while ((i=m_buffer.find("<Parameter",l))!=std::string::npos && i<(unsigned int)end) {
    j=m_buffer.find("name=",i);
    char separator=m_buffer[j+5];
    i=m_buffer.find(separator,j+6);
    name=m_buffer.substr(j+6,i-(j+6));
    if (name=="CREATIONTAG") name="TAG"; // RENAME!
    j=m_buffer.find('>',j);
    i=m_buffer.find("</",j);
    val=m_buffer.substr(j+1,i-j-1);
    pval.insert(std::pair<std::string,std::string>(name,val));
    l=j;
  }

  return pval;
}

void ConfigurationDatabaseImplXMLFile::disconnect() {
  m_lookup.clear();
  m_buffer.clear();
}

std::map<std::string, std::string> ConfigurationDatabaseImplXMLFile::parseWhere(const std::string& where) {
  std::string::size_type i,j=0,k,k2;
  std::map<std::string,std::string> itis;

  while ((i=where.find('=',j))!=std::string::npos) {
    k=where.rfind(' ',i);
    k2=where.rfind('(',i);
    if (k2!=std::string::npos && k2>k) k=k2;
    if (k==std::string::npos) k=0;
    else k++;
    std::string key = where.substr(k,i-k),value;
    if (where[i+1]=='\'' || where[i+1]=='\"') {
      j=where.find(where[i+1],i+2);
      value=where.substr(i+2,j-i-2);
    } else {
      j=where.find(' ',i);
      k=where.find(')',i);
      if (k!=std::string::npos && k<j) j=k;
      value=where.substr(i+1,j-i-1);
    }
    itis.insert(std::pair<std::string,std::string>(key,value));
  }
  return itis;
}

/*
hcal::ConfigurationDatabaseIterator* ConfigurationDatabaseImplXMLFile::query(const std::string& sector, const std::string& draftSelect, const std::string& draftWhere) throw (hcal::exception::ConfigurationDatabaseException) { 

  std::map<std::string,std::string> whereMap=parseWhere(draftWhere);
  if (sector=="PATTERN") whereMap["PATTERN_SPEC_NAME"]=whereMap["TAG"];
  std::string lookup=createKey(whereMap);
  //  printf("'%s'\n",lookup.c_str());
  std::map<std::string, std::pair<int,int> >::iterator j=m_lookup.find(lookup);
  if (j==m_lookup.end()) return new ConfigurationDatabaseImplXMLFileIterator("");
  std::string data="<?xml version='1.0'?>\n";
  data+=m_buffer.substr(j->second.first,j->second.second-j->second.first);
  return new ConfigurationDatabaseImplXMLFileIterator(data);
}
*/

unsigned int ConfigurationDatabaseImplXMLFile::getFirmwareChecksum(const std::string& board, unsigned int version) throw (hcal::exception::ConfigurationDatabaseException) {
  XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Unsupported");
}

void ConfigurationDatabaseImplXMLFile::getFirmwareMCS(const std::string& board, unsigned int version, std::vector<std::string>& mcsLines) throw (hcal::exception::ConfigurationDatabaseException) {

  std::string key=::toolbox::toString("%s:%x",board.c_str(),version);

  std::map<std::string, std::pair<int,int> >::iterator j=m_lookup.find(key);
  if (j==m_lookup.end()) {
    XCEPT_RAISE(hcal::exception::ConfigurationItemNotFoundException,"");
  }
  std::string data="<?xml version='1.0'?>\n";
  data+=m_buffer.substr(j->second.first,j->second.second-j->second.first);
  
  std::map<std::string, std::string> params;
  std::string encoding;
  m_parser.parse(data.c_str(),params,mcsLines,encoding);

}

void ConfigurationDatabaseImplXMLFile::getLUTChecksums(const std::string& tag, std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::MD5Fingerprint>& checksums) throw (hcal::exception::ConfigurationDatabaseException) {
  XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,"Unsupported");
}

void ConfigurationDatabaseImplXMLFile::getLUTs(const std::string& tag, int crate, int slot, std::map<hcal::ConfigurationDatabase::LUTId, hcal::ConfigurationDatabase::LUT >& LUTs) throw (hcal::exception::ConfigurationDatabaseException) {
  LUTs.clear();

  for (int tb=0; tb<=1; tb++) 
    for (int fiber=1; fiber<=8; fiber++) 
      for (int fiberChan=0; fiberChan<=2; fiberChan++) {
	int lut_type=1;

	std::string key=toolbox::toString("%s:%d:%d:%d:%d:%d:%d",
					  tag.c_str(), crate, slot, tb, lut_type, fiber, fiberChan);
	std::map<std::string, std::pair<int,int> >::iterator j=m_lookup.find(key);
	if (j==m_lookup.end()) continue;
	std::string data="<?xml version='1.0'?>\n";
	data+=m_buffer.substr(j->second.first,j->second.second-j->second.first);
	
	std::map<std::string, std::string> params;
	std::vector<std::string> values;
	std::string encoding;
	m_parser.parse(data.c_str(),params,values,encoding);
	
	hcal::ConfigurationDatabase::LUTId id(crate,slot,(hcal::ConfigurationDatabase::FPGASelection)tb,fiber,fiberChan,(hcal::ConfigurationDatabase::LUTType)lut_type);
	hcal::ConfigurationDatabase::LUT& lut=LUTs[id];
	lut.reserve(values.size());
	
	int strtol_base=0;
	if (encoding=="hex") strtol_base=16;
	else if (encoding=="dec") strtol_base=10;
	
	// convert the data
	for (unsigned int j=0; j<values.size(); j++) 
	  lut.push_back(strtol(values[j].c_str(),0,strtol_base));
      }
  for (int tb=0; tb<=1; tb++) 
    for (int slb=1; slb<=6; slb++) 
      for (int slbChan=0; slbChan<=3; slbChan++) {
	int lut_type=2;

	std::string key=toolbox::toString("%s:%d:%d:%d:%d:%d:%d",
					  tag.c_str(), crate, slot, tb, lut_type, slb, slbChan);

	std::map<std::string, std::pair<int,int> >::iterator j=m_lookup.find(key);
	if (j==m_lookup.end()) continue;
	std::string data="<?xml version='1.0'?>\n";
	data+=m_buffer.substr(j->second.first,j->second.second-j->second.first);
	
	std::map<std::string, std::string> params;
	std::vector<std::string> values;
	std::string encoding;
	m_parser.parse(data.c_str(),params,values,encoding);
	
	hcal::ConfigurationDatabase::LUTId id(crate,slot,(hcal::ConfigurationDatabase::FPGASelection)tb,slb,slbChan,(hcal::ConfigurationDatabase::LUTType)lut_type);
	hcal::ConfigurationDatabase::LUT& lut=LUTs[id];
	lut.reserve(values.size());
	
	int strtol_base=0;
	if (encoding=="hex") strtol_base=16;
	else if (encoding=="dec") strtol_base=10;
	
	// convert the data
	for (unsigned int j=0; j<values.size(); j++) 
	  lut.push_back(strtol(values[j].c_str(),0,strtol_base));
      }
}

void ConfigurationDatabaseImplXMLFile::getZSThresholds(const std::string& tag, int crate, int slot, std::map<hcal::ConfigurationDatabase::ZSChannelId, int>& thresholds) throw (hcal::exception::ConfigurationDatabaseException) {
  thresholds.clear();
  for (int tb=0; tb<=1; tb++) {

    std::string key=toolbox::toString("%s:%d:%d:%d",
				      tag.c_str(), crate, slot, tb);
    std::map<std::string, std::pair<int,int> >::iterator j=m_lookup.find(key);
    if (j==m_lookup.end()) continue;
    std::string data="<?xml version='1.0'?>\n";
    data+=m_buffer.substr(j->second.first,j->second.second-j->second.first);
      
    std::map<std::string, std::string> params;
    std::vector<std::string> values;
    std::string encoding;
    m_parser.parse(data.c_str(),params,values,encoding);
    
    if (values.size()!=24) {
      XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException,::toolbox::toString("Must have 24 items in ZS list.  Saw %d for %s",values.size(),key.c_str()));
    }
    for (int fiber=1; fiber<=8; fiber++) 
      for (int fc=0; fc<3; fc++) {
	hcal::ConfigurationDatabase::ZSChannelId id(crate,slot,(hcal::ConfigurationDatabase::FPGASelection)tb,fiber,fc);
	
	int strtol_base=0;
	if (encoding=="hex") strtol_base=16;
	else if (encoding=="dec") strtol_base=10;
      
	thresholds[id]=strtol(values[(fiber-1)*3+fc].c_str(),0,strtol_base);
      }
  }
}

void ConfigurationDatabaseImplXMLFile::getPatterns(const std::string& tag, int crate, int slot, std::map<hcal::ConfigurationDatabase::PatternId, hcal::ConfigurationDatabase::HTRPattern >& patterns) throw (hcal::exception::ConfigurationDatabaseException) {
  patterns.clear();
  for (int tb=0; tb<=1; tb++) 
    for (int fiber=1; fiber<=8; fiber++) {
      std::string key=toolbox::toString("%s:%d:%d:%d:%d",
				      tag.c_str(), crate, slot, tb, fiber);
      std::map<std::string, std::pair<int,int> >::iterator j=m_lookup.find(key);
      if (j==m_lookup.end()) continue;
      std::string data="<?xml version='1.0'?>\n";
      data+=m_buffer.substr(j->second.first,j->second.second-j->second.first);
      
      std::map<std::string, std::string> params;
      std::vector<std::string> values;
      std::string encoding;
      m_parser.parse(data.c_str(),params,values,encoding);
      
      hcal::ConfigurationDatabase::PatternId id(crate,slot,(hcal::ConfigurationDatabase::FPGASelection)tb,fiber);
      hcal::ConfigurationDatabase::HTRPattern& lut=patterns[id];
      lut.reserve(values.size());
      
    int strtol_base=0;
    if (encoding=="hex") strtol_base=16;
    else if (encoding=="dec") strtol_base=10;
      
    // convert the data
    for (unsigned int j=0; j<values.size(); j++) 
      lut.push_back(strtol(values[j].c_str(),0,strtol_base));
    }
}

/*
// added by Gena Kukartsev
oracle::occi::Connection * ConfigurationDatabaseImplXMLFile::getConnection( void ){
  return NULL;
}

oracle::occi::Environment * ConfigurationDatabaseImplXMLFile::getEnvironment( void ){
  return NULL;
}
*/
