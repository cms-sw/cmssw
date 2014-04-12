#include "EffSourceHandler.h"
#include "CondFormats/Calibration/interface/EfficiencyPayloads.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


//#include<iostream>
#include<sstream>
#include<vector>
#include<string>
#include <sstream>
#include <typeinfo>


popcon::ExEffSource::ExEffSource(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","ExEffSource")),
  m_since(pset.getUntrackedParameter<long long>("since",5)),
  m_type(pset.getUntrackedParameter<std::string>("type","NULL")),
  m_params(pset.getUntrackedParameter<std::vector<double> >("params", std::vector<double>() )){
}

popcon::ExEffSource::~ExEffSource()
{
 
}

void popcon::ExEffSource::getNewObjects() {

  edm::LogInfo("ExEffSource") << "------- " << m_name 
			      << " - > getNewObjects\n" 
    //check whats already inside of database
			      << "got offlineInfo"
			      << tagInfo().name << ", size " 
			      << tagInfo().size 
			      << ", last object valid since " 
			      << tagInfo().lastInterval.first << " token "   
			      << tagInfo().lastPayloadToken << std::endl;
  //
  //edm::LogInfo ("ExEffsSource")<< " ------ last entry info regarding the payload (if existing): " <<logDBEntry().usertext<< 
  //  "; last record with the correct tag (if existing) has been written in the db: " <<logDBEntry().destinationDB<< std::endl; 

  if (tagInfo().size>0) {
    Ref payload = lastPayload();
    edm::LogInfo   ("ExEffsSource")<<" type of last payload  "<< 
      typeid(*payload).name()<<std::endl;
  }


  std::cout<<"since = "<< m_since <<std::endl;
  
  // the most stupid factory It shall be not be anymore in fase with "record name"
  condex::Efficiency * p0=0;
  if (m_type.find("Eta")!=std::string::npos)
    p0 = new condex::ParametricEfficiencyInEta(m_params[0],m_params[1],m_params[2],m_params[3]);
  else
    p0 = new condex::ParametricEfficiencyInPt(m_params[0],m_params[1],m_params[2],m_params[3]);
  
  if (p0==0) {
    edm::LogInfo   ("ExEffsSource")<<" unable to build "<< m_type << std::endl; 
    return;
  }
  
  m_to_transfer.push_back(std::make_pair(p0,(unsigned long long)m_since));
  
  
  std::ostringstream ss;
  ss << "type=" << m_type 
         << ",since=" << m_since; 
  
  m_userTextLog = ss.str()+ ";" ;
  
  
  
  edm::LogInfo   ("ExEffsSource") << "------- " << m_name << " - > getNewObjects" << std::endl;
}
  
  
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::ExEffSource> ExPopConEfficiency;
//define this as a plug-in
DEFINE_FWK_MODULE(ExPopConEfficiency);
