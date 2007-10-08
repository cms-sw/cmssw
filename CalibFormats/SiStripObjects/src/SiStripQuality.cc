//
// Author:      Domenico Giordano
// Created:     Wed Sep 26 17:42:12 CEST 2007
// $Id$
//
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiStripQuality::SiStripQuality():toCleanUp(false){
  reader = new SiStripDetInfoFileReader(edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat").fullPath());
}

SiStripQuality::SiStripQuality(edm::FileInPath& file):toCleanUp(false){
  reader = new SiStripDetInfoFileReader(file.fullPath());
}

SiStripQuality::SiStripQuality(const SiStripBadStrip* base):toCleanUp(false){
  reader = new SiStripDetInfoFileReader(edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat").fullPath());
  add(base);
}

SiStripQuality::SiStripQuality(const SiStripBadStrip* base, edm::FileInPath& file):toCleanUp(false){
  reader = new SiStripDetInfoFileReader(file.fullPath());
  add(base);
}

void SiStripQuality::add(const SiStripBadStrip* base){
  SiStripBadStrip::RegistryIterator basebegin = base->getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator baseend   = base->getRegistryVectorEnd();
  SiStripBadStrip::Range  range, baserange;
  std::vector<unsigned int> vect, tmp;
  
  if (indexes.size()==0){
    //Case A - the Registry is empty
    edm::LogInfo("SiStripQuality") << "in CaseA" << std::endl;
    indexes.insert(indexes.end(),basebegin,baseend);
    v_badstrips.insert(v_badstrips.end(),base->getDataVectorBegin(),base->getDataVectorEnd());
      
  }else{
  
    //Case B - the Registry already contains data
    //Loop on detids
    for (SiStripBadStrip::RegistryIterator basep=basebegin; basep != baseend; ++basep) {
      uint32_t detid=basep->detid;
      edm::LogInfo("SiStripQuality") << "add detid " <<detid << std::endl;

      unsigned short Nstrips=reader->getNumberOfApvsAndStripLength(detid).first*128;
    
      baserange = SiStripBadStrip::Range( base->getDataVectorBegin()+basep->ibegin , base->getDataVectorBegin()+basep->iend );
  
      //Is this detid already in the collections owned by this class?
      range = getRange(detid);
    
      //Append bad strips  
      tmp.clear();
      if (range.first==range.second){
	edm::LogInfo("SiStripQuality") << "new detid" << std::endl;
	//It's a new detid
	tmp.insert(tmp.end(),baserange.first,baserange.second);
  	std::stable_sort(tmp.begin(),tmp.end());
	edm::LogInfo("SiStripQuality") << "ordered" << std::endl;
      } else {
	edm::LogInfo("SiStripQuality") << "already exists" << std::endl;
	//alredy existing detid
	
	//if full det is bad go to next detid
	edm::LogInfo("SiStripQuality") << "PIPPO " << range.second-range.first << " " << decode(*(range.first)).first << " " << decode(*(range.first)).second << "\n"<< std::endl;
	if(range.second-range.first==1
	   && decode(*(range.first)).first==0
	   && decode(*(range.first)).second>=Nstrips-1){
	  continue;
	}
	 	
 	tmp.insert(tmp.end(),baserange.first,baserange.second);
	tmp.insert(tmp.end(),range.first,range.second);
 	std::stable_sort(tmp.begin(),tmp.end());
	edm::LogInfo("SiStripQuality") << "ordered" << std::endl;
      }
      //Compact data
      compact(tmp,vect,Nstrips);
      edm::LogInfo("SiStripQuality") << "compacted" << std::endl;
		
      SiStripBadStrip::Range newrange(vect.begin(),vect.end());
      if ( ! put_replace(detid,newrange) )
	edm::LogError("SiStripQuality")<<"[" << __PRETTY_FUNCTION__ << "] " << std::endl;
    }
  }
}

bool SiStripQuality::put_replace(const uint32_t& DetId, Range input) {
  // put in SiStripQuality::v_badstrips of DetId
  Registry::iterator p = std::lower_bound(indexes.begin(),indexes.end(),DetId,SiStripBadStrip::StrictWeakOrdering());

  size_t sd= input.second-input.first;
  DetRegistry detregistry;
  detregistry.detid=DetId;
  detregistry.ibegin=v_badstrips.size();
  detregistry.iend=v_badstrips.size()+sd;

  v_badstrips.insert(v_badstrips.end(),input.first,input.second);

  if (p!=indexes.end() && p->detid==DetId){
    edm::LogInfo("SiStripQuality") << "[SiStripQuality::put_replace]  Replacing SiStripQuality for already stored DetID " << DetId << std::endl;
    toCleanUp=true;
    *p=detregistry;
  } else {
    indexes.insert(p,detregistry);
  }

  return true;
}

void SiStripQuality::compact(std::vector<unsigned int>& tmp,std::vector<unsigned int>& vect,unsigned short& Nstrips){
  std::pair<unsigned short,unsigned short> fs_0, fs_1, fs_m, fs_M; 
  vect.clear();

  ContainerIterator it=tmp.begin();
  fs_0=decode(*it);
  
  //Check if at the module end
  if (fs_0.first+fs_0.second>=Nstrips-1){
    vect.push_back(encode(fs_0.first,fs_0.second));
    return;
  }

  ++it;
  for(;it!=tmp.end();++it){
    fs_1=decode(*it);
    
    if (fs_0.first+fs_0.second>=fs_1.first+fs_1.second){
      //fs_0 includes fs_1, go ahead
    } else if (fs_0.first+fs_0.second>=fs_1.first){
      // contiguous or superimposed intervals
      //create new fs_0
      fs_0=std::make_pair(fs_0.first,fs_1.first+fs_1.second-fs_0.first);
      
      //Check if at the module end
      if (fs_0.first+fs_0.second>=Nstrips-1){
	vect.push_back(encode(fs_0.first,fs_0.second));
	return;
      }
    } else{
      //separated intervals
      vect.push_back(encode(fs_0.first,fs_0.second));
      fs_0=fs_1;
    }
  }
  vect.push_back(encode(fs_0.first,fs_0.second));
}

bool SiStripQuality::cleanUp(){

  if (!toCleanUp)
    return false;

  toCleanUp=false;
  
  std::vector<unsigned int> v_badstrips_tmp=v_badstrips;
  std::vector<DetRegistry> indexes_tmp=indexes;

  edm::LogInfo("SiStripQuality") << "[SiStripQuality::cleanUp] before cleanUp v_badstrips.size()= " << v_badstrips.size() << std::endl;

  v_badstrips.clear();
  indexes.clear();

  SiStripBadStrip::RegistryIterator basebegin = indexes_tmp.begin();
  SiStripBadStrip::RegistryIterator baseend   = indexes_tmp.end();

  for (SiStripBadStrip::RegistryIterator basep=basebegin; basep != baseend; ++basep) {

    SiStripBadStrip::Range range( v_badstrips_tmp.begin()+basep->ibegin, v_badstrips_tmp.begin()+basep->iend );
    if ( ! put(basep->detid,range) )
      edm::LogError("SiStripQuality")<<"[" << __PRETTY_FUNCTION__ << "] " << std::endl;
  }
  
  edm::LogInfo("SiStripQuality") << "[SiStripQuality::cleanUp] after cleanUp v_badstrips.size()= " << v_badstrips.size() << std::endl;
  return true;
}

bool SiStripQuality::IsStripBad(const uint32_t& detid, const short& strip) {
  bool result=false;
  SiStripBadStrip::Range range=getRange(detid);
  std::pair<unsigned short,unsigned short> fs;
  for(SiStripBadStrip::ContainerIterator it=range.first;it!=range.second;++it){
    fs=decode(*it);
    if ( fs.first<=strip && strip<=fs.first+fs.second ){
      result=true;
      break;
    }      
  }
  return result;
}




EVENTSETUP_DATA_REG(SiStripQuality);
