 //
// Author:      Domenico Giordano
// Created:     Wed Sep 26 17:42:12 CEST 2007
// $Id: SiStripQuality.cc,v 1.3 2007/10/18 08:42:18 giordano Exp $
//
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

 
SiStripQuality::SiStripQuality():
  toCleanUp(false),
  FileInPath_("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"){
  reader=new SiStripDetInfoFileReader(FileInPath_.fullPath());
}

SiStripQuality::SiStripQuality(edm::FileInPath& file):toCleanUp(false),FileInPath_(file){
  reader=new SiStripDetInfoFileReader(FileInPath_.fullPath());
}

SiStripQuality::SiStripQuality(const SiStripQuality& other){
  FileInPath_=other.FileInPath_;
  reader=new SiStripDetInfoFileReader(*(other.reader));
  toCleanUp=other.toCleanUp;
  indexes=other.indexes;
  v_badstrips=other.v_badstrips;
  BadComponentVect=other.BadComponentVect;
}


SiStripQuality& SiStripQuality::operator +=(const SiStripQuality& other){ 
  this->add(&other); 
  this->cleanUp(); 
  this->fillBadComponents(); 
  return *this; 
}

SiStripQuality& SiStripQuality::operator -=(const SiStripQuality& other){
    
  SiStripBadStrip::RegistryIterator rbegin = other.getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator rend   = other.getRegistryVectorEnd();
  std::vector<unsigned int> ovect,vect;
  uint32_t detid;
  unsigned short Nstrips;
    
  for (SiStripBadStrip::RegistryIterator rp=rbegin; rp != rend; ++rp) {
    
    detid=rp->detid;
    Nstrips=reader->getNumberOfApvsAndStripLength(detid).first*128;
    
    SiStripBadStrip::Range orange = SiStripBadStrip::Range( other.getDataVectorBegin()+rp->ibegin , other.getDataVectorBegin()+rp->iend );
    
    //Is this detid already in the collections owned by this class?
    SiStripBadStrip::Range range = getRange(detid);   
    if (range.first!=range.second){ //yes, it is

      vect.clear();
      ovect.clear();

      //if other full det is bad, remove det from this
      SiStripBadStrip::data data_=decode(*(orange.first));
      if(orange.second-orange.first!=1
	 || data_.firstStrip!=0
	 || data_.range<Nstrips){
	
	ovect.insert(ovect.end(),orange.first,orange.second);
	vect.insert(vect.end(),range.first,range.second);
	subtract(vect,ovect);
      } 
      SiStripBadStrip::Range newrange(vect.begin(),vect.end());
      if ( ! put_replace(detid,newrange) )
	edm::LogError("SiStripQuality")<<"[" << __PRETTY_FUNCTION__ << "] " << std::endl;
    }
  }
  cleanUp(); 
  fillBadComponents(); 
  return *this; 
}

const SiStripQuality SiStripQuality::operator -(const SiStripQuality& other) const {
  return SiStripQuality(*this) -= other; 
}

bool SiStripQuality::operator ==(const SiStripQuality& other) const{
  SiStripQuality a = (*this) - other ;
  return a.getRegistryVectorBegin()==a.getRegistryVectorEnd();
}
bool SiStripQuality::operator !=(const SiStripQuality& other) const { return !(*this == other) ; }


void SiStripQuality::add(const SiStripBadStrip* base){
  SiStripBadStrip::RegistryIterator basebegin = base->getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator baseend   = base->getRegistryVectorEnd();
  SiStripBadStrip::Range  range, baserange;
  std::vector<unsigned int> vect, tmp;
  
  if (indexes.size()==0){
    //Case A - the Registry is empty
    indexes.insert(indexes.end(),basebegin,baseend);
    v_badstrips.insert(v_badstrips.end(),base->getDataVectorBegin(),base->getDataVectorEnd());
      
  }else{
  
    //Case B - the Registry already contains data
    //Loop on detids
    for (SiStripBadStrip::RegistryIterator basep=basebegin; basep != baseend; ++basep) {
      uint32_t detid=basep->detid;
      LogTrace("SiStripQuality") << "add detid " <<detid << std::endl;

      unsigned short Nstrips=reader->getNumberOfApvsAndStripLength(detid).first*128;
    
      baserange = SiStripBadStrip::Range( base->getDataVectorBegin()+basep->ibegin , base->getDataVectorBegin()+basep->iend );
  
      //Is this detid already in the collections owned by this class?
      range = getRange(detid);
    
      //Append bad strips  
      tmp.clear();
      if (range.first==range.second){
	LogTrace("SiStripQuality") << "new detid" << std::endl;
	//It's a new detid
	tmp.insert(tmp.end(),baserange.first,baserange.second);
  	std::stable_sort(tmp.begin(),tmp.end());
	LogTrace("SiStripQuality") << "ordered" << std::endl;
      } else {
	LogTrace("SiStripQuality") << "already exists" << std::endl;
	//alredy existing detid
	
	//if full det is bad go to next detid
	SiStripBadStrip::data data_=decode(*(range.first));
	if(range.second-range.first==1
	   && data_.firstStrip==0
	   && data_.range>=Nstrips){
	  LogTrace("SiStripQuality") << "full det is bad.. " << range.second-range.first << " " << decode(*(range.first)).firstStrip << " " << decode(*(range.first)).range << " " << decode(*(range.first)).flag <<"\n"<< std::endl;
	  continue;
	}
	 	
 	tmp.insert(tmp.end(),baserange.first,baserange.second);
	tmp.insert(tmp.end(),range.first,range.second);
 	std::stable_sort(tmp.begin(),tmp.end());
	LogTrace("SiStripQuality") << "ordered" << std::endl;
      }
      //Compact data
      compact(tmp,vect,Nstrips);
      SiStripBadStrip::Range newrange(vect.begin(),vect.end());
      if ( ! put_replace(detid,newrange) )
	edm::LogError("SiStripQuality")<<"[" << __PRETTY_FUNCTION__ << "] " << std::endl;
    }
  }
}

void SiStripQuality::compact(unsigned int& detid, std::vector<unsigned int>& vect){
  std::vector<unsigned int> tmp=vect;
  vect.clear();
  std::stable_sort(tmp.begin(),tmp.end());
  unsigned short Nstrips=reader->getNumberOfApvsAndStripLength(detid).first*128;
  compact(tmp,vect,Nstrips);
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
    LogTrace("SiStripQuality") << "[SiStripQuality::put_replace]  Replacing SiStripQuality for already stored DetID " << DetId << std::endl;
    toCleanUp=true;
    *p=detregistry;
  } else {
    indexes.insert(p,detregistry);
  }

  return true;
}

void SiStripQuality::compact(std::vector<unsigned int>& tmp,std::vector<unsigned int>& vect,unsigned short& Nstrips){
  SiStripBadStrip::data fs_0, fs_1;
  vect.clear();

  ContainerIterator it=tmp.begin();
  fs_0=decode(*it);
   
  //Check if at the module end
  if (fs_0.firstStrip+fs_0.range>=Nstrips){
    vect.push_back(encode(fs_0.firstStrip,Nstrips-fs_0.firstStrip));
    return;
  }

  ++it;
  for(;it!=tmp.end();++it){
    fs_1=decode(*it);
    
    if (fs_0.firstStrip+fs_0.range>=fs_1.firstStrip+fs_1.range){
      //fs_0 includes fs_1, go ahead
    } else if (fs_0.firstStrip+fs_0.range>=fs_1.firstStrip){
      // contiguous or superimposed intervals
      //Check if at the module end
      if (fs_1.firstStrip+fs_1.range>=Nstrips){
	vect.push_back(encode(fs_0.firstStrip,Nstrips-fs_0.firstStrip));
	return;
      }else{
      	//create new fs_0
	fs_0.range=fs_1.firstStrip+fs_1.range-fs_0.firstStrip;
      }
    } else{
      //separated intervals
      vect.push_back(encode(fs_0.firstStrip,fs_0.range));
      fs_0=fs_1;
    }
  }
  vect.push_back(encode(fs_0.firstStrip,fs_0.range));
}

void SiStripQuality::subtract(std::vector<unsigned int>& A,const std::vector<unsigned int>& B){
  ContainerIterator it=B.begin();
  ContainerIterator itend=B.end();
  for(;it!=itend;++it){    
    subtraction(A,*it);
  }
}

void SiStripQuality::subtraction(std::vector<unsigned int>& A,const unsigned int& B){
  SiStripBadStrip::data fs_A, fs_B, fs_m, fs_M;
  std::vector<unsigned int> tmp;

  fs_B=decode(B);
  ContainerIterator jt=A.begin();
  ContainerIterator jtend=A.end();
  for(;jt!=jtend;++jt){
    fs_A=decode(*jt);
    if (B<*jt){
      fs_m=fs_B;
      fs_M=fs_A;
    }else{
      fs_m=fs_A;
      fs_M=fs_B;
    }
    //A) Verify the range to be subtracted crosses the new range
    if (fs_m.firstStrip+fs_m.range>fs_M.firstStrip){
      if (*jt<B){
	tmp.push_back(encode(fs_A.firstStrip,fs_B.firstStrip-fs_A.firstStrip));
      }
      if (fs_A.firstStrip+fs_A.range>fs_B.firstStrip+fs_B.range){
	tmp.push_back(encode(fs_B.firstStrip+fs_B.range,fs_A.firstStrip+fs_A.range-(fs_B.firstStrip+fs_B.range)));
      }
    }else{
      tmp.push_back(*jt);
    }
  } 
  A=tmp;
}

bool SiStripQuality::cleanUp(){

  if (!toCleanUp)
    return false;

  toCleanUp=false;

  std::vector<unsigned int> v_badstrips_tmp=v_badstrips;
  std::vector<DetRegistry> indexes_tmp=indexes;

  LogTrace("SiStripQuality") << "[SiStripQuality::cleanUp] before cleanUp v_badstrips.size()= " << v_badstrips.size() << " indexes.size()=" << indexes.size() << std::endl;

  v_badstrips.clear();
  indexes.clear();

  SiStripBadStrip::RegistryIterator basebegin = indexes_tmp.begin();
  SiStripBadStrip::RegistryIterator baseend   = indexes_tmp.end();

  for (SiStripBadStrip::RegistryIterator basep=basebegin; basep != baseend; ++basep) {
    if(basep->ibegin!=basep->iend){
      SiStripBadStrip::Range range( v_badstrips_tmp.begin()+basep->ibegin, v_badstrips_tmp.begin()+basep->iend );
      if ( ! put(basep->detid,range) )
	edm::LogError("SiStripQuality")<<"[" << __PRETTY_FUNCTION__ << "] " << std::endl;
    }
  }
  
  LogTrace("SiStripQuality") << "[SiStripQuality::cleanUp] after cleanUp v_badstrips.size()= " << v_badstrips.size() << " indexes.size()=" << indexes.size() << std::endl;
  return true;
}

void SiStripQuality::fillBadComponents(){
  
  BadComponentVect.clear();
  
  for (SiStripBadStrip::RegistryIterator basep=indexes.begin(); basep != indexes.end(); ++basep) {
    
    SiStripBadStrip::Range range( v_badstrips.begin()+basep->ibegin, v_badstrips.begin()+basep->iend );
    
    //Fill BadModules, BadFibers, BadApvs vectors
    unsigned short resultA=0, resultF=0;
    BadComponent result;
    
    SiStripBadStrip::data fs;
    unsigned short Nstrips=reader->getNumberOfApvsAndStripLength(basep->detid).first*128;
    
    //BadModules
    if (basep->iend - basep->ibegin == 1){
      fs=decode(*(range.first));
      if ( fs.firstStrip==0 && 
	   fs.range==Nstrips ){
	result.detid=basep->detid;
	result.BadModule=true;
	result.BadFibers=(1<< (Nstrips/256))-1; 
	result.BadApvs=(1<< (Nstrips/128))-1; 
	
	BadComponentVect.push_back(result);
      }  
    } else {

      //Bad Fibers and  Apvs
      for(SiStripBadStrip::ContainerIterator it=range.first;it!=range.second;++it){
	fs=decode(*it);
      
	//BadApvs
	for(short apvNb=0;apvNb<6;++apvNb){
	  if ( fs.firstStrip<=apvNb*128 && (apvNb+1)*128<=fs.firstStrip+fs.range ){
	    resultA=resultA | (1<<apvNb);
	  }     
	}
	//BadFibers
	for(short fiberNb=0;fiberNb<3;++fiberNb){
	  if ( fs.firstStrip<=fiberNb*256 && (fiberNb+1)*256<=fs.firstStrip+fs.range ){
	    resultF=resultF | (1<<fiberNb);
	  }     
	}
      }
      if (resultA!=0){
	result.detid=basep->detid;
	result.BadModule=false;
	result.BadFibers=resultF;
	result.BadApvs=resultA;
	BadComponentVect.push_back(result);    
      }
    }
  }
}

//--------------------------------------------------------------//

bool SiStripQuality::IsModuleBad(const uint32_t& detid) const{  

  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return p->BadModule;
  return false;
}

bool SiStripQuality::IsFiberBad(const uint32_t& detid, const short& fiberNb) const{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return ((p->BadFibers>>fiberNb)&0x1);
  return false;
}

bool SiStripQuality::IsApvBad(const uint32_t& detid, const short& apvNb) const{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return ((p->BadApvs>>apvNb)&0x1);
  return false;
}

bool SiStripQuality::IsStripBad(const uint32_t& detid, const short& strip) const{
  bool result=false;
  SiStripBadStrip::Range range=getRange(detid);
  SiStripBadStrip::data fs;
  for(SiStripBadStrip::ContainerIterator it=range.first;it!=range.second;++it){
    fs=decode(*it);
    if ( fs.firstStrip<=strip && strip<=fs.firstStrip+fs.range ){
      result=true;
      break;
    }      
  }
  return result;
}

short SiStripQuality::getBadApvs(const uint32_t& detid) const{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return p->BadApvs;
  return 0;
}

short SiStripQuality::getBadFibers(const uint32_t& detid) const{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return p->BadFibers;
  return 0;
} 

EVENTSETUP_DATA_REG(SiStripQuality);
