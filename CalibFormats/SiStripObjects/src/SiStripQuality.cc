//
// Author:      Domenico Giordano
// Created:     Wed Sep 26 17:42:12 CEST 2007
// $Id: SiStripQuality.cc,v 1.16 2009/07/25 11:34:40 demattia Exp $
//
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

// Uncomment this to activate all the debug code
// #define DEBUG_CODE

// Needed only for debug output
#ifdef DEBUG_CODE
#include "DataFormats/SiStripDetId/interface/TIDDetId.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/DetId/interface/DetId.h"
#endif

#include <boost/bind.hpp>
#include <boost/function.hpp>

SiStripQuality::SiStripQuality():
  toCleanUp(false),
  FileInPath_("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"),
  SiStripDetCabling_(NULL)
{
  reader=new SiStripDetInfoFileReader(FileInPath_.fullPath());
}

SiStripQuality::SiStripQuality(edm::FileInPath& file):toCleanUp(false),FileInPath_(file),SiStripDetCabling_(NULL)
{
  reader=new SiStripDetInfoFileReader(FileInPath_.fullPath());
}

SiStripQuality::SiStripQuality(const SiStripQuality& other)
{
  FileInPath_=other.FileInPath_;
  reader=new SiStripDetInfoFileReader(*(other.reader));
  toCleanUp=other.toCleanUp;
  indexes=other.indexes;
  v_badstrips=other.v_badstrips;
  BadComponentVect=other.BadComponentVect;
  SiStripDetCabling_=other.SiStripDetCabling_;
}


SiStripQuality& SiStripQuality::operator +=(const SiStripQuality& other)
{
  this->add(&other); 
  this->cleanUp(); 
  this->fillBadComponents(); 
  return *this; 
}

SiStripQuality& SiStripQuality::operator -=(const SiStripQuality& other)
{    
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

const SiStripQuality SiStripQuality::operator -(const SiStripQuality& other) const
{
  return SiStripQuality(*this) -= other; 
}

bool SiStripQuality::operator ==(const SiStripQuality& other) const
{
  SiStripQuality a = (*this) - other ;
  return a.getRegistryVectorBegin()==a.getRegistryVectorEnd();
}

bool SiStripQuality::operator !=(const SiStripQuality& other) const { return !(*this == other) ; }

void SiStripQuality::add(const SiStripDetVOff *Voff)
{
  std::vector<unsigned int> vect;
  short firstStrip=0;
  short range=0;

  //Get vector of Voff dets
  std::vector<uint32_t> vdets;
  Voff->getDetIds(vdets);
  std::vector<uint32_t>::const_iterator iter=vdets.begin();
  std::vector<uint32_t>::const_iterator iterEnd=vdets.end();

  for(;iter!=iterEnd;++iter){
    vect.clear();
    range = (short) (reader->getNumberOfApvsAndStripLength(*iter).second*128.);
    LogTrace("SiStripQuality") << "[add Voff] add detid " << *iter << std::endl;
    vect.push_back(encode(firstStrip,range));
    SiStripBadStrip::Range Range(vect.begin(),vect.end());
    add(*iter,Range);
  }
}

void SiStripQuality::add(const RunInfo *runInfo)
{
//   FEDNumbering numbering;
//   int siStripFedIdMin = numbering.getSiStripFEDIds().first;
//   int siStripFedIdMax = numbering.getSiStripFEDIds().second;
//   int siPixelFedIdMin  = FEDNumbering::MINSiPixelFEDID;
//   int siPixelFedIdMax  = FEDNumbering::MAXSiPixelFEDID;
  int siStripFedIdMin  = FEDNumbering::MINSiStripFEDID;
  int siStripFedIdMax  = FEDNumbering::MAXSiStripFEDID;
//   int cscFedIdMin      = FEDNumbering::MINCSCFEDID;
//   int cscFedIdMax      = FEDNumbering::MAXCSCFEDID;
//   int rpcFedIdMin      = 790;
//   int rpcFedIdMax      = 792;
//   int dtFedItMin       = 770;
//   int dtFedItMax       = 774;
//   int hcalFedIdMin     = FEDNumbering::MINHCALFEDID;
//   int hcalFedIdMax     = FEDNumbering::MAXHCALFEDID;
//   int l1TrigFedIdMin   = FEDNumbering::MINTriggerGTPFEDID;
//   int l1TrigFedIdMax   = FEDNumbering::MAXTriggerGTPFEDID;
//   int ecalFedIdMin     = 610;    
//   int ecalFedIdMax     = 645;
//   int ecalLowFedIdMin  = 601;
//   int ecalLowFedIdMax  = 609;
//   int ecalHighFedIdMin = 646;
//   int ecalHighFedIdMax = 654;

//   std::cout << "Adding RunInfo, with number of fed in = " << runInfo->m_fed_in.size() << std::endl;
//   std::cout << "m_run = " << runInfo->m_run << ", "
//             << "m_start_time_ll = " << runInfo->m_start_time_ll << ", "
//             << "m_start_time_str = " << runInfo->m_start_time_str << ", "
//             << "m_stop_time_ll = " << runInfo->m_stop_time_ll << ", "
//             << "m_stop_time_str = " << runInfo->m_stop_time_str << ", "
//             << "m_fed_in.size() = " << runInfo->m_fed_in.size() << ", "
//             << "m_start_current = " << runInfo->m_start_current << ", "
//             << "m_stop_current = " << runInfo->m_stop_current << ", "
//             << "m_avg_current = " << runInfo->m_avg_current << ", "
//             << "m_max_current = " << runInfo->m_max_current << ", "
//             << "m_min_current = " << runInfo->m_min_current << ", "
//             << "m_run_interval_micros = " << runInfo->m_run_intervall_micros << ", "
//             << "m_current.size() = " << runInfo->m_current.size() << ", "
//             << "m_times_of_currents.size() = " << runInfo->m_times_of_currents.size() << std::endl;
//   std::vector<int>::const_iterator it = runInfo->m_fed_in.begin();
//   for( ; it != runInfo->m_fed_in.end(); ++it ) {
    // If it is a SiStrip FED

//     if( *it >= siStripFedIdMin && *it <= siStripFedIdMax ) std::cout << "SiStripFed["<<*it<<"]" << std::endl;
//     if( *it >= siPixelFedIdMin && *it <= siPixelFedIdMax ) std::cout << "SiPixelFed["<<*it<<"]" << std::endl;
//     if( *it >= cscFedIdMin && *it <= cscFedIdMax ) std::cout << "cscFed["<<*it<<"]" << std::endl;
//     if( *it >= rpcFedIdMin && *it <= rpcFedIdMax ) std::cout << "rpcFed["<<*it<<"]" << std::endl;
//     if( *it >= dtFedItMin && *it <= dtFedItMax ) std::cout << "dtFed["<<*it<<"]" << std::endl;
//     if( *it >= hcalFedIdMin && *it <= hcalFedIdMax ) std::cout << "hcalFed["<<*it<<"]" << std::endl;
//     if( *it >= l1TrigFedIdMin && *it <= l1TrigFedIdMax ) std::cout << "l1TrigFed["<<*it<<"]" << std::endl;
//     if( *it >= ecalFedIdMin && *it <= ecalFedIdMax ) std::cout << "ecalFed["<<*it<<"]" << std::endl;
//     if( *it >= ecalLowFedIdMin && *it <= ecalLowFedIdMax ) std::cout << "ecalLowFed["<<*it<<"]" << std::endl;
//     if( *it >= ecalHighFedIdMin && *it <= ecalHighFedIdMax ) std::cout << "ecalHighFed["<<*it<<"]" << std::endl;
  //     if( *it >= siStripFedIdMin && *it <= siStripFedIdMax ) {
  //       std::cout << "it = " << *it << std::endl;
  //     }
//  }
  // const SiStripFedCabling * fedCabling = SiStripDetCabling_->fedCabling();



  // Take the list of active feds from fedCabling
  std::vector<uint16_t> activeFedsFromCabling = SiStripDetCabling_->fedCabling()->feds();

  // DEBUG code
  #ifdef DEBUG_CODE
  std::sort(activeFedsFromCabling.begin(), activeFedsFromCabling.end());
  std::cout << "activeFedsFromCabling:" << std::endl;
  std::copy(activeFedsFromCabling.begin(), activeFedsFromCabling.end(), std::ostream_iterator<uint16_t>(std::cout, " "));
  std::cout << std::endl;
  #endif
  // ----------

  // Take the list of active feds from RunInfo
  std::vector<int> activeFedsFromRunInfo;
  // Take only Tracker feds (remove all non Tracker)
  std::remove_copy_if( runInfo->m_fed_in.begin(),
                       runInfo->m_fed_in.end(),
                       std::back_inserter(activeFedsFromRunInfo),
                       !boost::bind(std::logical_and<bool>(),
                                    boost::bind(std::greater_equal<int>(), _1, siStripFedIdMin),
                                    boost::bind(std::less_equal<int>(), _1, siStripFedIdMax)) );

  // DEBUG code
  #ifdef DEBUG_CODE
  std::sort(activeFedsFromRunInfo.begin(), activeFedsFromRunInfo.end());
  std::cout << "activeFedsFromRunInfo:" << std::endl;
  std::copy(activeFedsFromRunInfo.begin(), activeFedsFromRunInfo.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  #endif
  // ----------

  // Compare the two. If a fedId from RunInfo is not present in the fedCabling we need to
  // get all the corresponding fedChannels and then the single apv pairs and use them to
  // turn off the corresponding strips (apvNumber*256).
  // set_difference returns the set of elements that are in the first and not in the second
  std::sort(activeFedsFromCabling.begin(), activeFedsFromCabling.end());
  std::sort(activeFedsFromRunInfo.begin(), activeFedsFromRunInfo.end());
  std::vector<int> differentFeds;
  // Take the feds active for cabling but not for runInfo
  std::set_difference(activeFedsFromCabling.begin(), activeFedsFromCabling.end(),
                      activeFedsFromRunInfo.begin(), activeFedsFromRunInfo.end(),
                      std::back_inserter(differentFeds));

  // DEBUG code
  #ifdef DEBUG_CODE
  std::cout << "differentFeds : ";
  std::copy(differentFeds.begin(), differentFeds.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "associated to detIds : " << std::endl;
  #endif
  // ----------

  // Feds in the differentFeds vector are now to be turned off as they are off according to runInfo
  // but where not in cabling and thus are still active for the SiStripQuality
  // Loop on all the fedIds, take the fedChannels and then the detId
  std::vector<int>::const_iterator fedIdIt = differentFeds.begin();
  for( ; fedIdIt != differentFeds.end(); ++fedIdIt ) {
    std::vector<FedChannelConnection>::const_iterator fedChIt = SiStripDetCabling_->fedCabling()->connections( *fedIdIt ).begin();
    for( ; fedChIt != SiStripDetCabling_->fedCabling()->connections( *fedIdIt ).end(); ++fedChIt ) {
      uint32_t detId = fedChIt->detId();
      if (detId == 0 || detId == 0xFFFFFFFF) continue;
      uint16_t apvPairNumber = fedChIt->apvPairNumber();

      // DEBUG code
      #ifdef DEBUG_CODE
      int layer = 0;
      int stereo = 0;
      std::string subDetName;
      std::cout << detId << " and apv = " << apvPairNumber << ", ";
      DetId detid(detId);
      switch (detid.subdetId()) {
      case StripSubdetector::TIB:
        {
          TIBDetId theTIBDetId(detid.rawId());
          layer = theTIBDetId.layer();
          stereo = theTIBDetId.stereo();
          subDetName = "TIB";
          break;
        }
      case StripSubdetector::TOB:
        {
          TOBDetId theTOBDetId(detid.rawId());
          layer = theTOBDetId.layer();
          stereo = theTOBDetId.stereo();
          subDetName = "TOB";
          break;
        }
      case StripSubdetector::TEC:
        {
          TECDetId theTECDetId(detid.rawId());
          // is this module in TEC+ or TEC-?
          layer = theTECDetId.wheel();
          stereo = theTECDetId.stereo();
          subDetName = "TEC";
          break;
        }
      case StripSubdetector::TID:
        {
          TECDetId theTIDDetId(detid.rawId());
          // is this module in TID+ or TID-?
          layer = theTIDDetId.wheel();
          stereo = theTIDDetId.stereo();
          subDetName = "TID";
          break;
        }
      }
      std::cout << "of subDet = " << subDetName << ", layer = " << layer << " stereo = " << stereo << std::endl;
      #endif
      // ----------


      // apvPairNumber == i it means that the i*256 strips are to be set off
      std::vector<unsigned int> vect;
      vect.push_back(encode(apvPairNumber*256,256));
      SiStripBadStrip::Range Range(vect.begin(), vect.end());
      add(detId,Range);
      LogTrace("SiStripQuality") << "[addOffForRunInfo] adding apvPairNumber "<<apvPairNumber<<" for detId "<<detId<<" off according to RunInfo" << std::endl;
    }
  }

  // DEBUG code
  #ifdef DEBUG_CODE
  std::cout << std::endl;
  #endif
  // ----------

  // Consistency check
  // -----------------
  std::vector<int> check;
  std::set_difference(activeFedsFromRunInfo.begin(), activeFedsFromRunInfo.end(),
                      activeFedsFromCabling.begin(), activeFedsFromCabling.end(),
                      std::back_inserter(check));
  // This must not happen
  if( !check.empty() ) {
    throw cms::Exception("LogicError")
      << "The cabling should always include the active feds in runInfo and possibly have some more"
      << "there are instead " << check.size() << " feds only active in runInfo";
  }
}

void SiStripQuality::add(const SiStripDetCabling *cab)
{
  SiStripDetCabling_=cab;
  addInvalidConnectionFromCabling();
  addNotConnectedConnectionFromCabling();
}

void SiStripQuality::addNotConnectedConnectionFromCabling()
{  
  std::map<uint32_t, SiStripDetInfoFileReader::DetInfo > allData = reader->getAllData();
  std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator iter=allData.begin();
  std::map<uint32_t, SiStripDetInfoFileReader::DetInfo >::const_iterator iterEnd=allData.end();
  std::vector<unsigned int> vect;
  short firstStrip=0;
  short range=0;
  for(;iter!=iterEnd;++iter)
    if (!SiStripDetCabling_->IsConnected(iter->first)){
      vect.clear();
      range=iter->second.nApvs*128;
      LogTrace("SiStripQuality") << "[addNotConnectedConnectionFromCabling] add detid " << iter->first << std::endl;
      vect.push_back(encode(firstStrip,range));
      SiStripBadStrip::Range Range(vect.begin(),vect.end());
      add(iter->first,Range);
    }
}

void SiStripQuality::addInvalidConnectionFromCabling()
{
  std::vector<uint32_t> connected_detids;
  SiStripDetCabling_->addActiveDetectorsRawIds(connected_detids);
  std::vector<uint32_t>::const_iterator itdet = connected_detids.begin();
  std::vector<uint32_t>::const_iterator itdetEnd = connected_detids.end();
  for(;itdet!=itdetEnd;++itdet){
    //LogTrace("SiStripQuality") << "[addInvalidConnectionFromCabling] looking at detid " <<*itdet << std::endl;
    const std::vector<FedChannelConnection>& fedconns=SiStripDetCabling_->getConnections(*itdet);
    std::vector<FedChannelConnection>::const_iterator itconns=fedconns.begin();
    std::vector<FedChannelConnection>::const_iterator itconnsEnd=fedconns.end();
    
    unsigned short nApvPairs=SiStripDetCabling_->nApvPairs(*itdet);
    short ngoodConn=0, goodConn=0;
    for(;itconns!=itconnsEnd;++itconns){
      //LogTrace("SiStripQuality") << "[addInvalidConnectionFromCabling] apvpair " << itconns->apvPairNumber() << " napvpair " << itconns->nApvPairs()<< " detid " << itconns->detId() << std::endl;
      if(itconns->nApvPairs()==sistrip::invalid_)
	continue;
      ngoodConn++;
      goodConn = goodConn | ( 0x1 << itconns->apvPairNumber() );
    }

    if (ngoodConn!=nApvPairs){
      std::vector<unsigned int> vect;
      for (size_t idx=0;idx<nApvPairs;++idx){
	if( !(goodConn & ( 0x1 << idx)) ) {
	  short firstStrip=idx*256;
	  short range=256;
	  LogTrace("SiStripQuality") << "[addInvalidConnectionFromCabling] add detid " <<*itdet << "firstStrip " << firstStrip<< std::endl;
	  vect.push_back(encode(firstStrip,range));
	}
      }
      if(!vect.empty()){
	SiStripBadStrip::Range Range(vect.begin(),vect.end());
	add(*itdet,Range);
      }
    }
  }
}

void SiStripQuality::add(const SiStripBadStrip* base)
{
  SiStripBadStrip::RegistryIterator basebegin = base->getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator baseend   = base->getRegistryVectorEnd();
  
  //the Registry already contains data
  //Loop on detids
  for (SiStripBadStrip::RegistryIterator basep=basebegin; basep != baseend; ++basep) {
    uint32_t detid=basep->detid;
    LogTrace("SiStripQuality") << "add detid " <<detid << std::endl;
    
    SiStripBadStrip::Range baserange = SiStripBadStrip::Range( base->getDataVectorBegin()+basep->ibegin , base->getDataVectorBegin()+basep->iend );
    
    add(detid,baserange);
  }
}

void SiStripQuality::add(const uint32_t& detid,const SiStripBadStrip::Range& baserange)
{
 std::vector<unsigned int> vect, tmp;

 unsigned short Nstrips=reader->getNumberOfApvsAndStripLength(detid).first*128;
    
 //Is this detid already in the collections owned by this class?
 SiStripBadStrip::Range range = getRange(detid);
 
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
     return;
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

void SiStripQuality::compact(unsigned int& detid, std::vector<unsigned int>& vect)
{
  std::vector<unsigned int> tmp=vect;
  vect.clear();
  std::stable_sort(tmp.begin(),tmp.end());
  unsigned short Nstrips=reader->getNumberOfApvsAndStripLength(detid).first*128;
  compact(tmp,vect,Nstrips);
}

bool SiStripQuality::put_replace(const uint32_t& DetId, Range input)
{
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

/*
Method to reduce the granularity of badcomponents:
if in an apv there are more than ratio*128 bad strips,
the full apv is declared as bad.
Method needed to help the 
 */
void SiStripQuality::ReduceGranularity(double threshold)
{
  SiStripBadStrip::RegistryIterator rp = getRegistryVectorBegin();
  SiStripBadStrip::RegistryIterator rend   = getRegistryVectorEnd();
  SiStripBadStrip::data data_;
  uint16_t BadStripPerApv[6], ipos;
  std::vector<unsigned int> vect;

  for (; rp != rend; ++rp) {
    uint32_t detid=rp->detid;

    BadStripPerApv[0]=0;    BadStripPerApv[1]=0;    BadStripPerApv[2]=0;    BadStripPerApv[3]=0;    BadStripPerApv[4]=0;    BadStripPerApv[5]=0;
    ipos=0;

    SiStripBadStrip::Range sqrange = SiStripBadStrip::Range( getDataVectorBegin()+rp->ibegin , getDataVectorBegin()+rp->iend );
    
    for(int it=0;it<sqrange.second-sqrange.first;it++){

      data_=decode( *(sqrange.first+it) );
      LogTrace("SiStripQuality") << "[SiStripQuality::ReduceGranularity] detid " << detid << " first strip " << data_.firstStrip << " lastStrip " << data_.firstStrip+data_.range-1  << " range " << data_.range;
      ipos= data_.firstStrip/128;
      while (ipos<=(data_.firstStrip+data_.range-1)/128){
	BadStripPerApv[ipos]+=std::min(data_.firstStrip+data_.range,(ipos+1)*128)-std::max(data_.firstStrip*1,ipos*128);
	LogTrace("SiStripQuality") << "[SiStripQuality::ReduceGranularity] ipos " << ipos << " Counter " << BadStripPerApv[ipos] << " min " << std::min(data_.firstStrip+data_.range,(ipos+1)*128) << " max " << std::max(data_.firstStrip*1,ipos*128) << " added " << std::min(data_.firstStrip+data_.range,(ipos+1)*128)-std::max(data_.firstStrip*1,ipos*128);
	ipos++;
      }
    }

    LogTrace("SiStripQuality") << "[SiStripQuality::ReduceGranularity] Total for detid " << detid << " values " << BadStripPerApv[0] << " " << BadStripPerApv[1] << " " << BadStripPerApv[2] << " " <<BadStripPerApv[3] << " " <<BadStripPerApv[4] << " " << BadStripPerApv[5];
    
    
    vect.clear();
    for(size_t i=0;i<6;++i){
      if (BadStripPerApv[i]>=threshold*128){
	vect.push_back(encode(i*128,128));
      }
    }
    if(!vect.empty()){
      SiStripBadStrip::Range Range(vect.begin(),vect.end());
      add(detid,Range);
    }
  }
}


void SiStripQuality::compact(std::vector<unsigned int>& tmp,std::vector<unsigned int>& vect,unsigned short& Nstrips)
{
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

void SiStripQuality::subtract(std::vector<unsigned int>& A,const std::vector<unsigned int>& B)
{
  ContainerIterator it=B.begin();
  ContainerIterator itend=B.end();
  for(;it!=itend;++it){    
    subtraction(A,*it);
  }
}

void SiStripQuality::subtraction(std::vector<unsigned int>& A,const unsigned int& B)
{
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

bool SiStripQuality::cleanUp(bool force)
{
  if (!toCleanUp && !force)
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

void SiStripQuality::fillBadComponents()
{
  BadComponentVect.clear();
  
  for (SiStripBadStrip::RegistryIterator basep=indexes.begin(); basep != indexes.end(); ++basep) {
    
    SiStripBadStrip::Range range( v_badstrips.begin()+basep->ibegin, v_badstrips.begin()+basep->iend );
    
    //Fill BadModules, BadFibers, BadApvs vectors
    unsigned short resultA=0, resultF=0;
    BadComponent result;
    
    SiStripBadStrip::data fs;
    unsigned short Nstrips=reader->getNumberOfApvsAndStripLength(basep->detid).first*128;
    
    //BadModules
    fs=decode(*(range.first));
    if (basep->iend - basep->ibegin == 1 &&
	fs.firstStrip==0 && 
	fs.range==Nstrips ){
      result.detid=basep->detid;
      result.BadModule=true;
      result.BadFibers=(1<< (Nstrips/256))-1; 
      result.BadApvs=(1<< (Nstrips/128))-1; 
	
      BadComponentVect.push_back(result);
      
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

bool SiStripQuality::IsModuleUsable(const uint32_t& detid) const
{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    if(p->BadModule)
      return false;

  if (SiStripDetCabling_!=NULL)
    if(!SiStripDetCabling_->IsConnected(detid))
      return false;

  return true;
}

bool SiStripQuality::IsModuleBad(const uint32_t& detid) const
{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return p->BadModule;
  return false;
}

bool SiStripQuality::IsFiberBad(const uint32_t& detid, const short& fiberNb) const
{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return ((p->BadFibers>>fiberNb)&0x1);
  return false;
}

bool SiStripQuality::IsApvBad(const uint32_t& detid, const short& apvNb) const
{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return ((p->BadApvs>>apvNb)&0x1);
  return false;
}

bool SiStripQuality::IsStripBad(const uint32_t& detid, const short& strip) const
{
  SiStripBadStrip::Range range=getRange(detid);
  return IsStripBad(range,strip);
}

bool SiStripQuality::IsStripBad(const Range& range, const short& strip) const
{
  bool result=false;
  SiStripBadStrip::data fs;
  for(SiStripBadStrip::ContainerIterator it=range.first;it!=range.second;++it){
    fs=decode(*it);
    if ( fs.firstStrip<=strip && strip<fs.firstStrip+fs.range ){
      result=true;
      break;
    }      
  }
  return result;
}

int SiStripQuality::nBadStripsOnTheLeft(const Range& range, const short& strip) const
{
  int result=0;
  SiStripBadStrip::data fs;
  for(SiStripBadStrip::ContainerIterator it=range.first;it!=range.second;++it){
    fs=decode(*it);
    if ( fs.firstStrip<=strip && strip<fs.firstStrip+fs.range ){
      result=strip-fs.firstStrip+1;
      break;
    }      
  }
  return result;
}

int SiStripQuality::nBadStripsOnTheRight(const Range& range, const short& strip) const
{
  int result=0;
  SiStripBadStrip::data fs;
  for(SiStripBadStrip::ContainerIterator it=range.first;it!=range.second;++it){
    fs=decode(*it);
    if ( fs.firstStrip<=strip && strip<fs.firstStrip+fs.range ){
      result=fs.firstStrip+fs.range-strip;
      break;
    }      
  }
  return result;
}

short SiStripQuality::getBadApvs(const uint32_t& detid) const
{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return p->BadApvs;
  return 0;
}

short SiStripQuality::getBadFibers(const uint32_t& detid) const
{
  std::vector<BadComponent>::const_iterator p = std::lower_bound(BadComponentVect.begin(),BadComponentVect.end(),detid,SiStripQuality::BadComponentStrictWeakOrdering());
  if (p!=BadComponentVect.end() && p->detid==detid)
    return p->BadFibers;
  return 0;
} 
