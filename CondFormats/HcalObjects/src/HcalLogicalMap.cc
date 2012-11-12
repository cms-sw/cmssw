#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <time.h>

using namespace std;

HcalLogicalMap::HcalLogicalMap(const HcalTopology* topo,std::vector<HBHEHFLogicalMapEntry>& HBHEHFEntries,
			       std::vector<HOHXLogicalMapEntry>& HOHXEntries,
			       std::vector<CALIBLogicalMapEntry>& CALIBEntries,
			       std::vector<ZDCLogicalMapEntry>& ZDCEntries,
			       std::vector<HTLogicalMapEntry>& HTEntries,
			       std::vector<uint32_t>& LinearIndex2Entry,
			       std::vector<uint32_t>& HbHash2Entry,
			       std::vector<uint32_t>& HeHash2Entry,
			       std::vector<uint32_t>& HfHash2Entry,
			       std::vector<uint32_t>& HtHash2Entry,
			       std::vector<uint32_t>& HoHash2Entry,
			       std::vector<uint32_t>& HxCalibHash2Entry,
			       std::vector<uint32_t>& ZdcHash2Entry) : topo_(topo)

{
  HBHEHFEntries_.resize(HBHEHFEntries.size());
  HOHXEntries_.resize(HOHXEntries.size());
  CALIBEntries_.resize(CALIBEntries.size());
  ZDCEntries_.resize(ZDCEntries.size());
  HTEntries_.resize(HTEntries.size());

  LinearIndex2Entry_.resize(LinearIndex2Entry.size());
  HbHash2Entry_.resize(HbHash2Entry.size());
  HeHash2Entry_.resize(HeHash2Entry.size());
  HfHash2Entry_.resize(HfHash2Entry.size());
  HtHash2Entry_.resize(HtHash2Entry.size());
  HoHash2Entry_.resize(HoHash2Entry.size());
  HxCalibHash2Entry_.resize(HxCalibHash2Entry.size());
  ZdcHash2Entry_.resize(ZdcHash2Entry.size());
  
  copy(HBHEHFEntries.begin(),HBHEHFEntries.end(),HBHEHFEntries_.begin());
  copy(HOHXEntries.begin(),HOHXEntries.end(),HOHXEntries_.begin());
  copy(CALIBEntries.begin(),CALIBEntries.end(),CALIBEntries_.begin());
  copy(ZDCEntries.begin(),ZDCEntries.end(),ZDCEntries_.begin());
  copy(HTEntries.begin(),HTEntries.end(),HTEntries_.begin());
  copy(LinearIndex2Entry.begin(),LinearIndex2Entry.end(),LinearIndex2Entry_.begin());
  copy(HbHash2Entry.begin(),HbHash2Entry.end(),HbHash2Entry_.begin());
  copy(HeHash2Entry.begin(),HeHash2Entry.end(),HeHash2Entry_.begin());
  copy(HfHash2Entry.begin(),HfHash2Entry.end(),HfHash2Entry_.begin());
  copy(HtHash2Entry.begin(),HtHash2Entry.end(),HtHash2Entry_.begin());
  copy(HoHash2Entry.begin(),HoHash2Entry.end(),HoHash2Entry_.begin());
  copy(HxCalibHash2Entry.begin(),HxCalibHash2Entry.end(),HxCalibHash2Entry_.begin());
  copy(ZdcHash2Entry.begin(),ZdcHash2Entry.end(),ZdcHash2Entry_.begin());
}

HcalLogicalMap::~HcalLogicalMap()
{
}

uint32_t HcalLogicalMap::makeEntryNumber(bool isvalid, int vectorid, int entry)
{
  uint32_t answer=0;
  answer|=isvalid;
  answer|=(vectorid&0x3)<<1;
  answer|=entry<<3;
  return answer;
}

void HcalLogicalMap::printMap( unsigned int mapIOV ){  
  using namespace std;

  static FILE* HBEFmap; 
  static FILE* HOXmap;
  static FILE* CALIBmap; 
  static FILE* ZDCmap; 
  static FILE* HTmap;  
  
  char tempbuff[30];

  stringstream mystream;
  string HBEFmapstr, HOXmapstr, CALIBmapstr, ZDCmapstr, HTmapstr;
  string date;
  string IOVlabel;
  
  time_t myTime;
  time(&myTime);
  
  strftime(tempbuff,128,"%d.%b.%Y",localtime(&myTime) );
  mystream<<tempbuff;
  date= mystream.str();

  mystream.str("");
  if      (mapIOV==1) IOVlabel = "A";
  else if (mapIOV==2) IOVlabel = "B";
  else if (mapIOV==3) IOVlabel = "C";
  else                IOVlabel = "D";

  HBEFmapstr  = "./HCALmapHBEF_"+IOVlabel+".txt";
  HOXmapstr   = "./HCALmapHO_"+IOVlabel+".txt";
  CALIBmapstr = "./HCALmapCALIB_"+IOVlabel+".txt";
  ZDCmapstr   = "./ZDCmap_"+IOVlabel+".txt";
  HTmapstr    = "./HCALmapHT_"+IOVlabel+".txt";

//  HBEFmapstr  = "./HCALmapHBEF_"+date+".txt";
//  HOXmapstr   = "./HCALmapHO_"+date+"_"+IOVlabel+".txt";
//  CALIBmapstr = "./HCALmapCALIB_"+date+".txt";
//  ZDCmapstr   = "./ZDCmap_"+date+".txt";
//  HTmapstr    = "./HCALmapHT_"+date+".txt";
  
  HBEFmap     = fopen(HBEFmapstr.c_str(),"w");
  HOXmap      = fopen(HOXmapstr.c_str(),"w");
  CALIBmap    = fopen(CALIBmapstr.c_str(),"w");
  ZDCmap      = fopen(ZDCmapstr.c_str(),"w");
  HTmap       = fopen(HTmapstr.c_str(),"w");
  /**********************/

  if(HBEFmap) {
    fprintf(HBEFmap,"## file created %s ##\n",date.c_str());
    printHBEFMap(HBEFmap);
  }
  else cout <<HBEFmapstr<<" not found!"<<endl;

  if(HOXmap) {
    fprintf(HOXmap,"## file created %s ##\n",date.c_str());
    printHOXMap(HOXmap);
  }
  else cout <<HOXmapstr<<" not found!"<<endl;

  if(CALIBmap) {
    fprintf(CALIBmap,"## file created %s ##\n",date.c_str());
    printCalibMap(CALIBmap);
  }
  else cout <<CALIBmapstr<<" not found!"<<endl;

  if(ZDCmap) {
    fprintf(ZDCmap,"## file created %s ##\n",date.c_str());
    printZDCMap(ZDCmap);
  }
  else cout <<ZDCmapstr<<" not found!"<<endl;

  if(HTmap) {
    fprintf(HTmap,"## file created %s ##\n",date.c_str());
    printHTMap(HTmap);
  }
  else cout <<HTmapstr<<" not found!"<<endl;

}

/**************/
HcalElectronicsMap HcalLogicalMap::generateHcalElectronicsMap()
{
  HcalElectronicsMap* theemap = new HcalElectronicsMap();
  
  for (std::vector<HBHEHFLogicalMapEntry>::iterator it = HBHEHFEntries_.begin(); it!=HBHEHFEntries_.end(); ++it) {
    theemap->mapEId2chId( it->getHcalElectronicsId(), it->getDetId() );}
  for (std::vector<HOHXLogicalMapEntry>::iterator it = HOHXEntries_.begin(); it!=HOHXEntries_.end(); ++it) {
    theemap->mapEId2chId( it->getHcalElectronicsId(), it->getDetId() );}
  for (std::vector<CALIBLogicalMapEntry>::iterator it = CALIBEntries_.begin(); it!=CALIBEntries_.end(); ++it) {
    theemap->mapEId2chId( it->getHcalElectronicsId(), it->getDetId() );}
  for (std::vector<ZDCLogicalMapEntry>::iterator it = ZDCEntries_.begin(); it!=ZDCEntries_.end(); ++it) {
    theemap->mapEId2chId( it->getHcalElectronicsId(), it->getDetId() );}
  for (std::vector<HTLogicalMapEntry>::iterator it = HTEntries_.begin(); it!=HTEntries_.end(); ++it) {
    theemap->mapEId2tId( it->getHcalTrigElectronicsId(), it->getDetId() );}
  theemap->sort();
  return *theemap;
}
/**************/

void HcalLogicalMap::printHBEFMap(FILE* hbefmapfile){
  int titlecounter = 0;

  for (std::vector<HBHEHFLogicalMapEntry>::iterator it = HBHEHFEntries_.begin(); it!=HBHEHFEntries_.end(); ++it) {
    titlecounter = titlecounter % 21;
    if (titlecounter == 0){
      fprintf(hbefmapfile,"#   side    eta    phi   dphi  depth    det     rbx  wedge     rm  pixel   qie    adc");
      fprintf(hbefmapfile,"  rm_fi  fi_ch  crate    htr   fpga  htr_fi  dcc_sl  spigo    dcc    slb  slbin  slbin2");
      fprintf(hbefmapfile,"           slnam    rctcra rctcar rctcon               rctnam     fedid\n");
    }
    titlecounter++;
    fprintf(hbefmapfile,"%s",it->printLMapLine());
  }
}

void HcalLogicalMap::printHOXMap(FILE* hoxmapfile){

  int titlecounter = 0;

  for (std::vector<HOHXLogicalMapEntry>::iterator it = HOHXEntries_.begin(); it!=HOHXEntries_.end(); ++it) {
    titlecounter = titlecounter % 21;
    if (titlecounter == 0){
      fprintf(hoxmapfile,"#   side    eta    phi   dphi  depth    det     rbx  sector    rm  pixel   qie    adc");
      fprintf(hoxmapfile,"  rm_fi  fi_ch let_code  crate    htr   fpga  htr_fi  dcc_sl  spigo    dcc  fedid\n");
    }
    titlecounter++;
    fprintf(hoxmapfile,"%s",it->printLMapLine());
  }    
}

void HcalLogicalMap::printCalibMap(FILE* calibmapfile){

  int titlecounter = 0;

  for (std::vector<CALIBLogicalMapEntry>::iterator it = CALIBEntries_.begin(); it!=CALIBEntries_.end(); ++it) {
    titlecounter = titlecounter % 21;
    if (titlecounter == 0){	  
      fprintf(calibmapfile,"#   side    eta    phi   dphi    det     rbx  sector  rm  rm_fi ");
      fprintf(calibmapfile," fi_ch  crate  htr  fpga  htr_fi  dcc_sl  spigo  dcc  fedid  ch_type      name\n");
    }
    titlecounter++;
    fprintf(calibmapfile,"%s",it->printLMapLine());
  }
}

void HcalLogicalMap::printZDCMap(FILE* zdcmapfile){

  int titlecounter = 0;

  for (std::vector<ZDCLogicalMapEntry>::iterator it = ZDCEntries_.begin(); it!=ZDCEntries_.end(); ++it) {
    titlecounter = titlecounter % 21;
    if (titlecounter == 0){
      fprintf(zdcmapfile,"#  side  x  y  dx  depth     det  det_ch  cable  rm  qie ");
      fprintf(zdcmapfile," adc  rm_fi  fi_ch  crate  htr  fpga  htr_fi  dcc_sl  spigo  dcc  fedid\n");
    }
    titlecounter++;
    fprintf(zdcmapfile,"%s",it->printLMapLine());
  }
}

void HcalLogicalMap::printHTMap(FILE* htmapfile){

  int titlecounter = 0;

  for (std::vector<HTLogicalMapEntry>::iterator it = HTEntries_.begin(); it!=HTEntries_.end(); ++it) {
    titlecounter = titlecounter % 21;
      if (titlecounter == 0){
	fprintf(htmapfile,"#  side  eta  phi  dphi  depth  det   wedge  crate");
        fprintf(htmapfile,"  htr  fpga  dcc_sl  spigo  dcc  slb  slbin  slbin2  nDat    ");
        fprintf(htmapfile,"     slnam  rctcra  rctcar  rctcon            rctnam  fedid\n");
      }
    titlecounter++;
    fprintf(htmapfile,"%s",it->printLMapLine());
  }
}


const DetId HcalLogicalMap::getDetId(const HcalElectronicsId& eid)
{
  const uint32_t entry=LinearIndex2Entry_.at(eid.linearIndex());
  if ( !(entry&0x1) ) return DetId(0);
  if ( ((entry>>1)&0x3)==0) return HBHEHFEntries_.at(entry>>3).getDetId();
  if ( ((entry>>1)&0x3)==1) return HOHXEntries_.at(entry>>3).getDetId();
  if ( ((entry>>1)&0x3)==2) return CALIBEntries_.at(entry>>3).getDetId();
  if ( ((entry>>1)&0x3)==3) return ZDCEntries_.at(entry>>3).getDetId();
  return DetId(0);
}

const HcalFrontEndId HcalLogicalMap::getHcalFrontEndId(const DetId& did)
{
  const HcalGenericDetId hgdi(did);
  
  const HcalGenericDetId::HcalGenericSubdetector hgsd=hgdi.genericSubdet();
  if (hgsd==HcalGenericDetId::HcalGenBarrel) {
    const int hashedId=topo_->detId2denseIdHB(did);
    const uint32_t entry=HbHash2Entry_.at(hashedId)-1;
    return HBHEHFEntries_.at(entry).getHcalFrontEndId();
  }
  else if (hgsd==HcalGenericDetId::HcalGenEndcap) {
    const int hashedId=topo_->detId2denseIdHE(did);
    const uint32_t entry=HeHash2Entry_.at(hashedId)-1;
    return HBHEHFEntries_.at(entry).getHcalFrontEndId();
  }
  else if (hgsd==HcalGenericDetId::HcalGenForward) {
    const int hashedId=topo_->detId2denseIdHF(did);
    const uint32_t entry=HfHash2Entry_.at(hashedId)-1;
    return HBHEHFEntries_.at(entry).getHcalFrontEndId();
  }
  else if (hgsd==HcalGenericDetId::HcalGenOuter) {
    const int hashedId=topo_->detId2denseIdHO(did);
    const uint32_t entry=HoHash2Entry_.at(hashedId)-1;
    return HOHXEntries_.at(entry).getHcalFrontEndId();
  }
  else if (hgsd==HcalGenericDetId::HcalGenCalibration) {
    HcalCalibDetId hcid(did);
    if (hcid.calibFlavor()==HcalCalibDetId::HOCrosstalk) {
      const int hashedId=topo_->detId2denseIdCALIB(did);
      const uint32_t entry=HxCalibHash2Entry_.at(hashedId)-1;
      return HOHXEntries_.at(entry).getHcalFrontEndId();
    }
    else if (hcid.calibFlavor()==HcalCalibDetId::CalibrationBox) {
      const int hashedId=topo_->detId2denseIdCALIB(did);
      const uint32_t entry=HxCalibHash2Entry_.at(hashedId)-1;
      return CALIBEntries_.at(entry).getHcalFrontEndId();
    }
  }
  return HcalFrontEndId(0);
}

void HcalLogicalMap::checkIdFunctions() {
  int HBHEHF_EID_pass=0;
  int HBHEHF_EID_fail=0;
  int HOHX_EID_pass=0;
  int HOHX_EID_fail=0;
  int CALIB_EID_pass=0;
  int CALIB_EID_fail=0;
  int ZDC_EID_pass=0;
  int ZDC_EID_fail=0;

  int HBHEHF_FEID_pass=0;
  int HBHEHF_FEID_fail=0;
  int HOHX_FEID_pass=0;
  int HOHX_FEID_fail=0;
  int CALIB_FEID_pass=0;
  int CALIB_FEID_fail=0;

  cout << "\nRunning the id function checker..." << endl;

  for (std::vector<HBHEHFLogicalMapEntry>::iterator it = HBHEHFEntries_.begin(); it!=HBHEHFEntries_.end(); ++it) {
    const HcalElectronicsId heid=it->getHcalElectronicsId();
    const DetId did0=it->getDetId();
    const DetId did1=getDetId(heid);
    if (did0==did1) HBHEHF_EID_pass++;
    else HBHEHF_EID_fail++;

    const HcalFrontEndId hfeid0=it->getHcalFrontEndId();
    const HcalFrontEndId hfeid1=getHcalFrontEndId(did0);
    if (hfeid0==hfeid1) HBHEHF_FEID_pass++;
    else HBHEHF_FEID_fail++;
  }
  for (std::vector<HOHXLogicalMapEntry>::iterator it = HOHXEntries_.begin(); it!=HOHXEntries_.end(); ++it) {
    const HcalElectronicsId heid=it->getHcalElectronicsId();
    const DetId did0=it->getDetId();
    const DetId did1=getDetId(heid);
    if (did0==did1) HOHX_EID_pass++;
    else HOHX_EID_fail++;

    const HcalFrontEndId hfeid0=it->getHcalFrontEndId();
    const HcalFrontEndId hfeid1=getHcalFrontEndId(did0);
    if (hfeid0==hfeid1) HOHX_FEID_pass++;
    else HOHX_FEID_fail++;
  }
  for (std::vector<CALIBLogicalMapEntry>::iterator it = CALIBEntries_.begin(); it!=CALIBEntries_.end(); ++it) {
    const HcalElectronicsId heid=it->getHcalElectronicsId();
    const DetId did0=it->getDetId();
    const DetId did1=getDetId(heid);
    if (did0==did1) CALIB_EID_pass++;
    else CALIB_EID_fail++;

    const HcalFrontEndId hfeid0=it->getHcalFrontEndId();
    const HcalFrontEndId hfeid1=getHcalFrontEndId(did0);
    if (hfeid0==hfeid1) CALIB_FEID_pass++;
    else CALIB_FEID_fail++;
  }
  for (std::vector<ZDCLogicalMapEntry>::iterator it = ZDCEntries_.begin(); it!=ZDCEntries_.end(); ++it) {
    const HcalElectronicsId heid=it->getHcalElectronicsId();
    const DetId did0=it->getDetId();
    const DetId did1=getDetId(heid);
    if (did0==did1) ZDC_EID_pass++;
    else ZDC_EID_fail++;
  }
  
  cout << "Checking detIds from electronics ids..." << endl;
  cout << "HBHEHF EID (pass,fail) = (" << HBHEHF_EID_pass << "," << HBHEHF_EID_fail << ")" << endl;
  cout << "HOHX EID (pass,fail) = (" << HOHX_EID_pass << "," << HOHX_EID_fail << ")" << endl;
  cout << "CALIB EID (pass,fail) = (" << CALIB_EID_pass << "," << CALIB_EID_fail << ")" << endl;
  cout << "ZDC EID (pass,fail) = (" << ZDC_EID_pass << "," << ZDC_EID_fail << ")" << endl;
  cout << endl;
  cout << "Checking frontEndIds from electronics ids..." << endl;
  cout << "HBHEHF FEID (pass,fail) = (" << HBHEHF_FEID_pass << "," << HBHEHF_FEID_fail << ")" << endl;
  cout << "HOHX FEID (pass,fail) = (" << HOHX_FEID_pass << "," << HOHX_FEID_fail << ")" << endl;
  cout << "CALIB FEID (pass,fail) = (" << CALIB_FEID_pass << "," << CALIB_FEID_fail << ")" << endl;
}


void HcalLogicalMap::checkHashIds() {
  std::vector<int> HB_Hashes_;     // index 0
  std::vector<int> HE_Hashes_;     // index 1
  std::vector<int> HF_Hashes_;     // index 2
  std::vector<int> HO_Hashes_;     // index 3
  std::vector<int> CALIBHX_Hashes_;// index 4
  std::vector<int> ZDC_Hashes_;    // index 5
  std::vector<int> HT_Hashes_;     // index 6

  int numfails[7]    = {0,0,0,0,0,0,0};
  int numpass[7]     = {0,0,0,0,0,0,0};
  int numnotdense[7] = {0,0,0,0,0,0,0};

  cout << "\nRunning the hash checker for detIds..." << endl;
  for (std::vector<HBHEHFLogicalMapEntry>::iterator it = HBHEHFEntries_.begin(); it!=HBHEHFEntries_.end(); ++it) {
    if (it->getDetId().subdetId()==HcalBarrel) {
      HB_Hashes_.push_back(topo_->detId2denseIdHB(it->getDetId()));
    }
    else if (it->getDetId().subdetId()==HcalEndcap) {
      HE_Hashes_.push_back(topo_->detId2denseIdHE(it->getDetId()));
    }
    else if (it->getDetId().subdetId()==HcalForward) {
      HF_Hashes_.push_back(topo_->detId2denseIdHF(it->getDetId()));
    }
  }
  for (std::vector<HOHXLogicalMapEntry>::iterator it = HOHXEntries_.begin(); it!=HOHXEntries_.end(); ++it) {
    if (HcalGenericDetId(it->getDetId().rawId()).isHcalCalibDetId() ) {
      CALIBHX_Hashes_.push_back(topo_->detId2denseIdCALIB(it->getDetId()));
    }
    else {
      HO_Hashes_.push_back(topo_->detId2denseIdHO(it->getDetId()));
    }
  }
  for (std::vector<CALIBLogicalMapEntry>::iterator it = CALIBEntries_.begin(); it!=CALIBEntries_.end(); ++it) {
    CALIBHX_Hashes_.push_back(topo_->detId2denseIdCALIB(it->getDetId()));
  }
  for (std::vector<ZDCLogicalMapEntry>::iterator it = ZDCEntries_.begin(); it!=ZDCEntries_.end(); ++it) {
    ZDC_Hashes_.push_back(HcalZDCDetId(it->getDetId()).denseIndex());
  }
  for (std::vector<HTLogicalMapEntry>::iterator it = HTEntries_.begin(); it!=HTEntries_.end(); ++it) {
    HT_Hashes_.push_back(topo_->detId2denseIdHT(it->getDetId()));
  }

  sort(HB_Hashes_.begin()     , HB_Hashes_.end());
  sort(HE_Hashes_.begin()     , HE_Hashes_.end());
  sort(HF_Hashes_.begin()     , HF_Hashes_.end());
  sort(HO_Hashes_.begin()     , HO_Hashes_.end());
  sort(CALIBHX_Hashes_.begin(), CALIBHX_Hashes_.end());
  sort(ZDC_Hashes_.begin()    , ZDC_Hashes_.end());
  sort(HT_Hashes_.begin()     , HT_Hashes_.end());

  for(unsigned int i = 0; i<HB_Hashes_.size()-1; i++) {
    int diff = HB_Hashes_.at(i+1)-HB_Hashes_.at(i);
    if (diff==0) numfails[0]++;
    else if (diff>1) numnotdense[0]++;
    else numpass[0]++;
  }
  for(unsigned int i = 0; i<HE_Hashes_.size()-1; i++) {
    int diff = HE_Hashes_.at(i+1)-HE_Hashes_.at(i);
    if (diff==0) numfails[1]++;
    else if (diff>1) numnotdense[1]++;
    else numpass[1]++;
  }
  for(unsigned int i = 0; i<HF_Hashes_.size()-1; i++) {
    int diff = HF_Hashes_.at(i+1)-HF_Hashes_.at(i);
    if (diff==0) numfails[2]++;
    else if (diff>1) numnotdense[2]++;
    else numpass[2]++;
  }
  for(unsigned int i = 0; i<HO_Hashes_.size()-1; i++) {
    int diff = HO_Hashes_.at(i+1)-HO_Hashes_.at(i);
    if (diff==0) numfails[3]++;
    else if (diff>1) numnotdense[3]++;
    else numpass[3]++;
  }
  for(unsigned int i = 0; i<CALIBHX_Hashes_.size()-1; i++) {
    int diff = CALIBHX_Hashes_.at(i+1)-CALIBHX_Hashes_.at(i);
    if (diff==0) numfails[4]++;
    else if (diff>1) numnotdense[4]++;
    else numpass[4]++;
  }
  for(unsigned int i = 0; i<ZDC_Hashes_.size()-1; i++) {
    int diff = ZDC_Hashes_.at(i+1)-ZDC_Hashes_.at(i);
    if (diff==0) numfails[5]++;
    else if (diff>1) numnotdense[5]++;
    else numpass[5]++;
  }
  for(unsigned int i = 0; i<HT_Hashes_.size()-1; i++) {
    int diff = HT_Hashes_.at(i+1)-HT_Hashes_.at(i);
    if (diff==0) numfails[6]++;
    else if (diff>1) numnotdense[6]++;
    else numpass[6]++;
  }
  cout << "HB HashIds (pass, collisions, non-dense) = (" << numpass[0] << "," << numfails[0] << "," << numnotdense[0] << ")" << endl;
  cout << "HE HashIds (pass, collisions, non-dense) = (" << numpass[1] << "," << numfails[1] << "," << numnotdense[1] << ")" << endl;
  cout << "HF HashIds (pass, collisions, non-dense) = (" << numpass[2] << "," << numfails[2] << "," << numnotdense[2] << ")" << endl;
  cout << "HO HashIds (pass, collisions, non-dense) = (" << numpass[3] << "," << numfails[3] << "," << numnotdense[3] << ")" << endl;
  cout << "CALIB/HX HashIds (pass, collisions, non-dense) = (" << numpass[4] << "," << numfails[4] << "," << numnotdense[4] << ")" << endl;
  cout << "ZDC HashIds (pass, collisions, non-dense) = (" << numpass[5] << "," << numfails[5] << "," << numnotdense[5] << ")" << endl;
  cout << "HT HashIds (pass, collisions, non-dense) = (" << numpass[6] << "," << numfails[6] << "," << numnotdense[6] << ")" << endl;
}

void HcalLogicalMap::checkElectronicsHashIds() {
  std::vector<int> Electronics_Hashes_;

  int numfails = 0;
  int numpass  = 0;
  int numnotdense = 0;

  cout << "\nRunning the hash checker for electronics Ids..." << endl;
  for (std::vector<HBHEHFLogicalMapEntry>::iterator it = HBHEHFEntries_.begin(); it!=HBHEHFEntries_.end(); ++it) {
    Electronics_Hashes_.push_back((it->getHcalElectronicsId()).linearIndex());
  }
  for (std::vector<ZDCLogicalMapEntry>::iterator it = ZDCEntries_.begin(); it!=ZDCEntries_.end(); ++it) {
    Electronics_Hashes_.push_back((it->getHcalElectronicsId()).linearIndex());
  }
  for (std::vector<CALIBLogicalMapEntry>::iterator it = CALIBEntries_.begin(); it!=CALIBEntries_.end(); ++it) {
    Electronics_Hashes_.push_back((it->getHcalElectronicsId()).linearIndex());
  }
  for (std::vector<HOHXLogicalMapEntry>::iterator it = HOHXEntries_.begin(); it!=HOHXEntries_.end(); ++it) {
    Electronics_Hashes_.push_back((it->getHcalElectronicsId()).linearIndex());
  }
  for (std::vector<HTLogicalMapEntry>::iterator it = HTEntries_.begin(); it!=HTEntries_.end(); ++it) {
    Electronics_Hashes_.push_back((it->getHcalTrigElectronicsId()).linearIndex());
  }

  sort(Electronics_Hashes_.begin() , Electronics_Hashes_.end());

  for(unsigned int i = 0; i<Electronics_Hashes_.size()-1; i++) {
    int diff = Electronics_Hashes_.at(i+1)-Electronics_Hashes_.at(i);
    if (diff==0) numfails++;
    else if (diff>1) numnotdense++;
    else numpass++;
  }
  cout << "Electronics Id linearIndex (pass, collisions, nondense) = (" << numpass << "," << numfails << "," << numnotdense << ")" << endl;
}

