#include "CalibTracker/SiPixelConnectivity/interface/PixelToLNKAssociateFromAscii.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelPannelType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <ostream>
#include <fstream>
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

PixelToLNKAssociateFromAscii::PixelToLNKAssociateFromAscii(const string & fn) {
  init(fn);
}
std::string PixelToLNKAssociateFromAscii::version() const
{
  return theVersion; 
}

const PixelToLNKAssociateFromAscii::CablingRocId * PixelToLNKAssociateFromAscii::operator()( 
    const PixelToLNKAssociateFromAscii::DetectorRocId& roc) const
{
//  bool deb = (roc.module->name()=="BPix_BpI_SEC1_LYR1_LDR1H_MOD1");
//  if (deb) cout <<"KUKU"<<endl;

  typedef std::vector< std::pair<DetectorRocId,CablingRocId> >::const_iterator IM;
  for (IM im = theConnection.begin(); im != theConnection.end(); im++) {
    if( ( *(im->first.module) == *roc.module ) && (im->first.rocDetId == roc.rocDetId)) {
      return &(im->second);  
    }
  }
  return 0;
}

void PixelToLNKAssociateFromAscii::init(const string & cfg_name)
{
  LogDebug("init, input file:") << cfg_name.c_str();

  std::ifstream file( cfg_name.c_str() );
  if ( !file ) {
    edm::LogError(" ** PixelToLNKAssociateFromAscii,init ** ")
         << " cant open data file: " << cfg_name;
    return;
  } else {
    edm::LogInfo("PixelToLNKAssociateFromAscii, read data from: ") <<cfg_name ;
  }

  string line;
  int fedId=-1;
  int linkId=-1;

  try {
  while (getline(file,line)) {
    //
    // treat # lines
    //
    string::size_type pos = line.find("#");
    if (pos != string::npos) line = line.erase(pos);

    string::size_type posF = line.find("FED:");
    string::size_type posL = line.find("LNK:");
    string::size_type posM = line.find("MOD:");
    string::size_type posR = line.find("ROC:");

    LogTrace("") <<" read line: "<< line;
    

    //
    // treat version lines, reset date
    //
    if (     line.compare(0,3,"VER") == 0 ) { 
      edm::LogInfo("version: ")<<line;
      theVersion = line;
    }

    //
    // fed id line
    //
    else if ( posF != string::npos) { 
      line = line.substr(posF+4);
      fedId = atoi(line.c_str());
    }

    //
    // link id linke
    //
    else if ( posL != string::npos) {
      string srtL = line.substr(posL+4);
      linkId = atoi(srtL.c_str());
    }

    //
    // module description
    //
    if ( posM != string::npos) {
      if (posR != string::npos) {
        string strM = line.substr(posM+4, posR-posM-5);
        string::size_type pos = strM.find(" "); 
        if(pos != string::npos) strM = strM.substr(pos+1);
        string strR = line.substr(posR+4);
        Range range = readRange(strR);
        addConnections( fedId, linkId, strM, range);
      } else { 
        string strM= line.substr(posM+4);
        string::size_type pos = strM.find(" "); 
        if(pos != string::npos) strM = strM.substr(pos+1);
        addConnections( fedId, linkId, strM, Range(0,0));
      }
    }
  }
  } 
  catch(exception& err) {
    edm::LogError("**PixelToLNKAssociateFromAscii**  exception")<<err.what();
  }

  //
  // for debug
  //
  std::ostringstream str;
  str <<" **PixelToLNKAssociateFromAscii ** CONNECTIONS: "<< endl;
  typedef vector< pair<DetectorRocId,CablingRocId> >::const_iterator ICON;
  for (ICON ic = theConnection.begin(); ic != theConnection.end(); ic++) {
    str<< (*ic).first.module->name()
       <<", rocDetId="<<(*ic).first.rocDetId
       <<", fedId="<<ic->second.fedId
       <<", linkId="<<ic->second.linkId
       <<", rocLinkId="<<ic->second.rocLinkId
       <<endl;
  }
  edm::LogInfo("PixelToLNKAssociateFromAscii")<<str.str();
}

void PixelToLNKAssociateFromAscii::addConnections(
    int fedId, int linkId,  std::string module, Range rocDetIds)
{
  string::size_type pos;

  // check for Barrel modules
  pos = module.find("BPix");
  if (pos != string::npos) { 

     // shell
     string strP = module.substr(pos+6,2);
     PixelBarrelName::Shell part;
         if (strP=="mO") part = PixelBarrelName::mO; 
     else if(strP=="mI") part = PixelBarrelName::mI;
     else if(strP=="pO") part = PixelBarrelName::pO;
     else                part = PixelBarrelName::pI;
     module = module.substr(pos+9);

     // sector
     pos = module.find("_");
     if (pos ==  string::npos) throw cms::Exception("problem with sector formatting");
     // int sector = atoi( module.substr(3,pos-3).c_str());
     module = module.substr(pos+1);

     // layer
     pos = module.find("_");
     if (pos ==  string::npos) throw cms::Exception("problem with layer formatting");
     int layer = atoi( module.substr(3,pos-3).c_str());
     module = module.substr(pos+1);

     // ladder
     pos = module.find("_");
     if (pos ==  string::npos) throw cms::Exception("problem with ladder formatting");
     int ladder = atoi( module.substr(3,pos-3).c_str());
     module = module.substr(pos+1);

     // z-module
     int zmodule = atoi( module.substr(3,pos-3).c_str());

     // place modules in connections
     int rocLnkId = 0; 
     for (int rocDetId=rocDetIds.min(); rocDetId <= rocDetIds.max(); rocDetId++) {
       rocLnkId++;
       DetectorRocId  detectorRocId;
       PixelBarrelName * name = new PixelBarrelName(part, layer, zmodule, ladder);
       detectorRocId.module = name;
       detectorRocId.rocDetId = rocDetId;
       CablingRocId   cablingRocId;
       cablingRocId.fedId = fedId;
       cablingRocId.linkId = linkId;
       cablingRocId.rocLinkId = rocLnkId;
       // fix for type-B modules in barrel
       if (name->isHalfModule() && (rocDetIds.min()>7)  
           && (part==PixelBarrelName::mO || PixelBarrelName::mI) ) {
	 //cablingRocId.rocLinkId = 9-rocLnkId;
	 // rocDetId=8,...,15
         cablingRocId.rocLinkId = rocLnkId;   // 1...8    19/11/08 d.k.
         detectorRocId.rocDetId = rocDetId-8; // 0...7
       }
       theConnection.push_back( make_pair(detectorRocId,cablingRocId));
     } 
  }

  // check for endcap modules
  // check for Barrel modules
  pos = module.find("FPix");
  if (pos != string::npos) {
     string strH = module.substr(pos+6,2);
     PixelEndcapName::HalfCylinder part;
         if (strH=="mO") part = PixelEndcapName::mO;
     else if(strH=="mI") part = PixelEndcapName::mI;
     else if(strH=="pO") part = PixelEndcapName::pO;
     else                part = PixelEndcapName::pI;
     module = module.substr(pos+9);

     // disk
     pos = module.find("_");
     if (pos ==  string::npos) throw cms::Exception("problem with disk formatting");
     int disk = atoi( module.substr(1,pos-1).c_str());
     module = module.substr(pos+1);

     // blade
     pos = module.find("_");
     if (pos ==  string::npos) throw cms::Exception("problem with blade formatting");
     int blade = atoi( module.substr(3,pos-3).c_str());
     module = module.substr(pos+1);

     //pannel
     pos = module.find("_");
     if (pos ==  string::npos) throw cms::Exception("problem with pannel formatting");
     int pannel = atoi( module.substr(3,pos-3).c_str());
     module = module.substr(pos+1);

     // plaquete
//     pos = module.find("_");
//     if (pos ==  string::npos) throw cms::Exception("problem with plaquette formatting");
//     int plaq = atoi( module.substr(3,pos-3).c_str());

     // pannel type
     pos = module.find("TYP:");
     if (pos ==  string::npos) throw cms::Exception("problem with pannel type formatting");
     string strT = module.substr(pos+5,3);

     PixelPannelType::PannelType pannelType; 
          if (strT=="P3R") pannelType=PixelPannelType::p3R;
     else if (strT=="P3L") pannelType=PixelPannelType::p3L;
     else if (strT=="P4R") pannelType=PixelPannelType::p4R;
     else if (strT=="P4L") pannelType=PixelPannelType::p4L;
     else throw cms::Exception("problem with pannel type formatting (unrecoginzed word)");

     if ( pannelType==PixelPannelType::p4L) {
//     cout <<"----------- p4L"<<endl;
       int rocLnkId =0;
       for (int plaq = 1; plaq <= 4; plaq++) {
         Range rocs; int firstRoc=0; int step=0;
         if (plaq==1) { rocs = Range(0,1); firstRoc=1; step=-1; }
         if (plaq==2) { rocs = Range(0,5); firstRoc=0; step=+1; }
         if (plaq==3) { rocs = Range(0,7); firstRoc=0; step=+1; }
         if (plaq==4) { rocs = Range(0,4); firstRoc=0; step=+1; }
         for (int iroc =rocs.min(); iroc<=rocs.max(); iroc++) {
           rocLnkId++;
           int rocDetId = firstRoc + step*iroc; 

           DetectorRocId  detectorRocId;
           detectorRocId.module = new PixelEndcapName(part,disk,blade,pannel,plaq);
           detectorRocId.rocDetId = rocDetId;

           CablingRocId   cablingRocId;
           cablingRocId.fedId = fedId;
           cablingRocId.linkId = linkId;
           cablingRocId.rocLinkId = rocLnkId;

           theConnection.push_back( make_pair(detectorRocId,cablingRocId));
//         cout <<"PLAQ:"<<plaq<<" rocDetId: "<<rocDetId<<" rocLnkId:"<<rocLnkId<<endl;
         }
       } 
     } 
     else if ( pannelType==PixelPannelType::p4R) {
//     cout <<"----------- p4R"<<endl;
       int rocLnkId =0;
       for (int plaq = 4; plaq >= 1; plaq--) {
         Range rocs; int firstRoc=0; int step=0;
         if (plaq==1) { rocs = Range(0,1); firstRoc=1; step=-1; }
         if (plaq==2) { rocs = Range(0,5); firstRoc=3; step=+1; }
         if (plaq==3) { rocs = Range(0,7); firstRoc=4; step=+1; }
         if (plaq==4) { rocs = Range(0,4); firstRoc=0; step=+1; }
         for (int iroc =rocs.min(); iroc-rocs.max() <= 0; iroc++) {
           rocLnkId++;
           int rocDetId = firstRoc + step*iroc;
           if (rocDetId > rocs.max()) rocDetId = (rocDetId-1)%rocs.max();

           DetectorRocId  detectorRocId;
           detectorRocId.module = new PixelEndcapName(part,disk,blade,pannel,plaq);
           detectorRocId.rocDetId = rocDetId;

           CablingRocId   cablingRocId;
           cablingRocId.fedId = fedId;
           cablingRocId.linkId = linkId;
           cablingRocId.rocLinkId = rocLnkId;

           theConnection.push_back( make_pair(detectorRocId,cablingRocId));
//         cout <<"PLAQ:"<<plaq<<" rocDetId: "<<rocDetId<<" rocLnkId:"<<rocLnkId<<endl;
         }
       }
     }
     else if ( pannelType==PixelPannelType::p3L) {
//     cout <<"----------- p3L"<<endl;
       int rocLnkId =0;
       for (int plaq = 1; plaq <= 3; plaq++) {
         Range rocs; int firstRoc=0; int step=0;
         if (plaq==1) { rocs = Range(0,5); firstRoc=0; step=1; }
         if (plaq==2) { rocs = Range(0,7); firstRoc=0; step=1; }
         if (plaq==3) { rocs = Range(0,9); firstRoc=0; step=1; }
         for (int iroc =rocs.min(); iroc<=rocs.max(); iroc++) {
           rocLnkId++;
           int rocDetId = firstRoc + step*iroc; 

           DetectorRocId  detectorRocId;
           detectorRocId.module = new PixelEndcapName(part,disk,blade,pannel,plaq);
           detectorRocId.rocDetId = rocDetId;

           CablingRocId   cablingRocId;
           cablingRocId.fedId = fedId;
           cablingRocId.linkId = linkId;
           cablingRocId.rocLinkId = rocLnkId;

           theConnection.push_back( make_pair(detectorRocId,cablingRocId));
//         cout <<"PLAQ:"<<plaq<<" rocDetId: "<<rocDetId<<" rocLnkId:"<<rocLnkId<<endl;
         }
       } 
     } 
     else if ( pannelType==PixelPannelType::p3R) {
//     cout <<"----------- p3R"<<endl;
       int rocLnkId =0;
       for (int plaq = 3; plaq >= 1; plaq--) {
         Range rocs; int firstRoc=0; int step=0;
         if (plaq==1) { rocs = Range(0,5); firstRoc=3; step=1; }
         if (plaq==2) { rocs = Range(0,7); firstRoc=4; step=1; }
         if (plaq==3) { rocs = Range(0,9); firstRoc=5; step=1; }
         for (int iroc =rocs.min(); iroc<=rocs.max(); iroc++) {
           rocLnkId++;
           int rocDetId = firstRoc + step*iroc;
           if (rocDetId > rocs.max()) rocDetId = (rocDetId-1)%rocs.max();

           DetectorRocId  detectorRocId;
           detectorRocId.module = new PixelEndcapName(part,disk,blade,pannel,plaq);
           detectorRocId.rocDetId = rocDetId;

           CablingRocId   cablingRocId;
           cablingRocId.fedId = fedId;
           cablingRocId.linkId = linkId;
           cablingRocId.rocLinkId = rocLnkId;

           theConnection.push_back( make_pair(detectorRocId,cablingRocId));
//         cout <<"PLAQ:"<<plaq<<" rocDetId: "<<rocDetId<<" rocLnkId:"<<rocLnkId<<endl;
         }
       }
     }

  }
}

PixelToLNKAssociateFromAscii::Range 
    PixelToLNKAssociateFromAscii::readRange( const string & l) const
{
  bool first = true;
  int num1 = -1;
  int num2 = -1;
  const char * line = l.c_str();
  while (line) {
    char * evp = 0;
    int num = strtol(line, &evp, 10);
    { stringstream s; s<<"raad from line: "; s<<num; LogTrace("") << s.str(); }
    if (evp != line) {
      line = evp +1;
      if (first) { num1 = num; first = false; }
      num2 = num;
    } else line = 0;
  }
  if (first) {
    string s = "** PixelToLNKAssociateFromAscii, read data, cant intrpret: " ;
    edm::LogInfo(s) << endl 
              << l << endl 
              <<"=====> send exception " << endl;
    s += l;
    throw cms::Exception(s);
  }
  return Range(num1,num2);
}

