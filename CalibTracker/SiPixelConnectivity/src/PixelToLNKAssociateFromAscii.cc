#include "CalibTracker/SiPixelConnectivity/interface/PixelToLNKAssociateFromAscii.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelPannelType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <ostream>
#include <fstream>
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

PixelToLNKAssociateFromAscii::PixelToLNKAssociateFromAscii(const string & fn, const bool phase) {
  phase1_ = phase;
  init(fn);

}
std::string PixelToLNKAssociateFromAscii::version() const {
  return theVersion; 
}

const PixelToLNKAssociateFromAscii::CablingRocId * PixelToLNKAssociateFromAscii::operator()( 
    const PixelToLNKAssociateFromAscii::DetectorRocId& roc) const {

  typedef std::vector< std::pair<DetectorRocId,CablingRocId> >::const_iterator IM;
  for (IM im = theConnection.begin(); im != theConnection.end(); im++) {
    if( ( *(im->first.module) == *roc.module ) && (im->first.rocDetId == roc.rocDetId)) {
      return &(im->second);  
    }
  }
  return nullptr;
}

// This is where the reading and interpretation of the ascci cabling input file is
void PixelToLNKAssociateFromAscii::init(const string & cfg_name) {

  edm::LogInfo(" init, input file: ") << cfg_name;

  std::ifstream file( cfg_name.c_str() );
  if ( !file ) {
    edm::LogError(" ** PixelToLNKAssociateFromAscii,init ** ")
         << " cant open data file: " << cfg_name;
    return;
  } else {
    edm::LogInfo("PixelToLNKAssociateFromAscii, read data from: ") <<cfg_name;
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

    edm::LogInfo("") <<" read line: "<< line;
    

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
	//cout<<" range find "<<strR<<" "<<strR.size()<<" "<<range.min()<<" "<<range.max()<<endl;
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
    int fedId, int linkId,  std::string module, Range rocDetIds) {

  string::size_type pos;


  // check for Barrel modules
  pos = module.find("BPix");
  if (pos != string::npos) { 
    
    string module0=module;
    // strip the trailing spaces 
    string::size_type p = module0.find(" ");
    //string::size_type p1 = module0.find_first_of(" ");
    //string::size_type p2 = module0.find_last_not_of(" ");
    //cout<<p<<" "<<p1<<" "<<p2<<endl;
    if(p != string::npos) module0 = module0.substr(0,p);
    PixelBarrelName * name = new PixelBarrelName(module0,phase1_);

    // shell
    string strP = module.substr(pos+6,2);
    PixelBarrelName::Shell part;
    if (strP=="mO")     part = PixelBarrelName::mO; 
    else if(strP=="mI") part = PixelBarrelName::mI;
    else if(strP=="pO") part = PixelBarrelName::pO;
    else                part = PixelBarrelName::pI;

    // // all this can be skipped ----------------------------------- 
    // module = module.substr(pos+9);
    // // sector
    // pos = module.find("_");
    // if (pos ==  string::npos) throw cms::Exception("problem with sector formatting");
    // // int sector = atoi( module.substr(3,pos-3).c_str());
    // module = module.substr(pos+1);
    // // layer
    // pos = module.find("_");
    // if (pos ==  string::npos) throw cms::Exception("problem with layer formatting");
    // int layer = atoi( module.substr(3,pos-3).c_str());
    // module = module.substr(pos+1);
    // // ladder
    // pos = module.find("_");
    // if (pos ==  string::npos) throw cms::Exception("problem with ladder formatting");
    // int ladder = atoi( module.substr(3,pos-3).c_str());
    // module = module.substr(pos+1);
    // // z-module
    // int zmodule = atoi( module.substr(3,pos-3).c_str());    
    // // place modules in connections
    // PixelBarrelName * name0 = new PixelBarrelName(part, layer, zmodule, ladder, phase1_);
    // if(name->name() != module0) cout<<" wrong name "<<fedId<<" "<<linkId<<" "
    // 				     <<module0<<" "<<name->name()<<" "<<name0->name()<<endl;
    // if(name->name() != name0->name()) cout<<" wrong name "<<fedId<<" "<<linkId<<" "
    // 				     <<module0<<" "<<name->name()<<" "<<name0->name()<<endl;
    // //edm::LogInfo(" module ")<<fedId<<" "<<linkId<<" "<<module0<<" "
    // //			     <<name0->name()<<" "<<rocDetIds.max()<<endl;
    // // until here 
    
    
     int rocLnkId = 0; 
     bool loopExecuted = false;
     for (int rocDetId=rocDetIds.min(); rocDetId <= rocDetIds.max(); rocDetId++) {
       loopExecuted = true;
       rocLnkId++;
       DetectorRocId  detectorRocId;
       //detectorRocId.module = name0;
       detectorRocId.module = name;
       detectorRocId.rocDetId = rocDetId;

       CablingRocId   cablingRocId;
       cablingRocId.fedId = fedId;
       cablingRocId.linkId = linkId;
       cablingRocId.rocLinkId = rocLnkId;
       // fix for type-B modules in barrel
       edm::LogInfo(" roc ")<<rocDetId<<" "<<rocLnkId<<" "<<name->isHalfModule()<<endl;
       if (name->isHalfModule() && (rocDetIds.min()>7)  
           && (part==PixelBarrelName::mO || part==PixelBarrelName::mI) ) {
	 //cablingRocId.rocLinkId = 9-rocLnkId;
	 // rocDetId=8,...,15
	 edm::LogInfo(" special for half modules ");
         cablingRocId.rocLinkId = rocLnkId;   // 1...8    19/11/08 d.k.
         detectorRocId.rocDetId = rocDetId-8; // 0...7
       }
       theConnection.push_back( make_pair(detectorRocId,cablingRocId));
     } 
     if (!loopExecuted) delete name;
  }


  // check for endcap modules
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

     int ring=1; // preset to 1 so it is ok for pilot blades
     // pannel type
     PixelPannelType::PannelType pannelType; 
     if(phase1_) {
       pannelType=PixelPannelType::p2x8; // only 1 type for phase1
       
       // this is not really needed, just for testing 
       // ring 
       pos = module.find("RNG");
       if (pos ==  string::npos) throw cms::Exception("problem with ring  formatting");
       ring  = atoi( module.substr(pos+3,1).c_str()); //
       //cout<<" ring "<<ring<<" "<<module<<endl;

     } else { // phase0
       pos = module.find("TYP:");
       if (pos ==  string::npos) throw cms::Exception("problem with pannel type formatting");
       string strT = module.substr(pos+5,3);
       string strT4 = module.substr(pos+5,4);
       ring=1;
       if (strT=="P3R") pannelType=PixelPannelType::p3R;
       else if (strT=="P3L") pannelType=PixelPannelType::p3L;
       else if (strT=="P4R") pannelType=PixelPannelType::p4R;
       else if (strT=="P4L") pannelType=PixelPannelType::p4L;
       else if (strT4=="P2X8") pannelType=PixelPannelType::p2x8;  // for pilot blades
       else throw cms::Exception("problem with pannel type formatting (unrecoginzed word)");      
     }
     

     // Cabling accoring to the panle type
     if ( pannelType==PixelPannelType::p4L) {
//     cout <<"----------- p4L"<<endl;
       int rocLnkId =0;
       for (int plaq = 1; plaq <= 4; plaq++) {
         Range rocs; int firstRoc=0; int step=0;
         if (plaq==1) { rocs = Range(0,1); firstRoc=1; step=-1; }
         if (plaq==2) { rocs = Range(0,5); firstRoc=0; step=+1; }
         if (plaq==3) { rocs = Range(0,7); firstRoc=0; step=+1; }
         if (plaq==4) { rocs = Range(0,4); firstRoc=0; step=+1; }
         PixelEndcapName * name  = new PixelEndcapName(part,disk,blade,pannel,plaq);
         for (int iroc =rocs.min(); iroc<=rocs.max(); iroc++) {
           rocLnkId++;
           int rocDetId = firstRoc + step*iroc; 

           DetectorRocId  detectorRocId;
           //detectorRocId.module = new PixelEndcapName(part,disk,blade,pannel,plaq);
           detectorRocId.module = name;
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
         PixelEndcapName * name  = new PixelEndcapName(part,disk,blade,pannel,plaq);
         for (int iroc =rocs.min(); iroc-rocs.max() <= 0; iroc++) {
           rocLnkId++;
           int rocDetId = firstRoc + step*iroc;
           if (rocDetId > rocs.max()) rocDetId = (rocDetId-1)%rocs.max();

           DetectorRocId  detectorRocId;
           //detectorRocId.module = new PixelEndcapName(part,disk,blade,pannel,plaq);
           detectorRocId.module = name;
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
         PixelEndcapName * name  = new PixelEndcapName(part,disk,blade,pannel,plaq);
         for (int iroc =rocs.min(); iroc<=rocs.max(); iroc++) {
           rocLnkId++;
           int rocDetId = firstRoc + step*iroc; 

           DetectorRocId  detectorRocId;
           detectorRocId.module = name;
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
         PixelEndcapName * name  = new PixelEndcapName(part,disk,blade,pannel,plaq);
         for (int iroc =rocs.min(); iroc<=rocs.max(); iroc++) {
           rocLnkId++;
           int rocDetId = firstRoc + step*iroc;
           if (rocDetId > rocs.max()) rocDetId = (rocDetId-1)%rocs.max();

           DetectorRocId  detectorRocId;
           detectorRocId.module = name;
           detectorRocId.rocDetId = rocDetId;

           CablingRocId   cablingRocId;
           cablingRocId.fedId = fedId;
           cablingRocId.linkId = linkId;
           cablingRocId.rocLinkId = rocLnkId;

           theConnection.push_back( make_pair(detectorRocId,cablingRocId));
//         cout <<"PLAQ:"<<plaq<<" rocDetId: "<<rocDetId<<" rocLnkId:"<<rocLnkId<<endl;
         } // for
       } // for

     } else if ( pannelType==PixelPannelType::p2x8) { // phase-1 blades
       //       cout <<"----------- p2x8"<<endl;
       int rocLnkId = 0; 
       //       Range rocs = Range(0, 15); 
       //       for (int rocDetId=rocs.min(); rocDetId <= rocs.max(); rocDetId++) {
       PixelEndcapName * name = new PixelEndcapName(part, disk, blade, pannel, ring, phase1_); 
       bool loopExecuted = false;
       for (int rocDetId=rocDetIds.min(); rocDetId <= rocDetIds.max(); rocDetId++) {
           loopExecuted = true;
	 rocLnkId++;
	 DetectorRocId  detectorRocId;
	 detectorRocId.module = name;
	 detectorRocId.rocDetId = rocDetId;
	 CablingRocId   cablingRocId;
	 cablingRocId.fedId = fedId;
	 cablingRocId.linkId = linkId;
	 cablingRocId.rocLinkId = rocLnkId;
	 theConnection.push_back( make_pair(detectorRocId,cablingRocId));
	 edm::LogInfo("PixelToLNKAssociateFromAscii FPix ")
	   << " rocDetId: " << rocDetId 
	   << " rocLnkId:" << rocLnkId 
	   << " fedId = " << fedId 
	   << " linkId = " << linkId
	   << " name = " << name->name();
	 // cout << " rocDetId: " << rocDetId 
	 //      << " rocLnkId:" << rocLnkId 
	 //      << " fedId = " << fedId 
	 //      << " linkId = " << linkId
	 //      << " name = " << name0->name()
	 //      << endl;
       } // end for 
       if (!loopExecuted) {
           delete name;
       }
       
     } // end of type

  }
}

PixelToLNKAssociateFromAscii::Range 
    PixelToLNKAssociateFromAscii::readRange( const string & l) const {
  //cout<<l<<" in range "<<l.size()<<endl;
  string l1,l2;
  int i1=-1, i2=-1;
  int len = l.size();
  //for(int i=0; i<len;i++) {
  // cout<<i<<" "<<l[i]<<endl;
  //}
  string::size_type p = l.find(",");
  if(p != string::npos) {
    //cout<<p<<" "<<len<<endl;
    l1 = l.substr(0,p-1+1);
    l2 = l.substr(p+1,len-1-p);
    i1 = stoi(l1);
    i2 = stoi(l2);
    //cout<<l1<<" "<<l2<<" "<<i1<<" "<<i2<<endl;
  }

  return Range(i1,i2);

  // this method is very stupid it relies on a space being present after the last number!
  // exchange with string opertaions (above)
  //bool first = true;
  //int num1 = -1;
  //int num2 = -1;
  // const char * line = l.c_str();
  // int i=0;
  // while (line) {
  //   i++;
  //   char * evp = 0;
  //   int num = strtol(line, &evp, 10);
  //   //cout<<i<<" "<<num<<" "<<evp<<" "<<line<<endl;
  //   //{ stringstream s; s<<"read from line: "; s<<num; LogTrace("") << s.str(); }
  //   if (evp != line) {
  //     line = evp +1;
  //     //cout<<i<<" "<<num<<" "<<evp<<" "<<line<<endl;
  //     if (first) { num1 = num; first = false;}
  //     num2 = num;
  //     //cout<<" not first "<<num2<<endl;
  //   } else line = 0;
  // }
  // if (first) {
  //   string s = "** PixelToLNKAssociateFromAscii, read data, cant intrpret: " ;
  //   edm::LogInfo(s) << endl 
  // 		    << l << endl 
  //             <<"=====> send exception " << endl;
  //   s += l;
  //   throw cms::Exception(s);
  // }
  //if(i1!=num1) cout<<" something wrong with min range "<<i1<<" "<<num1<<endl;
  //if(!phase1_ && (i2!=num2)) cout<<" something wrong with max range "<<i2<<" "<<num2<<endl;
  //cout<<" min max "<<num1<<" "<<num2<<endl;
  //return Range(num1,num2);

}

