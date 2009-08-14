//
// This class stores the name and related
// hardware mapings for a ROC 
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTimeFormatter.h"
//#include "PixelUtilities/PixelTestStandUtilities/include/PixelTimer.h"
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>
#include <assert.h>
#include <stdexcept>

using namespace pos;
using namespace std;


PixelNameTranslation::PixelNameTranslation(std::vector< std::vector<std::string> > &tableMat):PixelConfigBase(" "," "," "){
  std::string mthn = "[PixelNameTranslation::PixelNameTranslation()]\t\t    " ;
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;

/*
  EXTENSION_TABLE_NAME: PIXEL_NAME_TRANS (VIEW: CONF_KEY_NAME_TRANS_V)

  CONFIG_KEY				    NOT NULL VARCHAR2(80)
  KEY_TYPE				    NOT NULL VARCHAR2(80)
  KEY_ALIAS				    NOT NULL VARCHAR2(80)
  VERSION					     VARCHAR2(40)
  KIND_OF_COND  			    NOT NULL VARCHAR2(40)
  ROC_NAME				    NOT NULL VARCHAR2(200)
  PXLFEC_NAME				    NOT NULL NUMBER(38)
  MFEC_POSN				    NOT NULL NUMBER(38)
  MFEC_CHAN				    NOT NULL NUMBER(38)
  HUB_ADDRS					     NUMBER(38)
  PORT_NUM				    NOT NULL NUMBER(38)
  ROC_I2C_ADDR  			    NOT NULL NUMBER(38)
  PXLFED_NAME				    NOT NULL NUMBER(38)
  FED_CHAN				    NOT NULL NUMBER(38)
  FED_ROC_NUM				    NOT NULL NUMBER(38)
  TBM_MODE					     VARCHAR2(200)
*/

  colNames.push_back("CONFIG_KEY"  ); //0
  colNames.push_back("KEY_TYPE"    ); //1
  colNames.push_back("KEY_ALIAS"   ); //2
  colNames.push_back("VERSION"     ); //3
  colNames.push_back("KIND_OF_COND"); //4
  colNames.push_back("ROC_NAME"    ); //5
  colNames.push_back("PXLFEC_NAME" ); //6 
  colNames.push_back("MFEC_POSN"   ); //7 
  colNames.push_back("MFEC_CHAN"   ); //8 
  colNames.push_back("HUB_ADDRS"   ); //9 
  colNames.push_back("PORT_NUM"    ); //10
  colNames.push_back("ROC_I2C_ADDR"); //11
  colNames.push_back("PXLFED_NAME" ); //12
  colNames.push_back("FED_CHAN"    ); //13
  colNames.push_back("FED_ROC_NUM" ); //14
  colNames.push_back("TBM_MODE"    ); //15

  for(unsigned int c = 0 ; c < tableMat[0].size() ; c++){
    for(unsigned int n=0; n<colNames.size(); n++){
      if(tableMat[0][c] == colNames[n]){
	colM[colNames[n]] = c;
	break;
      }
    }
  }//end for
  /*
  for(unsigned int n=0; n<colNames.size(); n++){
    if(colM.find(colNames[n]) == colM.end()){
      std::cerr << __LINE__ << "]\t" << mthn << "Couldn't find in the database the column with name " << colNames[n] << std::endl;
      assert(0);
    }
  }
  */

 for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix
   std::string rocname       = tableMat[r][colM["ROC_NAME"]] ;
   std::string TBMChannel    = tableMat[r][colM["TBM_MODE"]] ; // assert(0); // need to add this to the input table
   if(TBMChannel == "")
     {
       TBMChannel = "A" ;
     }
   /* // modified by MR on 13-07-2008 11:32:50
      Umesh changed the content of the column and 
      stripped out the FPix_Pxl_FEC_ part of the "number"
   tableMat[r][colM["PXLFEC_NAME"]].erase(0 , 13);//PIXFEC
   unsigned int fecnumber    = (unsigned int)atoi(tableMat[r][colM["PXLFEC_NAME"]].c_str());
   */
   unsigned int fecnumber    = (unsigned int)atoi(tableMat[r][colM["PXLFEC_NAME"]].c_str());
   unsigned int mfec         = (unsigned int)atoi(tableMat[r][colM["MFEC_POSN"]].c_str());
   unsigned int mfecchannel  = (unsigned int)atoi(tableMat[r][colM["MFEC_CHAN"]].c_str());
   unsigned int hubaddress   = (unsigned int)atoi(tableMat[r][colM["HUB_ADDRS"]].c_str());
   unsigned int portaddress  = (unsigned int)atoi(tableMat[r][colM["PORT_NUM"]].c_str());
   unsigned int rocid	     = (unsigned int)atoi(tableMat[r][colM["ROC_I2C_ADDR"]].c_str());
   // modified by MR on 13-07-2008 11:47:32
   /* Umesh changed the content of the column and 
      stripped out the PxlFED_ part of the "number"
     
   tableMat[r][colM["PXLFED_NAME"]].erase(0,7);//FED
   */
   unsigned int fednumber    = (unsigned int)atoi(tableMat[r][colM["PXLFED_NAME"]].c_str());
   unsigned int fedchannel   = (unsigned int)atoi(tableMat[r][colM["FED_CHAN"]].c_str());
   unsigned int fedrocnumber = (unsigned int)(atoi(tableMat[r][colM["FED_ROC_NUM"]].c_str()));
	
	
   PixelROCName aROC(rocname);
   if (aROC.rocname()!=rocname){
     std::cout << __LINE__ << "]\t" << mthn << "Rocname  : " << rocname        << std::endl;
     std::cout << __LINE__ << "]\t" << mthn << "Parsed to: " << aROC.rocname() << std::endl;
     assert(0);
   }

   if (ROCNameFromFEDChannelROCExists(fednumber,fedchannel,
				      fedrocnumber)){
     std::cout << __LINE__ << "]\t" << mthn 
               << "ROC with fednumber=" << fednumber
	       << " fedchannel="	<< fedchannel
	       << " roc number="	<< fedrocnumber
	       << " already exists"     << std::endl;
     std::cout << __LINE__ << "]\t" << mthn << "Fix this inconsistency in the name translation"
	       << std::endl;
     assert(0);
   }
   
   PixelHdwAddress hdwAdd(fecnumber,mfec,mfecchannel,
			  hubaddress,portaddress,
			  rocid,
			  fednumber,fedchannel,fedrocnumber);
//    std::cout << "[PixelNameTranslation::PixelNameTranslation()] aROC: " << aROC << std::endl;
   translationtable_[aROC]=hdwAdd;
   fedlookup_[hdwAdd]=aROC;

   PixelModuleName aModule(rocname);
   PixelChannel aChannel(aModule, TBMChannel);

   hdwTranslationTable_[hdwAdd] = aChannel;
   

   // Look for this channel in channelTransaltionTable.  If it is found, check that the hardware address agrees.  If not, add it to the table.  Also, if another channel on that module is found, check that the FEC part agrees, and the FED part doesn't.
   bool foundChannel = false;

   std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.find(aChannel);

   if ( channelTranslationTable_itr != channelTranslationTable_.end()) {

     if (!(channelTranslationTable_itr->second |= hdwAdd))
	   {
	     cout << "Found two ROCs on the same channe, but not same hdw"<<endl;
	     cout << "Hdw1:"<<endl<<channelTranslationTable_itr->second<<endl;
	     cout << "Hdw2:"<<endl<<hdwAdd<<endl;
	   }
     assert( channelTranslationTable_itr->second |= hdwAdd );
     foundChannel = true;
   }
   else if ( channelTranslationTable_itr->first.module() == aModule ) 
     {
       assert( channelTranslationTable_itr->second.fecnumber() == hdwAdd.fecnumber() );
       assert( channelTranslationTable_itr->second.mfec() == hdwAdd.mfec() );
       assert( channelTranslationTable_itr->second.mfecchannel() == hdwAdd.mfecchannel() );
       //assert( channelTranslationTable_itr->second.portaddress() == hdwAdd.portaddress() );
       assert( channelTranslationTable_itr->second.hubaddress() == hdwAdd.hubaddress() );
       assert( channelTranslationTable_itr->second.fednumber() != hdwAdd.fednumber() || channelTranslationTable_itr->second.fedchannel() != hdwAdd.fedchannel() );
     }
   
   if ( foundChannel == false ) {
     channelTranslationTable_[aChannel] = hdwAdd;
   }     

 }//end for r

  const std::map<unsigned int, std::set<unsigned int> > fedsAndChannels=getFEDsAndChannels();


  std::vector<PixelROCName> tmp(24);

  std::map<unsigned int, std::map<unsigned int, int > > counter;
    //       FED id                  FED channel

  std::map<unsigned int, std::map<unsigned int, int > > maxindex;

  std::map<PixelROCName,PixelHdwAddress>::const_iterator it=translationtable_.begin();
    
  for(;it!=translationtable_.end();it++){

    int index=it->second.fedrocnumber();

    unsigned int fednumber=it->second.fednumber();
    unsigned int fedchannel=it->second.fedchannel();

    std::vector<PixelROCName>& tmp= rocsFromFEDidAndChannel_[fednumber][fedchannel];
    
    if (tmp.size()==0){
      tmp.resize(24);
      counter[fednumber][fedchannel]=0;
      maxindex[fednumber][fedchannel]=0;
    }

    if (index>maxindex[fednumber][fedchannel]) maxindex[fednumber][fedchannel]=index;
    tmp[index]=it->first;
    counter[fednumber][fedchannel]++;

  }


  it=translationtable_.begin();
    
  for(;it!=translationtable_.end();it++){

    unsigned int fednumber=it->second.fednumber();
    unsigned int fedchannel=it->second.fedchannel();
    
    std::vector<PixelROCName>& tmp= rocsFromFEDidAndChannel_[fednumber][fedchannel];
    
    assert(counter[fednumber][fedchannel]==maxindex[fednumber][fedchannel]+1);

    tmp.resize(counter[fednumber][fedchannel]);

  }

}//end contructor
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PixelNameTranslation::PixelNameTranslation(std::string filename):
  PixelConfigBase(" "," "," "){

  static std::string mthn = "[PixelNameTranslation::PixelNameTranslation()]\t\t    " ;
  
  std::ifstream in(filename.c_str());
  
  if (!in.good()){
    std::cout << __LINE__ << "]\t" << mthn << "Could not open: " << filename << std::endl;
    throw std::runtime_error("Failed to open file "+filename);
  }
  else {
    std::cout << __LINE__ << "]\t" << mthn << "Reading from: "   << filename << std::endl;
  }
  
  std::string dummy;

  getline(in, dummy); // skip the column headings


  do {

    std::string rocname;
    std::string TBMChannel;
    unsigned int fecnumber;
    unsigned int mfec;
    unsigned int mfecchannel;
    unsigned int hubaddress;
    unsigned int portaddress;
    unsigned int rocid;
    unsigned int fednumber;
    unsigned int fedchannel;
    unsigned int fedrocnumber;


    in >> rocname;
    in >> TBMChannel;
    if ( TBMChannel != "A" && TBMChannel != "B" ) // no TBM channel was specified, so default to A and set fecnumber to the value of this string
      {
	fecnumber = atoi(TBMChannel.c_str());
	TBMChannel = "A";
      }
    else // TBM channel was specified, now read fecnumber
      {
	in >> fecnumber;
      }
    in >> mfec >> mfecchannel 
       >> hubaddress >> portaddress >> rocid >> fednumber 
       >> fedchannel >> fedrocnumber;

    if (!in.eof() ){

      PixelROCName aROC(rocname);

      //debug
      //if (aROC.rocname()!=rocname){
      //std::cout << __LINE__ << "]\t" << mthn << "Rocname  : " << rocname        << std::endl;
      //std::cout << __LINE__ << "]\t" << mthn << "Parsed to: " << aROC.rocname() << std::endl;
      //assert(0);
      //}

      if (ROCNameFromFEDChannelROCExists(fednumber,fedchannel,
					 fedrocnumber)){
	std::cout << __LINE__ << "]\t"     << mthn 
	          << "ROC with fednumber=" << fednumber
		  << " fedchannel="	   << fedchannel
		  << " roc number="	   << fedrocnumber
		  << " already exists"     << std::endl;
	std::cout << __LINE__ << "]\t"     << mthn 
	          << "Fix this inconsistency in the name translation"
		  << std::endl;
	assert(0);
	
      }

      PixelHdwAddress hdwAdd(fecnumber,mfec,mfecchannel,
			     hubaddress,portaddress,
			     rocid,
			     fednumber,fedchannel,fedrocnumber);
      //std::cout << aROC << std::endl;
      // modified by MR on 18-01-2008 11:18:53
      //       std::cout << hdwAdd << std::endl ;
//cout << "[PixelNameTranslation::PixelNameTranslation()]\t\t-----------------------------"  << endl ;
// A fecnumber    
// B mfec	  
// C mfecchannel  
// D hubaddress   
// E portaddress  
// F rocid	  
// G fednumber    
// H fedchannel   
// I fedrocnumber 
//cout << "[PixelNameTranslation::PixelNameTranslation()]\t\t"
//     << " A " << fecnumber   
//     << " B " << mfec	      
//     << " C " << mfecchannel  
//     << " D " << hubaddress   
//     << " E " << portaddress  
//     << " F " << rocid        
//     << " G " << fednumber    
//     << " H " << fedchannel   
//     << " I " << fedrocnumber << endl ;    


      translationtable_[aROC]=hdwAdd;
      fedlookup_[hdwAdd]=aROC;
	    
      PixelModuleName aModule(rocname);
      PixelChannel aChannel(aModule, TBMChannel);

      hdwTranslationTable_[hdwAdd] = aChannel;

      // Look for this channel in channelTransaltionTable.  If it is found, 
      // check that the hardware address agrees.  If not, add it to the table.
      // Also, if another channel on that module is found, check that the FEC 
      // part agrees, and the FED part doesn't.
      bool foundChannel = false;


      std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.find(aChannel);

      if ( channelTranslationTable_itr != channelTranslationTable_.end()) {
	if (!(channelTranslationTable_itr->second |= hdwAdd)){
		
	  cout << __LINE__ << "]\t" << mthn << "Found two ROCs on the same channe, but not same hdw" << endl;
	  cout << __LINE__ << "]\t" << mthn << "Hdw1: " << endl << channelTranslationTable_itr->second << endl;
	  cout << __LINE__ << "]\t" << mthn << "Hdw2: " << endl << hdwAdd << endl;
	}
	assert( channelTranslationTable_itr->second |= hdwAdd );
	foundChannel = true;
      }
      else if ( channelTranslationTable_itr->first.module() == aModule ) {
	assert( channelTranslationTable_itr->second.fecnumber() == hdwAdd.fecnumber() );
	assert( channelTranslationTable_itr->second.mfec() == hdwAdd.mfec() );
	assert( channelTranslationTable_itr->second.mfecchannel() == hdwAdd.mfecchannel() );
	//assert( channelTranslationTable_itr->second.portaddress() == hdwAdd.portaddress() );
	assert( channelTranslationTable_itr->second.hubaddress() == hdwAdd.hubaddress() );
	assert( channelTranslationTable_itr->second.fednumber() != hdwAdd.fednumber() || channelTranslationTable_itr->second.fedchannel() != hdwAdd.fedchannel() );
      }
    
    
     
      if ( foundChannel == false ){
	channelTranslationTable_[aChannel] = hdwAdd;
      }

    }
  }
  while (!in.eof());
  in.close();


  const std::map<unsigned int, std::set<unsigned int> > fedsAndChannels=getFEDsAndChannels();


  std::vector<PixelROCName> tmp(24);

  std::map<unsigned int, std::map<unsigned int, int > > counter;
    //       FED id                  FED channel

  std::map<unsigned int, std::map<unsigned int, int > > maxindex;

  std::map<PixelROCName,PixelHdwAddress>::const_iterator it=translationtable_.begin();
    
  for(;it!=translationtable_.end();it++){

    int index=it->second.fedrocnumber();

    unsigned int fednumber=it->second.fednumber();
    unsigned int fedchannel=it->second.fedchannel();

    std::vector<PixelROCName>& tmp= rocsFromFEDidAndChannel_[fednumber][fedchannel];
    
    if (tmp.size()==0){
      tmp.resize(24);
      counter[fednumber][fedchannel]=0;
      maxindex[fednumber][fedchannel]=0;
    }

    if (index>maxindex[fednumber][fedchannel]) maxindex[fednumber][fedchannel]=index;
    tmp[index]=it->first;
    counter[fednumber][fedchannel]++;

  }


  it=translationtable_.begin();
    
  for(;it!=translationtable_.end();it++){

    unsigned int fednumber=it->second.fednumber();
    unsigned int fedchannel=it->second.fedchannel();
    
    std::vector<PixelROCName>& tmp= rocsFromFEDidAndChannel_[fednumber][fedchannel];
    
    assert(counter[fednumber][fedchannel]==maxindex[fednumber][fedchannel]+1);

    tmp.resize(counter[fednumber][fedchannel]);

  }


}

std::ostream& operator<<(std::ostream& s, const PixelNameTranslation& table){

  //for (unsigned int i=0;i<table.translationtable_.size();i++){
  //	s << table.translationtable_[i]<<std::endl;
  //   }
  return s;

}

std::list<const PixelROCName*> PixelNameTranslation::getROCs() const
{
  std::list<const PixelROCName*> listOfROCs;
  for ( std::map<PixelROCName, PixelHdwAddress>::const_iterator translationTableEntry = translationtable_.begin();
	translationTableEntry != translationtable_.end(); ++translationTableEntry ) {
    listOfROCs.push_back(&(translationTableEntry->first));
  }

  return listOfROCs;
}

std::list<const PixelModuleName*> PixelNameTranslation::getModules() const
{
  std::list<const PixelModuleName*> listOfModules;
  for ( std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.begin(); channelTranslationTable_itr != channelTranslationTable_.end(); channelTranslationTable_itr++ )
    {
      bool foundOne = false;
      for ( std::list<const PixelModuleName*>::const_iterator listOfModules_itr = listOfModules.begin(); listOfModules_itr != listOfModules.end(); listOfModules_itr++ )
	{
	  if ( *(*listOfModules_itr) == channelTranslationTable_itr->first.module() )
	    {
	      foundOne = true;
	      break;
	    }
	}
      if (!foundOne) listOfModules.push_back( &(channelTranslationTable_itr->first.module()) );
    }

  return listOfModules;
}

std::set<PixelChannel> PixelNameTranslation::getChannels() const
{
  std::set<PixelChannel> channelSet;
  for ( std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.begin(); channelTranslationTable_itr != channelTranslationTable_.end(); channelTranslationTable_itr++ )
    {
      channelSet.insert(channelTranslationTable_itr->first);
    }
  return channelSet;
}

std::set<PixelChannel> PixelNameTranslation::getChannels(const PixelDetectorConfig& aDetectorConfig) const
{
  std::set<PixelChannel> channelSet;
  for ( std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.begin(); channelTranslationTable_itr != channelTranslationTable_.end(); channelTranslationTable_itr++ )
    {
      if ( aDetectorConfig.containsModule(channelTranslationTable_itr->first.module()) ) channelSet.insert(channelTranslationTable_itr->first);
    }
  return channelSet;
}

const PixelHdwAddress* PixelNameTranslation::getHdwAddress(const PixelROCName& aROC) const{

  static std::string mthn = "[PixelNameTranslation::getHdwAddress()]\t\t    " ;
  std::map<PixelROCName,PixelHdwAddress>::const_iterator it=
    translationtable_.find(aROC);

  if (it==translationtable_.end()){
    std::cout<< __LINE__ << "]\t" << mthn << "Could not look up ROC: " << aROC << std::endl;
    assert(0);
  }
    
  return &(it->second);

}

// Added for Debbie (used there only) to allow integrity checks (Dario)
bool PixelNameTranslation::checkROCExistence(const PixelROCName& aROC) const{

  std::string mthn = "[PixelNameTranslation::checkROCExistence()]\t\t    " ;
  if (translationtable_.find(aROC)==translationtable_.end()) return false ;
  return true ;
}

const bool PixelNameTranslation::checkFor(const PixelROCName& aROC) const{ 
  if (translationtable_.find(aROC)==translationtable_.end())
    {
      return false ;
    }
  else 
    {
      return true ;
    }
  }

const PixelHdwAddress& PixelNameTranslation::getHdwAddress(const PixelChannel& aChannel) const
{
  std::map<PixelChannel, PixelHdwAddress >::const_iterator channelHdwAddress_itr = channelTranslationTable_.find(aChannel);
  assert( channelHdwAddress_itr != channelTranslationTable_.end() );
  return channelHdwAddress_itr->second;
}

const PixelHdwAddress& PixelNameTranslation::firstHdwAddress(const PixelModuleName& aModule) const
{
        std::string mthn = "[PixelNameTranslation::firstHdwAddress()]\t\t    " ;
	std::set<PixelChannel> channelsOnModule = getChannelsOnModule(aModule);
	if (channelsOnModule.size() == 0 ){
	  cout << __LINE__ << "]\t" << mthn << "module=" << aModule << " has zero channels!" << endl;
	  cout << __LINE__ << "]\t" << mthn << "Will terminate" << endl;
	  ::abort();
	}
	std::set<PixelChannel>::const_iterator firstChannel = channelsOnModule.begin();
	assert( firstChannel != channelsOnModule.end() );
	return getHdwAddress( *firstChannel );
}

const PixelChannel& PixelNameTranslation::getChannelForROC(const PixelROCName& aROC) const
{
  std::map<PixelROCName,PixelHdwAddress>::const_iterator foundEntry = translationtable_.find(aROC);
  assert( foundEntry != translationtable_.end() );
  return getChannelFromHdwAddress( foundEntry->second );
}

std::set< PixelChannel > PixelNameTranslation::getChannelsOnModule(const PixelModuleName& aModule) const
{
  std::set< PixelChannel > returnThis;
  for ( std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.begin(); channelTranslationTable_itr != channelTranslationTable_.end(); channelTranslationTable_itr++ )
    {
      if ( channelTranslationTable_itr->first.module() == aModule ) returnThis.insert(channelTranslationTable_itr->first);
    }
  assert( returnThis.size() <= 2 );
  return returnThis;
}


const std::vector<PixelROCName>& PixelNameTranslation::getROCsFromFEDChannel(unsigned int fednumber, 
								      unsigned int fedchannel) const{

  std::map<unsigned int, std::map<unsigned int, std::vector<PixelROCName> > >::const_iterator it=rocsFromFEDidAndChannel_.find(fednumber);

  assert(it!=rocsFromFEDidAndChannel_.end());

  std::map<unsigned int, std::vector<PixelROCName> >::const_iterator it2=it->second.find(fedchannel);

  assert(it2!=it->second.end());

  return it2->second;

}


//Will return ROC names sorted by FED readout order.
//This (private) method will be called once to build this list
//when the data is read in.

std::vector<PixelROCName> PixelNameTranslation::buildROCsFromFEDChannel(unsigned int fednumber, 
									unsigned int fedchannel) const{

  std::vector<PixelROCName> tmp(24);

  int counter=0;        

  int maxindex=0;

  std::map<PixelROCName,PixelHdwAddress>::const_iterator it=translationtable_.begin();
    
  for(;it!=translationtable_.end();it++){

    if (it->second.fednumber()==fednumber&&
	it->second.fedchannel()==fedchannel){
      int index=it->second.fedrocnumber();
      if (index>maxindex) maxindex=index;
      //std::cout << "Found one:"<<index<<" "<<it->first<<std::endl;
      tmp[index]=it->first;
      counter++;
    }

  }

  assert(counter==maxindex+1);

  tmp.resize(counter);

  return tmp;

}


bool PixelNameTranslation::ROCNameFromFEDChannelROCExists(unsigned int fednumber, 
                                                          unsigned int channel, 
                                                          unsigned int roc) const {


  PixelHdwAddress tmp(0,0,0,0,0,0,fednumber,channel,roc);

  return (fedlookup_.find(tmp)!=fedlookup_.end());

}


PixelROCName PixelNameTranslation::ROCNameFromFEDChannelROC(unsigned int fednumber, 
							    unsigned int channel, 
							    unsigned int roc) const {


  std::string mthn = "[PixelNameTranslation::ROCNameFromFEDChannelROC()]\t\t    " ;
  PixelHdwAddress tmp(0,0,0,0,0,0,fednumber,channel,roc);

  std::map<PixelHdwAddress,PixelROCName,PixelHdwAddress>::const_iterator it1=fedlookup_.find(tmp);

  if (it1!=fedlookup_.end()){
    return it1->second;
  }

  std::cout << __LINE__ << "]\t" << mthn << "could not find ROCName "
	    << " for FED#" << fednumber << " chan=" << channel << " roc#=" << roc << std::endl;

  assert(0);

  PixelROCName tmp1;

  return tmp1;

}

PixelChannel PixelNameTranslation::ChannelFromFEDChannel(unsigned int fednumber, unsigned int fedchannel) const
{
        std::string mthn = "[PixelNameTranslation::ChannelFromFEDChannel()]\t\t    " ;
	std::map<PixelChannel,PixelHdwAddress>::const_iterator toReturn;
	bool foundOne = false;
	for(std::map<PixelChannel,PixelHdwAddress>::const_iterator it=channelTranslationTable_.begin(); it!=channelTranslationTable_.end();it++)
	{
		if (it->second.fednumber()==fednumber && it->second.fedchannel()==fedchannel)
		{
			if ( foundOne )
			{
				std::cout << __LINE__ << "]\t" << mthn 
				          << "ERROR: multiple channels on FED#" << fednumber << ", chan=" << fedchannel << std::endl;
				assert(0);
			}
			else
			{
				toReturn = it;
				foundOne = true;
			}
		}
	}
	
	if ( !foundOne )
	{
		std::cout << __LINE__ << "]\t" << mthn 
		          << "ERROR: no channel found for FED#" << fednumber << ", chan=" << fedchannel << std::endl;
		assert(0);
	}
	
	return toReturn->first;
}

bool PixelNameTranslation::FEDChannelExist(unsigned int fednumber, unsigned int fedchannel) const
{
        std::string mthn = "[PixelNameTranslation::FEDChannelExist()]\t\t    " ;
	std::map<PixelChannel,PixelHdwAddress>::const_iterator toReturn;
	bool foundOne = false;
	for(std::map<PixelChannel,PixelHdwAddress>::const_iterator it=channelTranslationTable_.begin(); it!=channelTranslationTable_.end();it++)
	{
		if (it->second.fednumber()==fednumber && it->second.fedchannel()==fedchannel)
		{
			if ( foundOne )
			{
				std::cout << __LINE__ << "]\t" << mthn 
				          << "ERROR: multiple channels on FED#" << fednumber << ", chan=" << fedchannel << std::endl;
				assert(0);
			}
			else
			{
				toReturn = it;
				foundOne = true;
			}
		}
	}
	return foundOne;
}

const PixelChannel& PixelNameTranslation::getChannelFromHdwAddress(const PixelHdwAddress& aHdwAddress) const
{
// modified by MR on 30-01-2008 10:38:22
  std::string mthn = "[PixelNameTranslation::getChannelFromHdwAddress()]\t\t    " ;

  std::map<PixelHdwAddress, PixelChannel >::const_iterator it=
    hdwTranslationTable_.find(aHdwAddress);

  if (it==hdwTranslationTable_.end()){
    std::cout << __LINE__ << "]\t" << mthn 
	      << "ERROR: no channel found for hardware address " << aHdwAddress << std::endl;
    assert(0);
  }

  return it->second;

  /*
    for ( std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.begin(); 
  	channelTranslationTable_itr != channelTranslationTable_.end(); channelTranslationTable_itr++ )
     {
      if ( aHdwAddress |= channelTranslationTable_itr->second )
  	{
  	  return channelTranslationTable_itr->first;
  	}
     }
// modified by MR on 30-01-2008 13:56:34
// if you get here then there was NO match on the previous loop!!
  std::cout << __LINE__ << "]\t" << mthn 
            << "ERROR: no channel found for hardware address " << aHdwAddress << std::endl;
  assert(0);
  */
}

void PixelNameTranslation::writeASCII(std::string dir) const {

  std::string mthn = "[PixelNameTranslation::writeASCII()]\t\t\t    " ;
  if (dir!="") dir+="/";
  std::string filename=dir+"translation.dat";

  std::ofstream out(filename.c_str());
  
  //std::cout << "[PixelNameTranslation::writeASCII()]\t\tfilename: " 
  //	    << filename 
  //	    << " status: " 
  //	    << out 
  //	    << "   " 
  //	    << out.is_open() 
  //	    <<endl ;

  out << "# name                          TBMchannel  FEC      mfec  mfecchannel hubaddress portadd rocid     FED     channel     roc#"<<endl;

  std::map<PixelROCName,PixelHdwAddress>::const_iterator iroc=translationtable_.begin();

  for (;iroc!=translationtable_.end();++iroc) {
  
    // Find the PixelChannel for this ROC, in order to get the TBM channel.
    std::string TBMChannel = getChannelFromHdwAddress(iroc->second).TBMChannelString();
  
    out << iroc->first.rocname()<<"       "
	<< TBMChannel<<"       "
	<< iroc->second.fecnumber()<<"       "
	<< iroc->second.mfec()<<"       "
	<< iroc->second.mfecchannel()<<"       "
	<< iroc->second.hubaddress()<<"       "
	<< iroc->second.portaddress()<<"       "
	<< iroc->second.rocid()<<"         "
	<< iroc->second.fednumber()<<"       "
	<< iroc->second.fedchannel()<<"       "
	<< iroc->second.fedrocnumber()
	<< endl;
  }



  out.close();

}

//=============================================================================================
void PixelNameTranslation::writeXMLHeader(pos::PixelConfigKey key, 
                                  	  int version, 
                                  	  std::string path, 
                                  	  std::ofstream *outstream,
                                  	  std::ofstream *out1stream,
                                  	  std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelNameTranslation:::writeXMLHeader()]\t\t\t    " ;
  std::stringstream fullPath ;
  fullPath << path << "/Pixel_NameTranslation_" << PixelTimeFormatter::getmSecTime() << ".xml" ;
  cout << __LINE__ << "]\t" << mthn << "Writing to: " << fullPath.str() << endl ;
  
  outstream->open(fullPath.str().c_str()) ;
  *outstream << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"			 	     << endl ;
  *outstream << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 		 	             << endl ;
  *outstream << " <HEADER>"								         	     << endl ;
  *outstream << "  <TYPE>"								         	     << endl ;
  *outstream << "   <EXTENSION_TABLE_NAME>PIXEL_NAME_TRANSLATION</EXTENSION_TABLE_NAME>"          	     << endl ;
  *outstream << "   <NAME>Pixel Name Translation</NAME>"				         	     << endl ;
  *outstream << "  </TYPE>"								         	     << endl ;
  *outstream << "  <RUN>"								         	     << endl ;
  *outstream << "   <RUN_TYPE>Pixel Name Translation</RUN_TYPE>" 		                             << endl ;
  *outstream << "   <RUN_NUMBER>1</RUN_NUMBER>"					         	             << endl ;
  *outstream << "   <RUN_BEGIN_TIMESTAMP>" << pos::PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << endl ;
  *outstream << "   <LOCATION>CERN P5</LOCATION>"                                                            << endl ; 
  *outstream << "  </RUN>"								         	     << endl ;
  *outstream << " </HEADER>"								         	     << endl ;
  *outstream << "  "								         	             << endl ;
  *outstream << " <DATA_SET>"								         	     << endl ;
  *outstream << "  <PART>"                                                                                   << endl ;
  *outstream << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"                                                 << endl ;
  *outstream << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"                                              << endl ;
  *outstream << "  </PART>"                                                                                  << endl ;
  *outstream << "  <VERSION>"             << version      << "</VERSION>"				     << endl ;
  *outstream << "  <COMMENT_DESCRIPTION>" << getComment() << "</COMMENT_DESCRIPTION>"			     << endl ;
  *outstream << "  <INITIATED_BY_USER>"   << getAuthor()  << "</INITIATED_BY_USER>"			     << endl ;
  *outstream << "  "								         	             << endl ;

}  

//=============================================================================================
void PixelNameTranslation::writeXML(std::ofstream *outstream,
                                    std::ofstream *out1stream,
                                    std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelNameTranslation::writeXML()]\t\t\t    " ;
  
  std::map<PixelROCName,PixelHdwAddress>::const_iterator iroc=translationtable_.begin();

  for (;iroc!=translationtable_.end();++iroc) 
    {
      // Find the PixelChannel for this ROC, in order to get the TBM channel.
      std::string TBMChannel = getChannelFromHdwAddress(iroc->second).TBMChannelString();

      *outstream << "  <DATA>" 								 	      	     << endl ;
      *outstream << "   <ROC_NAME>"     << iroc->first.rocname()	<< "</ROC_NAME>"	      	     << endl ;
      *outstream << "   <TBM_MODE>"     << TBMChannel	                << "</TBM_MODE>"	      	     << endl ;
      *outstream << "   <PXLFEC_NAME>"  << iroc->second.fecnumber()	<< "</PXLFEC_NAME>"	      	     << endl ;
      *outstream << "   <MFEC_POSN>"    << iroc->second.mfec() 	<< "</MFEC_POSN>"	 	      	     << endl ;
      *outstream << "   <MFEC_CHAN>"    << iroc->second.mfecchannel()  << "</MFEC_CHAN>"	      	     << endl ;
      *outstream << "   <HUB_ADDRS>"    << iroc->second.hubaddress()	<< "</HUB_ADDRS>"	      	     << endl ;
      *outstream << "   <PORT_NUM>"     << iroc->second.portaddress()  << "</PORT_NUM>" 	      	     << endl ;
      *outstream << "   <ROC_I2C_ADDR>" << iroc->second.rocid()	<< "</ROC_I2C_ADDR>"	 	      	     << endl ;
      *outstream << "   <PXLFED_NAME>"  << iroc->second.fednumber()	<< "</PXLFED_NAME>"	      	     << endl ;
      *outstream << "   <FED_CHAN>"     << iroc->second.fedchannel()	<< "</FED_CHAN>" 	      	     << endl ;
      *outstream << "   <FED_ROC_NUM>"  << iroc->second.fedrocnumber() << "</FED_ROC_NUM>"	      	     << endl ;
      *outstream << "  </DATA>"	 							 	      	     << endl ;
      *outstream << ""								         	      	     << endl ;
    }
}

//=============================================================================================
void PixelNameTranslation::writeXMLTrailer(std::ofstream *outstream,
                                  	   std::ofstream *out1stream,
                                  	   std::ofstream *out2stream) const 
{
  std::string mthn = "[PixelNameTranslation::writeXMLTrailer()]\t\t\t    " ;
  
  *outstream << " </DATA_SET>" 						    	 	              	     << endl ;
  *outstream << "</ROOT> "								              	     << endl ;

  outstream->close() ;
}

//=============================================================================================
void PixelNameTranslation::writeXML(pos::PixelConfigKey key, int version, std::string path) const {
  std::string mthn = "[PixelNameTranslation::writeXML]\t\t\t    " ;
  std::stringstream fullPath ;

  fullPath << path << "/Pixel_NameTranslation.xml" ;
  cout << __LINE__ << "]\t" << mthn << "Writing to: " << fullPath.str()  << endl ;
  
  std::ofstream out(fullPath.str().c_str()) ;

  out << "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"				      << endl ;
  out << "<ROOT xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>" 			      << endl ;
  out << ""											      << endl ;
  out << " <HEADER>"										      << endl ;
  out << "  <HINTS mode='only-det-root' />"                                                           << endl ; 
  out << "  <TYPE>"										      << endl ;
  out << "   <EXTENSION_TABLE_NAME>PIXEL_NAME_TRANSLATION</EXTENSION_TABLE_NAME>"		      << endl ;
  out << "   <NAME>Pixel Name Translation</NAME>"						      << endl ;
  out << "  </TYPE>"										      << endl ;
  out << "  <RUN>"										      << endl ;
  out << "   <RUN_TYPE>Pixel Name Translation</RUN_TYPE>" 					      << endl ;
  out << "   <RUN_NUMBER>1</RUN_NUMBER>"							      << endl ;
  out << "   <RUN_BEGIN_TIMESTAMP>" << pos::PixelTimeFormatter::getTime() << "</RUN_BEGIN_TIMESTAMP>" << endl ;
  out << "   <COMMENT_DESCRIPTION>Test of Name Translation xml</COMMENT_DESCRIPTION>"		      << endl ;
  out << "   <LOCATION>CERN TAC</LOCATION>"							      << endl ;
  out << "   <INITIATED_BY_USER>Dario Menasce</INITIATED_BY_USER>"				      << endl ;
  out << "  </RUN>"										      << endl ;
  out << " </HEADER>"										      << endl ;
  out << ""											      << endl ;
  out << " <DATA_SET>"  									      << endl ;
  out << "  <VERSION>" << version << "</VERSION>"						      << endl ;
  out << "  <PART>"										      << endl ;
  out << "   <NAME_LABEL>CMS-PIXEL-ROOT</NAME_LABEL>"						      << endl ;
  out << "   <KIND_OF_PART>Detector ROOT</KIND_OF_PART>"					      << endl ;
  out << "  </PART>"										      << endl ;
  out << ""											      << endl ;

  std::map<PixelROCName,PixelHdwAddress>::const_iterator iroc=translationtable_.begin();

  for (;iroc!=translationtable_.end();++iroc) 
    {
      // Find the PixelChannel for this ROC, in order to get the TBM channel.
      std::string TBMChannel = getChannelFromHdwAddress(iroc->second).TBMChannelString();

      out << "  <DATA>" 								 	      << endl ;
      out << "   <PXLFEC_NAME>"  << iroc->second.fecnumber()	<< "</PXLFEC_NAME>"	 	      << endl ;
      out << "   <MFEC_POSN>"    << iroc->second.mfec() 	<< "</MFEC_POSN>"	 	      << endl ;
      out << "   <MFEC_CHAN>"    << iroc->second.mfecchannel()  << "</MFEC_CHAN>"	 	      << endl ;
      out << "   <HUB_ADDRS>"    << iroc->second.hubaddress()	<< "</HUB_ADDRS>"	 	      << endl ;
      out << "   <PORT_NUM>"     << iroc->second.portaddress()  << "</PORT_NUM>" 	 	      << endl ;
      out << "   <ROC_I2C_ADDR>" << iroc->second.rocid()	<< "</ROC_I2C_ADDR>"	 	      << endl ;
      out << "   <PXLFED_NAME>"  << iroc->second.fednumber()	<< "</PXLFED_NAME>"	 	      << endl ;
      out << "   <FED_CHAN>"     << iroc->second.fedchannel()	<< "</FED_CHAN>" 	 	      << endl ;
      out << "   <FED_RO_NUM>"  << iroc->second.fedrocnumber() << "</FED_ROC_NUM>"	 	      << endl ;
      out << "  </DATA>"	 							 	      << endl ;
      out << ""								         	              << endl ;
    }
  out << " </DATA_SET> "								              << endl ;
  out << "</ROOT> "								         	      << endl ;
  out.close() ;
  assert(0) ;
}

const std::vector<PixelROCName>& 
PixelNameTranslation::getROCsFromChannel(const PixelChannel& aChannel) const {

  const PixelHdwAddress& channelHdwAddress = getHdwAddress(aChannel);
  return getROCsFromFEDChannel( channelHdwAddress.fednumber(), 
				channelHdwAddress.fedchannel() );

}

std::vector<PixelROCName> PixelNameTranslation::getROCsFromModule(const PixelModuleName& aModule) const
{
  std::vector<PixelROCName> returnThis;
	
  std::set<PixelChannel> channelsOnThisModule = getChannelsOnModule(aModule);
  for ( std::set<PixelChannel>::const_iterator channelsOnThisModule_itr = channelsOnThisModule.begin(); channelsOnThisModule_itr != channelsOnThisModule.end(); channelsOnThisModule_itr++ )
    {
      std::vector<PixelROCName> ROCsOnThisChannel = getROCsFromChannel( *channelsOnThisModule_itr );
      for ( std::vector<PixelROCName>::const_iterator ROCsOnThisChannel_itr = ROCsOnThisChannel.begin(); ROCsOnThisChannel_itr != ROCsOnThisChannel.end(); ROCsOnThisChannel_itr++ )
	{
	  returnThis.push_back(*ROCsOnThisChannel_itr);
	}
    }

  return returnThis;
}

//====================================================================================
// Added by Dario
bool PixelNameTranslation::ROCexists(PixelROCName theROC)
{
  if (translationtable_.find(theROC)==translationtable_.end()) {return false ;}
  return true ;
}


std::map <unsigned int, std::set<unsigned int> > PixelNameTranslation::getFEDsAndChannels() const {

  std::map <unsigned int, std::set<unsigned int> > tmp;

std::map<PixelChannel, PixelHdwAddress >::const_iterator 
  channelTranslationTable_itr = channelTranslationTable_.begin();

  for ( ; channelTranslationTable_itr != channelTranslationTable_.end(); 
          channelTranslationTable_itr++ ) {

    unsigned int fednumber=channelTranslationTable_itr->second.fednumber();
    unsigned int fedchannel=channelTranslationTable_itr->second.fedchannel();

    tmp[fednumber].insert(fedchannel);

  }

  return tmp;
  
}
