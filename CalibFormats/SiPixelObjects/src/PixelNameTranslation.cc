//
// This class stores the name and related
// hardware mapings for a ROC 
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <assert.h>

using namespace pos;
using namespace std;


PixelNameTranslation::PixelNameTranslation(std::vector< std::vector<std::string> > &tableMat):PixelConfigBase(" "," "," "){
  std::vector< std::string > ins = tableMat[0];
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  colNames.push_back("CONFIG_KEY_ID" );//0
  colNames.push_back("CONFG_KEY"     );//1
  colNames.push_back("VERSION"       );//2
  colNames.push_back("KIND_OF_COND"  );//3
  colNames.push_back("SERIAL_NUMBER" );//4
  colNames.push_back("ROC_NAME"      );//5
  colNames.push_back("PXLFEC_NAME"   );//6
  colNames.push_back("MFEC_POSN"     );//7
  colNames.push_back("MFEC_CHAN"     );//8
  colNames.push_back("HUB_ADDRS"     );//9
  colNames.push_back("PORT_NUM"      );//10
  colNames.push_back("ROC_I2C_ADDR"  );//11
  colNames.push_back("PXLFED_NAME"   );//12
  colNames.push_back("FED_CHAN"      );//13
  colNames.push_back("FED_ROC_NUM"   );//14


  for(unsigned int c = 0 ; c < ins.size() ; c++){
    for(unsigned int n=0; n<colNames.size(); n++){
      if(tableMat[0][c] == colNames[n]){
	colM[colNames[n]] = c;
	break;
      }
    }
  }//end for
  for(unsigned int n=0; n<colNames.size(); n++){
    if(colM.find(colNames[n]) == colM.end()){
      std::cerr << "[PixelNameTranslation::PixelNameTranslation()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
      assert(0);
    }
  }
 

 for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix
   std::string rocname       = tableMat[r][colM["ROC_NAME"]] ;
   std::string TBMChannel = "A"; // assert(0); // need to add this to the input table
   tableMat[r][colM["PXLFEC_NAME"]].erase(0 , 8);//PIXFEC
   unsigned int fecnumber    = (unsigned int)atoi(tableMat[r][colM["PXLFEC_NAME"]].c_str());
   unsigned int mfec         = (unsigned int)atoi(tableMat[r][colM["MFEC_POSN"]].c_str());
   unsigned int mfecchannel  = (unsigned int)atoi(tableMat[r][colM["MFEC_CHAN"]].c_str());
   unsigned int hubaddress   = (unsigned int)atoi(tableMat[r][colM["HUB_ADDRS"]].c_str());
   unsigned int portaddress  = (unsigned int)atoi(tableMat[r][colM["PORT_NUM"]].c_str());
   unsigned int rocid	     = (unsigned int)atoi(tableMat[r][colM["ROC_I2C_ADDR"]].c_str());
   tableMat[r][colM["PXLFED_NAME"]].erase(0,7);//FED
   unsigned int fednumber    = (unsigned int)atoi(tableMat[r][colM["PXLFED_NAME"]].c_str());
   unsigned int fedchannel   = (unsigned int)atoi(tableMat[r][colM["FED_CHAN"]].c_str());
   unsigned int fedrocnumber = (unsigned int)atoi(tableMat[r][colM["FED_ROC_NUM"]].c_str());
	
	
   PixelROCName aROC(rocname);
   if (aROC.rocname()!=rocname){
     std::cout << "[PixelNameTranslation::PixelNameTranslation()]\tRocname:"<<rocname<<std::endl;
     std::cout << "[PixelNameTranslation::PixelNameTranslation()]\tParsed to:"<<aROC.rocname()<<std::endl;
     assert(0);
   }
   PixelHdwAddress hdwAdd(fecnumber,mfec,mfecchannel,
			  hubaddress,portaddress,
			  rocid,
			  fednumber,fedchannel,fedrocnumber);
   std::cout << aROC << std::endl;
   translationtable_[aROC]=hdwAdd;
   
   PixelModuleName aModule(rocname);
   
	    PixelChannel aChannel(aModule, TBMChannel);
	    // Look for this channel in channelTransaltionTable.  If it is found, check that the hardware address agrees.  If not, add it to the table.  Also, if another channel on that module is found, check that the FEC part agrees, and the FED part doesn't.
	    bool foundChannel = false;
	    for ( std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.begin(); channelTranslationTable_itr != channelTranslationTable_.end(); channelTranslationTable_itr++ )
	    {
	        if ( channelTranslationTable_itr->first == aChannel )
	        {
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
	    }
	    if ( foundChannel == false ) channelTranslationTable_[aChannel] = hdwAdd;
	    
  }//end for r
}//end contructor
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PixelNameTranslation::PixelNameTranslation(std::string filename):
  PixelConfigBase(" "," "," "){

  std::ifstream in(filename.c_str());
  
  if (!in.good()){
    std::cout << "[PixelNameTranslation::PixelNameTranslation()]\tCould not open: " << filename <<std::endl;
    assert(0);
  }
  else {
    std::cout << "[PixelNameTranslation::PixelNameTranslation()]\tReading from: "   << filename <<std::endl;
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
      if (aROC.rocname()!=rocname){
	std::cout << "[PixelNameTranslation::PixelNameTranslation()]\tRocname:"<<rocname<<std::endl;
	std::cout << "[PixelNameTranslation::PixelNameTranslation()]\tParsed to:"<<aROC.rocname()<<std::endl;
	assert(0);
      }

      if (ROCNameFromFEDChannelROCExists(fednumber,fedchannel,
					 fedrocnumber)){
	std::cout << "ROC with fednumber="<<fednumber
		  << " fedchannel="<<fedchannel
		  << " roc number="<<fedrocnumber
		  << " already exists"<<std::endl;
	std::cout << "Fix this inconsistency in the name translation"
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
      translationtable_[aROC]=hdwAdd;
	    
      PixelModuleName aModule(rocname);
      PixelChannel aChannel(aModule, TBMChannel);
      // Look for this channel in channelTransaltionTable.  If it is found, check that the hardware address agrees.  If not, add it to the table.  Also, if another channel on that module is found, check that the FEC part agrees, and the FED part doesn't.
      bool foundChannel = false;
      for ( std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.begin(); channelTranslationTable_itr != channelTranslationTable_.end(); channelTranslationTable_itr++ )
	{
	  if ( channelTranslationTable_itr->first == aChannel )
	    {
	      if (!(channelTranslationTable_itr->second |= hdwAdd)){
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
	}
      if ( foundChannel == false )	channelTranslationTable_[aChannel] = hdwAdd;
    }
  }
  while (!in.eof());
  in.close();

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

  if (translationtable_.find(aROC)==translationtable_.end()){
    std::cout<<"Could not look up ROC:"<<aROC<<std::endl;
    assert(0);
  }
    
  return &(translationtable_.find(aROC))->second;

}

const PixelHdwAddress& PixelNameTranslation::getHdwAddress(const PixelChannel& aChannel) const
{
  std::map<PixelChannel, PixelHdwAddress >::const_iterator channelHdwAddress_itr = channelTranslationTable_.find(aChannel);
  assert( channelHdwAddress_itr != channelTranslationTable_.end() );
  return channelHdwAddress_itr->second;
}

const PixelHdwAddress& PixelNameTranslation::firstHdwAddress(const PixelModuleName& aModule) const
{
	std::set<PixelChannel> channelsOnModule = getChannelsOnModule(aModule);
	assert( channelsOnModule.size() > 0 );
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


//Will return ROC names sorted by FED readout order.
std::vector<PixelROCName> PixelNameTranslation::getROCsFromFEDChannel(unsigned int fednumber, 
								      unsigned int fedchannel) const{

  //FIXME this should have a proper map to directly look up things in!

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

  //FIXME this should have a proper map to directly look up things in!

  std::map<PixelROCName,PixelHdwAddress>::const_iterator it=translationtable_.begin();
    
  for(;it!=translationtable_.end();it++){

    if (it->second.fednumber()==fednumber&&
        it->second.fedchannel()==channel&&
        it->second.fedrocnumber()==roc){
      return true;
    }

  }
  
  return false;
}


PixelROCName PixelNameTranslation::ROCNameFromFEDChannelROC(unsigned int fednumber, 
							    unsigned int channel, 
							    unsigned int roc) const {

  //FIXME this should have a proper map to directly look up things in!

  std::map<PixelROCName,PixelHdwAddress>::const_iterator it=translationtable_.begin();
    
  for(;it!=translationtable_.end();it++){

    if (it->second.fednumber()==fednumber&&
	it->second.fedchannel()==channel&&
	it->second.fedrocnumber()==roc){
      return it->first;
    }

  }

  std::cout << "PixelNameTranslation::ROCNameFromFEDChannelROC: could not find ROCName "
	    << " for FED#" << fednumber <<" chan=" << channel << " roc#=" << roc << std::endl;

  assert(0);

  PixelROCName tmp;

  return tmp;

}

PixelChannel PixelNameTranslation::ChannelFromFEDChannel(unsigned int fednumber, unsigned int fedchannel) const
{
	std::map<PixelChannel,PixelHdwAddress>::const_iterator toReturn;
	bool foundOne = false;
	for(std::map<PixelChannel,PixelHdwAddress>::const_iterator it=channelTranslationTable_.begin(); it!=channelTranslationTable_.end();it++)
	{
		if (it->second.fednumber()==fednumber && it->second.fedchannel()==fedchannel)
		{
			if ( foundOne )
			{
				std::cout << "ERROR: multiple channels on FED#" << fednumber <<", chan=" << fedchannel << std::endl;
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
		std::cout << "ERROR: no channel found for FED#" << fednumber <<", chan=" << fedchannel << std::endl;
		assert(0);
	}
	
	return toReturn->first;
}

const PixelChannel& PixelNameTranslation::getChannelFromHdwAddress(const PixelHdwAddress& aHdwAddress) const
{
// modified by MR on 30-01-2008 10:38:22
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
  std::cout << "[PixelNameTranslation::getChannelFromHdwAddress()]\tERROR: no channel found for hardware address " << aHdwAddress << std::endl;
  assert(0);
}

void PixelNameTranslation::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  std::string filename=dir+"translation.dat";

  std::ofstream out(filename.c_str());
  
  std::cout << "[PixelNameTranslation::writeASCII()]\t\tfilename: " 
	    << filename 
	    << " status: " 
	    << out 
	    << "   " 
	    << out.is_open() 
	    <<endl ;

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

std::vector<PixelROCName> PixelNameTranslation::getROCsFromChannel(const PixelChannel& aChannel) const
{
  const PixelHdwAddress& channelHdwAddress = getHdwAddress(aChannel);
  return getROCsFromFEDChannel( channelHdwAddress.fednumber(), channelHdwAddress.fedchannel() );
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
