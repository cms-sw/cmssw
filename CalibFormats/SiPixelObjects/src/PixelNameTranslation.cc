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
 colNames.push_back("ROC_NAME");//0
 colNames.push_back("PIXFEC");//1
 colNames.push_back("MFEC_POSN");//2
 colNames.push_back("MFEC_CHAN");//3
 colNames.push_back("HUB_ADDRS");//4
 colNames.push_back("PORT_NUMBER");//5
 colNames.push_back("ROC_POSN");//6
 colNames.push_back("PIXFED");//7
 colNames.push_back("FED_CHAN");//8
 colNames.push_back("FED_ROC_NUM");//9

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
   std::string rocname       = tableMat[r][colM[colNames[0]]];//ROC_NAME
   std::string TBMChannel = "A"; assert(0); // need to add this to the input table
   tableMat[r][colM[colNames[1]]].erase(0 , 8);//PIXFEC
   unsigned int fecnumber    = (unsigned int)atoi(tableMat[r][colM[colNames[1]]].c_str());//"PIXFEC"
   unsigned int mfec         = (unsigned int)atoi(tableMat[r][colM[colNames[2]]].c_str());//"MFEC_POSN"
   unsigned int mfecchannel  = (unsigned int)atoi(tableMat[r][colM[colNames[3]]].c_str());//"MFEC_CHAN"
   unsigned int hubaddress   = (unsigned int)atoi(tableMat[r][colM[colNames[4]]].c_str());//"HUB_ADDRS"
   unsigned int portaddress  = (unsigned int)atoi(tableMat[r][colM[colNames[5]]].c_str());//"PORT_NUMBER"
   unsigned int rocid        = (unsigned int)atoi(tableMat[r][colM[colNames[6]]].c_str());//"ROC_POSN"
   tableMat[r][colM[colNames[7]]].erase(0,7);//FED
   unsigned int fednumber    = (unsigned int)atoi(tableMat[r][colM[colNames[7]]].c_str());//"FED"	
   unsigned int fedchannel   = (unsigned int)atoi(tableMat[r][colM[colNames[8]]].c_str());//"FED_CHAN"
   unsigned int fedrocnumber = (unsigned int)atoi(tableMat[r][colM[colNames[9]]].c_str());//"FED_ROC_NUM"
	
	
   PixelROCName aROC(rocname);
   if (aROC.rocname()!=rocname){
     std::cout << "Rocname:"<<rocname<<std::endl;
     std::cout << "Parsed to:"<<aROC.rocname()<<std::endl;
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
	           assert( channelTranslationTable_itr->second.portaddress() == hdwAdd.portaddress() );
	           assert( channelTranslationTable_itr->second.hubaddress() == hdwAdd.hubaddress() );
	           assert( channelTranslationTable_itr->second.fednumber() != hdwAdd.fednumber() || channelTranslationTable_itr->second.fedchannel() != hdwAdd.fedchannel() );
	        }
	    }
	    if ( foundChannel == false ) channelTranslationTable_[aChannel] = hdwAdd;
	    
	// Fill moduleTranslationtable_ below
   
//    const PixelHdwAddress* aHdwAdd=getHdwAddress(aModule);
   const std::vector<PixelHdwAddress>& aHdwAdd(*getHdwAddress(aModule));
   if (&aHdwAdd==0){
//      std::cout << "Inserting new module:"<<aModule<<std::endl;
     std::vector<PixelHdwAddress> tmp;
     tmp.push_back(hdwAdd);
     moduleTranslationtable_[aModule]=tmp;
   }
   else{
     //std::cout << "Module:"<<aModule<<" already existing."<<std::endl;
     assert(aHdwAdd.size()<3);
     assert(aHdwAdd.size()>0);
     
     assert(aHdwAdd[0].fecnumber()==fecnumber);
     assert(aHdwAdd[0].mfec()==mfec);
     assert(aHdwAdd[0].mfecchannel()==mfecchannel);
     assert(aHdwAdd[0].hubaddress()==hubaddress);
     
     if (aHdwAdd.size()==1){
       if (!(aHdwAdd[0].fednumber()==fednumber&&
	     aHdwAdd[0].fedchannel()==fedchannel)){
	 moduleTranslationtable_[aModule].push_back(hdwAdd);
       }
     }
     else{
       assert((aHdwAdd[0].fednumber()==fednumber&&
	       aHdwAdd[0].fedchannel()==fedchannel)||
	      (aHdwAdd[1].fednumber()==fednumber&&
	       aHdwAdd[1].fedchannel()==fedchannel));
     }
   }
 }//end for r
}//end contructor
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PixelNameTranslation::PixelNameTranslation(std::string filename):
    PixelConfigBase(" "," "," "){

    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << "Could not open:"<<filename<<std::endl;
	assert(0);
    }
    else {
	std::cout << "Opened:"<<filename<<std::endl;
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
	      std::cout << "Rocname:"<<rocname<<std::endl;
	      std::cout << "Parsed to:"<<aROC.rocname()<<std::endl;
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
	           assert( channelTranslationTable_itr->second.portaddress() == hdwAdd.portaddress() );
	           assert( channelTranslationTable_itr->second.hubaddress() == hdwAdd.hubaddress() );
	           assert( channelTranslationTable_itr->second.fednumber() != hdwAdd.fednumber() || channelTranslationTable_itr->second.fedchannel() != hdwAdd.fedchannel() );
	        }
	    }
	    if ( foundChannel == false ) channelTranslationTable_[aChannel] = hdwAdd;
	    
	    // Fill moduleTranslationtable_
	    const std::vector<PixelHdwAddress>& aHdwAdd(*getHdwAddress(aModule));
	    if (&aHdwAdd==0){
	      //std::cout << "Inserting new module:"<<aModule<<std::endl;
	      std::vector<PixelHdwAddress> tmp;
	      tmp.push_back(hdwAdd);
	      moduleTranslationtable_[aModule]=tmp;
	    }
	    else{
	      //std::cout << "Module:"<<aModule<<" already existing."<<std::endl;
	      assert(aHdwAdd.size()<3);
	      assert(aHdwAdd.size()>0);

	      assert(aHdwAdd[0].fecnumber()==fecnumber);
	      assert(aHdwAdd[0].mfec()==mfec);
	      assert(aHdwAdd[0].mfecchannel()==mfecchannel);
	      assert(aHdwAdd[0].hubaddress()==hubaddress);
	      
	      if (aHdwAdd.size()==1){
                if (!(aHdwAdd[0].fednumber()==fednumber&&
		      aHdwAdd[0].fedchannel()==fedchannel)){
		  moduleTranslationtable_[aModule].push_back(hdwAdd);
		}
	      }
	      else{
                assert((aHdwAdd[0].fednumber()==fednumber&&
			aHdwAdd[0].fedchannel()==fedchannel)||
		       (aHdwAdd[1].fednumber()==fednumber&&
			aHdwAdd[1].fedchannel()==fedchannel));
	      }

	    }
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
	
	assert( listOfModules.size() == moduleTranslationtable_.size() );

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

const std::vector<PixelHdwAddress>* PixelNameTranslation::getHdwAddress(const PixelModuleName& aModule) const{

    if (moduleTranslationtable_.find(aModule)==moduleTranslationtable_.end()){
      //std::cout<<"Could not look up module:"<<aModule<<std::endl;
      //std::map<PixelModuleName,std::vector<PixelHdwAddress> >::const_iterator it=moduleTranslationtable_.begin();
      //for(;it!=moduleTranslationtable_.end();++it){
      //  std::cout << "Module name:"<<it->first<<std::endl;
      //}
      return 0;
    }
    
    return &(moduleTranslationtable_.find(aModule))->second;

}

const PixelHdwAddress& PixelNameTranslation::getHdwAddress(const PixelChannel& aChannel) const
{
	std::map<PixelChannel, PixelHdwAddress >::const_iterator channelHdwAddress_itr = channelTranslationTable_.find(aChannel);
	assert( channelHdwAddress_itr != channelTranslationTable_.end() );
	return channelHdwAddress_itr->second;
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

const PixelChannel& PixelNameTranslation::getChannelFromHdwAddress(const PixelHdwAddress& aHdwAddress) const
{
	for ( std::map<PixelChannel, PixelHdwAddress >::const_iterator channelTranslationTable_itr = channelTranslationTable_.begin(); channelTranslationTable_itr != channelTranslationTable_.end(); channelTranslationTable_itr++ )
	{
		if ( aHdwAddress |= channelTranslationTable_itr->second )
		{
			return channelTranslationTable_itr->first;
		}
	}
	std::cout << "ERROR: no channel found for hardware address " << aHdwAddress << std::endl;
	assert(0);
}

void PixelNameTranslation::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  std::string filename=dir+"translation.dat";

  std::ofstream out(filename.c_str());

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

std::vector<PixelROCName> PixelNameTranslation::getROCsFromModule(const PixelModuleName& aModule) const
{
	const std::vector<PixelHdwAddress>* hdwAddresses = getHdwAddress(aModule);

	std::vector<PixelROCName> returnThis;

	// Return empty vector if this module was not found in the configuration.
	if ( hdwAddresses == 0 ) return returnThis;

	for ( std::vector<PixelHdwAddress>::const_iterator hdwAddress_itr = hdwAddresses->begin(); hdwAddress_itr != hdwAddresses->end(); ++hdwAddress_itr)
	{
		unsigned int fednumber = (*hdwAddress_itr).fednumber();
		unsigned int fedchannel = (*hdwAddress_itr).fedchannel();
		std::vector<PixelROCName> ROCsAtThisHdwAddress = getROCsFromFEDChannel(fednumber, fedchannel);
		for ( std::vector<PixelROCName>::const_iterator ROCName_itr = ROCsAtThisHdwAddress.begin(); ROCName_itr != ROCsAtThisHdwAddress.end(); ++ROCName_itr )
		{
			returnThis.push_back(*ROCName_itr);
		}
	}

	return returnThis;
}
