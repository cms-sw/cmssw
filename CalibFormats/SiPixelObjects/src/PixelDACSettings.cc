//
// This class provide a base class for the
// pixel ROC DAC data for the pixel FEC configuration
//
//
//
//


#include "CalibFormats/SiPixelObjects/interface/PixelDACSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCDACSettings.h"
#include <fstream>
#include <iostream>
#include <ios>
#include <assert.h>
#include <map>
#include <sstream>

using namespace pos;

PixelDACSettings::PixelDACSettings(std::string filename):
  PixelConfigBase("","",""){


    if (filename[filename.size()-1]=='t'){

	std::ifstream in(filename.c_str());

	if (!in.good()){
	    std::cout << "Could not open:"<<filename<<std::endl;
	    assert(0);
	}
	else {
	    // std::cout << "Opened:"<<filename<<std::endl;
	}
	
	
	dacsettings_.clear();
	
	std::string tag;
	
	in >> tag;

	while (!in.eof()){
	    
	    
	    PixelROCName rocid(in);

//	    std::cout << "[PixelDACSettings::PixelDACSettings()] DAC setting ROC id:"<<rocid<<std::endl;
	    
	    PixelROCDACSettings tmp;

	    tmp.read(in,rocid);
	    
//	    std::cout << "[PixelDACSettings::PixelDACSettings()] DACSetting: " << std::endl << tmp << std::endl ;
	    dacsettings_.push_back(tmp);
	    
	    in >> tag;
	    
	    assert(dacsettings_.size()<100);
	    
	}
	
	in.close();

    }
    else{

	std::ifstream in(filename.c_str(),std::ios::binary);

	if (!in.good()){
	    std::cout << "Could not open:"<<filename<<std::endl;
	    assert(0);
	}
	else {
	    std::cout << "Opened:"<<filename<<std::endl;
	}

        char nchar;

	in.read(&nchar,1);
	
       	std::string s1;

        //wrote these lines of code without ref. needs to be fixed
	for(int i=0;i< nchar; i++){
	    char c;
	    in >>c;
	    s1.push_back(c);
	}

	//std::cout << "READ ROC name:"<<s1<<std::endl;

	dacsettings_.clear();


	while (!in.eof()){

	    //std::cout << "PixelDACSettings::PixelDACSettings read s1:"<<s1<<std::endl;

	    PixelROCName rocid(s1);

	    //td::cout << "PixelDACSettings::PixelDACSettings read rocid:"<<rocid<<std::endl;
	    
	    PixelROCDACSettings tmp;
      
	    tmp.readBinary(in, rocid);

	    dacsettings_.push_back(tmp);


	    in.read(&nchar,1);

	    s1.clear();

	    if (in.eof()) continue;

	    //wrote these lines of code without ref. needs to be fixed
	    for(int i=0;i< nchar; i++){
		char c;
		in >>c;
		s1.push_back(c);
	    }


	}

	in.close();



    }


    //std::cout << "Read dac settings for "<<dacsettings_.size()<<" ROCs"<<std::endl;


}
// modified by MR on 10-01-2008 14:48:19
PixelDACSettings::PixelDACSettings(PixelROCDACSettings &rocname):
  PixelConfigBase("","","") {
  dacsettings_.push_back(rocname) ;
}

// modified by MR on 24-01-2008 14:27:35a
void PixelDACSettings::addROC(PixelROCDACSettings &rocname)
{
  dacsettings_.push_back(rocname) ;
}

PixelDACSettings::PixelDACSettings(std::vector< std::vector<std::string> > &tableMat): PixelConfigBase("","","")
{
 
  std::multimap<std::string,std::pair<std::string,int > > pDSM;
//  std::stringstream currentRocName;
  std::vector< std::string > ins = tableMat[0];
  std::string dacName;
  int dacValue;
  std::map<std::string , int > colM;
  std::vector<std::string > colNames;
  colNames.push_back("CONFIG_KEY_ID");//0
  colNames.push_back("CONFG_KEY");//1
  colNames.push_back("VERSION");//2
  colNames.push_back("KIND_OF_COND");
  colNames.push_back("ROC_NAME");
  colNames.push_back("HUB_ADDRS");
  colNames.push_back("PORT_NUMBER");
  colNames.push_back("I2C_ADDR");
  colNames.push_back("GEOM_ROC_NUM");
  colNames.push_back("DAC_NAME");
  colNames.push_back("DAC_VALUE");

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
      std::cerr << "[PixelDACSettings::PixelDACSettings()]\tCouldn't find in the database the column with name " << colNames[n] << std::endl;
      assert(0);
    }
  }

	
  for(unsigned int r = 1 ; r < tableMat.size() ; r++){    //Goes to every row of the Matrix
  
   // currentRocName.str("");
    
   // currentRocName << tableMat[r][colM["NAME"]] ; 
   
   
    //currentRocName << "FPix_BmI_D" << tableMat[r][colM["HDISK_POSN"]]                 
	//	   << "_BLD"       << tableMat[r][colM["BLD_POSN"]]                  
	//	   << "_PNL"       << tableMat[r][colM["PANEL_POSITION"]]            
	//	   << "_PLQ"       << tableMat[r][colM["PLAQ_POS"]]                 
	//	   << "_ROC"       << tableMat[r][colM["ROC_POSN"]];                
		   
    dacName  = tableMat[r][colM["DAC_NAME"]];
    dacValue = atoi(tableMat[r][colM["DAC_VALUE"]].c_str());
    
    pDSM.insert(std::pair<std::string,std::pair<std::string,int> >(tableMat[r][colM["ROC_NAME"]],std::pair<std::string,int>(dacName,dacValue)));
    
  }//end for r
  
  dacsettings_.clear();
  std::string currentRocName2 = "";
  for(std::multimap<std::string,std::pair<std::string,int> >::iterator tableMapIt=pDSM.begin(); tableMapIt!= pDSM.end(); tableMapIt++){
    if(currentRocName2 != tableMapIt->first){
//       std::cout << tableMapIt->first << std::endl;
      currentRocName2 = tableMapIt->first;
      PixelROCName rocid(currentRocName2);
      
     // std::cout << "DAC setting ROC id:"<<rocid<<std::endl;
  
      PixelROCDACSettings tmp(rocid);
      
//       tmp.read(in,rocid);
	    
      dacsettings_.push_back(tmp);
    }//end if
    dacsettings_[dacsettings_.size()-1].setDac(tableMapIt->second.first,tableMapIt->second.second);
  }//end for 
  
  
     for(unsigned int w = 0 ; w < dacsettings_.size() ; w++)
  {
  
   PixelROCDACSettings tmp2 = dacsettings_[w];
//   std::cout<<tmp2<<std::endl;
  }   
//  std::cout<<"Number of ROCs in the PixelDACSettings::PixelDACSettings(vector <vector<string> >):"<<dacsettings_.size()<<std::endl; 
//  std::cout << "[PixelDACSettings::PixelDACSettings(std::vector)] before end of constructor" << std::endl ;
}//end PDSMatrix constructor
//end added by Umesh
PixelROCDACSettings PixelDACSettings::getDACSettings(int ROCId) const {

  return dacsettings_[ROCId];

}
 
PixelROCDACSettings* PixelDACSettings::getDACSettings(PixelROCName name){

  for(unsigned int i=0;i<dacsettings_.size();i++){
    if (dacsettings_[i].getROCName()==name) return &(dacsettings_[i]);
  }

  return 0;

}
 
void PixelDACSettings::writeBinary(std::string filename) const {

    std::ofstream out(filename.c_str(),std::ios::binary);

    for(unsigned int i=0;i<dacsettings_.size();i++){
	dacsettings_[i].writeBinary(out);
    }

}


void PixelDACSettings::writeASCII(std::string dir) const {

  PixelModuleName module(dacsettings_[0].getROCName().rocname());

  std::string filename=dir+"/ROC_DAC_module_"+module.modulename()+".dat";
  std::cout << "[PixelDACSettings::writeASCII()] Writing to file " << filename << std::endl ; 
  std::ofstream out(filename.c_str());
  
  for(unsigned int i=0;i<dacsettings_.size();i++){
    dacsettings_[i].writeASCII(out);
  }

}



void PixelDACSettings::generateConfiguration(PixelFECConfigInterface* pixelFEC,
					     PixelNameTranslation* trans) const{

    std::vector<unsigned int> dacs;

    for(unsigned int i=0;i<dacsettings_.size();i++){

	dacsettings_[i].getDACs(dacs);

	PixelHdwAddress theROC=*(trans->getHdwAddress(dacsettings_[i].getROCName()));

/*     Now moved to PixelDACSettings

	if (i==0) {

	    //For now, set the TBM initialization here --FIXME--
	    //As implemented below these methods will be called
	    //more than once per FEC

	    int mfec=theROC.mfec();
	    int mfecchannel=theROC.mfecchannel();
	    int tbmchannel=14; //??
	    int hubaddress=theROC.hubaddress();

	    pixelFEC->injectrsttbm(mfec, 1);
	    pixelFEC->injectrstroc(mfec,1);
	    pixelFEC->clockphaseselect(mfec,0);
	    pixelFEC->enablecallatency(mfec,0);
	    pixelFEC->disableexttrigger(mfec,0);
	    pixelFEC->injecttrigger(mfec,0);
	    pixelFEC->callatencycount(mfec,79);
	    //pixelFEC->synccontrolregister(mfec);

	    //setting speed to 40MHz
	    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 0, 1, 0);
	    //set mode (sync/clear evt counter/pre-cal) to pre-calibrate
	    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 1, 0xc0, 0);
	    //Reset TBM and reset ROC
	    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 2, 0x14, 0);
	    //TBM Analog input amplifier bias
	    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 5, 0x7f, 0);
	    //TBM Analog output driver bias
	    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 6, 0x7f, 0);
	    //TBM output DAC gain
	    pixelFEC->tbmcmd(mfec, mfecchannel, tbmchannel, hubaddress, 4, 7, 150, 0);

	}

*/

	//Need to set readout speed (40MHz) and Vcal range (0-1800 mV) and enable the chip

	int controlreg=dacsettings_[i].getControlRegister();
	//std::cout << "ROC control reg to be set to: " <<  controlreg <<std::endl;

	pixelFEC->progdac(theROC.mfec(),
			  theROC.mfecchannel(),
			  theROC.hubaddress(),
			  theROC.portaddress(),
			  theROC.rocid(),
			  0xfd,
			  controlreg);

	pixelFEC->setAllDAC(theROC,dacs);

	// start with no pixels on for calibration
	pixelFEC->clrcal(theROC.mfec(), theROC.mfecchannel(), theROC.hubaddress(), theROC.portaddress(), theROC.rocid());

	// enable all the double columns
	for(int dcol=0;dcol<26;dcol++){
	    pixelFEC->dcolenable(theROC.mfec(),
				 theROC.mfecchannel(),
				 theROC.hubaddress(),
				 theROC.portaddress(),
				 theROC.rocid(),
				 dcol,
				 1);
	}


    }

} 


std::ostream& operator<<(std::ostream& s, const PixelDACSettings& dacs){

  s << dacs.getDACSettings(0) <<std::endl; 

  return s;

}

