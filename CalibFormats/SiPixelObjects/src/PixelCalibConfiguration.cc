//
// This class is a first attempt at writing a configuration
// object that will perform a calibration loop.
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelCalibConfiguration.h"
#include "CalibFormats/SiPixelObjects/interface/PixelDACNames.h"
#include <fstream>
#include <iostream>
#include <ios>
#include <assert.h>
#include <stdlib.h>

using namespace pos;


PixelCalibConfiguration::PixelCalibConfiguration(std::string filename):
  PixelCalibBase(), PixelConfigBase("","","") {

  _bufferData=true; 
  
  channelListBuilt_ = false;

    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << "Could not open:"<<filename<<std::endl;
	assert(0);
    }
    else {
	std::cout << "Opened:"<<filename<<std::endl;
    }

    std::string tmp;

    in >> tmp;

    if (tmp=="Mode:"){
      in >> mode_;
      std::cout << "PixelCalibConfiguration mode="<<mode_
		<< std::endl;

      assert(mode_=="FEDBaselineWithTestDACs"||
	     mode_=="FEDAddressLevelWithTestDACs"||
	     mode_=="FEDBaselineWithPixels"||
	     mode_=="AOHBias"||
	     mode_=="TBMUB"||
	     mode_=="ROCUBEqualization"||
	     mode_=="Iana"||
	     mode_=="FEDAddressLevelWithPixels"||
	     mode_=="GainCalibration"||
             mode_=="GainCalibrationWithSLink"||
	     mode_=="PixelAlive"||
             mode_=="PixelAliveWithSLink"||
	     mode_=="SCurve"||
             mode_=="SCurveWithSLink"||
	     mode_=="Delay25"||
	     mode_=="ClockPhaseCalibration"||
	     mode_=="TemperatureCalibration"||
	     mode_=="ThresholdCalDelayFIFO1"||
             mode_=="ThresholdCalDelay");
      in >>tmp;
    } else {
      mode_="FEDChannelOffsetPixel";
      std::cout << "PixelCalibCOnfiguration mode not set, is this an old file? "
		<< std::endl;
      assert(0);
    }

    singleROC_=false;
      
    if (tmp=="SingleROC") {
      singleROC_=true;
      in >> tmp;
    }

	// Read in parameters, if any.
	if (tmp=="Parameters:") {
		in >> tmp;
		while (tmp!="Rows:")
		{
			assert( !in.eof() );
			std::string paramName = tmp;
			in >> tmp; // tmp contains the parameter value
			parameters_[paramName] = tmp;
			in >> tmp; // tmp contains the next parameter's name, or "Rows:"
		}
	}

    assert(tmp=="Rows:");

    in >> tmp;

    std::vector <unsigned int> rows;
    while (tmp!="Cols:"){
	if (tmp=="|") {
	    rows_.push_back(rows);
            rows.clear();
	}
	else{
          if (tmp!="*"){
	    rows.push_back(atoi(tmp.c_str()));
          }
	}
	in >> tmp;
    }
    rows_.push_back(rows);
    rows.clear();
    
    in >> tmp;

    std::vector <unsigned int> cols;
    while ((tmp!="VcalLow:")&&(tmp!="VcalHigh:")&&
	   (tmp!="Vcal:")&&(tmp!="VcalHigh")&&(tmp!="VcalLow")){
	if (tmp=="|") {
	    cols_.push_back(cols);
            cols.clear();
	}
	else{
          if (tmp!="*"){
	    cols.push_back(atoi(tmp.c_str()));
          }
	}
	in >> tmp;
    }
    cols_.push_back(cols);
    cols.clear();

    highVCalRange_=true;

    if (tmp=="VcalLow") {
      highVCalRange_=false;
      in >> tmp;
    }

    if (tmp=="VcalHigh") {
      highVCalRange_=true;
      in >> tmp;
    }

    if (tmp=="VcalLow:") {
	highVCalRange_=false;
    }

    if ((tmp=="VcalLow:")||(tmp=="VcalHigh:")||(tmp=="Vcal:")){
      unsigned int  first,last,step;
      in >> first >> last >> step;
      unsigned int index=1;
      if (dacs_.size()>0) {
        index=dacs_.back().index()*dacs_.back().getNPoints();
      }
      PixelDACScanRange dacrange(pos::k_DACName_Vcal,first,last,step,index);
      dacs_.push_back(dacrange);
      in >> tmp;
    }
    else{

      //in >> tmp;
      while(tmp=="Scan:"){
        in >> tmp;
        unsigned int  first,last,step;
        in >> first >> last >> step;
        unsigned int index=1;
        if (dacs_.size()>0) {
          index=dacs_.back().index()*dacs_.back().getNPoints();
        }
        PixelDACScanRange dacrange(tmp,first,last,step,index);
        dacs_.push_back(dacrange);
        in >> tmp;
      }
      
      while (tmp=="Set:"){
        in >> tmp;
        unsigned int val;
        in >> val;
        unsigned int index=1;
        if (dacs_.size()>0) index=dacs_.back().index()*dacs_.back().getNPoints();
        PixelDACScanRange dacrange(tmp,val,val,1,index);
        dacs_.push_back(dacrange);
        in >> tmp;
      }
    }

    assert(tmp=="Repeat:");

    in >> ntrigger_;

    in >> tmp;

    bool buildROCListNow = false;
    if ( tmp=="Rocs:" ) buildROCListNow = true;
    else { assert(tmp=="ToCalibrate:"); buildROCListNow = false; }

    while (!in.eof())
    {
       tmp = "";
       in >> tmp;

       // added by F.Blekman to be able to deal with POS245 style calib.dat files in CMSSW
       // these files use the syntax: 
       // Rocs:
       // all

       if( tmp=="all" || tmp=="+" || tmp=="-" ){
	 buildROCListNow=false;
       }
       // end of addition by F.B.
	 
       if ( tmp=="" ) continue;
       rocListInstructions_.push_back(tmp);
    }

    in.close();
    
    rocAndModuleListsBuilt_ = false;
    if ( buildROCListNow )
    {
       std::set<PixelROCName> rocSet;
       for(std::vector<std::string>::iterator rocListInstructions_itr = rocListInstructions_.begin(); rocListInstructions_itr != rocListInstructions_.end(); rocListInstructions_itr++)
       {
          PixelROCName rocname(*rocListInstructions_itr);
          rocSet.insert(rocname);
       }
       buildROCAndModuleListsFromROCSet(rocSet);
    }
    
    return;

}


PixelCalibConfiguration::~PixelCalibConfiguration(){}

void PixelCalibConfiguration::buildROCAndModuleLists(const PixelNameTranslation* translation, const PixelDetectorConfig* detconfig)
{
	assert( translation != 0 );
	assert( detconfig != 0 );
	
	if ( rocAndModuleListsBuilt_ ) return;
	
	// Build the ROC set from the instructions.
	std::set<PixelROCName> rocSet;
	bool addNext = true;
	for(std::vector<std::string>::iterator rocListInstructions_itr = rocListInstructions_.begin(); rocListInstructions_itr != rocListInstructions_.end(); rocListInstructions_itr++)
	{
		std::string instruction = *rocListInstructions_itr;
		
		if ( instruction == "+" )
		{
			addNext = true;
			continue;
		}
		if ( instruction == "-" )
		{
			addNext = false;
			continue;
		}
		
		if ( instruction == "all" )
		{
			if ( addNext ) // add all ROCs in the configuration
			{
				const std::vector <PixelModuleName>& moduleList = detconfig->getModuleList();
				for ( std::vector <PixelModuleName>::const_iterator moduleList_itr = moduleList.begin(); moduleList_itr != moduleList.end(); moduleList_itr++ )
				{
					std::vector<PixelROCName> ROCsOnThisModule = translation->getROCsFromModule( *moduleList_itr );
					for ( std::vector<PixelROCName>::iterator ROCsOnThisModule_itr = ROCsOnThisModule.begin(); ROCsOnThisModule_itr != ROCsOnThisModule.end(); ROCsOnThisModule_itr++ )
					{
						rocSet.insert(*ROCsOnThisModule_itr);
					}
				}
			}
			else // remove all ROCs
			{
				rocSet.clear();
			}
			addNext = true;
			continue;
		}
		
		// Assume it's a ROC or module name.
		PixelModuleName modulename(instruction);
		
		// Skip if this module (or the module this ROC is on) isn't in the detector config.
		if ( !(detconfig->containsModule(modulename)) )
		{
			addNext = true;
			continue;
		}
		
		if ( modulename.modulename() == instruction ) // it's a module
		{
			std::vector<PixelROCName> ROCsOnThisModule = translation->getROCsFromModule( modulename );
			for ( std::vector<PixelROCName>::iterator ROCsOnThisModule_itr = ROCsOnThisModule.begin(); ROCsOnThisModule_itr != ROCsOnThisModule.end(); ROCsOnThisModule_itr++ )
			{
				if ( addNext ) rocSet.insert(*ROCsOnThisModule_itr);
				else           rocSet.erase( *ROCsOnThisModule_itr);
			}
			addNext = true;
			continue;
		}
		else // it's a ROC
		{
			PixelROCName rocname(instruction);
			if ( addNext )
			{
				// Only add this ROC if it's in the configuration.
				bool foundIt = false;
				std::list<const PixelROCName*> allROCs = translation->getROCs();
				for ( std::list<const PixelROCName*>::iterator allROCs_itr = allROCs.begin(); allROCs_itr != allROCs.end(); allROCs_itr++ )
				{
					if ( (*(*allROCs_itr)) == rocname )
					{
						foundIt = true;
						break;
					}
				}
				if (foundIt) rocSet.insert(rocname);
			}
			else
			{
				rocSet.erase(rocname);
			}
			addNext = true;
			continue;
		}
		
		// should never get here
		assert(0);
	}
	// done building ROC set
	
	buildROCAndModuleListsFromROCSet(rocSet);
}

void PixelCalibConfiguration::buildROCAndModuleListsFromROCSet(const std::set<PixelROCName>& rocSet)
{
	assert( !rocAndModuleListsBuilt_ );
	
	// Build the ROC list from the ROC set.
	for (std::set<PixelROCName>::iterator rocSet_itr = rocSet.begin(); rocSet_itr != rocSet.end(); rocSet_itr++ )
	{
		rocs_.push_back(*rocSet_itr);
	}
	
	// Build the module set from the ROC set.
	for (std::set<PixelROCName>::iterator rocSet_itr = rocSet.begin(); rocSet_itr != rocSet.end(); rocSet_itr++ )
	{
		PixelModuleName modulename(rocSet_itr->rocname());
		modules_.insert( modulename );
		countROC_[modulename]++;
	}
	
	// Test printout.
	/*cout << "\nROC list:\n";
	for ( std::vector<PixelROCName>::iterator rocs_itr = rocs_.begin(); rocs_itr != rocs_.end(); rocs_itr++ )
	{
		cout << rocs_itr->rocname() << "\n";
	}
	cout << "\nModule list:\n";
	for ( std::set<PixelModuleName>::iterator modules_itr = modules_.begin(); modules_itr != modules_.end(); modules_itr++ )
	{
		cout << modules_itr->modulename() << "\n";
	}
	cout << "\n";*/

	// Determine max ROCs on a module for singleROC mode.
	nROC_=1;
	if (singleROC_)
	{
		unsigned maxROCs=0;
		for (std::map<PixelModuleName,unsigned int>::iterator imodule=countROC_.begin();imodule!=countROC_.end();++imodule)
		{
			if (imodule->second>maxROCs) maxROCs=imodule->second;
      }
		nROC_=maxROCs;

		std::cout << "Max ROCs on a module="<<nROC_<<std::endl;
	}
	
	for(unsigned int irocs=0;irocs<rocs_.size();irocs++){
		old_irows.push_back(-1);
		old_icols.push_back(-1);
	}
	
	rocAndModuleListsBuilt_ = true;
}

const std::set <PixelChannel>& PixelCalibConfiguration::channelList(const PixelNameTranslation* aNameTranslation)
{
	assert( rocAndModuleListsBuilt_ );
	if ( !channelListBuilt_ )
	{
		assert( aNameTranslation != 0 );
		for (std::vector<PixelROCName>::const_iterator rocs_itr = rocs_.begin(); rocs_itr != rocs_.end(); ++rocs_itr)
		{
			channels_.insert( aNameTranslation->getChannelForROC(*rocs_itr) );
		}
		channelListBuilt_ = true;
	}
	return channels_;
}

unsigned int PixelCalibConfiguration::iScan(std::string dac) const{

  for (unsigned int i=0;i<dacs_.size();i++){
    if (dac==dacs_[i].name()) return i;
  }

  std::cout << "In PixelCalibConfiguration::iScan could not find dac="
            << dac <<std::endl; 

  assert(0);

  return 0;

}

unsigned int PixelCalibConfiguration::scanValue(unsigned int iscan,
                                               unsigned int state) const{

  
    assert(state<nConfigurations());

    unsigned int i_scan=state%nScanPoints();

    for(unsigned int i=0;i<iscan;i++){
      i_scan/=nScanPoints(i);
    }

    unsigned int i_threshold=i_scan%nScanPoints(iscan);

    unsigned int threshold=dacs_[iscan].first()+
      i_threshold*dacs_[iscan].step();

    return threshold;

}


unsigned int PixelCalibConfiguration::scanCounter(unsigned int iscan,
                                                 unsigned int state) const{


    assert(state<nConfigurations());

    unsigned int i_scan=state%nScanPoints();

    for(unsigned int i=0;i<iscan;i++){
      i_scan/=nScanPoints(i);
    }

    unsigned int i_threshold=i_scan%nScanPoints(iscan);

    return i_threshold;

}



void PixelCalibConfiguration::nextFECState(PixelFECConfigInterface* pixelFEC,
					   PixelDetectorConfig* detconfig,
					   PixelNameTranslation* trans,
					   std::map<pos::PixelModuleName,pos::PixelMaskBase*>* masks,
					   std::map<pos::PixelModuleName,pos::PixelTrimBase*>* trims,
					   unsigned int state) const {

  std::string modeName=parameterValue("ScanMode");

  int mode=-1;

  if (modeName=="maskAllPixel")  mode=0;
  if (modeName=="useAllPixel"||modeName=="")  mode=1;
  if (modeName=="default")  mode=2;

  static bool first=true;

  if (first) {
    cout << "PixelCalibConfiguration::nextFECState mode="<<mode<<endl;
    first=false;
  }
  
  if (mode==-1) {
    cout << "In Pxielcalibconfiguration: ScanMode="<<modeName
	 << " not understood."<<endl;
    ::abort();
  }
  
  pixelFEC->fecDebug(1);

  //unsigned long version=0;
  //pixelFEC->getversion(&version);
  //std::cout<<"mfec firmware version:"<<version<<std::endl;

  assert(rocAndModuleListsBuilt_);
    
  assert(state<nConfigurations());

  unsigned int i_ROC=state/(cols_.size()*rows_.size()*
			    nScanPoints());

  unsigned int i_row=(state-i_ROC*cols_.size()*rows_.size()*
		      nScanPoints())/
    (cols_.size()*nScanPoints());

  unsigned int i_col=(state-i_ROC*cols_.size()*rows_.size()*
		      nScanPoints()-
		      i_row*cols_.size()*nScanPoints())/
    (nScanPoints());


  std::vector<unsigned int> dacvalues;

  unsigned int first_scan=true;

  for (unsigned int i=0;i<dacs_.size();i++){
    dacvalues.push_back(scanValue(i,state));
    if (scanCounter(i,state)!=0) first_scan=false;
  }

  assert(i_row<rows_.size());
  assert(i_col<cols_.size());

  if (state==0&&(mode==0||mode==1)) {

    for(unsigned int i=0;i<rocs_.size();i++){
      const PixelHdwAddress* hdwadd=trans->getHdwAddress(rocs_[i]);

      assert(hdwadd!=0);
      PixelHdwAddress theROC=*hdwadd;
          
      //Turn off all pixels
      disablePixels(pixelFEC, theROC);

    }
   
  }


  if (first_scan){

    if (state!=0){

      unsigned int statetmp=state-1;
	
      unsigned int i_ROC=statetmp/(cols_.size()*rows_.size()*
				   nScanPoints());

      unsigned int i_row=(statetmp-i_ROC*cols_.size()*rows_.size()*
			  nScanPoints())/
	(cols_.size()*nScanPoints());

      unsigned int i_col=(statetmp-i_ROC*cols_.size()*rows_.size()*
			  nScanPoints()-
			  i_row*cols_.size()*nScanPoints())/
	(nScanPoints());


      assert(i_row<rows_.size());
      assert(i_col<cols_.size());

      if (mode!=2){
	for(unsigned int i=0;i<rocs_.size();i++){
	  const PixelHdwAddress* hdwadd=trans->getHdwAddress(rocs_[i]);
	  
	  // std::cout << "Got Hdwadd" << std::endl;

	  assert(hdwadd!=0);
	  PixelHdwAddress theROC=*hdwadd;
          
	  if (singleROC_&&theROC.fedrocnumber()!=i_ROC) continue;
          
	  disablePixels(pixelFEC, i_row, i_col, theROC);

	}
      }
    }
  }
    
  for(unsigned int i=0;i<rocs_.size();i++){

    //	std::cout << "Will configure roc:"<<rocs_[i] << std::endl;

    const PixelHdwAddress* hdwadd=trans->getHdwAddress(rocs_[i]);

    // std::cout << "Got Hdwadd" << std::endl;

    assert(hdwadd!=0);
    PixelHdwAddress theROC=*hdwadd;
        
    if (singleROC_&&theROC.fedrocnumber()!=i_ROC) continue;

    //	std::cout << "Will call progdac for vcal:"<< vcal << std::endl;
    
    for (unsigned int ii=0;ii<dacs_.size();ii++){
      pixelFEC->progdac(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			dacs_[ii].dacchannel(),
			dacvalues[ii],_bufferData);
      //          std::cout << "Will set dac "<<dacchannel_[i]
      //          <<" to "<<dacvalues[i]<<std::endl;
    }

    //std::cout << "Will set Vcal="<<vcal_<<std::endl;
    //
    //pixelFEC->progdac(theROC.mfec(),
    //		  theROC.mfecchannel(),
    //		  theROC.hubaddress(),
    //		  theROC.portaddress(),
    //		  theROC.rocid(),
    //		  25,
    //		  vcal_);
    //

    //	std::cout << "Done with progdac" << std::endl;
    if (first_scan){

      if (mode!=2){

	//FIXME This is very inefficient
	PixelModuleName module(rocs_[i].rocname());
	
	PixelMaskBase* moduleMasks=(*masks)[module];
	PixelTrimBase* moduleTrims=(*trims)[module];

	PixelROCMaskBits* rocMasks=moduleMasks->getMaskBits(rocs_[i]);
	PixelROCTrimBits* rocTrims=moduleTrims->getTrimBits(rocs_[i]);

	if (mode==1) rocMasks=0;

	//std::cout << "Will enable pixels!" <<std::endl;
	enablePixels(pixelFEC, i_row, i_col, rocMasks, rocTrims, theROC);

      }

      //FIXME!!!!
      //TODO retrieve ROC control register from configuration
      //range is controlled here by one bit, but rest must match config
      //bit 0 on/off= 20/40 MHz speed; bit 1 on/off=disabled/enable; bit 3=Vcal range
      int range=0;  //MUST replace this line with desired control register setting
      if (highVCalRange_) range|=0x4;  //turn range bit on
      else range&=0x3;                 //turn range bit off
      

      pixelFEC->progdac(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			0xfd,
			range,_bufferData);
      
      pixelFEC->clrcal(theROC.mfec(),
		       theROC.mfecchannel(),
		       theROC.hubaddress(),
		       theROC.portaddress(),
		       theROC.rocid(),_bufferData);
            
      unsigned int nrow=rows_[i_row].size();
      unsigned int ncol=cols_[i_col].size();
      unsigned int nmax=std::max(nrow,ncol);
      if (nrow==0||ncol==0) nmax=0;
      for (unsigned int n=0;n<nmax;n++){
	unsigned int irow=n;
	unsigned int icol=n;
	if (irow>=nrow) irow=nrow-1;
	if (icol>=ncol) icol=ncol-1;
	unsigned int row=rows_[i_row][irow];
	unsigned int col=cols_[i_col][icol];
	/*		std::cout << "Will do a calpix on roc, col, row:"
			<<theROC.rocid()<<" "<<col<<" "<<row<<std::endl;
	*/
	
	pixelFEC->calpix(theROC.mfec(),
			 theROC.mfecchannel(),
			 theROC.hubaddress(),
			 theROC.portaddress(),
			 theROC.rocid(),
			 col,
			 row,
			 1,_bufferData);
      }
    }
  }

  if (_bufferData) {
    pixelFEC->qbufsend();
  }
  
  return;

} 

// FIXME This code breaks if it is called more than once with different crate numbers!
std::vector<std::pair<unsigned int, std::vector<unsigned int> > >& PixelCalibConfiguration::fedCardsAndChannels(unsigned int crate,
														PixelNameTranslation* translation,
														PixelFEDConfig* fedconfig,
														PixelDetectorConfig* detconfig) const{

    assert(rocAndModuleListsBuilt_);
    
    assert(rocs_.size()!=0);

    for(unsigned int i=0;i<rocs_.size();i++){
      PixelModuleName module(rocs_[i].rocname());
      if (!detconfig->containsModule(module)) continue;
      const PixelHdwAddress* hdw=translation->getHdwAddress(rocs_[i]);
	assert(hdw!=0);
	//std::cout << "ROC, fednumber:"<<rocs_[i]<<" "<<hdw->fednumber()
	//	  << std::endl;
        //first check if fed associated with the roc is in the right crate
	if (fedconfig->crateFromFEDNumber(hdw->fednumber())!=crate) continue;
	//next look if we have already found fed number
	unsigned int index=fedCardsAndChannels_.size();
	for(unsigned int j=0;j<fedCardsAndChannels_.size();j++){
	    if (fedCardsAndChannels_[j].first==hdw->fednumber()){
		index=j;
		break;
	    }
	}
        //If we didn't find the fedcard we will create it
	if (index==fedCardsAndChannels_.size()){
	    std::vector<unsigned int> tmp;
	    tmp.push_back(hdw->fedchannel());
	    std::pair<unsigned int, std::vector<unsigned int> > tmp2(hdw->fednumber(),tmp);
	    fedCardsAndChannels_.push_back(tmp2);
            continue;
	}
	//Now look and see if the channel has been added
	std::vector<unsigned int>& channels=fedCardsAndChannels_[index].second;
	bool found=false;
	for(unsigned int k=0;k<channels.size();k++){
	    if (channels[k]==hdw->fedchannel()) {
		found=true;
		break;
	    }
	}
	if (found) continue;
	channels.push_back(hdw->fedchannel());

    }


    return fedCardsAndChannels_;

}

std::map <unsigned int, std::set<unsigned int> > PixelCalibConfiguration::getFEDsAndChannels (PixelNameTranslation *translation) {

  assert(rocAndModuleListsBuilt_);
  
  std::map <unsigned int, std::set<unsigned int> > fedsChannels;
  assert(rocs_.size()!=0);
  std::vector<PixelROCName>::iterator iroc=rocs_.begin();

  for (;iroc!=rocs_.end();++iroc){
    const PixelHdwAddress *roc_hdwaddress=translation->getHdwAddress(*iroc);
    unsigned int fednumber=roc_hdwaddress->fednumber();
    unsigned int fedchannel=roc_hdwaddress->fedchannel();
    fedsChannels[fednumber].insert(fedchannel);
  }

  return fedsChannels;
}

std::set <unsigned int> PixelCalibConfiguration::getFEDCrates(const PixelNameTranslation* translation, const PixelFEDConfig* fedconfig) const{

	assert(rocAndModuleListsBuilt_);
	
	std::set<unsigned int> fedcrates;
	assert(modules_.size()!=0);
	std::set<PixelModuleName>::iterator imodule=modules_.begin();

	for (;imodule!=modules_.end();++imodule)
	{
		const std::vector<PixelHdwAddress> *module_hdwaddress=translation->getHdwAddress(*imodule);
		for (unsigned int i=0;i<module_hdwaddress->size();i++){
		  unsigned int fednumber=(*module_hdwaddress)[i].fednumber();
		  fedcrates.insert(fedconfig->crateFromFEDNumber(fednumber));
		}
	}

	return fedcrates;
}

std::set <unsigned int> PixelCalibConfiguration::getFECCrates(const PixelNameTranslation* translation, const PixelFECConfig* fecconfig) const{

	assert(rocAndModuleListsBuilt_);
	
	std::set<unsigned int> feccrates;
	assert(modules_.size()!=0);
	std::set<PixelModuleName>::iterator imodule=modules_.begin();

	for (;imodule!=modules_.end();++imodule)
	{
		const std::vector<PixelHdwAddress> *module_hdwaddress=translation->getHdwAddress(*imodule);
		for (unsigned int i=0;i<module_hdwaddress->size();i++){
		  unsigned int fecnumber=(*module_hdwaddress)[i].fecnumber();
		  feccrates.insert(fecconfig->crateFromFECNumber(fecnumber));
		}
	}

	return feccrates;
}

std::set <unsigned int> PixelCalibConfiguration::getTKFECCrates(const PixelPortcardMap *portcardmap, const std::map<std::string,PixelPortCardConfig*>& mapNamePortCard, const PixelTKFECConfig* tkfecconfig) const{

	assert(rocAndModuleListsBuilt_);
	
	std::set<unsigned int> tkfeccrates;
	assert(modules_.size()!=0);
	std::set<PixelModuleName>::iterator imodule=modules_.begin();

	for (;imodule!=modules_.end();++imodule)
	{
		// implement this by module --(PixelPortcardMap)-> port card(s) --(PixelPortCardConfig)-> FEC # --(PixelFECConfig theTKFECConfiguration_)-> crate
		const std::set< std::string > portCards = portcardmap->portcards(*imodule);
		for ( std::set< std::string >::const_iterator portCards_itr = portCards.begin(); portCards_itr != portCards.end(); ++portCards_itr)
		{
			const std::string portcardname=*portCards_itr;
			std::map<std::string,PixelPortCardConfig*>::const_iterator portcardconfig_itr = mapNamePortCard.find(portcardname);
			assert(portcardconfig_itr != mapNamePortCard.end());
			PixelPortCardConfig* portcardconfig = portcardconfig_itr->second;
			std::string TKFECID = portcardconfig->getTKFECID();
			tkfeccrates.insert(tkfecconfig->crateFromTKFECID(TKFECID));
		}
	}

	return tkfeccrates;
}

std::ostream& pos::operator<<(std::ostream& s, const PixelCalibConfiguration& calib){
     if (!calib.parameters_.empty())
    {
       s<< "Parameters:"<<std::endl;
       for ( std::map<std::string, std::string>::const_iterator paramItr = calib.parameters_.begin(); paramItr != calib.parameters_.end(); ++paramItr )
       {
          s<< paramItr->first << " " << paramItr->second << std::endl;
       }
    }
    
    s<< "Rows:"<<std::endl;
    for (unsigned int i=0;i<calib.rows_.size();i++){
	for (unsigned int j=0;j<calib.rows_[i].size();j++){
	    s<<calib.rows_[i][j]<<" "<<std::endl;
	}
	s<< "|"<<std::endl;
    }

    s<< "Cols:"<<std::endl;
    for (unsigned int i=0;i<calib.cols_.size();i++){
	for (unsigned int j=0;j<calib.cols_[i].size();j++){
	  s<<calib.cols_[i][j]<<" "<<std::endl;
	}
	s<< "|"<<std::endl;
    }

    s << "Vcal:"<<std::endl;

    //s << calib.vcal_<<std::endl;

    s << "Vcthr:"<<std::endl;

    s << calib.dacs_[0].first() << " " << calib.dacs_[0].last() 
      << " "<< calib.dacs_[0].step()<<std::endl;

    s << "CalDel:"<<std::endl;

    s << calib.dacs_[1].first() << " " << calib.dacs_[0].last() 
      << " "<< calib.dacs_[1].step()<<std::endl;

    s << "Repeat:"<<std::endl;
    
    s << calib.ntrigger_<<std::endl;

    return s;

}


void PixelCalibConfiguration::enablePixels(PixelFECConfigInterface* pixelFEC,
					   unsigned int irows, 
					   unsigned int icols,
					   pos::PixelROCMaskBits* masks,
					   pos::PixelROCTrimBits* trims, 
					   PixelHdwAddress theROC) const{

  for (unsigned int irow=0;irow<rows_[irows].size();irow++){
    for (unsigned int icol=0;icol<cols_[icols].size();icol++){
      /*	    std::cout << "Will turn on pixel col="
		    <<cols_[icols][icol]
		    <<" row="<<rows_[irows][irow]<<std::endl;
      */
      unsigned int bits=trims->trim(cols_[icols][icol],rows_[irows][irow]);

      //if masks==0 always enable pixel
      if (masks==0||
	  masks->mask(cols_[icols][icol],rows_[irows][irow])) bits|=0x80;

      pixelFEC->progpix(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			cols_[icols][icol],
			rows_[irows][irow],
			bits,_bufferData);
    }
  }  
}

void PixelCalibConfiguration::disablePixels(PixelFECConfigInterface* pixelFEC,
			      unsigned int irows, unsigned int icols,
			      PixelHdwAddress theROC) const{

	for (unsigned int irow=0;irow<rows_[irows].size();irow++){
	    for (unsigned int icol=0;icol<cols_[icols].size();icol++){
	      /*		std::cout << "Will turn off pixel col="
			  <<cols_[old_icols][icol]
			  <<" row="<<rows_[old_irows][irow]<<std::endl;
	      */
		pixelFEC->progpix(theROC.mfec(),
				  theROC.mfecchannel(),
				  theROC.hubaddress(),
				  theROC.portaddress(),
				  theROC.rocid(),
				  cols_[icols][icol],
				  rows_[irows][irow],
				  0x0,_bufferData);
	    }
	}
}


void PixelCalibConfiguration::disablePixels(PixelFECConfigInterface* pixelFEC,
					    PixelHdwAddress theROC) const{

  //FIXME This should be done with more efficient commands!
  for (unsigned int row=0;row<80;row++){
    for (unsigned int col=0;col<52;col++){
      pixelFEC->progpix(theROC.mfec(),
			theROC.mfecchannel(),
			theROC.hubaddress(),
			theROC.portaddress(),
			theROC.rocid(),
			col,
			row,
			0x0,_bufferData);
    }
  }
}

std::string PixelCalibConfiguration::parameterValue(std::string parameterName) const
{
	std::map<std::string, std::string>::const_iterator itr = parameters_.find(parameterName);
	if (itr == parameters_.end()) // parameterName is not in the list
	{
		return "";
	}
	else
	{
		return itr->second;
	}
}

void PixelCalibConfiguration::writeASCII(std::string dir) const {


  //FIXME this is not tested for all the use cases...

  if (dir!="") dir+="/";
  std::string filename=dir+"calib.dat";
  std::ofstream out(filename.c_str());

  out << "Mode: "<<mode_<<endl;
  if (singleROC_) out << "SingleROC"<<endl;
  out << "Rows:" <<endl;
  for (unsigned int i=0;i<rows_.size();i++){
    for (unsigned int j=0;j<rows_[i].size();j++){
      out << rows_[i][j] <<" ";
    }
    if (i!=rows_.size()-1) out <<"|";
    out <<endl;
  }
  out << "Cols:" <<endl;
  for (unsigned int i=0;i<cols_.size();i++){
    for (unsigned int j=0;j<cols_[i].size();j++){
      out << cols_[i][j] <<" ";
    }
    if (i!=cols_.size()-1) out <<"|";
    out <<endl;
  }

  for (unsigned int i=0;i<dacs_.size();i++){
    out << "Scan: "<<dacs_[i].name()<<" "
	<<dacs_[i].first()<<" "
	<<dacs_[i].last()<<" "
	<<dacs_[i].step()<<endl;
  }

  out << "Repeat:" <<endl;
  out << ntrigger_ << endl;

  out << "Rocs:"<< endl;
  for (unsigned int i=0;i<rocs_.size();i++){
    out << rocs_[i].rocname() <<endl;
  }  

  out.close();

}
