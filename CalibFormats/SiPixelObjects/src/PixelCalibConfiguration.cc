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


PixelCalibConfiguration::PixelCalibConfiguration(std::string filename):
  PixelCalibBase(), PixelConfigBase("","","") {


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
	     mode_=="FEDAddressLevelWithPixels"||
	     mode_=="GainCalibration"||
	     mode_=="PixelAlive"||
	     mode_=="SCurve"||
	     mode_=="Delay25"||
	     mode_=="ClockPhaseCalibration"||
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

    if (tmp=="VcalLow") {
      highVCalRange_=false;
      in >> tmp;
    }

    if (tmp=="VcalHigh") {
      highVCalRange_=true;
      in >> tmp;
    }

    highVCalRange_=true;
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

    if (in.eof()){
	roclistfromconfig_=false;
	in.close();
	return;
    }

    assert(tmp=="Rocs:");

    in >> tmp;

    while (!in.eof()){
        PixelROCName rocname(tmp);
	PixelModuleName modulename(tmp);
	rocs_.push_back(rocname);
	modules_.insert(modulename);
	countROC_[modulename]++;
	//std::cout << "modulename, rocname:"<<modulename<<" "
	//	  <<rocname<<" "<<countROC_[modulename]<<std::endl;
	in >> tmp;
    }

    nROC_=1;

    if (singleROC_){

      std::map<PixelModuleName,unsigned int>::iterator imodule=countROC_.begin();

      unsigned maxROCs=0;

      for (;imodule!=countROC_.end();++imodule){
	//std::cout << "module, roc:"<<imodule->first<<" "<<imodule->second
	//	  <<std::endl;
	if (imodule->second>maxROCs) maxROCs=imodule->second;
      }
      
      nROC_=maxROCs;
      
      std::cout << "Max ROCs on a module="<<nROC_<<std::endl;

    }

    in.close();

    for(unsigned int irocs=0;irocs<rocs_.size();irocs++){
      old_irows.push_back(-1);
      old_icols.push_back(-1);
    }
    
    return;

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
			     unsigned int state) const {

    if ((!roclistfromconfig_)&&rocs_.size()==0){

      //This code is not at all tested
      assert(0);
      /*
	int nmodule=detconfig->getNModules();
	for (int imodule=0;imodule<nmodule;imodule++){
	    PixelModuleName module=detconfig->getModule(imodule);
	    //This is ugly need to fix this somehow
	    for (unsigned int iplq=1;iplq<5;iplq++){
		for (unsigned int iroc=1;iroc<5;iroc++){
       		    std::string name=module.modulename()+"_PLQ"+itoa(iplq)+"_ROC"+itoa(iroc);
		    const PixelHdwAddress* hdwadd=0;
                    PixelROCName rocname(name);
		    hdwadd=trans->getHdwAddress(rocname);
		    if (hdwadd!=0){
			rocs_.push_back(rocname);
		    }
		}
	    }
	}
      */
    }

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
    
    for(unsigned int i=0;i<rocs_.size();i++){

      //	std::cout << "Will configure roc:"<<rocs_[i] << std::endl;

      const PixelHdwAddress* hdwadd=trans->getHdwAddress(rocs_[i]);

      // std::cout << "Got Hdwadd" << std::endl;

      assert(hdwadd!=0);
      PixelHdwAddress theROC=*hdwadd;
        
      if (singleROC_&&theROC.fedrocnumber()!=i_ROC) continue;

	//	std::cout << "Will call progdac for vcal:"<< vcal << std::endl;

        for (unsigned int i=0;i<dacs_.size();i++){
          pixelFEC->progdac(theROC.mfec(),
                            theROC.mfecchannel(),
                            theROC.hubaddress(),
                            theROC.portaddress(),
                            theROC.rocid(),
                            dacs_[i].dacchannel(),
                            dacvalues[i]);
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

          //std::cout << "Will enable pixels!" <<std::endl;
	    enablePixels(pixelFEC, i_row, i_col, theROC);
	    //            std::cout << "Will do a clrcal on roc:"<<theROC.rocid()<<std::endl;

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
			      range);


	    pixelFEC->clrcal(theROC.mfec(),
			     theROC.mfecchannel(),
			     theROC.hubaddress(),
			     theROC.portaddress(),
			     theROC.rocid());
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
				 1);
	    }
	}
    }
    
    return;

} 

// This code breaks if it is called more than once with different crate numbers!
std::vector<std::pair<unsigned int, std::vector<unsigned int> > >& PixelCalibConfiguration::fedCardsAndChannels(unsigned int crate,
												   PixelNameTranslation* translation,
												   PixelFEDConfig* fedconfig) const{

    assert(rocs_.size()!=0);

    for(unsigned int i=0;i<rocs_.size();i++){
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

std::ostream& operator<<(std::ostream& s, const PixelCalibConfiguration& calib){

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
			      unsigned int irows, unsigned int icols,
			      PixelHdwAddress theROC) const{

  //std::cout << "irows, icols:"<<irows<<" "<<icols<<std::endl;

    for (unsigned int irow=0;irow<rows_[irows].size();irow++){
	for (unsigned int icol=0;icol<cols_[icols].size();icol++){
	  /*	    std::cout << "Will turn on pixel col="
		      <<cols_[icols][icol]
		      <<" row="<<rows_[irows][irow]<<std::endl;
	  */
	    pixelFEC->progpix(theROC.mfec(),
			      theROC.mfecchannel(),
			      theROC.hubaddress(),
			      theROC.portaddress(),
			      theROC.rocid(),
			      cols_[icols][icol],
			      rows_[irows][irow],
			      0x80);
		
	}
    }


    //std::cout << "Done"<<std::endl;

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
				  0x0);
		
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
