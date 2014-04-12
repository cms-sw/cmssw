#include "CondTools/Ecal/interface/EcalADCToGeVXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalLinearCorrectionsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalIntercalibErrorsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalWeightGroupXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalTBWeightsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalLaserAPDPNRatiosXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalTimeCalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalTimeCalibErrorsXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"

#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"

#include "CondTools/Ecal/interface/EcalGainRatiosXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalChannelStatusXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalWeightSetXMLTranslator.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"



#include <iostream>
using std::cout;
using std::endl;

#include <string>

int main(){

  // Test ADCtoGeV
  
  EcalCondHeader header;
  EcalCondHeader header2;


  header.method_="testmethod";
  header.version_="testversion";
  header.datasource_="testdata";
  header.since_=123;
  header.tag_="testtag";
  header.date_="Mar 24 1973";

 
  
  EcalADCToGeVConstant adctogev_constant;
  EcalADCToGeVConstant adctogev_constant2;
 
  adctogev_constant.setEBValue(1.1);
  adctogev_constant.setEEValue(2.2);

  std::string adctogevfile("/tmp/EcalADCToGeVConstant.xml");
  std::string adctogevfile2("/tmp/adctogev2.xml");

  EcalADCToGeVXMLTranslator::writeXML(adctogevfile,header,adctogev_constant);
  EcalADCToGeVXMLTranslator::readXML(adctogevfile,header2,adctogev_constant2);
  EcalADCToGeVXMLTranslator::writeXML(adctogevfile2,header2,adctogev_constant2);
  

  // Test Intercalibration
  
  
  EcalLinearCorrections lin_constants;
  EcalIntercalibConstants intercalib_constants;
  EcalIntercalibErrors    intercalib_errors;

  std::string linfile("/tmp/EcalLinearCorrections.xml");
  std::string intercalibfile("/tmp/EcalIntercalibConstants.xml");
  std::string intercaliberrfile("/tmp/EcalIntercalibErrors.xml");
  std::string intercalibfiledb("/tmp/EcalIntercalibConstantsDB.xml");

  for (int cellid = 0; 
	     cellid < EBDetId::kSizeForDenseIndexing; 
	     ++cellid){// loop on EB cells
    
    
    uint32_t rawid = EBDetId::unhashIndex(cellid);

    EcalIntercalibConstant intercalib_constant = 
      EBDetId::unhashIndex(cellid).iphi();

    EcalIntercalibConstant lin_constant = 
      EBDetId::unhashIndex(cellid).iphi();

    EcalIntercalibError    intercalib_error    = intercalib_constant +1;
    
    lin_constants[rawid]=lin_constant;
    intercalib_constants[rawid]=intercalib_constant;
    intercalib_errors[rawid]   =intercalib_error;
  } 
  
  for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EB cells
    
    

    if (EEDetId::validHashIndex(cellid)){  
      uint32_t rawid = EEDetId::unhashIndex(cellid);
      EcalLinearCorrection lin_constant = EEDetId::unhashIndex(cellid).ix();;
      EcalIntercalibConstant intercalib_constant = EEDetId::unhashIndex(cellid).ix();;
      EcalIntercalibError    intercalib_error    = intercalib_constant +1;

      lin_constants[rawid]=lin_constant;
      intercalib_constants[rawid]=intercalib_constant;
      intercalib_errors[rawid]=intercalib_error;
    } // if
  } 


 
  EcalLinearCorrectionsXMLTranslator::writeXML(linfile,header,
						 lin_constants);

  EcalIntercalibConstantsXMLTranslator::writeXML(intercalibfile,header,
						 intercalib_constants);
  
  EcalIntercalibErrorsXMLTranslator::writeXML(intercaliberrfile,header,
                                              intercalib_errors);

  EcalIntercalibConstants lin_constants2;
  EcalIntercalibConstants intercalib_constants2;
  EcalIntercalibErrors    intercalib_errors2;


  EcalLinearCorrectionsXMLTranslator::readXML(linfile,header2,
						lin_constants2);

  EcalIntercalibConstantsXMLTranslator::readXML(intercalibfile,header2,
						intercalib_constants2);

  EcalIntercalibErrorsXMLTranslator::readXML(intercaliberrfile,header2,
					     intercalib_errors2);

  std::string linfile2("/tmp/linfile-2.xml");
  std::string intercalibfile2("/tmp/intercalibfile-2.xml");
  std::string intercaliberrfile2("/tmp/intercaliberrfile-2.xml");

  EcalLinearCorrectionsXMLTranslator::writeXML(linfile2,
			   header2,
			   lin_constants2);
  EcalIntercalibConstantsXMLTranslator::writeXML(intercalibfile2,
			   header2,
			   intercalib_constants2);

  EcalIntercalibErrorsXMLTranslator::writeXML(intercaliberrfile,header2,
					     intercalib_errors2);

  cout << "Done testing Intercalib abd Linear Corrections" << endl;

  // Test Timing Intercalibration
  
  
  EcalTimeCalibConstants timeCalib_constants;
  EcalTimeCalibErrors    timeCalib_errors;

  std::string timeCalibFile("/tmp/EcalTimeCalibConstants.xml");
  std::string timeCalibErrFile("/tmp/EcalTimeCalibErrors.xml");
  std::string timeCalibFileDB("/tmp/EcalTimeCalibConstantsDB.xml");

  for (int cellid = 0; 
	     cellid < EBDetId::kSizeForDenseIndexing; 
	     ++cellid){// loop on EB cells
    
    
    uint32_t rawid = EBDetId::unhashIndex(cellid);

    EcalTimeCalibConstant timeCalib_constant = 
      EBDetId::unhashIndex(cellid).iphi();

    EcalTimeCalibError timeCalib_error    = timeCalib_constant +1;
    
    timeCalib_constants[rawid]=timeCalib_constant;
    timeCalib_errors[rawid]   =timeCalib_error;
  } 
  
  for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EB cells
    
    

    if (EEDetId::validHashIndex(cellid)){  
      uint32_t rawid = EEDetId::unhashIndex(cellid);
      EcalTimeCalibConstant timeCalib_constant = EEDetId::unhashIndex(cellid).ix();;
      EcalTimeCalibError    timeCalib_error    = timeCalib_constant +1;

      timeCalib_constants[rawid]=timeCalib_constant;
      timeCalib_errors[rawid]=timeCalib_error;
    } // if
  } 


 
  EcalTimeCalibConstantsXMLTranslator::writeXML(timeCalibFile,header,
          				 timeCalib_constants);
  
  EcalTimeCalibErrorsXMLTranslator::writeXML(timeCalibErrFile,header,
                                        timeCalib_errors);

  EcalTimeCalibConstants timeCalib_constants2;
  EcalTimeCalibErrors    timeCalib_errors2;


  EcalTimeCalibConstantsXMLTranslator::readXML(timeCalibFile,header2,
          				timeCalib_constants2);

  EcalTimeCalibErrorsXMLTranslator::readXML(timeCalibErrFile,header,
					     timeCalib_errors2);

  std::string timeCalibFile2("/tmp/timeCalibFile-2.xml");
  std::string timeCalibErrFile2("/tmp/timeCalibErrFile-2.xml");

  EcalTimeCalibConstantsXMLTranslator::writeXML(timeCalibFile2,
			   header2,
			   timeCalib_constants2);

  EcalTimeCalibErrorsXMLTranslator::writeXML(timeCalibErrFile2,header2,
					     timeCalib_errors2);
  cout << "Done testing timing intercalib " << endl;

  // test xtalgroup
  
  EcalWeightGroupXMLTranslator grouptrans;

  EcalWeightXtalGroups    groups;

  std::string groupfile("/tmp/EcalWeightXtalGroups.xml");
  
  for (int cellid = 0 ;
	     cellid < EBDetId::kSizeForDenseIndexing; 
	     ++cellid){// loop on EB cells
    
    
    uint32_t rawid = EBDetId::unhashIndex(cellid);
    // a random gid
    EcalXtalGroupId gid( EBDetId::unhashIndex(cellid).iphi());
    groups[rawid]=gid;
 
  } 
  
  for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EB cells
    
    

    if (EEDetId::validHashIndex(cellid)){  
      uint32_t rawid = EEDetId::unhashIndex(cellid);
      // a random gid
      EcalXtalGroupId gid( EEDetId::unhashIndex(cellid).ix());
      groups[rawid]=gid;
    } // if
  } 


  
  grouptrans.writeXML(groupfile,header,groups);
  
  EcalWeightXtalGroups    groups2;

  grouptrans.readXML(groupfile,header2,groups2);

  std::string groupfile2("/tmp/group-2.xml");

  grouptrans.writeXML(groupfile2,header2,groups2);
    
  cout << "Done testing groups " << endl;

  EcalGainRatiosXMLTranslator transGainRatios;
  
  EcalGainRatios gainratios;

  std::string filenamegr("/tmp/EcalGainRatios.xml");
  std::string newfilegr("/tmp/gainratios-2.xml");


  for (int cellid = 0; 
	     cellid < EBDetId::kSizeForDenseIndexing; 
	     ++cellid){// loop on EB cells
    
    
    EcalMGPAGainRatio ecalGR; 
    ecalGR.setGain12Over6(2.);
    ecalGR.setGain6Over1(5.); 

    gainratios.insert(std::make_pair(EBDetId::unhashIndex(cellid),ecalGR));
    
  } 
  
  for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EE cells
    
    if (EEDetId::validHashIndex(cellid)){  
      EcalMGPAGainRatio ecalGR; 
      ecalGR.setGain12Over6(2.);
      ecalGR.setGain6Over1(5.);       
      gainratios.insert(std::make_pair(EEDetId::unhashIndex(cellid),ecalGR));
    } // if
  } 
 
 
  transGainRatios.writeXML(filenamegr,header,gainratios);
 
  EcalGainRatios gainratios2;
  transGainRatios.readXML(filenamegr,header2,gainratios2);

  
  transGainRatios.writeXML(newfilegr,header2,gainratios2);

  
  cout << "Done testing gainratios " << endl;

  EcalChannelStatusXMLTranslator transChannelStatus;
  
  EcalChannelStatus channelstatus;


  for (int cellid = 0; 
	     cellid < EBDetId::kSizeForDenseIndexing; 
	     ++cellid){// loop on EB cells
    
    
    EcalChannelStatusCode ecalCSC = EcalChannelStatusCode(16);

    uint32_t rawid= EBDetId::unhashIndex(cellid);

    channelstatus[rawid]=ecalCSC;
    
  } 
  
  for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on Ee cells
    
    if (EEDetId::validHashIndex(cellid)){  
      
      uint32_t rawid= EEDetId::unhashIndex(cellid);
      EcalChannelStatusCode ecalCSC = EcalChannelStatusCode(35); 
      channelstatus[rawid]=ecalCSC;
    } // if
  } 
 
  std::string cscfile("/tmp/EcalChannelStatus.xml");
 
  transChannelStatus.writeXML(cscfile,header,channelstatus);
  
  EcalChannelStatus channelstatus2;

  transChannelStatus.readXML(cscfile,header2,channelstatus2);

  std::string cscfile2("/tmp/cscfile-2.xml");

  transChannelStatus.writeXML(cscfile2,header2,channelstatus2);


  cout << "Done testing channelstatus " << endl;
  
  EcalTBWeightsXMLTranslator transWeight;
  
  EcalWeightSet weightset;


  for(int i=0;i<3;i++)
    {
      for(int k=0;k<10;k++)
	{
	  weightset.getWeightsBeforeGainSwitch()(i,k) = 1.2*k;
	  weightset.getWeightsAfterGainSwitch()(i,k) = 1.2*k;
	}
    }



  for(int i=0;i<10;i++)
    {
      for(int k=0;k<10;k++)
	{
	  weightset.getChi2WeightsBeforeGainSwitch()(i,k) = 1.2*k;
	  weightset.getChi2WeightsAfterGainSwitch()(i,k) = 1.2*k;
	}
    }

  EcalXtalGroupId          gid=1;
  EcalTBWeights::EcalTDCId tid=2;


  EcalTBWeights tbw;
  tbw.setValue(gid,tid,weightset);
 
  
  std::string filew("/tmp/EcalTBWeights.xml");
  std::string filew2("/tmp/tbweight2.xml");

  transWeight.writeXML(filew,header,tbw);


  EcalTBWeights tbw2;

  transWeight.readXML(filew,header2,tbw2);

  transWeight.writeXML(filew2,header2,tbw2);
    

  // test laser
  
  std::string filelaser("/tmp/EcalLaserAPDPNratios.xml");
  std::string filelaser2("/tmp/EcalLaserAPDPNratios-2.xml");

  EcalLaserAPDPNRatios laserrecord1;
  EcalLaserAPDPNRatios laserrecord2;

  for (int cellid = 0; 
	     cellid < EBDetId::kSizeForDenseIndexing; 
	     ++cellid){// loop on EB cells
    
    
    uint32_t rawid= EBDetId::unhashIndex(cellid);

    EcalLaserAPDPNRatios::EcalLaserAPDPNpair pair;
    pair.p1 =1;
    pair.p2 =2;
    pair.p3 =3;

    laserrecord1.setValue(rawid,pair);
    
  } 
  
  for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on Ee cells
    
    if (EEDetId::validHashIndex(cellid)){  
      
      uint32_t rawid= EEDetId::unhashIndex(cellid);
      EcalLaserAPDPNRatios::EcalLaserAPDPNpair pair;
      pair.p1 =1;
      pair.p2 =2;
      pair.p3 =3;
      
      laserrecord1.setValue(rawid,pair);
    
    } // if
  } 

  EcalLaserAPDPNRatiosXMLTranslator::writeXML(filelaser,header,laserrecord1);
  EcalLaserAPDPNRatiosXMLTranslator::readXML(filelaser,header2,laserrecord2);
  EcalLaserAPDPNRatiosXMLTranslator::writeXML(filelaser2,header2,laserrecord2);
  return 0;  
}
