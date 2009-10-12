#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include "CalibCalorimetry/HcalTPGAlgos/interface/HcaluLUTTPGCoder.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObjects.h"
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObject.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/LutXml.h"


const float HcaluLUTTPGCoder::nominal_gain = 0.177; 

HcaluLUTTPGCoder::HcaluLUTTPGCoder(const char* filename, bool read_Ascii_LUTs, bool read_XML_LUTs) {
  AllocateLUTs();
  // std::cout << " filename:" << filename << " read_Ascii_LUTs " << read_Ascii_LUTs << std::endl;
  if (read_Ascii_LUTs) {
    update(filename);
  }
  else if (read_XML_LUTs) {
     updateXML(filename);
  }
  else {
    getRecHitCalib(filename);
  }
//  CaloTPGTranscoderULUT("CalibCalorimetry/CaloTPG/data/outputLUTtranscoder_CRUZET_part3_v2.dat","CalibCalorimetry/CaloTPG/data/TPGcalcDecompress_CRUZET4_v2.txt");
//  CaloTPGTranscoderULUT("CalibCalorimetry/CaloTPG/data/outputLUTtranscoder_CRUZET_part3_v2.dat","");
//  CaloTPGTranscoderULUT();
	LUTGenerationMode = false;
      DumpL1TriggerObjects = false;
	TagName = "";
	AlgoName = "";
}

void HcaluLUTTPGCoder::PrintTPGMap() {
	HcalTopology theTopo;
	HcalDetId did;
//	std::string HCAL[3] = {"HB", "HE", "HF"};  

	//std::cout << "HB Calorimeter Tower Mapping" << std::endl;
	//std::cout << "ieta depth phi span" << std::endl;
	for (int ieta=1; ieta <= 41; ieta++) {
		for (int depth = 1; depth <= 3; depth++) {
			bool newdepth = true, OK = false;
			int lastphi = 0;
			for (int iphi = 1; iphi <= 72; iphi++) {
		       	did=HcalDetId(HcalBarrel,ieta,iphi,depth);
			    if (theTopo.valid(did)) {
					if (newdepth) {
					  std::cout << "eta/depth = " << ieta << "/" << depth << ": ";
						newdepth = false;
						OK = true;
					}
					if (lastphi == 0) std::cout << iphi << " ";
					lastphi = iphi;
	        	}  else lastphi = 0;
			}
			if (lastphi != 0) std::cout << "- " << lastphi;
			if (OK) std::cout << std::endl;
		}
	}
	std::cout << "--------------------------------------------------------------" << std::endl;

	std::cout << "HE Calorimeter Tower Mapping" << std::endl;
	std::cout << "ieta depth phi span" << std::endl;
	for (int ieta=-41; ieta <= 41; ieta++) {
		for (int depth = 1; depth <= 3; depth++) {
			bool newdepth = true, OK = false;
			int lastphi = 0;
			for (int iphi = 1; iphi <= 72; iphi++) {
		       	did=HcalDetId(HcalEndcap,ieta,iphi,depth);
			    if (theTopo.valid(did)) {
					if (newdepth) {
						std::cout << "eta/depth = " << ieta << "/" << depth << ": ";
						newdepth = false;
						OK = true;
					}
					if (lastphi == 0) std::cout << iphi << " ";
					lastphi = iphi;
	        	}  else lastphi = 0;
			}
			if (lastphi != 0) std::cout << "- " << lastphi;
			if (OK) std::cout << std::endl;
		}
	}
	std::cout << "--------------------------------------------------------------" << std::endl;

	std::cout << "HF Calorimeter Tower Mapping" << std::endl;
	std::cout << "ieta depth phi span" << std::endl;
	for (int ieta=1; ieta <= 41; ieta++) {
		for (int depth = 1; depth <= 3; depth++) {
			bool newdepth = true, OK = false;
			int lastphi = 0;
			for (int iphi = 1; iphi <= 72; iphi++) {
		      	did=HcalDetId(HcalForward,ieta,iphi,depth);
			    if (theTopo.valid(did)) {
					if (newdepth) {
						std::cout << "eta/depth = " << ieta << "/" << depth << ": ";
						newdepth = false;
						OK = true;
					}
					if (lastphi == 0) std::cout << iphi << " ";
					lastphi = iphi;
	        	}  else lastphi = 0;
			}
			if (lastphi != 0) std::cout << "- " << lastphi;
			if (OK) std::cout << std::endl;
		}
	}
	std::cout << "--------------------------------------------------------------" << std::endl;

	HcalDetId dd;
	HcalTrigTowerDetId tt;
	HcalTrigTowerGeometry tg;
	std::vector<HcalTrigTowerDetId> towerids;
	std::vector<HcalDetId> detids;
	std::vector<HcalDetId> ttmap[64][72];
	
//	std::cout << "HB Tower Mapping:" << std::endl;
	for (int ieta=1; ieta <= 41; ieta++) {
		for (int depth = 1; depth <= 3; depth++) {
			for (int iphi = 1; iphi <= 72; iphi++) {
				did=HcalDetId(HcalBarrel,ieta,iphi,depth);
				if (theTopo.valid(did)) {
					tt = HcalTrigTowerDetId(did.rawId());
//					std::cout << "ieta, depth, iphi = " << ieta << ", " << depth << ", " << iphi;
					towerids = tg.towerIds(did.rawId());
					for(unsigned int n = 0; n < towerids.size(); ++n)
					{
//						std::cout << "Towerid[" << n << "]:" << towerids[n] << " HcalDetId:" << did << std::endl;
						int ie = towerids[n].ieta();
						if (ie < 0) ie = 32 - ie;
						int ip = towerids[n].iphi();
						ttmap[ie-1][ip-1].push_back(did);
					}
				}
			}
		}
	}
//	std::cout << "HE Tower Mapping:" << std::endl;
	for (int ieta=1; ieta <= 41; ieta++) {
		for (int depth = 1; depth <= 3; depth++) {
			for (int iphi = 1; iphi <= 72; iphi++) {
				did=HcalDetId(HcalEndcap,ieta,iphi,depth);
				if (theTopo.valid(did)) {
					tt = HcalTrigTowerDetId(did.rawId());
//					std::cout << "ieta, depth, iphi = " << ieta << ", " << depth << ", " << iphi;
					towerids = tg.towerIds(did.rawId());
					for(unsigned int n = 0; n < towerids.size(); ++n)
					{
						int ie = towerids[n].ieta();
						if (ie < 0) ie = 32 - ie;
						int ip = towerids[n].iphi();
						ttmap[ie-1][ip-1].push_back(did);
//						std::cout << "Towerid[" << n << "]:" << towerids[n] << " HcalDetId:" << did << std::endl;
					}
				}
			}
		}
	}
//	std::cout << "HF Tower Mapping:" << std::endl;
	for (int ieta=1; ieta <= 41; ieta++) {
		for (int iphi = 1; iphi <= 72; iphi++) {
			for (int depth = 1; depth <= 3; depth++) {
				did=HcalDetId(HcalForward,ieta,iphi,depth);
				if (theTopo.valid(did)) {
					tt = HcalTrigTowerDetId(did.rawId());
//					std::cout << "ieta, iphi, depth = " << ieta << ", " << iphi << ", " << depth << " ";
					towerids = tg.towerIds(did.rawId());
					for(unsigned int n = 0; n < towerids.size(); ++n)
					{
						int ie = towerids[n].ieta();
						if (ie < 0) ie = 32 - ie;
						int ip = towerids[n].iphi();
						ttmap[ie-1][ip-1].push_back(did);
//						detids = tg.detIds(towerids[n]);
//						for(unsigned int m = 0; m < detids.size(); ++m) std::cout << detids[m] << "; ";
//						std::cout << towerids[n] << " HcalDetId:" << did << "; ";
					}
//					std::cout << std::endl;
				}
			}
		}
	}
	
	for (int ieta = 1; ieta <= 32; ieta++) {
		for (int iphi = 1; iphi <=72; iphi++) {
			if (ttmap[ieta-1][iphi-1].size() > 0) {
				std::cout << "Trigger tower [" << ieta << "," << iphi << "] contains: ";
				for(std::vector<HcalDetId>::const_iterator itr = (ttmap[ieta-1][iphi-1]).begin(); itr != (ttmap[ieta-1][iphi-1]).end(); ++itr) std::cout << (*itr) << ", ";
				std::cout << std::endl;
			}
		}
	}
}

void HcaluLUTTPGCoder::compress(const IntegerCaloSamples& ics, const std::vector<bool>& featureBits, HcalTriggerPrimitiveDigi& tp) const {
  throw cms::Exception("PROBLEM: This method should never be invoked!");
}

HcaluLUTTPGCoder::~HcaluLUTTPGCoder() {
  for (int i = 0; i < nluts; i++) {
    if (inputLUT[i] != 0) delete [] inputLUT[i];
  }
  delete [] _gain;
  delete [] _ped;
}

void HcaluLUTTPGCoder::AllocateLUTs() {
  HcalTopology theTopo;
  HcalDetId did;

//  PrintTPGMap();

  _ped = new float[nluts];
  _gain = new float[nluts];
  for (int i = 0; i < nluts; i++) inputLUT[i] = 0;
  int maxid = 0, minid = 0x7FFFFFFF, rawid = 0;
  for (int ieta=-41; ieta <= 41; ieta++) {
    for (int iphi = 1; iphi <= 72; iphi++) {
      for (int depth = 1; depth <= 3; depth++) {
	did=HcalDetId(HcalBarrel,ieta,iphi,depth);
	if (theTopo.valid(did)) {
	  rawid = GetLUTID(HcalBarrel, ieta, iphi, depth);
	  if (inputLUT[rawid] != 0) std::cout << "Error: LUT with (ieta,iphi,depth) = (" << ieta << "," << iphi << "," << depth << ") has been previously allocated!" << std::endl;
	  else inputLUT[rawid] = new LUT[INPUT_LUT_SIZE];
	  if (rawid < minid) minid = rawid;
	  if (rawid > maxid) maxid = rawid;
	}

	did=HcalDetId(HcalEndcap,ieta,iphi,depth);
	if (theTopo.valid(did)) {
	  rawid = GetLUTID(HcalEndcap, ieta, iphi, depth);
	  if (inputLUT[rawid] != 0) std::cout << "Error: LUT with (ieta,iphi,depth) = (" << ieta << "," << iphi << "," << depth << ") has been previously allocated!" << std::endl;
	  else inputLUT[rawid] = new LUT[INPUT_LUT_SIZE];
	  if (rawid < minid) minid = rawid;
	  if (rawid > maxid) maxid = rawid;
	}
	did=HcalDetId(HcalForward,ieta,iphi,depth);
	if (theTopo.valid(did)) {
	  rawid = GetLUTID(HcalForward, ieta, iphi, depth);
	  if (inputLUT[rawid] != 0) std::cout << "Error: LUT with (ieta,iphi,depth) = (" << ieta << "," << iphi << "," << depth << ") has been previously allocated!" << std::endl;
	  else inputLUT[rawid] = new LUT[INPUT_LUT_SIZE];
	  if (rawid < minid) minid = rawid;
	  if (rawid > maxid) maxid = rawid;
	}
      }
    }
  }

}

int HcaluLUTTPGCoder::GetLUTID(HcalSubdetector id, int ieta, int iphi, int depth) const {
  int detid = 0;
  if (id == HcalEndcap) detid = 1;
  else if (id == HcalForward) detid = 2;
  return iphi + 72 * ((ieta + 41) + 83 * (depth + 3 * detid)) - 7777;
}

int HcaluLUTTPGCoder::GetLUTID(uint32_t rawid) const {
   HcalDetId detid(rawid);
   return GetLUTID(detid.subdet(), detid.ieta(), detid.iphi(), detid.depth());
}

void HcaluLUTTPGCoder::getRecHitCalib(const char* filename) {

   std::ifstream userfile;
   userfile.open(filename);
   int tool;
   float Rec_calib_[87];
 
   if (userfile) {
	       userfile >> tool;

	       if (tool != 86) {
		 std::cout << "Wrong RecHit calibration filesize: " << tool << " (expect 86)" << std::endl;
	       }
     for (int j=1; j<87; j++) {
       userfile >> Rec_calib_[j]; // Read the Calib factors
       Rcalib[j] = Rec_calib_[j] ;
     }
   
     userfile.close();  
   }
   else  std::cout << "File " << filename << " with RecHit calibration factors not found" << std::endl;
}

void HcaluLUTTPGCoder::update(const char* filename) {
  HcalTopology theTopo;
  int tool;
  //std::string HCAL[3] = {"HB", "HE", "HF"};  
  std::ifstream userfile;
  userfile.open(filename);
  //std::cout << filename << std::endl;
  if( userfile ) {
    int nluts = 0;
	std::string s;
    std::vector<int> loieta,hiieta;
    std::vector<int> loiphi,hiiphi;    
	std::vector<int> loidep,hiidep;
	std::vector<int> idet;
    getline(userfile,s);
	//std::cout << "Reading LUT's for: " << s << std::endl;
    getline(userfile,s);
//
	unsigned int index = s.find("H",0);	
	while (index < s.length()) {
		std::string det = s.substr(index,2);
		if (det == "HB") idet.push_back(0); // HB
		else if (det == "HE") idet.push_back(1); //HE
		else if (det == "HF") idet.push_back(2); // HF
		//else std::cout << "Wrong LUT detector description in " << s << std::endl;
		//std::cout << det.data() << " ";
		nluts++;
		index +=2;
		index = s.find("H",index);
	}
	//if (nluts != 0) std::cout << std::endl;
	//std::cout << "Found " << nluts << " LUTs" << std::endl;

    inputluts_.resize(nluts);
    for (int i=0; i<nluts; i++) {
      inputluts_[i].resize(INPUT_LUT_SIZE); 
    }
    
	//std::cout << "EtaMin = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loieta.push_back(tool);
	  //std::cout << tool << " ";
    }
	//std::cout << std::endl << "EtaMax = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiieta.push_back(tool);
	  //std::cout << tool << " ";
    }
	//std::cout << std::endl << "PhiMin = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loiphi.push_back(tool);
	  //std::cout << tool << " ";
    }
	//std::cout << std::endl << "PhiMax = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiiphi.push_back(tool);
	  //std::cout << tool << " ";
    }
	//std::cout << std::endl << "DepMin = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loidep.push_back(tool);
	  //std::cout << tool << " ";
    }
	//std::cout << std::endl << "DepMax = ";
    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiidep.push_back(tool);
	  //std::cout << tool << " ";
    }    
	//std::cout << std::endl;
	
    for (int j=0; j<INPUT_LUT_SIZE; j++) { 
      for(int i = 0; i <nluts; i++) {
		userfile >> inputluts_[i][j];
		//if (userfile.eof()) std::cout << "Error: LUT file is truncated or has a wrong format: " << i << "," << j << std::endl;
	  }
    }
    userfile.close();
	
	HcalDetId cell;
	int id, ntot = 0;
	for (int i=0; i < nluts; i++) {
		int nini = 0;
    	for (int depth = loidep[i]; depth <= hiidep[i]; depth++) {
     		for (int iphi = loiphi[i]; iphi <= hiiphi[i]; iphi++) {      
       			for (int ieta=loieta[i]; ieta <= hiieta[i]; ieta++) {
	 				if (idet[i] == 0) cell = HcalDetId(HcalBarrel,ieta,iphi,depth);
	 				else if (idet[i] == 1) cell = HcalDetId(HcalEndcap,ieta,iphi,depth);
	 				else if (idet[i] == 2) cell = HcalDetId(HcalForward,ieta,iphi,depth);
	 				if (theTopo.valid(cell)) {  
	   					if (idet[i] == 0) id = GetLUTID(HcalBarrel,ieta,iphi,depth);
		 				else if (idet[i] == 1) id = GetLUTID(HcalEndcap,ieta,iphi,depth);
		 				else if (idet[i] == 2) id = GetLUTID(HcalForward,ieta,iphi,depth);
	   					if (inputLUT[id] == 0) throw cms::Exception("PROBLEM: inputLUT has not been initialized for idet, ieta, iphi, depth, id = ") << idet[i] << "," << ieta << "," << iphi << "," << depth << "," << id << std::endl;
		    			for (int j = 0; j <= 0x7F; j++) inputLUT[id][j] = inputluts_[i][j];
						nini++;
						ntot++;
	 				}
       			}
			}
       }
	   //std::cout << nini << " LUT's have been initialized for " << HCAL[idet[i]] << ": eta = [" << loieta[i] << "," << hiieta[i] << "]; iphi = [" << loiphi[i] << "," << hiiphi[i] << "]; depth = [" << loidep[i] << "," << hiidep[i] << "]" << std::endl;
    }
    //std::cout << "Total of " << ntot << " have been initialized" << std::endl;
  } 
}

void HcaluLUTTPGCoder::updateXML(const char* filename) {
   HcalTopology theTopo;
   LutXml * _xml = new LutXml(filename);
   _xml->create_lut_map();
   HcalSubdetector subdet[3] = {HcalBarrel, HcalEndcap, HcalForward};
   for (int ieta=-41; ieta<=41; ++ieta){
      for (int iphi=1; iphi<=72; ++iphi){
         for (int depth=1; depth<=3; ++depth){
            for (int isub=0; isub<3; ++isub){
               HcalDetId detid(subdet[isub], ieta, iphi, depth);
               if (!theTopo.valid(detid)) continue;
               int id = GetLUTID(subdet[isub], ieta, iphi, depth);
               std::vector<unsigned int>* lut = _xml->getLutFast(detid);
               if (lut==0) throw cms::Exception("PROBLEM: No inputLUT in xml file for ") << detid << std::endl;
               if (lut->size()!=128) throw cms::Exception ("PROBLEM: Wrong inputLUT size in xml file for ") << detid << std::endl;
               for (int i=0; i<128; ++i) inputLUT[id][i] = (LUT)lut->at(i);
            }
         }
      }
   }
   delete _xml;
   XMLProcessor::getInstance()->terminate();
}

void HcaluLUTTPGCoder::update(const HcalDbService& conditions) {
	HcalL1TriggerObjects *HcalL1TrigObjCol = new HcalL1TriggerObjects();
   const HcalQIEShape* shape = conditions.getHcalShape();
   HcalCalibrations calibrations;
   int id;
   float divide;
   HcalTopology theTopo;

	//debug
	//std::ofstream ofdebug("debug_LUTGeneration.txt");
	//ofdebug.setf(std::ios::fixed,std::ios::floatfield);
	//ofdebug.precision(6);
	
   float cosheta_[41], lsb_ = 1./16.;
   for (int i = 0; i < 13; i++) cosheta_[i+29] = cosh((theHFEtaBounds[i+1] + theHFEtaBounds[i])/2.);
    
   for (int depth = 1; depth <= 3; depth++) {
     for (int iphi = 1; iphi <= 72; iphi++) {
       divide = 1.*nominal_gain;
       for (int ieta=-16; ieta <= 16; ieta++) {
	 HcalDetId cell(HcalBarrel,ieta,iphi,depth);
	 if (theTopo.valid(cell)) {  
	   id = GetLUTID(HcalBarrel,ieta,iphi,depth);
	   if (inputLUT[id] == 0) throw cms::Exception("PROBLEM: inputLUT has not been initialized for HB, ieta, iphi, depth, id = ") << ieta << "," << iphi << "," << depth << "," << id << std::endl;
	   //conditions.makeHcalCalibration (cell, &calibrations);
		//
		const HcalQIECoder* channelCoder = conditions.getHcalCoder (cell);
		HcalCoderDb coder (*channelCoder, *shape);
		HBHEDataFrame frame(cell);
	   frame.setSize(1);
	   CaloSamples samples(cell, 1);
		float ped_ = 0;
		float gain_ = 0;

		if (LUTGenerationMode){
			HcalCalibrations calibrations = conditions.getHcalCalibrations(cell);
	   	ped_ = (calibrations.pedestal(0)+calibrations.pedestal(1)+calibrations.pedestal(2)+calibrations.pedestal(3))/4;
	   	gain_= (calibrations.LUTrespcorrgain(0)+calibrations.LUTrespcorrgain(1)+calibrations.LUTrespcorrgain(2)+calibrations.LUTrespcorrgain(3))/4;          

			//Add HcalL1TriggerObject to its container
			HcalL1TriggerObject HcalL1TrigObj(cell.rawId(), ped_, gain_);
			HcalL1TrigObjCol->addValues(HcalL1TrigObj);
		}
		else{
			const HcalL1TriggerObject* myL1TObj = conditions.getHcalL1TriggerObject(cell);
			ped_ = myL1TObj->getPedestal();
			gain_ = myL1TObj->getRespGain();
			//debug
			//ofdebug << cell.rawId() << '\t' << ped_ << '\t' << gain_ << '\n';
		}
		
		_ped[id] = ped_;
		_gain[id] = gain_;

	   for (int j = 0; j <= 0x7F; j++) {
	     HcalQIESample adc(j);
	     frame.setSample(0,adc);
	     coder.adc2fC(frame,samples);
	     float adc2fC_ = samples[0];
	     if (ieta <0 )inputLUT[id][j] = (LUT) std::min(std::max(0,int((adc2fC_ - ped_)*gain_*Rcalib[abs(ieta)]/divide)), 0x3FF);
	     else inputLUT[id][j] = (LUT) std::min(std::max(0,int((adc2fC_ - ped_)*gain_*Rcalib[abs(ieta)+43]/divide)), 0x3FF);
	   }
	 }
       }
       for (int ieta=-29; ieta <= 29; ieta++) {
	 HcalDetId cell(HcalEndcap,ieta,iphi,depth);
	 if (theTopo.valid(cell)) {  
	   if (abs(ieta) < 18) divide = 1.*nominal_gain;
	   else if (abs(ieta) < 27) divide = 2.*nominal_gain;
	   else divide = 5.*nominal_gain;
	   id = GetLUTID(HcalEndcap,ieta,iphi,depth);
	   if (inputLUT[id] == 0) throw cms::Exception("PROBLEM: inputLUT has not been initialized for HE, ieta, iphi, depth, id = ") << ieta << "," << iphi << "," << depth << "," << id << std::endl;
	   //conditions.makeHcalCalibration (cell, &calibrations);
		//
		const HcalQIECoder* channelCoder = conditions.getHcalCoder (cell);
		HcalCoderDb coder (*channelCoder, *shape);
		HBHEDataFrame frame(cell);
	   frame.setSize(1);
	   CaloSamples samples(cell, 1);
		float ped_ = 0;
		float gain_ = 0;

		if (LUTGenerationMode){
			HcalCalibrations calibrations = conditions.getHcalCalibrations(cell);
	   	ped_ = (calibrations.pedestal(0)+calibrations.pedestal(1)+calibrations.pedestal(2)+calibrations.pedestal(3))/4;
	   	gain_= (calibrations.LUTrespcorrgain(0)+calibrations.LUTrespcorrgain(1)+calibrations.LUTrespcorrgain(2)+calibrations.LUTrespcorrgain(3))/4;          

			//Add HcalL1TriggerObject to its container
			HcalL1TriggerObject HcalL1TrigObj(cell.rawId(), ped_, gain_);
			HcalL1TrigObjCol->addValues(HcalL1TrigObj);
		}
		else{
			const HcalL1TriggerObject* myL1TObj = conditions.getHcalL1TriggerObject(cell);
			ped_ = myL1TObj->getPedestal();
			gain_ = myL1TObj->getRespGain();
			//debug
			//ofdebug << cell.rawId() << '\t' << ped_ << '\t' << gain_ << '\n';
		}
		
		_ped[id] = ped_;
		_gain[id] = gain_;

	   for (int j = 0; j <= 0x7F; j++) {
	     HcalQIESample adc(j);
	     frame.setSample(0,adc);
	     coder.adc2fC(frame,samples);
	     float adc2fC_ = samples[0];
	     if ( ieta < 0 ) inputLUT[id][j] = (LUT) std::min(std::max(0,int((adc2fC_ - ped_)*gain_*Rcalib[abs(ieta)+1]/divide)), 0x3FF);
	     else inputLUT[id][j] = (LUT) std::min(std::max(0,int((adc2fC_ - ped_)*gain_*Rcalib[abs(ieta)+44]/divide)), 0x3FF);
	   }
	 }
       }        
       for (int ieta=-41; ieta <= 41; ieta++) {
		HcalDetId cell(HcalForward,ieta,iphi,depth);
		if (theTopo.valid(cell)) {  
			id = GetLUTID(HcalForward,ieta,iphi,depth);
			if (inputLUT[id] == 0) throw cms::Exception("PROBLEM: inputLUT has not been initialized for HF, ieta, iphi, depth, id = ") << ieta << "," << iphi << "," << depth << "," << id << std::endl;
	   //conditions.makeHcalCalibration (cell, &calibrations);
		//
		const HcalQIECoder* channelCoder = conditions.getHcalCoder (cell);
		HcalCoderDb coder (*channelCoder, *shape);
		HBHEDataFrame frame(cell);
	   frame.setSize(1);
	   CaloSamples samples(cell, 1);
		float ped_ = 0;
		float gain_ = 0;

		if (LUTGenerationMode){
			HcalCalibrations calibrations = conditions.getHcalCalibrations(cell);
	   	ped_ = (calibrations.pedestal(0)+calibrations.pedestal(1)+calibrations.pedestal(2)+calibrations.pedestal(3))/4;
	   	gain_= (calibrations.LUTrespcorrgain(0)+calibrations.LUTrespcorrgain(1)+calibrations.LUTrespcorrgain(2)+calibrations.LUTrespcorrgain(3))/4;          

			//Add HcalL1TriggerObject to its container
			HcalL1TriggerObject HcalL1TrigObj(cell.rawId(), ped_, gain_);
			HcalL1TrigObjCol->addValues(HcalL1TrigObj);
		}
		else{
			const HcalL1TriggerObject* myL1TObj = conditions.getHcalL1TriggerObject(cell);
			ped_ = myL1TObj->getPedestal();
			gain_ = myL1TObj->getRespGain();
			//debug
			//ofdebug << cell.rawId() << '\t' << ped_ << '\t' << gain_ << '\n';
		}
		
		_ped[id] = ped_;
		_gain[id] = gain_;

				
			int offset = (abs(ieta) >= 33 && abs(ieta) <= 36) ? 1 : 0; // Lumi offset of 1 for the four rings used to measure lumi
			for (int j = 0; j <= 0x7F; j++) {
				HcalQIESample adc(j);
				frame.setSample(0,adc);
				coder.adc2fC(frame,samples);
				float adc2fC_ = samples[0];
				if (ieta < 0 ) inputLUT[id][j] = (LUT) std::min(std::max(0,int((adc2fC_ - ped_ + offset)*Rcalib[abs(ieta)+2]*gain_/lsb_/cosheta_[abs(ieta)])), 0x3FF);
				else inputLUT[id][j] = (LUT) std::min(std::max(0,int((adc2fC_ - ped_ + offset)*Rcalib[abs(ieta)+45]*gain_/lsb_/cosheta_[abs(ieta)])), 0x3FF);
			}
		}
       }
     }
   }

	if(LUTGenerationMode && DumpL1TriggerObjects){
		//Test Dump HcalL1TriggerObjects
		HcalL1TrigObjCol->setTagString(TagName);
		HcalL1TrigObjCol->setAlgoString(AlgoName);
		std::string outfilename = "Dump_L1TriggerObjects_";
		outfilename += TagName;
		outfilename += ".txt";
		std::ofstream of(outfilename.c_str());
		HcalDbASCIIIO::dumpObject(of, *HcalL1TrigObjCol);
	}

	//debug
//	for (int i=0;i<nluts;++i){
//		if (inputLUT[i] != 0){
//			ofdebug  << i << ":\t";
//			for (int j=0;j<INPUT_LUT_SIZE;++j) ofdebug << (LUT) inputLUT[i][j] << ' ';
//			ofdebug  << std::endl;
//		}
//	}
}

 void HcaluLUTTPGCoder::adc2Linear(const HBHEDataFrame& df, IntegerCaloSamples& ics) const {
   int id = GetLUTID(df.id().subdet(), df.id().ieta(), df.id().iphi(), df.id().depth());
   if (inputLUT[id]==0) {
     throw cms::Exception("Missing Data") << "No LUT for " << df.id();
   } 
   else {
     for (int i=0; i<df.size(); i++){
       if (df[i].adc() >= INPUT_LUT_SIZE || df[i].adc() < 0) throw cms::Exception("ADC overflow for HBHE tower: ") << i << " adc= " << df[i].adc();
       if (inputLUT[id] !=0) ics[i]=inputLUT[id][df[i].adc()];
     }  
   }
 }

 void HcaluLUTTPGCoder::adc2Linear(const HFDataFrame& df, IntegerCaloSamples& ics)  const{
   int id = GetLUTID(df.id().subdet(), df.id().ieta(), df.id().iphi(), df.id().depth());
   if (inputLUT[id]==0) {
     throw cms::Exception("Missing Data") << "No LUT for " << df.id();
   } else {
     for (int i=0; i<df.size(); i++){
       if (df[i].adc() >= INPUT_LUT_SIZE || df[i].adc() < 0)
	 throw cms::Exception("ADC overflow for HF tower: ") << i << " adc= " << df[i].adc();
       if (inputLUT[id] !=0) ics[i]=inputLUT[id][df[i].adc()];
     }
   }
 }

unsigned short HcaluLUTTPGCoder::adc2Linear(HcalQIESample sample, HcalDetId id) const {
  int ref = GetLUTID(id.subdet(), id.ieta(), id.iphi(), id.depth());
  return inputLUT[ref][sample.adc()];
}

float HcaluLUTTPGCoder::getLUTPedestal(HcalDetId id) const {
  int ref = GetLUTID(id.subdet(), id.ieta(), id.iphi(), id.depth());
  return _ped[ref];
}

float HcaluLUTTPGCoder::getLUTGain(HcalDetId id) const {
  int ref = GetLUTID(id.subdet(), id.ieta(), id.iphi(), id.depth());
  return _gain[ref];
}
