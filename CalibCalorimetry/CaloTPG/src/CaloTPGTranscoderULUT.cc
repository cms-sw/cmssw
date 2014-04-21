#include "CalibCalorimetry/CaloTPG/src/CaloTPGTranscoderULUT.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <fstream>
#include <math.h>

//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"

using namespace std;


CaloTPGTranscoderULUT::CaloTPGTranscoderULUT(const std::string& compressionFile,
                                             const std::string& decompressionFile)
                                                : isLoaded_(false), nominal_gain_(0.), rctlsb_factor_(0.),
                                                  compressionFile_(compressionFile),
                                                  decompressionFile_(decompressionFile)
{
  for (int i = 0; i < NOUTLUTS; i++) outputLUT_[i] = 0;
}

CaloTPGTranscoderULUT::~CaloTPGTranscoderULUT() {
  for (int i = 0; i < NOUTLUTS; i++) {
    if (outputLUT_[i] != 0) delete [] outputLUT_[i];
  }
}

void CaloTPGTranscoderULUT::loadHCALCompress() const{
// Initialize analytical compression LUT's here
   // TODO cms::Error log
  if (OUTPUT_LUT_SIZE != (unsigned int) 0x400) std::cout << "Error: Analytic compression expects 10-bit LUT; found LUT with " << OUTPUT_LUT_SIZE << " entries instead" << std::endl;

  const HcalTopology& topo=theTrigTowerGeometry->topology();

  std::vector<unsigned int> analyticalLUT(OUTPUT_LUT_SIZE, 0);
  std::vector<unsigned int> identityLUT(OUTPUT_LUT_SIZE, 0);

  // Compute compression LUT
  for (unsigned int i=0; i < OUTPUT_LUT_SIZE; i++) {
	analyticalLUT[i] = (unsigned int)(sqrt(14.94*log(1.+i/14.94)*i) + 0.5);
	identityLUT[i] = min(i,0xffu);
  }

   // Version 0 loop
  for (int ieta=-32; ieta <= 32; ieta++){
    //const int version_of_hcal_TPs = 0;
     for (int iphi = 1; iphi <= 72; iphi++){
        HcalTrigTowerDetId id(ieta, iphi);
        if (!topo.validHT(id)) continue;
        int lutId = topo.detId2denseIdHT(id);
        // TODO cms::Error log
        if (outputLUT_[lutId] != 0){
           std::cout << "Error: LUT with (ieta,iphi) = (" << ieta << "," << iphi << ") has been previously allocated!" << std::endl;
           continue;
        }

        outputLUT_[lutId] = new LUT[OUTPUT_LUT_SIZE];

        const HcalLutMetadatum *meta = lutMetadata_->getValues(id);
        int threshold = meta->getOutputLutThreshold();

        for (int i = 0; i < threshold; ++i)
           outputLUT_[lutId][i] = 0;

        for (unsigned int i = threshold; i < OUTPUT_LUT_SIZE; ++i)
	  outputLUT_[lutId][i] = (abs(ieta) < theTrigTowerGeometry->firstHFTower(id.version())) ? analyticalLUT[i] : identityLUT[i];
     } //for iphi
  } //for ieta
   // Version 1 loop
  for (int ieta=-41; ieta <= 41; ieta++){
      const int version_of_hcal_TPs = 1;
      for (int iphi = 1; iphi <= 72; iphi++){
          HcalTrigTowerDetId id(ieta, iphi);
          id.setVersion(version_of_hcal_TPs);

          if (!topo.validHT(id)) continue;
          int lutId = topo.detId2denseIdHT(id);
          // TODO cms::Error log
          if (outputLUT_[lutId] != 0){
              std::cout << "Error: LUT with (ieta,iphi) = (" << ieta << "," << iphi << ") has been previously allocated!" << std::endl;
              continue;
          }

          outputLUT_[lutId] = new LUT[OUTPUT_LUT_SIZE];

          const HcalLutMetadatum *meta = lutMetadata_->getValues(id);
          int threshold = meta->getOutputLutThreshold();

          for (int i = 0; i < threshold; ++i)
              outputLUT_[lutId][i] = 0;

          for (unsigned int i = threshold; i < OUTPUT_LUT_SIZE; ++i)
              outputLUT_[lutId][i] = (abs(ieta) < theTrigTowerGeometry->firstHFTower(id.version())) ? analyticalLUT[i] : identityLUT[i];
      } //for iphi
  } //for ieta
}

void CaloTPGTranscoderULUT::loadHCALCompress(const std::string& filename) const{
  // TODO: Add Version 1 support
  //  const int version_of_hcal_TPs = 0;
  int tool;
  std::ifstream userfile;
  std::vector< std::vector<LUT> > outputluts;
  const HcalTopology& topo=theTrigTowerGeometry->topology();

  std::cout << "Initializing compression LUT's from " << (char *)filename.data() << std::endl;
  for (int i = 0; i < NOUTLUTS; i++) outputLUT_[i] = 0;
  int maxid = 0, minid = 0x7FFFFFFF, rawid = 0;
  for (int ieta=-32; ieta <= 32; ieta++) {
    for (int iphi = 1; iphi <= 72; iphi++) {
      HcalTrigTowerDetId id(ieta,iphi);
      if (topo.validHT(id)) {
        rawid = topo.detId2denseIdHT(id);
        if (outputLUT_[rawid] != 0) std::cout << "Error: LUT with (ieta,iphi) = (" << ieta << "," << iphi << ") has been previously allocated!" << std::endl;
        else outputLUT_[rawid] = new LUT[OUTPUT_LUT_SIZE];
        if (rawid < minid) minid = rawid;
        if (rawid > maxid) maxid = rawid;
      }
    }
  }

  userfile.open((char *)filename.data());

  if( userfile ) {
    int nluts = 0;
	std::string s;
    std::vector<int> loieta,hiieta;
    std::vector<int> loiphi,hiiphi;
    getline(userfile,s);

    getline(userfile,s);

	unsigned int index = 0;
	while (index < s.length()) {
	  while (isspace(s[index])) index++;
	  if (index < s.length()) nluts++;
	  while (!isspace(s[index])) index++;
	}
	for (unsigned int i=0; i<=s.length(); i++) userfile.unget(); //rewind last line
    outputluts.resize(nluts);
    for (int i=0; i<nluts; i++) outputluts[i].resize(OUTPUT_LUT_SIZE);


    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loieta.push_back(tool);
    }

    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiieta.push_back(tool);
    }

    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      loiphi.push_back(tool);

    }

    for (int i=0; i<nluts; i++) {
      userfile >> tool;
      hiiphi.push_back(tool);

    }

    for (unsigned int j=0; j<OUTPUT_LUT_SIZE; j++) {
      for(int i=0; i <nluts; i++) {
		userfile >> tool;
		if (tool < 0) {
			std::cout << "Error: LUT can't have negative numbers; 0 used instead: " << i << ", " << j << " : = " << tool << std::endl;
			tool = 0;
		} else if (tool > 0xff) {
			std::cout << "Error: LUT can't have >8-bit numbers; 0xff used instead: " << i << ", " << j << " : = " << tool << std::endl;
			tool = 0xff;
		}
		outputluts[i][j] = tool;
		if (userfile.eof()) std::cout << "Error: LUT file is truncated or has a wrong format: " << i << "," << j << std::endl;
	  }
    }
    userfile.close();

	HcalDetId cell;
	int id, ntot = 0;
    for (int i=0; i < nluts; i++) {
      int nini = 0;
      for (int iphi = loiphi[i]; iphi <= hiiphi[i]; iphi++) {
        for (int ieta=loieta[i]; ieta <= hiieta[i]; ieta++) {
          HcalTrigTowerDetId detid(ieta,iphi);
          if (topo.validHT(detid)) {
            id = topo.detId2denseIdHT(detid);
            if (outputLUT_[id] == 0) throw cms::Exception("PROBLEM: outputLUT_ has not been initialized for ieta, iphi, id = ") << ieta << ", " << iphi << ", " << id << std::endl;
            for (int j = 0; j <= 0x3ff; j++) outputLUT_[id][j] = outputluts[i][j];
            nini++;
            ntot++;
          }
        }
      }
    }
  } else {
      loadHCALCompress();
  }
}

void CaloTPGTranscoderULUT::loadHCALUncompress() const {
   hcaluncomp_.clear();
   for (int i = 0; i < NOUTLUTS; i++){
      RCTdecompression decompressionTable(TPGMAX,0);
      hcaluncomp_.push_back(decompressionTable);
   }
   const HcalTopology& topo=theTrigTowerGeometry->topology();
   
   // Version 0 loop
   for (int ieta = -32; ieta <= 32; ++ieta){

     const int version_of_hcal_TPs = 0;
     
     double eta_low = 0., eta_high = 0.;
     theTrigTowerGeometry->towerEtaBounds(ieta,version_of_hcal_TPs,eta_low,eta_high);
     double cosh_ieta = fabs(cosh((eta_low + eta_high)/2.));
     
     for (int iphi = 1; iphi <= 72; iphi++) {
       HcalTrigTowerDetId id(ieta, iphi);
       
       if (!topo.validHT(id)) continue; 
       int lutId = topo.detId2denseIdHT(id);
       
       double factor = 0.;
       
       // HF
       if (abs(ieta) >= theTrigTowerGeometry->firstHFTower(version_of_hcal_TPs)) {
         factor = rctlsb_factor_;
       }
       // HBHE
       else {
         const HcalLutMetadatum *meta = lutMetadata_->getValues(id);
         factor = nominal_gain_ / cosh_ieta * meta->getLutGranularity();
       }
       
       // tpg - compressed value
       unsigned int tpg = outputLUT_[lutId][0];
       int low = 0;
       for (unsigned int i = 0; i < OUTPUT_LUT_SIZE; ++i){
	 if (outputLUT_[lutId][i] != tpg){
	   unsigned int mid = (low + i)/2;
	   hcaluncomp_[lutId][tpg] = (tpg == 0 ? low : factor * mid);
	   low = i;
	   tpg = outputLUT_[lutId][i];
	 }
       }
       hcaluncomp_[lutId][tpg] = factor * low;
     } // for phi
   } // for ieta
   
   // Version 1 loop
   for (int ieta = -41; ieta <= 41; ++ieta) {

       const int version_of_hcal_TPs = 1;
       if (abs(ieta) < 28) { continue; }

       double eta_low = 0., eta_high = 0.;
       theTrigTowerGeometry->towerEtaBounds(ieta, version_of_hcal_TPs, eta_low, eta_high);
       double cosh_ieta = fabs(cosh((eta_low + eta_high)/2.));

       for (int iphi = 1; iphi <= 72; iphi++) {
           HcalTrigTowerDetId id(ieta, iphi);
           id.setVersion(version_of_hcal_TPs);
           if (!topo.validHT(id)) continue; 
           int lutId = topo.detId2denseIdHT(id);

           double factor = 0.;
           // HF
           if (abs(ieta) >= theTrigTowerGeometry->firstHFTower(version_of_hcal_TPs)) {
               factor = rctlsb_factor_;
           }
           // HBHE
           else {
               const HcalLutMetadatum *meta = lutMetadata_->getValues(id);
               factor = nominal_gain_ / cosh_ieta * meta->getLutGranularity();
           }

           // tpg - compressed value
           unsigned int tpg = outputLUT_[lutId][0];
           int low = 0;
           for (unsigned int i = 0; i < OUTPUT_LUT_SIZE; ++i) {
               if (outputLUT_[lutId][i] != tpg) {
                   unsigned int mid = (low + i)/2;
                   if (tpg == 0) {
                       hcaluncomp_[lutId][tpg] = low;
                   } else {
                       hcaluncomp_[lutId][tpg] = factor * mid;
                   }
                   low = i;
                   tpg = outputLUT_[lutId][i];
               }
           }
           hcaluncomp_[lutId][tpg] = factor * low;
       }  // for phi
   } // for ieta
}

void CaloTPGTranscoderULUT::loadHCALUncompress(const std::string& filename) const {
  // TODO: Add Version 1 support
  //  const int version_of_hcal_TPs = 0;
  std::ifstream userfile;
  userfile.open(filename.c_str());

  hcaluncomp_.resize(NOUTLUTS);
  for (int i = 0; i < NOUTLUTS; i++) hcaluncomp_[i].resize(TPGMAX);

  static const int etabound = 32;
  if( userfile ) {
    double et;
    for (int i=0; i<TPGMAX; i++) {
      for(int j = 1; j <=etabound; j++) {
        userfile >> et;
        for (int iphi = 1; iphi <= 72; iphi++) {
          HcalTrigTowerDetId id(j,iphi);
          int itower = theTrigTowerGeometry->topology().detId2denseIdHT(id);

          if (itower >= 0) hcaluncomp_[itower][i] = et;
          HcalTrigTowerDetId id2(-j,iphi);
          itower = theTrigTowerGeometry->topology().detId2denseIdHT(id2);
          if (itower >= 0) hcaluncomp_[itower][i] = et;
        }
      }
    }
    userfile.close();
  }
  else {
    loadHCALUncompress();
  }
}

HcalTriggerPrimitiveSample CaloTPGTranscoderULUT::hcalCompress(const HcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const {

  int itower = theTrigTowerGeometry->topology().detId2denseIdHT(id);

  if (itower < 0) cms::Exception("Invalid Data") << "No trigger tower found for ieta, iphi = " << id.ieta() << ", " << id.iphi() << " v"<<id.version();
  if (sample >= OUTPUT_LUT_SIZE) {

    throw cms::Exception("Out of Range") << "LUT has 1024 entries for " << itower << " but " << sample << " was requested.";
    sample=OUTPUT_LUT_SIZE - 1;
  }

  return HcalTriggerPrimitiveSample(outputLUT_[itower][sample],fineGrain,0,0);
}

double CaloTPGTranscoderULUT::hcaletValue(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc) const {
  if (hcaluncomp_.empty()) loadHCALUncompress(decompressionFile_);

  int compET = hc.compressedEt();	// to be within the range by the class
  int itower = theTrigTowerGeometry->topology().detId2denseIdHT(hid);
  double etvalue = hcaluncomp_[itower][compET];
  return(etvalue);
}

EcalTriggerPrimitiveSample CaloTPGTranscoderULUT::ecalCompress(const EcalTrigTowerDetId& id, unsigned int sample, bool fineGrain) const {
  throw cms::Exception("Not Implemented") << "CaloTPGTranscoderULUT::ecalCompress";
}

void CaloTPGTranscoderULUT::rctEGammaUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
						const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec,
						unsigned int& et, bool& egVecto, bool& activity) const {
  throw cms::Exception("Not Implemented") << "CaloTPGTranscoderULUT::rctEGammaUncompress";
}
void CaloTPGTranscoderULUT::rctJetUncompress(const HcalTrigTowerDetId& hid, const HcalTriggerPrimitiveSample& hc,
					     const EcalTrigTowerDetId& eid, const EcalTriggerPrimitiveSample& ec,
					     unsigned int& et) const {
  throw cms::Exception("Not Implemented") << "CaloTPGTranscoderULUT::rctJetUncompress";
}

std::vector<unsigned char> CaloTPGTranscoderULUT::getCompressionLUT(HcalTrigTowerDetId id) const {
   std::vector<unsigned char> lut;
   int itower = theTrigTowerGeometry->topology().detId2denseIdHT(id);
   if (itower >= 0) {
         lut.resize(OUTPUT_LUT_SIZE);
	 for (unsigned int i = 0; i < OUTPUT_LUT_SIZE; i++) lut[i]=outputLUT_[itower][i];
   }
   return lut;
}

void CaloTPGTranscoderULUT::setup(const edm::EventSetup& es, Mode mode=All) const{
   if (isLoaded_) return;
   // TODO Try/except
   es.get<HcalLutMetadataRcd>().get(lutMetadata_);
   es.get<CaloGeometryRecord>().get(theTrigTowerGeometry);

   nominal_gain_ = lutMetadata_->getNominalGain();
   float rctlsb =lutMetadata_->getRctLsb();
   if (rctlsb != 0.25 && rctlsb != 0.5)
      throw cms::Exception("RCTLSB") << " value=" << rctlsb << " (should be 0.25 or 0.5)" << std::endl;
   rctlsb_factor_ = rctlsb;

   if (compressionFile_.empty() && decompressionFile_.empty()) {
      loadHCALCompress();
   }
   else {
      // TODO Message to discourage using txt.
      std::cout << "From Text File:" << std::endl;
      loadHCALCompress(compressionFile_);
   }
   isLoaded_ = true;
}

void CaloTPGTranscoderULUT::printDecompression() const{
   std::cout << "RCT Decompression table" << std::endl;
   const int version_of_hcal_TPs = 0; // appropriate for RCT
   for (int i=0; i < 256; i++) {
      for (int j=1; j <= theTrigTowerGeometry->nTowers(version_of_hcal_TPs); j++)
         std::cout << int(hcaletValue(j,i)*100. + 0.5)/100. << " ";
      std::cout << std::endl;
      for (int j=1; j <= theTrigTowerGeometry->nTowers(version_of_hcal_TPs); j++)
         if (hcaletValue(j,i) != hcaletValue(-j,i))
            cout << "Error: decompression table for ieta = +/- " << j << " disagree! " << hcaletValue(-j,i) << ", " << hcaletValue(j,i) << endl;
   }
}

