#ifndef SiPixelFedFillerWordEventNumber_H
#define SiPixelFedFillerWordEventNumber_H

// user include files
#include <stdio.h>
#include <vector>
#include <iostream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

//===== class decleration
class SiPixelFedFillerWordEventNumber  : public edm::EDProducer {
   public:
      explicit SiPixelFedFillerWordEventNumber (const edm::ParameterSet&);
      ~SiPixelFedFillerWordEventNumber ();
      std::string label;
      std::string instance;
      bool SaveFillerWordsbool;
      
   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::ParameterSet config_;
      int status; 
      unsigned int EventNum;
      
      // ============= member data =========================================
      int PwordSlink64(uint64_t *, const int, uint32_t &);
      unsigned int CalibStatFillWord(unsigned int, int);
      unsigned int CalibStatFill; 
      std::vector<uint32_t>		      vecSaveFillerWords;
      std::vector<uint32_t>::iterator	      vecSaveFillerWords_It;
      std::vector<uint32_t>		      vecFillerWordsEventNumber1;
      std::vector<uint32_t>::iterator	      vecFillerWordsEventNumber1_It;
      std::vector<uint32_t>		      vecFillerWordsEventNumber2;
      std::vector<uint32_t>::iterator	      vecFillerWordsEventNumber2_It;        
};
#endif
