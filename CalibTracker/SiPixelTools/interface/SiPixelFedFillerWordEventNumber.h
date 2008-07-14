#ifndef SIPIXELFEDFILLERWORDEVENTNUMBER_H
#define SIPIXELFEDFILLERWORDEVENTNUMBER_H

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
class SiPixelFedFillerWordEventNumber : public edm::EDProducer {
   public:
      explicit SiPixelFedFillerWordEventNumber(const edm::ParameterSet&);
      ~SiPixelFedFillerWordEventNumber();
      std::string label;
      std::string instance;
      bool SaveFillerWords_bool;
      
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::ParameterSet config_;
      int status; 
      
      // ============= member data =========================================
      int PwordSlink64(uint64_t * ldata, const int length, uint32_t &totword); 
      std::vector<uint32_t>		      vecSaveFillerWords;
      std::vector<uint32_t>::iterator	      vecSaveFillerWords_It;
      std::vector<uint32_t>		      vecFillerWordsEventNumber;
      std::vector<uint32_t>::iterator	      vecFillerWordsEventNumber_It;      
};
#endif
