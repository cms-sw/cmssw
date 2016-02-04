#include "DQM/SiStripCommissioningSummary/interface/ViewTranslator.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <fstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void ViewTranslator::buildMaps( const SiStripFedCabling& cabling, 
				Mapping& det_to_fec, 
				Mapping& fed_to_fec ) {
  
//   if ( !cabling ) {
//     edm::LogWarning(mlCabling_) 
//       << "[ViewTranslator::" << __func__ << "]"
//       << " NULL pointer to FED cabling object!";
//     return;
//   }
  
  // Iterator through cabling, construct keys and push back into std::map
  std::vector<uint16_t>::const_iterator ifed = cabling.feds().begin();
  for ( ; ifed != cabling.feds().end(); ifed++ ) { 

    const std::vector<FedChannelConnection>& conns = cabling.connections( *ifed );
    std::vector<FedChannelConnection>::const_iterator ichan;
    for( ichan = conns.begin(); ichan != conns.end(); ichan++ ) {
      if( ichan->fedId() ) { 
	
	uint32_t fed = SiStripFedKey( *ifed, 
				      SiStripFedKey::feUnit(ichan->fedCh()),
				      SiStripFedKey::feChan(ichan->fedCh()) ).key();
	
	uint32_t fec = SiStripFecKey( ichan->fecCrate(),
				      ichan->fecSlot(),
				      ichan->fecRing(),
				      ichan->ccuAddr(),
				      ichan->ccuChan(),
				      ichan->lldChannel() ).key();
	
	SiStripDetId det_id( ichan->detId(),
			     ichan->apvPairNumber() ); 
	uint32_t det = SiStripDetKey( det_id ).key();
	
	det_to_fec[det] = fec;
	fed_to_fec[fed] = fec;
	
      } 
    } 
  } 
  
  LogTrace(mlCabling_) 
    << "[ViewTranslator::" << __func__ << "]"
    << " Size of FedToFec std::map: " << fed_to_fec.size()
    << ", size of DetToFec std::map: " << det_to_fec.size(); 
  
}


// -----------------------------------------------------------------------------
//
uint32_t ViewTranslator::fedToFec( const uint32_t& fed_key_mask, 
				   const Mapping& input,
				   Mapping& output ) {
  
  if( input.empty() ) { 
    edm::LogWarning(mlCabling_) 
      << "[ViewTranslator::" << __func__ << "]"
      << " Input std::map is empty!";
    return 0; 
  }
  
//   Mapping::iterator iter;
//   SiStripFedKey fed_key( fed_key_mask );
  
//   if( fed_key.detId() == sistrip::invalid_ ||
//       fed_key.apvPair() == sistrip::invalid_ ) {
//     edm::LogWarning(mlCabling_) 
//       << "[ViewTranslator::" << __func__ << "]"
//       << " DetKey is not defined!";
//     output = input;
//     return output.size(); 
//   }
  
//   if( fed_key.detId() != sistrip::invalid_ && 
//       fed_key.apvPair() != sistrip::invalid_ ) {
//     iter=input->find( fed_key_mask );
//     output[ (*iter).first ] = (*iter).second;
//     LogTrace(mlSummaryPlots_) << "both are not masked";
//   }
  
//   if( fed_key.detId()!=0xFFFFFFFF && fed_key.apvPair()==0xFFFF ) {
//     LogTrace(mlSummaryPlots_) << "apv is masked";
//     for(iter=input->begin() ; iter!=input->end() ; iter++) {
//       DetKey = SiStripFedKey( (*iter).first );
//       if(fed_key.detId()==DetKey.detId())
// 	output[ (*iter).first ]=( (*iter).second );
//     } //for(iter=input->begin() ; iter!=input->end() ; iter++)
//   }//if( fed_key.detId_!=0xFFFFFFFF && fed_key.apvPair_==0xFFFF )
//   else LogTrace(mlSummaryPlots_) << "Cannot find the det to fec std::map in the root file. ";

  return 0;
  
}

// -----------------------------------------------------------------------------
//
uint32_t ViewTranslator::detToFec( const uint32_t& det_key_mask, 
				   const Mapping& input,
				   Mapping& output ) {
  
//   if( input.empty() ) { 
//     edm::LogWarning(mlCabling_) 
//       << "[ViewTranslator::" << __func__ << "]"
//       << " Input std::map is empty!";
//     return 0 ;
//   }
  
//   Mapping::iterator iter;
//   SiStripDetKey::Path det_key = SiStripDetKey::path( det_key_mask );
  
//   if( det_key.detId_ == sistrip::invalid_ ||
//       det_key.apvPair_ == sistrip::invalid_ ) {
//     edm::LogWarning(mlCabling_) 
//       << "[ViewTranslator::" << __func__ << "]"
//       << " DetKey is not defined!";
//     output = input;
//     return output.size(); 
//   }
  
//   if( det_key.detId_ != sistrip::invalid_ && 
//       det_key.apvPair_ != sistrip::invalid_ ) {
//     iter=input->find( det_key_mask );
//     output[ (*iter).first ] = (*iter).second;
//     LogTrace(mlSummaryPlots_) << "both are not masked";
//   }
  
//   if( det_key.detId_!=0xFFFFFFFF && det_key.apvPair_==0xFFFF ) {
//     LogTrace(mlSummaryPlots_) << "apv is masked";
//     for(iter=input->begin() ; iter!=input->end() ; iter++) {
//       DetKey = SiStripDetKey::path( (*iter).first );
//       if(det_key.detId_==DetKey.detId_)
// 	output[ (*iter).first ]=( (*iter).second );
//     } //for(iter=input->begin() ; iter!=input->end() ; iter++)
//   }//if( det_key.detId_!=0xFFFFFFFF && det_key.apvPair_==0xFFFF )
//   else LogTrace(mlSummaryPlots_) << "Cannot find the det to fec std::map in the root file. ";

  return 0; //@@ temp!

}

// -----------------------------------------------------------------------------
//
// void ViewTranslator::fedToFec( const uint32_t& fed_key_mask, 
// 			       const Mapping& input,
// 			       Mapping& output ) {

  
//       Mapping::iterator iter;
//       uint16_t fedId=0;   //fedId
//       uint16_t feUnit=0;  //Front End Unit
//       uint16_t feChan=0;   //Front End Channel
//       uint16_t fedApv=0;
      
//       //unpack the FED key to tranlsate into the corresponding FEC key(s)
//       SiStripFedKey::Path FedKey = SiStripFedKey::path(fed_key_mask);
//       fedId=FedKey.fedId_;
//       feUnit=FedKey.feUnit_;
//       feChan=FedKey.feChan_;
//       fedApv=FedKey.fedApv_;
            
       
//       if(fedId==0 && feUnit==0 && feChan==0 && fedApv == sistrip::invalid_ ) {
// 	output=*(input);
//       }
      
//       if(fedId!=0 && feUnit!=0 && feChan!=0 && fedApv == sistrip::invalid_) {
//        	 iter=input->find(fed_key_mask);
// 	 output[fed_key_mask]=(*iter).second;
//        } 
       
       
//        if(fedId!=0 && feUnit!=0 && feChan==0 && fedApv == sistrip::invalid_ ) {
// 	 for(iter=input->begin(); iter!=input->end(); iter++) {
// 	   FedKey = SiStripFedKey::path( (*iter).first );
// 	   if(fedId==FedKey.fedId_ && feUnit==FedKey.feUnit_) {
// 	     output[ (*iter).first ] = (*iter).second;
// 	   }// if
// 	 }// for 
//        }// if 
       
//        if(fedId!=0 && feUnit==0 && feChan==0) {
// 	 for(iter=input->begin(); iter!=input->end(); iter++) {
// 	   FedKey = SiStripFedKey::path( (*iter).first ); //FedKey is the key from the std::map
// 	   if( fedId==FedKey.fedId_ ) {
// 	     output[ (*iter).first ] = (*iter).second;
// 	   } else LogTrace(mlSummaryPlots_) << "The fedId cannot be found. Please check readout path";
// 	 } //for
//        }//if 
       
      
      
//     } else LogTrace(mlSummaryPlots_) << "The fed to fec std::map could not be found in the root file" << endl << "Please load the ViewTranslator module to create the requisite std::maps";
    
    
//     f->Close();
        
//   } //if(TFile::Open(fname.cstr())
//   else LogTrace(mlSummaryPlots_) << "Error:Cannot open root file.";

// }

// -----------------------------------------------------------------------------
//
void ViewTranslator::writeMapsToFile( std::string fname, 
				      Mapping& det, 
				      Mapping& fed ) {
  
//   Mapping *det_to_fec;
//   Mapping *fed_to_fec;
  
//   det_to_fec = &det;
//   fed_to_fec = &fed;
  
//   if(TFile::Open(fname.c_str())!=NULL) {
//     TFile *f=TFile::Open(fname.c_str(), "UPDATE");
//     gDirectory->cd();
//     TDirectory *std::mapdir = gDirectory;
//     gDirectory->cd("/DQMData/SiStrip");
//     std::mapdir=gDirectory;
//     std::mapdir->WriteObject(det_to_fec, "det_to_fec");
//     std::mapdir->WriteObject(fed_to_fec, "fed_to_fec");
//     LogTrace(mlSummaryPlots_) << "Wrote the std::maps";
//     f->Close();
//   } else LogTrace(mlSummaryPlots_) << "Cannot find root file. Maps not written.";

}

