#include "DQM/SiStripCommissioningSummary/interface/ViewTranslator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"
//
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
//
#include "TFile.h"
//
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
void ViewTranslator::generateMaps( SiStripFedCabling* fed_cabling, 
				   Mapping& det_to_fec, 
				   Mapping& fed_to_fec ) {
  
//   if ( !fed_cabling ) {
//     edm::LogWarning(mlCabling_) 
//       << "[ViewTranslator::" << __func__ << "]"
//       << " NULL pointer to FED cabling object!";
//     return;
//   }
  
//   // Iterator over FED cabling, construct keys and push back into map
//   vector<uint16_t>::const_iterator ifed; 
//   for ( ifed = fed_cabling->feds().begin();ifed != fed_cabling->feds().end(); ifed++ ) { 

//     const vector<FedChannelConnection>& conns = fed_cabling->connections( *ifed );
//     vector<FedChannelConnection>::iterator ichan;
//     for( ichan = conns.begin(); ichan != conns.end(); ichan++ ) {
//       if( ichan->fedId() ) { 
	
// 	uint32_t fed = SiStripFedKey::key( *ifed, 
// 					   ichan->fedCh() );
	
// 	uint32_t fec = SiStripFecKey::key( ichan->fecCrate(),
// 					   ichan->fecSlot(),
// 					   ichan->fecRing(),
// 					   ichan->ccuAddr(),
// 					   ichan->ccuChan(),
// 					   ichan->lldChannel() );
	
// 	uint32_t det = SiStripDetKey::key( ichan->detId(),
// 					   ichan->apvPairNumber() );
	
// 	det_to_fec[det] = fec;
// 	fed_to_fec[fed] = fec;
	
//       } 
//     } 
//   } 

//   LogTrace(mlCabling_) 
//     << "[ViewTranslator::" << __func__ << "]"
//     << " Map sizes: DetToFec=" << det_to_fec.size() 
//     << " FedToFec=" << fed_to_fec.size();
  
}

// -----------------------------------------------------------------------------
//
uint32_t ViewTranslator::detToFec( const uint32_t& det_key_mask, 
				   const Mapping& input,
				   Mapping& output ) {
  
//   if( input.empty() ) { 
//     edm::LogWarning(mlCabling_) 
//       << "[ViewTranslator::" << __func__ << "]"
//       << " Input map is empty!";
//     return 0; 
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
//     cout << "both are not masked" << endl;
//   }
  
//   if( det_key.detId_!=0xFFFFFFFF && det_key.apvPair_==0xFFFF ) {
//     cout << "apv is masked" << endl;
//     for(iter=input->begin() ; iter!=input->end() ; iter++) {
//       DetKey = SiStripDetKey::path( (*iter).first );
//       if(det_key.detId_==DetKey.detId_)
// 	output[ (*iter).first ]=( (*iter).second );
//     } //for(iter=input->begin() ; iter!=input->end() ; iter++)
//   }//if( det_key.detId_!=0xFFFFFFFF && det_key.apvPair_==0xFFFF )
//   else cout << "Cannot find the det to fec map in the root file. " << endl;

}

// -----------------------------------------------------------------------------
//
void ViewTranslator::fedToFec( const uint32_t& fed_key_mask, 
			       const Mapping& input,
			       Mapping& output ) {

  
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
// 	   FedKey = SiStripFedKey::path( (*iter).first ); //FedKey is the key from the map
// 	   if( fedId==FedKey.fedId_ ) {
// 	     output[ (*iter).first ] = (*iter).second;
// 	   } else cout << "The fedId cannot be found. Please check readout path" << endl;
// 	 } //for
//        }//if 
       
      
      
//     } else cout << "The fed to fec map could not be found in the root file" << endl << "Please load the ViewTranslator module to create the requisite maps" << endl;
    
    
//     f->Close();
        
//   } //if(TFile::Open(fname.cstr())
//   else cout << "Error:Cannot open root file." << endl;

}

// -----------------------------------------------------------------------------
//
void ViewTranslator::writeMapsToFile( string fname, 
				      Mapping& det, 
				      Mapping& fed ) {
  
//   Mapping *det_to_fec;
//   Mapping *fed_to_fec;
  
//   det_to_fec = &det;
//   fed_to_fec = &fed;
  
//   if(TFile::Open(fname.c_str())!=NULL) {
//     TFile *f=TFile::Open(fname.c_str(), "UPDATE");
//     gDirectory->cd();
//     TDirectory *mapdir = gDirectory;
//     gDirectory->cd("/DQMData/SiStrip");
//     mapdir=gDirectory;
//     mapdir->WriteObject(det_to_fec, "det_to_fec");
//     mapdir->WriteObject(fed_to_fec, "fed_to_fec");
//     cout << "Wrote the maps" << endl;
//     f->Close();
//   } else cout << "Cannot find root file. Maps not written." << endl;

}

