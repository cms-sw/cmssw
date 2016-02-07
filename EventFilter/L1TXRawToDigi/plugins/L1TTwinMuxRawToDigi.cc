//-------------------------------------------------
//
//   Class: DTTM7FEDReader
//
//   L1 DT TwinMux Raw-to-Digi
//
//
//
//   Author :
//   C. F. Bedoya  - CIEMAT
//   G. Codispoti -- INFN Bologna
//   J. Pazzini   -- INFN Padova
//
//--------------------------------------------------

#include "L1TTwinMuxRawToDigi.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>

L1TTwinMuxRawToDigi::L1TTwinMuxRawToDigi(const edm::ParameterSet& pset) :

  debug_( pset.getUntrackedParameter<bool>("debug", false) ), 
  passbc0_( pset.getUntrackedParameter<bool>("passbc0", false) ), 
  nfeds_(0),
  DTTM7InputTag_( pset.getParameter<edm::InputTag>("DTTM7_FED_Source") ),
  feds_( pset.getUntrackedParameter<std::vector<int> >("feds", std::vector<int>()) ),
  wheels_( pset.getUntrackedParameter<std::vector<int> >("wheels", std::vector<int>())),
  amcsecmap_( pset.getUntrackedParameter<std::vector<long long int> >("amcsecmap", std::vector<long long int>()))

{
    
  produces<L1MuDTChambPhContainer>();
  produces<L1MuDTChambThContainer>();
  Raw_token = consumes<FEDRawDataCollection> (DTTM7InputTag_);
 
  nfeds_ = feds_.size();
  
  if ( nfeds_ != wheels_.size() )
    throw cms::Exception("TwinMux_unpacker") << "Configuration file error. Size of \'wheels\' and \'feds\' differs.\n";  

  if ( amcsecmap_.size() != wheels_.size() )
    throw cms::Exception("TwinMux_unpacker") << "Configuration file error. Size of \'wheels\' and \'amcsecmap\' differs.\n";  
    
  for (size_t wh_i = 0; wh_i < amcsecmap_.size(); ++wh_i){
    std::array<short, 12> whmap;      
    for (size_t amc_i = 1; amc_i < 13; ++amc_i ){
      short shift = (12-amc_i)*4;
      whmap[amc_i-1] = ( amcsecmap_[wh_i] >> shift ) & 0xF;
    }
    amcsec_.push_back(whmap);
  }
   
}

L1TTwinMuxRawToDigi::~L1TTwinMuxRawToDigi(){}

void L1TTwinMuxRawToDigi::produce(edm::Event& e, 
                             const edm::EventSetup& c) {

  std::auto_ptr<L1MuDTChambPhContainer> TM7phi_product(new L1MuDTChambPhContainer);
  std::auto_ptr<L1MuDTChambThContainer> TM7the_product(new L1MuDTChambThContainer);

  L1MuDTChambPhContainer::Phi_Container phi_data;
  L1MuDTChambThContainer::The_Container the_data;

  if ( !fillRawData(e, phi_data, the_data) ) return;

  TM7phi_product->setContainer(phi_data);
  TM7the_product->setContainer(the_data);

  e.put(TM7phi_product);
  e.put(TM7the_product);

}


bool L1TTwinMuxRawToDigi::fillRawData( edm::Event& e,
                                  L1MuDTChambPhContainer::Phi_Container& phi_data,
                                  L1MuDTChambThContainer::The_Container& the_data ) {

  edm::Handle<FEDRawDataCollection> data;
  e.getByToken( Raw_token, data );

  for ( size_t w_i = 0; w_i < nfeds_; ++w_i ) {
    processFed( feds_[w_i], wheels_[w_i], amcsec_[w_i], data, phi_data, the_data );
  }
  
  return true;
}

int L1TTwinMuxRawToDigi::normBx( int bx_, 
                            int bxCnt_ ){
    
    int bxNorm_ = bx_ - bxCnt_;    
    if ( abs( bxNorm_ ) < 3000 ) return bxNorm_; 
    
    if ( bxNorm_ > 0 ) return bxNorm_ - 3564;
    if ( bxNorm_ < 0 ) return bxNorm_ + 3564;
    
    return -99;
    
}

int L1TTwinMuxRawToDigi::radAngConversion( int radAng_  ) {
    
    if (radAng_>2047) 
        return radAng_-4096;

    return radAng_;
    
}

int L1TTwinMuxRawToDigi::benAngConversion( int benAng_  ) {
    
    if (benAng_>511) 
        return benAng_-1024;

    return benAng_;
    
}

void L1TTwinMuxRawToDigi::processFed( int twinMuxFed, 
                                 int twinMuxWheel,
                                 std::array<short, 12> twinMuxAmcSec,
                                 edm::Handle<FEDRawDataCollection> data,
                                 L1MuDTChambPhContainer::Phi_Container& phiSegments,
                                 L1MuDTChambThContainer::The_Container& theSegments ) {

  /// Container
  std::vector<long> DTTM7WordContainer;

  /// Debug
  std::ofstream logfile;
  if ( debug_ ) {
    std::ostringstream fname;
    fname << "eventDump_" <<  twinMuxFed << ".txt";
    logfile.open( fname.str() );
  }

  /// Header
  FEDRawData TM7data = data->FEDData(twinMuxFed); 
  if ( TM7data.size() == 0 ) return;

  /// Variables
  LineFED_ = TM7data.data();
  int nline  = 0; // counting already include header
  long dataWord = 0;
  int newCRC = 0xFFFF;

  readline( nline, dataWord );
  calcCRC( dataWord, newCRC );

  int TM7fedId = ( dataWord >> 8 ) & 0xFFF;  // positions 8 -> 19
  /*** NOT UNPACKED  
  int bunchCnt = ( dataWord >> 20 ) & 0xFFF;  // positions 20 -> 31
  int eventCnt = ( dataWord >> 32 ) & 0xFFFFFF;  // positions 32 -> 55
  ***/
  int BOEevTy  = ( dataWord >> 56 ) & 0xFf;  // positions 60 -> 63

  int linecounter = 0;
  if ( debug_ ) logfile << '[' << ++linecounter << "]\t"
                        << std::hex << dataWord << std::dec << "\t|\t"
                        << "BOEevTy " << BOEevTy << '\t'
                        << "TM7fedId "  << TM7fedId << '\n';

  if ( (BOEevTy != 0x51) || ( TM7fedId != twinMuxFed ) ) {
            
    edm::LogWarning("TwinMux_unpacker") << "Not a TM7 of FED " 
                                        << twinMuxFed << " header "
                                        << std::hex << dataWord;
    return;
    
  }

  readline( nline, dataWord );
  calcCRC( dataWord, newCRC );

  std::map<int, int> AMCsizes;
  /*** NOT UNPACKED  
  int orbit = ( dataWord >> 4 ) & 0xFFFFFFFF;  // positions 4 -> 35
  ***/
  int nAMC = ( dataWord >> 52 ) & 0xF;  // positions 52 -> 55

  if ( debug_ ) logfile << '[' << ++linecounter << "]\t" << std::hex
                        << dataWord << std::dec << "\t|\t"
                        << "nAMC " << nAMC << '\n';

  for ( int j = 0; j < nAMC; ++j ) {
  
    readline( nline, dataWord ); 
    calcCRC( dataWord, newCRC );
   
    int TM7boardID = dataWord & 0xFFFF;  // positions 0 -> 15
    int AMCno = (dataWord >> 16 ) & 0xF;  // positions 16 -> 19
    /*** NOT UNPACKED  
    int bulkno = (dataWord >> 20 ) & 0xFF;  // positions 20 -> 27
    ***/
    if ( AMCno < 12 ) {
      AMCsizes[AMCno] = ( dataWord >> 32 ) & 0xFFFFFF;  // positions 32 -> 55

      edm::LogInfo("TwinMux_unpacker") << "reading line " << nline 
                                       << " AMCno " << AMCno 
                                       << " boardId " << TM7boardID 
                                       << " size " << AMCsizes[AMCno] << " ";
    }

    if ( debug_ ) logfile << '[' << ++linecounter << "]\t"
                          << std::hex << dataWord
                          << std::dec << "\t|\t"
                          << "AMCsizes[" << AMCno << "] "
                          << AMCsizes[AMCno]
                          << std::dec << '\n';
    //If we want to do something with the subheaders, here is the place
   
  }

  int chkEOE = 0;
  readline( nline, dataWord );

  if ( debug_ ) logfile << '[' << ++linecounter << "]\t" << std::hex
                        << dataWord << std::dec << '\n';

  while ( chkEOE != 0xA0 ) {

    calcCRC( dataWord, newCRC);
    DTTM7WordContainer.push_back( dataWord );

    readline( nline, dataWord );
    if ( debug_ ) logfile << '[' << ++linecounter << "]\t" << std::hex
                          << dataWord << std::dec << '\n';
    chkEOE = ( dataWord >> 56 ) & 0xFF;  // positions 56 -> 64

    if ( nline > 3026 ) { 
      // 3026 = 1(header) + 3024(max # PHTF-ETTF 64 bits words) + 1(trailer)
      edm::LogWarning("TwinMux_unpacker") << "Warning : number of TwinMux nline > 3026 "; 
      return;
    }

  } /// later on read RPC until crc

  calcCRC( dataWord & 0xFFFFFFFF0000FFFF, newCRC); /// needed not to put crc in crc calc
  edm::LogInfo("TwinMux_unpacker") << "TM7 of FED " << TM7fedId 
                                   << ", asking " << twinMuxFed 
                                   << " BOEevTy " << BOEevTy 
                                   << " DTTM7WordContainer.size() " << DTTM7WordContainer.size();

  //--> Trailer
  int CRC = ( dataWord >> 16 ) & 0xFFFF; // positions 17 ->32
  int evtLgth = ( dataWord >> 32 ) & 0xFFFFFF; // positions 33 ->56

  if ( debug_ ) logfile << "\tevtLgth " << std::hex
                        << evtLgth << "\tCRC " << CRC << std::dec << '\n';

  if ( nline != evtLgth ) {
    edm::LogWarning("TwinMux_unpacker") << "Number of words read " << std::dec << nline 
                                        << " and event length " << std::dec << evtLgth 
                                        << " differ ";
    return;
  }

  if ( newCRC != CRC ) {
    edm::LogWarning("TwinMux_unpacker") << "Calculated CRC " << std::hex << newCRC 
                                        << " differs from CRC in trailer " << std::hex << CRC;
    return;
  }

  // --> Analyze event 
  std::vector<long>::iterator DTTM7iterator = DTTM7WordContainer.begin();
  std::vector<long>::iterator DTTM7itend = DTTM7WordContainer.end();

  int lcounter = 0;
  for ( ; DTTM7iterator != DTTM7itend; ++DTTM7iterator ) {

    dataWord  = (*DTTM7iterator);
    int dataLenght = (dataWord & 0xFFFFF);         // positions 0 -> 19
    int bxCounter  = (dataWord >> 20 ) & 0xFFF;    // positions 20 -> 31
    int event      = (dataWord >> 32 ) & 0xFFFFFF; // positions 32 -> 55
    int AMC_ID     = (dataWord >> 56 ) & 0xF;      // positions 56 -> 59
    int control    = (dataWord >> 60 ) & 0xF;      // positions 59 -> 63 
    int wheel      = twinMuxWheel;
    
    if( ( AMC_ID < 1 ) or ( AMC_ID > 12 ) ) {
      edm::LogWarning("TwinMux_unpacker") << "%%%%%% AMC_ID OUT OF RANGE \n"
                                          << " TM7fedId "     << TM7fedId
                                          << " AMC_ID "       << AMC_ID;
      break;
    }
    
    int sector     = twinMuxAmcSec[AMC_ID-1];
    
    if( ( sector < 1 ) or ( sector > 12 ) ) {
      if( sector != 15 ) edm::LogWarning("TwinMux_unpacker") << "%%%%%% VALID AMC_ID POINTS TO SECTOR OUT OF RANGE \n"
                                                             << " TM7fedId "     << TM7fedId
                                                             << " AMC_ID "       << AMC_ID
                                                             << " wheel "        << wheel
                                                             << " sector "       << sector;         
      break;
    }

    if ( debug_ ) logfile << '[' << ++lcounter << "]\t"
                          << std::hex << dataWord << std::dec << "\t|\t"
                          << "AMC_ID "     << AMC_ID << '\t' 
                          << "control "    << control   << '\t' 
                          << "event "      << event  << '\t' 
                          << "bxCounter "  << bxCounter  << '\t' 
                          << "dataLenght " << dataLenght << '\n';

    ++DTTM7iterator; // User word empty  /// ==>> increment 2
    if( DTTM7iterator == DTTM7itend ) {
      edm::LogInfo("TwinMux_unpacker") << "TRAILING WORD AS A PAYLOAD END in FED " 
                                       << std::hex << TM7fedId 
                                       << std::hex << dataWord 
                                       << std::dec<< " [it pos " 
                                       << int(DTTM7iterator - DTTM7itend)  << " ]";
      break;
    }

    dataWord = (*DTTM7iterator);
    int boardID   = (dataWord & 0xFFFF); // positions  0 -> 15
    int orbit     = (dataWord >> 16 ) & 0xFFFF; // positions 15 -> 32
    
    if ( DTTM7iterator == DTTM7itend ) {
      edm::LogWarning("TwinMux_unpacker") << "%%%%%% AMC_ID " << AMC_ID
                                          << " control "      << control
                                          << " event "        << event
                                          << " bxCounter "    << bxCounter
                                          << " size "         << dataLenght
                                          << " orbit "        << orbit
                                          << " board "        << boardID
                                          << " AMCsizes "     << AMCsizes[AMC_ID]
                                          << " it pos "       << int(DTTM7iterator - DTTM7itend);
      break;
    }

    if (debug_ ) logfile << '[' << ++lcounter << "]\t" 
                         << std::hex << dataWord 
                         << std::dec << "\t|\t" 
                         << " orbit " << orbit
                         << " board " << boardID << '\n';

    int AMCsize = AMCsizes[AMC_ID] - 1; /// do not consider the trailer
    int bxID =  99;
    int bc0  = -99;
    int bxNr = -99;
    
    /// 2 words already read, last removed because trailer with CRC
    for ( int tm7eventsize = 2; tm7eventsize < AMCsize; ++tm7eventsize ) {
  
      ++DTTM7iterator; /// ==>> increment 3   
      if ( DTTM7iterator == DTTM7itend ) {
          
        edm::LogWarning("TwinMux_unpacker") << "UNEXPECTED END OF PAYLOAD INSIDE CHAMBER DESCRIPTION"
                                          << " [it pos " << int(DTTM7iterator - DTTM7itend)  << " ]" ;
        break;
        
      }

      long dataWordSub = (*DTTM7iterator);
      int selector = ( dataWordSub >> 60 ) & 0xF; // positions 60 -> 63

      if ( selector == 0x4 ) { //TSC word

        bxID = ( dataWordSub >> 48 ) & 0xFFF; // positions 48 -> 60
        bc0  = ( dataWordSub >> 22 ) & 0x1; // positions 22 -> 23
		
        if ( debug_ ) {
            logfile << '[' << ++lcounter << "]\t" << std::hex
                    << dataWordSub << '\t' << dataWordSub
                    << std::dec << "\t|\t"
                    << "bxID " << bxID << '\t'
                    << "bc0  " << bc0  << '\n';
        }

        bxNr = normBx(bxID, bxCounter); /// bx normalized to the bxcounter

      }//TSC WORD
 
      else if ( selector == 0x1 ) { //MB1/2 word
 
        int mb2_phi =    ( dataWordSub & 0xFFF);        // positions  0 -> 11
        int mb2_phib =   ( dataWordSub >> 12 ) & 0x3FF; // positions 12 -> 21
        int mb2_qual =   ( dataWordSub >> 22 ) & 0x7;   // positions 22 -> 24
        int mb2_ts2tag = ( dataWordSub >> 26 ) & 0x1;   // positions 26
        /*** NOT UNPACKED  
        int mb2_parity = ( dataWordSub >> 27) & 0x1;    // positions 27
        ***/

        int mb1_phi  =   ( dataWordSub >> 32 ) & 0xFFF; // positions 32 -> 43
        int mb1_phib =   ( dataWordSub >> 44 ) & 0x3FF; // positions 44 -> 53
        int mb1_qual =   ( dataWordSub >> 54 ) & 0x7;   // positions 54 -> 56
        int mb1_ts2tag = ( dataWordSub >> 58 ) & 0x1;   // positions 58
        /*** NOT UNPACKED  
        int mb1_parity = ( dataWordSub >> 59 ) &0x1;    // positions 59
        ***/
        
        int mb1_phi_conv  = radAngConversion(mb1_phi);
        int mb1_phib_conv = benAngConversion(mb1_phib);

        int mb2_phi_conv  = radAngConversion(mb2_phi);
        int mb2_phib_conv = benAngConversion(mb2_phib);

        phiSegments.push_back( L1MuDTChambPhDigi( bxNr, wheel, sector-1, 
                                                1, mb1_phi_conv, mb1_phib_conv, 
                                                mb1_qual, mb1_ts2tag, bxCounter ) );
        phiSegments.push_back( L1MuDTChambPhDigi( bxNr, wheel, sector-1, 
                                                2, mb2_phi_conv, mb2_phib_conv, 
                                                mb2_qual, mb2_ts2tag, bxCounter ) );

        if ( debug_ ) logfile << '[' << ++lcounter << "]\t"<< std::hex 
                              << dataWordSub   << std::dec      << "\t|\t"
                              << "mb1_ts2tag " << mb1_ts2tag    << '\t'
                              << "mb1_qual "   << mb1_qual      << '\t'
                              << "mb1_phib "   << mb1_phib_conv << '\t'
                              << "mb1_phi "    << mb1_phi_conv  << '\t'
                              << "mb2_ts2tag " << mb2_ts2tag    << '\t'
                              << "mb2_qual "   << mb2_qual      << '\t'
                              << "mb2_phib "   << mb2_phib_conv << '\t'
                              << "mb2_phi "    << mb2_phi_conv  << '\n';
      }//MB1/2 word
 
      else if ( selector == 0x2 ) {

        int mb4_phi =    ( dataWordSub & 0xFFF);        // positions  0 -> 11
        int mb4_phib =   ( dataWordSub >> 12 ) & 0x3FF; // positions 12 -> 21
        int mb4_qual =   ( dataWordSub >> 22 ) & 0x7;   // positions 22 -> 24
        int mb4_ts2tag = ( dataWordSub >> 26 ) & 0x1;   // positions 26
        /*** NOT UNPACKED  
        int mb4_parity = ( dataWordSub >> 27) & 0x1;    // positions 27
        ***/

        int mb3_phi  =   ( dataWordSub >> 32 ) & 0xFFF; // positions 32 -> 43
        int mb3_qual =   ( dataWordSub >> 54 ) & 0x7;   // positions 54 -> 56
        int mb3_ts2tag = ( dataWordSub >> 58 ) & 0x1;   // positions 58
        /*** NOT UNPACKED  
        int mb3_parity = ( dataWordSub >> 59 ) &0x1;    // positions 59
        ***/

        int mb3_phi_conv  = radAngConversion(mb3_phi);
        int mb3_phib_conv = 0;
        if (passbc0_)
            mb3_phib_conv = bc0; // fill 'mb3_phib' with the 'bc0' information

        int mb4_phi_conv  = radAngConversion(mb4_phi);
        int mb4_phib_conv = benAngConversion(mb4_phib);

        phiSegments.push_back( L1MuDTChambPhDigi( bxNr, wheel, sector-1, 
                                                  3, mb3_phi_conv, mb3_phib_conv, 
                                                  mb3_qual, mb3_ts2tag, bxCounter) );
        phiSegments.push_back( L1MuDTChambPhDigi( bxNr, wheel, sector-1, 
                                                  4, mb4_phi_conv, mb4_phib_conv, 
                                                  mb4_qual, mb4_ts2tag, bxCounter) );

        if ( debug_ ) logfile << '[' << ++lcounter << "]\t"<< std::hex
                              << dataWordSub   << std::dec      << "\t|\t"
                              << "mb3_ts2tag " << mb3_ts2tag    << '\t'
                              << "mb3_qual "   << mb3_qual      << '\t'
                              << "mb3_phib "   << mb3_phib_conv << '\t'
                              << "mb3_phi "    << mb3_phi_conv  << '\t'
                              << "mb4_ts2tag " << mb4_ts2tag    << '\t'
                              << "mb4_qual "   << mb4_qual      << '\t'
                              << "mb4_phib "   << mb4_phib_conv << '\t'
                              << "mb4_phi "    << mb4_phi_conv  << '\n';

      }//MB3/4 word
 
 
      else if ( selector == 0x3 ) { //etha word
       
        int posALL, posBTI[7];
    
        int mb3_eta    = ( dataWordSub & 0xFF );        // positions  0 -> 7
        int mb3_eta_HQ = ( dataWordSub >> 8  ) & 0xFF;  // positions  8 -> 15
        int mb2_eta    = ( dataWordSub >> 16 ) & 0xFF;  // positions 16 -> 23

        int mb2_eta_HQ = ( dataWordSub >> 32 ) & 0xFF;  // positions 32 -> 39
        int mb1_eta    = ( dataWordSub >> 40 ) & 0xFF;  // positions 40 -> 47
        int mb1_eta_HQ = ( dataWordSub >> 48 ) & 0xFF;  // positions 48 -> 55

        if ( debug_ ) logfile << '[' << ++lcounter << "]\t" << std::hex
                              << dataWordSub << std::dec << "\t|\t"
                              << "mb1_eta_HQ " <<  mb1_eta_HQ   << '\t'
                              << "mb1_eta " << mb1_eta  << '\t'
                              << "mb2_eta_HQ " <<  mb2_eta_HQ   << '\t'
                              << "mb2_eta " << mb2_eta  << '\t'
                              << "mb3_eta_HQ " <<  mb3_eta_HQ   << '\t'
                              << "mb3_eta " << mb3_eta  << '\n';
        
        posALL    = mb1_eta_HQ & 0x7F;
        posBTI[0] = mb1_eta_HQ & 0x01;
        posBTI[1] = (mb1_eta_HQ & 0x02)>>1;
        posBTI[2] = (mb1_eta_HQ & 0x04)>>2;
        posBTI[3] = (mb1_eta_HQ & 0x08)>>3;
        posBTI[4] = (mb1_eta_HQ & 0x10)>>4;
        posBTI[5] = (mb1_eta_HQ & 0x20)>>5;
        posBTI[6] = (mb1_eta_HQ & 0x40)>>6;

        if ( posALL ) {
            
          theSegments.push_back( L1MuDTChambThDigi( bxNr, wheel, sector-1, 1, posBTI) );
            
        }
        
        posALL    = mb2_eta_HQ & 0x7F;
        posBTI[0] = mb2_eta_HQ & 0x01;
        posBTI[1] = (mb2_eta_HQ & 0x02)>>1;
        posBTI[2] = (mb2_eta_HQ & 0x04)>>2;
        posBTI[3] = (mb2_eta_HQ & 0x08)>>3;
        posBTI[4] = (mb2_eta_HQ & 0x10)>>4;
        posBTI[5] = (mb2_eta_HQ & 0x20)>>5;
        posBTI[6] = (mb2_eta_HQ & 0x40)>>6;

        if ( posALL ) {
            
          theSegments.push_back( L1MuDTChambThDigi( bxNr, wheel, sector-1, 2, posBTI) );
            
        }

        posALL    = mb3_eta_HQ & 0x7F;
        posBTI[0] = mb3_eta_HQ & 0x01;
        posBTI[1] = (mb3_eta_HQ & 0x02)>>1;
        posBTI[2] = (mb3_eta_HQ & 0x04)>>2;
        posBTI[3] = (mb3_eta_HQ & 0x08)>>3;
        posBTI[4] = (mb3_eta_HQ & 0x10)>>4;
        posBTI[5] = (mb3_eta_HQ & 0x20)>>5;
        posBTI[6] = (mb3_eta_HQ & 0x40)>>6;

        if ( posALL ) {
            
          theSegments.push_back( L1MuDTChambThDigi( bxNr, wheel, sector-1, 3, posBTI) );
            
        }
    
      }//etha word
    
      else if ( selector == 0xB || selector == 0xC || selector == 0xD ) { //output etha word

        if ( debug_ ) logfile << '[' << ++lcounter << "]\t" << std::hex
                              << dataWordSub << std::dec << '\n';

      }//output etha word
 
      else if ( selector == 0xF ) { //ERROR word

        edm::LogInfo("TwinMux_unpacker") << "ERROR WORD [" << std::dec << tm7eventsize << "] : "
                                         << std::hex << dataWordSub << std::dec
                                         << " it pos " << int(DTTM7iterator - DTTM7itend);
  
        if ( debug_ ) logfile << '[' << ++lcounter << "]\t" << std::hex 
                              << dataWordSub << std::dec
                              << "\t ERROR WORD\n";
      }//ERROR word
 
      else { //unkown word

        edm::LogInfo("TwinMux_unpacker") << "UNKNOWN WORD received " << std::hex << dataWordSub 
                                           << " in FED " << std::hex << TM7fedId;

   	    if ( debug_ ) logfile << '[' << ++lcounter << "]\t" << std::hex 
                              << dataWordSub << std::dec
                              << "\t UNKNOWN WORD\n";
      }
  
      if( DTTM7iterator == DTTM7itend ) break;
      
    } //end of loop over AMCsize


    /// Trailer AMC with CRC
    ++DTTM7iterator;
    /*** NOT UNPACKED  
    dataWord = (*DTTM7iterator);
    int amcCrc = ( dataWord >> 32 ); // positions 32 -> 63
    int eventCountdataEnd = ( dataWord >> 24 ) & 0xFF; // positions 24 -> 31
    int dataLenghtEnd = ( dataWord & 0xFFFFF ); // positions 0 -> 20
    ***/

    if( DTTM7iterator == DTTM7itend ) break;

  } // end for-loop container content

  return;
}



void L1TTwinMuxRawToDigi::calcCRC( long word, int & myC ) {

  int myCRC[16], D[64], C[16];

  for ( int i = 0; i < 64; ++i ) { D[i]    = (word >> i) & 0x1; }
  for ( int i = 0; i < 16; ++i ) { C[i]    = (myC>>i)  & 0x1; }

  myCRC[0] = ( D[63] + D[62] + D[61] + D[60] + D[55] + D[54] +
               D[53] + D[52] + D[51] + D[50] + D[49] + D[48] +
               D[47] + D[46] + D[45] + D[43] + D[41] + D[40] +
               D[39] + D[38] + D[37] + D[36] + D[35] + D[34] +
               D[33] + D[32] + D[31] + D[30] + D[27] + D[26] +
               D[25] + D[24] + D[23] + D[22] + D[21] + D[20] +
               D[19] + D[18] + D[17] + D[16] + D[15] + D[13] +
               D[12] + D[11] + D[10] + D[9]  + D[8]  + D[7]  +
               D[6]  + D[5]  + D[4]  + D[3]  + D[2]  + D[1]  +
               D[0]  + C[0]  + C[1]  + C[2]  + C[3]  + C[4]  +
               C[5]  + C[6]  + C[7]  + C[12] + C[13] + C[14] +
               C[15] )%2;

  myCRC[1] = ( D[63] + D[62] + D[61] + D[56] + D[55] + D[54] +
	           D[53] + D[52] + D[51] + D[50] + D[49] + D[48] +
	           D[47] + D[46] + D[44] + D[42] + D[41] + D[40] +
	           D[39] + D[38] + D[37] + D[36] + D[35] + D[34] +
    	       D[33] + D[32] + D[31] + D[28] + D[27] + D[26] +
    	       D[25] + D[24] + D[23] + D[22] + D[21] + D[20] +
    	       D[19] + D[18] + D[17] + D[16] + D[14] + D[13] +
    	       D[12] + D[11] + D[10] + D[9]  + D[8]  + D[7]  +
	           D[6]  + D[5]  + D[4]  + D[3]  + D[2]  + D[1]  +
	           C[0]  + C[1]  + C[2]  + C[3]  + C[4]  + C[5]  +
	           C[6]  + C[7]  + C[8]  + C[13] + C[14] + C[15] )%2;

  myCRC[2] = ( D[61] + D[60] + D[57] + D[56] + D[46] + D[42] +
	           D[31] + D[30] + D[29] + D[28] + D[16] + D[14] +
	           D[1]  + D[0]  + C[8]  + C[9]  + C[12] + C[13] )%2;

  myCRC[3] = ( D[62] + D[61] + D[58] + D[57] + D[47] + D[43] +
	           D[32] + D[31] + D[30] + D[29] + D[17] + D[15] +
	           D[2]  + D[1]  + C[9]  + C[10] + C[13] + C[14] )%2;

  myCRC[4] = ( D[63] + D[62] + D[59] + D[58] + D[48] + D[44] +
    	       D[33] + D[32] + D[31] + D[30] + D[18] + D[16] + 
	           D[3]  + D[2]  + C[0]  + C[10] + C[11] + C[14] +
	           C[15] )%2;

  myCRC[5] = ( D[63] + D[60] + D[59] + D[49] + D[45] + D[34] +
	           D[33] + D[32] + D[31] + D[19] + D[17] + D[4]  +
    	       D[3]  + C[1]  + C[11] + C[12] + C[15] )%2;

  myCRC[6] = ( D[61] + D[60] + D[50] + D[46] + D[35] + D[34] +
	           D[33] + D[32] + D[20] + D[18] + D[5]  + D[4]  +
	           C[2]  + C[12] + C[13] )%2;

  myCRC[7] = ( D[62] + D[61] + D[51] + D[47] + D[36] + D[35] +
    	       D[34] + D[33] + D[21] + D[19] + D[6]  + D[5]  +
	           C[3]  + C[13] + C[14] )%2;

  myCRC[8] = ( D[63] + D[62] + D[52] + D[48] + D[37] + D[36] +
	           D[35] + D[34] + D[22] + D[20] + D[7]  + D[6]  +
    	       C[0]  + C[4]  + C[14] + C[15] )%2;

  myCRC[9] = ( D[63] + D[53] + D[49] + D[38] + D[37] + D[36] +
	           D[35] + D[23] + D[21] + D[8]  + D[7]  + C[1]  +
	           C[5]  + C[15] )%2;

  myCRC[10] = ( D[54] + D[50] + D[39] + D[38] + D[37] + D[36] + 
       		    D[24] + D[22] + D[9]  + D[8]  + C[2]  + C[6] )%2;

  myCRC[11] = ( D[55] + D[51] + D[40] + D[39] + D[38] + D[37] +
		        D[25] + D[23] + D[10] + D[9]  + C[3]  + C[7] )%2;

  myCRC[12] = ( D[56] + D[52] + D[41] + D[40] + D[39] + D[38] +
        		D[26] + D[24] + D[11] + D[10] + C[4]  + C[8] )%2;

  myCRC[13] = ( D[57] + D[53] + D[42] + D[41] + D[40] + D[39] +
		        D[27] + D[25] + D[12] + D[11] + C[5]  + C[9] )%2;

  myCRC[14] = ( D[58] + D[54] + D[43] + D[42] + D[41] + D[40] +
        		D[28] + D[26] + D[13] + D[12] + C[6]  + C[10] )%2;

  myCRC[15] = ( D[63] + D[62] + D[61] + D[60] + D[59] + D[54] +
		        D[53] + D[52] + D[51] + D[50] + D[49] + D[48] + 
	        	D[47] + D[46] + D[45] + D[44] + D[42] + D[40] +
        		D[39] + D[38] + D[37] + D[36] + D[35] + D[34] + 
		        D[33] + D[32] + D[31] + D[30] + D[29] + D[26] +
        		D[25] + D[24] + D[23] + D[22] + D[21] + D[20] + 
        		D[19] + D[18] + D[17] + D[16] + D[15] + D[14] +
        		D[12] + D[11] + D[10] + D[9]  + D[8]  + D[7]  + 
        		D[6]  + D[5]  + D[4]  + D[3]  + D[2]  + D[1]  +
        		D[0]  + C[0]  + C[1]  + C[2]  + C[3]  + C[4]  + 
	        	C[5]  + C[6]  + C[11] + C[12] + C[13] + C[14] +
	        	C[15] )%2;

  int tempC = 0x0;  
  for ( int i = 0; i < 16 ; ++i) { tempC = tempC + ( myCRC[i] << i ); }
  myC = tempC;
  return;
}


//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TTwinMuxRawToDigi);
