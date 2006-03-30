#ifndef RPCFEDData_h
#define RPCFEDData_h

/** \class RPCFEDData
 *  Container for the RPC Pay Load unpacked by RPCFormatter. 
 *  The RPCUnpackingModule creates the container and fills it with the DCC
 *  header and trailer info. The RPCRecordFormatter fills it
 *  with the RPC RMB Data.
 *  
 *
 *  $Date: 2005/12/15 17:41:44 $
 *  $Revision: 1.3 $
 * \author Ilaria Segoni - CERN
 */

#include <EventFilter/RPCRawToDigi/interface/RPCLinkBoardData.h>
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>


#include <map>
#include <vector>

class FEDHeader;
class FEDTrailer;

struct RPCFEDData{

public:
   RPCFEDData(){DCCDiscarded=0;}
   ~RPCFEDData(){}

   /// Insert new Header
   void addCdfHeader(FEDHeader & header);  
   /// Insert new Trailer
   void addCdfTrailer(FEDTrailer & trailer);  
   /// Insert BX
   void addBXData(int bx);  
   /// Insert RMB Data
   void addRMBData(int rmb, int chn, RPCLinkBoardData lbData);  
   /// Insert RMBand Channel number from wich Data was discarded  
   void addRMBDiscarded(int rmb, int chn);  
   /// Insert RMBand Channel number with corrupted data  
   void addRMBCorrupted(int rmb, int chn);  
   /// update counter of DCC discarded events
   void addDCCDiscarded();  
   
   /// Access Methods to  unpacked information
   
   /// Get Header(s) info
   std::vector<FEDHeader> fedHeaders() const;
   /// Get Trailer(s) info
   std::vector<FEDTrailer>  fedTrailers() const;
   /// Get List of BX numbers
   std::vector<int> bxData() const;
   /// Get Map between RMB and map bewteen Channel ID's and Link Board Payload.
   std::map<int , std::map<int, std::vector<RPCLinkBoardData> > > rmbData() const;
   /// Get RMB and Channel number from wich Data was discarded  
   std::map<int,std::vector<int> > dataRMBDiscarded() const;
   /// Get RMB and Channel number with corrupted data 
   std::map<int,std::vector<int> > dataRMBCorrupted() const;
   /// Get counter of DCC discarded events
   int dccDiscarded() const;


private:
   
   std::vector<FEDHeader> cdfHeaders;
   std::vector<FEDTrailer> cdfTrailers;   
   std::vector<int>  bxCounts;
   
   std::map<int , std::map<int, std::vector<RPCLinkBoardData> > > rmbDataMap;
   
   std::map<int ,std::vector<int> > RMBDiscarded;
   std::map<int ,std::vector<int> > RMBCorrupted;
   int DCCDiscarded;

};


#endif
