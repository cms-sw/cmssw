#ifndef RPCFILEREADER_RPCPACDATA_H
#define RPCFILEREADER_RPCPACDATA_H

/** \class RPCPacData
 *
 *  Muxed PAC data
 *
 *  $Date: $
 *  $Revision: $
 * \author K. Bunkowski & M. Bluj - Warsaw
*/

#include <string>
#include <vector>

class RPCPacData {
public:
  RPCPacData();
  
  RPCPacData(unsigned int rawData);
  
  RPCPacData(unsigned int partData,
	     unsigned int partNo, 
	     unsigned int partDelay,
	     unsigned int eofData,
	     unsigned int halfPart,
	     unsigned int lbNo);
	
  unsigned int PartitionData() const { return partitionData_; }
  unsigned int PartitionNum() const { return partitionNum_; }
  unsigned int PartitionDelay() const { return partitionDelay_; }
  unsigned int EndOfData() const { return endOfData_; }
  unsigned int HalfPartition() const { return halfPartition_; }
  unsigned int LbNum() const { return lbNum_; }

  void SetPartitionData(unsigned int partData) { partitionData_=partData; }
  void SetPartitionNum(unsigned int partNo) { partitionNum_=partNo; }
  void SetPartitionDelay(unsigned int partDelay) { partitionDelay_=partDelay; }
  void SetEndOfData(unsigned int eofData) { endOfData_=eofData; }
  void SetHalfPartition(unsigned int halfPart) { halfPartition_=halfPart; }
  void SetLbNum(unsigned int lbNo) { lbNum_=lbNo; }

  unsigned int ToRaw();
  
  std::string ToString();
  
  bool operator == (const RPCPacData& right) const;
  
  bool operator != (const RPCPacData& right) const;

 private:

  unsigned int partitionData_;
  unsigned int partitionNum_; 
  unsigned int partitionDelay_;
  unsigned int endOfData_;
  unsigned int halfPartition_;
  unsigned int lbNum_;
	
};

/*
typedef std::vector<RPCPacData> RPCPacDataVec;
typedef std::vector<RPCPacDataVec> RPCPacDataVec2;
*/

#endif 
