#ifndef RPCFILEREADER_RPCPACDATA_H
#define RPCFILEREADER_RPCPACDATA_H

/** \class RPCPacData
 *
 *  Muxed PAC data
 *
 *  $Date: 2006/07/31 14:47:56 $
 *  $Revision: 1.1 $
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
	
  unsigned int partitionData() const { return partitionData_; }
  unsigned int partitionNum() const { return partitionNum_; }
  unsigned int partitionDelay() const { return partitionDelay_; }
  unsigned int endOfData() const { return endOfData_; }
  unsigned int halfPartition() const { return halfPartition_; }
  unsigned int lbNum() const { return lbNum_; }

  void setPartitionData(unsigned int partData) { partitionData_=partData; }
  void setPartitionNum(unsigned int partNo) { partitionNum_=partNo; }
  void setPartitionDelay(unsigned int partDelay) { partitionDelay_=partDelay; }
  void setEndOfData(unsigned int eofData) { endOfData_=eofData; }
  void setHalfPartition(unsigned int halfPart) { halfPartition_=halfPart; }
  void setLbNum(unsigned int lbNo) { lbNum_=lbNo; }

  unsigned int toRaw();
  
  std::string toString();
  
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
