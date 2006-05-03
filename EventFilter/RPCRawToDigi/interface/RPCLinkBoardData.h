#ifndef RPCLinkBoardData_h
#define RPCLinkBoardData_h


/** \class RPCLinkBoardData
 *
 * Container for data of RPC Link Board Record 
 *
 * 
 *
 *  $Date: 2006/03/30 15:13:58 $
 *  $Revision: 1.1 $
 * \author Ilaria Segoni - CERN
 */

#include<vector>

struct RPCLinkBoardData {

public:
  
  /// Constructor
  RPCLinkBoardData();
  /// Constructor with data
  RPCLinkBoardData(std::vector<int> bits, int halfP, int eod, int partitionNumber, int lbNumber);
  
  /// Destructor
  virtual ~RPCLinkBoardData() {};

  /// data setter methods
  void setBits(std::vector<int> bits);
  void setHalfP(int hp);
  void setEod(int eod);
  void setPartitionNumber(int partNumb);
  void setLbNumber(int lbNumb);

  /// data access methods
  std::vector<int> bitsOn() const;
  int halfP() const;
  int eod() const;
  int partitionNumber() const;
  int lbNumber() const;

private:
 
  std::vector<int> bitsOn_;
  int halfP_;
  int eod_;
  int partitionNumber_;
  int lbNumber_;
  
};

#endif
