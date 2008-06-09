#ifndef RPCLinkBoardData_h
#define RPCLinkBoardData_h


/** \class RPCLinkBoardData
 *
 * Container for data of RPC Link Board Record 
 *
 * 
 *
 *  $Date: 2006/05/03 16:23:25 $
 *  $Revision: 1.2 $
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
  int lbData() const;
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
