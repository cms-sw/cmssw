#ifndef EventFilter_RPCRawToDigi_RPCLBRecord_h
#define EventFilter_RPCRawToDigi_RPCLBRecord_h

#include <cstdint>

class RPCLBRecord {
public:
  /* https://twiki.cern.ch/twiki/bin/viewauth/CMS/DtUpgradeTwinMux#RPC_payload RPC_optical_links_data_format.pdf */
  /* 4b partition is split in 3b connector + 1b partition */
  static std::uint32_t const bcn_mask_ = 0xfe000000;
  static std::uint32_t const bc0_mask_ = 0x01000000;
  static std::uint32_t const link_board_mask_ = 0x00060000;
  static std::uint32_t const eod_mask_ = 0x00008000;
  static std::uint32_t const delay_mask_ = 0x00007000;
  static std::uint32_t const connector_mask_ = 0x00000e00;
  static std::uint32_t const partition_mask_ = 0x00000100;
  static std::uint32_t const partition_data_mask_ = 0x000000ff;

  static unsigned int const bcn_offset_ = 25;
  static unsigned int const bc0_offset_ = 24;
  static unsigned int const link_board_offset_ = 17;
  static unsigned int const eod_offset_ = 15;
  static unsigned int const delay_offset_ = 12;
  static unsigned int const connector_offset_ = 9;
  static unsigned int const partition_offset_ = 8;
  static unsigned int const partition_data_offset_ = 0;

public:
  RPCLBRecord(std::uint32_t const record = 0x0);
  RPCLBRecord(unsigned int bcn,
              bool bc0,
              unsigned int link_board,
              bool eod,
              unsigned int delay,
              unsigned int connector,
              unsigned int partition,
              std::uint8_t data);

  void set(std::uint32_t const record = 0x0);
  void reset();

  std::uint32_t const& getRecord() const;

  unsigned int getBCN() const;
  bool isBC0() const;
  unsigned int getLinkBoard() const;
  bool isEOD() const;
  unsigned int getDelay() const;
  unsigned int getConnector() const;
  unsigned int getPartition() const;
  std::uint8_t getPartitionData() const;

  void setBCN(unsigned int bcn);
  void setBC0(bool bc0);
  void setLinkBoard(unsigned int link_board);
  void setEOD(bool eod);
  void setDelay(unsigned int delay);
  void setConnector(unsigned int connector);
  void setPartition(unsigned int partition);
  void setPartitionData(std::uint8_t data);

  bool operator<(RPCLBRecord const& rhs) const;

protected:
  std::uint32_t record_;
};

#include "EventFilter/RPCRawToDigi/interface/RPCLBRecord.icc"

#endif  // EventFilter_RPCRawToDigi_RPCLBRecord_h
