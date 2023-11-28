/**********************************************************
*
* This is a part of the TOTEM offline software.
* Authors:
*    Jan Kaspar (jan.kaspar@gmail.com)
*
**********************************************************/

#ifndef EventFilter_CTPPSRawToDigi_VFATFrame
#define EventFilter_CTPPSRawToDigi_VFATFrame

#include <vector>
#include <cstddef>
#include <cstdint>

/**
 * Representation of VFAT frame plus extra info added by DAQ.
**/
class VFATFrame {
public:
  typedef uint16_t word;

public:
  VFATFrame(const word* _data = nullptr);

  VFATFrame(const VFATFrame& copy) {
    setData(copy.data);
    presenceFlags = copy.presenceFlags;
    daqErrorFlags = copy.daqErrorFlags;
    numberOfClusters = copy.numberOfClusters;
  }

  virtual ~VFATFrame() {}

  /// Copies a memory block to data buffer.
  void setData(const word* _data);

  VFATFrame::word* getData() { return data; }

  const VFATFrame::word* getData() const { return data; }

  /// Returns Bunch Crossing number (BC<11:0>).
  VFATFrame::word getBC() const { return data[11] & 0x0FFF; }

  /// Returns Event Counter (EV<7:0>).
  VFATFrame::word getEC() const { return (data[10] & 0x0FF0) >> 4; }

  /// Returns flags.
  VFATFrame::word getFlags() const { return data[10] & 0x000F; }

  /// Returns ChipID (ChipID<11:0>).
  VFATFrame::word getChipID() const { return data[9] & 0x0FFF; }

  /// Returns the CRC.
  VFATFrame::word getCRC() const { return data[0]; }

  /// Returns the CRC, for non-reversed TOTEM T2.
  VFATFrame::word getCRCT2() const { return data[11]; }

  /// Sets presence flags.
  void setPresenceFlags(uint8_t v) { presenceFlags = v; }

  /// Returns true if the BC word is present in the frame.
  bool isBCPresent() const { return presenceFlags & 0x1; }

  /// Returns true if the EC word is present in the frame.
  bool isECPresent() const { return presenceFlags & 0x2; }

  /// Returns true if the ID word is present in the frame.
  bool isIDPresent() const { return presenceFlags & 0x4; }

  /// Returns true if the CRC word is present in the frame.
  bool isCRCPresent() const { return presenceFlags & 0x8; }

  /// Returns true if the CRC word is present in the frame.
  bool isNumberOfClustersPresent() const { return presenceFlags & 0x10; }

  /// Sets DAQ error flags.
  void setDAQErrorFlags(uint8_t v) { daqErrorFlags = v; }

  void setNumberOfClusters(uint8_t v) { numberOfClusters = v; }

  /// Returns the number of clusters as given by the "0xD0 frame".
  /// Returns 0, if not available.
  uint8_t getNumberOfClusters() const { return numberOfClusters; }

  /// Checks the fixed bits in the frame.
  /// Returns false if any of the groups (in BC, EC and ID words) is present but wrong.
  bool checkFootprint() const;

  /// Checks the fixed bits in the frame, for the TOTEM T2 non-inverse word ordering.
  /// Returns false if any of the groups (in BC, EC and ID words) is present but wrong.
  bool checkFootprintT2() const;

  /// Checks the validity of frame (CRC and daqErrorFlags).
  /// Returns false if daqErrorFlags is non-zero.
  /// Returns false if the CRC is present and invalid.
  virtual bool checkCRC() const;

  /// Checks the validity of Totem T2 non-reversed CRC.
  /// Returns false if the CRC is present and invalid.
  virtual bool checkCRCT2() const;

  /// Checks if channel number 'channel' was active.
  /// Returns positive number if it was active, 0 otherwise.
  virtual bool channelActive(unsigned char channel) const {
    return (data[1 + (channel / 16)] & (1 << (channel % 16))) ? true : false;
  }

  /// Returns list  of active channels.
  /// It's more efficient than the channelActive(char) for events with low channel occupancy.
  virtual std::vector<unsigned char> getActiveChannels() const;

  /// Prints the frame.
  /// If binary is true, binary format is used.
  void Print(bool binary = false) const;

  //Follow the VFAT2 manual format, not reversed
  void PrintT2(bool binary = false) const;

  /// internaly used to check CRC
  static word calculateCRC(word crc_in, word dato);

protected:
  /** Raw data frame as sent by electronics.
    * The container is organized as follows (reversed Figure 8 at page 23 of VFAT2 manual):
    * \verbatim
    * buffer index   content       size
    * ---------------------------------------------------------------
    *   0            CRC           16 bits
    *   1->8         Channel data  128 bits, channel 0 first
    *   9            ChipID        4 constant bits (1110) + 12 bits
    *   10           EC, Flags     4 constant bits (1100) + 8, 4 bits
    *   11           BC            4 constant bits (1010) + 12 bits
    * \endverbatim
    **/
  word data[12];

private:
  /// Flag indicating the presence of various components.
  ///   bit 1: "BC word" (buffer index 11)
  ///   bit 2: "EC word" (buffer index 10)
  ///   bit 3: "ID word" (buffer index 9)
  ///   bit 4: "CRC word" (buffer index 0)
  ///   bit 5: "number of clusters word" (prefix 0xD0)
  uint8_t presenceFlags;

  /// Error flag as given by certain versions of DAQ.
  uint8_t daqErrorFlags;

  /// Number of clusters.
  /// Only available in cluster mode and if the number of clusters exceeds a limit (10).
  uint8_t numberOfClusters;
};

#endif
