#ifndef UCTAMCRawData_hh
#define UCTAMCRawData_hh

class UCTAMCRawData {
public:
  UCTAMCRawData(const uint32_t *d) : myDataPtr(d) {}

  virtual ~UCTAMCRawData() { ; }

  // Access functions for convenience

  const uint32_t *dataPtr() const { return myDataPtr; }

  const uint32_t *header() const { return &myDataPtr[0]; }
  const uint32_t *payload() const { return &myDataPtr[4]; }
  const uint32_t *trailer() const { return &myDataPtr[trailerOffset()]; }

  uint32_t dataLength() const { return (myDataPtr[0] & 0x000FFFFF); }

  uint32_t BXID() { return ((myDataPtr[0] & 0xFFF00000) >> 20); }
  uint32_t L1ID() { return (myDataPtr[1] & 0x00FFFFFF); }
  uint32_t amcNo() { return ((myDataPtr[1] & 0x0F000000) >> 24); }
  uint32_t layer1Phi() { return (myDataPtr[2] & 0x0000FFFF); }
  uint32_t orbitNo() { return ((myDataPtr[2] & 0xFFFF0000) >> 16); }

  uint32_t trailerOffset() const { return (dataLength() - 1) * 2; }

  uint32_t dataLengthTrailer() { return (myDataPtr[trailerOffset()] & 0x000FFFFF); }
  uint32_t L1IDTrailer() { return ((myDataPtr[trailerOffset()] & 0xFF000000) >> 24); }
  uint32_t crc32() { return (myDataPtr[trailerOffset() + 1]); }

  void print() {
    using namespace std;
    cout << "AMC Payload Header:" << endl;
    cout << "Data Length.. = " << dec << dataLength() << endl;
    cout << "BXID......... = " << dec << BXID() << endl;
    cout << "L1ID......... = " << internal << setfill('0') << setw(8) << hex << L1ID() << endl;
    cout << "AMC No ...... = " << dec << amcNo() << endl;
    cout << "Layer-1 Phi.. = " << dec << layer1Phi() << endl;
    cout << "Orbit No..... = " << dec << orbitNo() << endl;
    cout << "AMC Payload Trailer:" << endl;
    cout << "Data Length.. = " << dec << dataLengthTrailer() << endl;
    cout << "L1ID......... = " << internal << setfill('0') << setw(8) << hex << L1IDTrailer() << endl;
    cout << "CRC32........ = " << internal << setfill('0') << setw(10) << hex << crc32() << endl;
  }

private:
  // No copy constructor and equality operator are needed

  UCTAMCRawData(const UCTAMCRawData &) = delete;
  const UCTAMCRawData &operator=(const UCTAMCRawData &i) = delete;

  // RawData data

  const uint32_t *myDataPtr;
};

#endif
