#ifndef __PHIMEMORYIMAGE_
#define __PHIMEMORYIMAGE_ 

class PhiMemoryImage{

 public:

  typedef unsigned long int value_type;
  typedef PhiMemoryImage::value_type *value_ptr;

  static const int STATIONS = 4; // number of stations;
  static const int UNITS    = 3; // number of value_types per station
  static const int TOTAL_UNITS = UNITS * STATIONS;


  ///constructors///
  PhiMemoryImage();
  PhiMemoryImage(PhiMemoryImage::value_ptr buffer, int offset);
  
  PhiMemoryImage (value_type s1a,value_type s1b,value_type s1c,value_type s2a, value_type s2b, value_type s2c, value_type s3a, value_type s3b, value_type s3c, value_type s4a,value_type s4b, value_type s4c){
  _buffer[0] = s1a;_buffer[1] = s1b;_buffer[2] = s1c;_buffer[3] = s2a;_buffer[4] = s2b;_buffer[5] = s2c;_buffer[6] = s3a;_buffer[7] = s3b;_buffer[8] = s3c;_buffer[9] = s4a;_buffer[10] = s4b;_buffer[11] = s4c;
  }

  ///functions///
  void CopyFromBuffer (PhiMemoryImage::value_ptr rhs, int offset);
  
  void SetBit (int station, int bitNumber, bool value = true);
  bool GetBit (int station, int bitNumber) const;
  
  void BitShift (int nBits); // nBits > 0 executes << nbits, nBits <0 is >> nBits 
  void Print();
  
  void SetBuff(int chunk, int value){_buffer[chunk] = value;}
  
  void printbuff();

 // const PhiMemoryImage::value_type & operator [] (int index) const 
//  {return _buffer[index];}
  
  PhiMemoryImage::value_type & operator [] (int index) 
  {return _buffer[index];}
  
  
 private:

  PhiMemoryImage::value_type _buffer[PhiMemoryImage::TOTAL_UNITS];
  int    _keyStationOffset;
  
};

#endif
