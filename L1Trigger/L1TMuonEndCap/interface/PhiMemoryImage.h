#ifndef __PHIMEMORYIMAGE_
#define __PHIMEMORYIMAGE_ 

class PhiMemoryImage{

 public:

  typedef unsigned long int value_type;
  typedef PhiMemoryImage::value_type *value_ptr;

  static const int STATIONS = 4; // number of stations;
  static const int UNITS    = 2; // number of value_types per station
  static const int TOTAL_UNITS = UNITS * STATIONS;


  ///constructors///
  PhiMemoryImage();
  PhiMemoryImage(PhiMemoryImage::value_ptr buffer, int offset);
  
  PhiMemoryImage (int z,int o,int tw, int th,int fo, int fi, int si,int se){
  _buffer[0] = z;_buffer[1] = o;_buffer[2] = tw;_buffer[3] = th;_buffer[4] = fo;_buffer[5] = fi;_buffer[6] = si;_buffer[7] = se;
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
