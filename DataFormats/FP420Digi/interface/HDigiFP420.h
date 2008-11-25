#ifndef DataFormats_HDigiFP420_h
#define DataFormats_HDigiFP420_h

class HDigiFP420 {
public:

  //typedef unsigned int ChannelType;

  HDigiFP420() : strip_(0), adc_(0) {
}

  HDigiFP420( int strip, int adc) : strip_(strip), adc_(adc) {
}
    HDigiFP420( short strip, short adc) : strip_(strip), adc_(adc) {
//    numStripsY = 200;        // Y plane number of strips:200*0.050=10mm (zside=1) H
//    numStripsX = 400;        // X plane number of strips:400*0.050=20mm (zside=2) V
}

  // Access to digi information
  int strip() const   {return strip_;}
  int adc() const     {return adc_;}
  int channel() const {return strip();}

  int stripVW() const {return (strip_/numStripsX);}
  //int stripVW() const {return (strip_/401);}
  int stripV() const {return (strip_-stripVW()*numStripsX);}
  //int stripHW() const {return (strip_/201) ;}
  int stripHW() const {return (strip_/numStripsY) ;}
  int stripH() const {return (strip_-stripHW()*numStripsY) ;}
  //                                             //
  //		  int iy= istrip.channel()/numStripsY;
  //		  int ix= istrip.channel() - iy*numStripsY;
  //                                             //
private:
  static const int  numStripsY = 144;        // Y plate number of strips:144*0.050=7.2mm (xytype=1)
  static const int  numStripsX = 160;        // X plate number of strips:160*0.050=8.0mm (xytype=2)
  //static const int  numStripsY= 200 ;        // Y plate number of strips:200*0.050=10mm (zside=1)
  //static const int  numStripsX= 400 ;        // X plate number of strips:400*0.050=20mm (zside=2)
  short strip_;
  short adc_;
};

// Comparison operators
inline bool operator<( const HDigiFP420& one, const HDigiFP420& other) {
  return one.channel() < other.channel();
}
//std::ostream& operator<<(std::ostream& s, const HDigiFP420& hit) {
//  return s << hit.channel() << ": " << hit.adc() << " adc, " << hit.strip() << " number";
//}

#endif
