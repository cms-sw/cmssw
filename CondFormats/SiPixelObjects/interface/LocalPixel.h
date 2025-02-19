#ifndef CondFormats_SiPixelObjects_LocalPixel_H
#define CondFormats_SiPixelObjects_LocalPixel_H

namespace sipixelobjects {

  /// identify pixel inside single ROC
  class LocalPixel { 

    public:

      static const int numRowsInRoc = 80;
      static const int numColsInRoc = 52;

      /// row and collumn in ROC representation 
      struct RocRowCol { 
        int rocRow, rocCol; 
        bool valid() const { return    0 <= rocRow && rocRow < numRowsInRoc  
                                    && 0 <= rocCol && rocCol < numColsInRoc; }
      };

      /// double collumn and pixel ID in double collumn representation
      struct DcolPxid { 
        int dcol, pxid; 
        bool valid() const { return (0 <= dcol && dcol < 26 &&  2 <= pxid && pxid < 162 ); }
      }; 

      LocalPixel( const DcolPxid & pixel) {
        thePixel.rocCol = pixel.dcol*2 + pixel.pxid%2;
        thePixel.rocRow = numRowsInRoc - pixel.pxid/2;
      }

      LocalPixel( const RocRowCol & pixel) : thePixel(pixel) {} 

      int  dcol() const { return thePixel.rocCol/2; }
      int  pxid() const { return 2*(numRowsInRoc-thePixel.rocRow)+ (thePixel.rocCol%2); }

      int  rocRow() const { return thePixel.rocRow; }
      int  rocCol() const { return thePixel.rocCol; }

      bool valid() const { return thePixel.valid(); }
    private:
      RocRowCol thePixel;
  };
}

#endif
