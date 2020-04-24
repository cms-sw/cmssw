#ifndef Geometry_VeryForwardGeometry_CTPPSPixelSimTopology_h
#define Geometry_VeryForwardGeometry_CTPPSPixelSimTopology_h

#include "TMath.h"
#include "Geometry/VeryForwardGeometry/interface/CTPPSPixelTopology.h"

class CTPPSPixelSimTopology : public CTPPSPixelTopology
{
  public:
    /* simX and simY are the coordinates as in the simulation:
        _________
       |         |
       |         |  y
       |_________|

       x
    */
    class PixelInfo
    {
      public:
        PixelInfo( double lower_simX_border, double higher_simX_border, double lower_simY_border, double higher_simY_border, double eff_factor, unsigned short pixel_row_no, unsigned short pixel_col_no ) :
          lower_simX_border_( lower_simX_border ), higher_simX_border_( higher_simX_border ),
          lower_simY_border_( lower_simY_border ), higher_simY_border_( higher_simY_border ),
          eff_factor_( eff_factor ),
          pixel_row_no_( pixel_row_no ), pixel_col_no_( pixel_col_no ),
          pixel_index_( pixel_col_no*CTPPSPixelTopology::no_of_pixels_simX_+pixel_row_no )
        {}

        inline double higherSimXBorder() const { return higher_simX_border_; }
        inline double lowerSimXBorder() const { return lower_simX_border_; }
        inline double higherSimYBorder() const { return higher_simY_border_; }
        inline double lowerSimYBorder() const { return lower_simY_border_; }
        inline double effFactor() const { return eff_factor_; }
        inline unsigned short pixelRowNo() const { return pixel_row_no_; }
        inline unsigned short pixelColNo() const { return pixel_col_no_; }
        inline unsigned short pixelIndex() const { return pixel_index_; }

      private:
        double lower_simX_border_;
        double higher_simX_border_;
        double lower_simY_border_;
        double higher_simY_border_;
        double eff_factor_;
        unsigned short pixel_row_no_;
        unsigned short pixel_col_no_;
        unsigned short pixel_index_;
    };

  public:
    CTPPSPixelSimTopology();
    ~CTPPSPixelSimTopology() {}

    PixelInfo getPixelsInvolved( double x, double y, double sigma, double& hit_pos_x, double& hit_pos_y ) const;

    inline void pixelRange( unsigned int arow, unsigned int acol, double& lower_x, double& higher_x, double& lower_y, double& higher_y ) const {
      // x and y in the system  of Geant4 SIMULATION
      arow = 159 - arow;
      if ( arow > 159 || acol > 155 )
        throw cms::Exception("CTPPSPixelSimTopology")<< "rows or columns exceeding limits";

      // rows (x segmentation)
      if ( arow == 0 ) {
        lower_x = 0;
        higher_x = 0.3;
      }
      else if ( arow <= 78 ) {
        lower_x =  0.3 + ( arow-1 )*pitch_simX_;
        higher_x = 0.3 +   arow    *pitch_simX_;
      }
      else if ( arow == 79 ) {
        lower_x =  0.3 + ( arow-1 )*pitch_simX_;
        higher_x = 0.3 + ( arow+1 )*pitch_simX_;
      }
      else if ( arow == 80 ) {
        lower_x =  0.3 +   arow    *pitch_simX_;
        higher_x = 0.3 + ( arow+2 )*pitch_simX_;
      }
      else if ( arow <= 158 ) {
        lower_x =  0.3 + ( arow+1 )*pitch_simX_;
        higher_x = 0.3 + ( arow+2 )*pitch_simX_;
      }
      else if ( arow == 159) {
        lower_x =  0.3 + ( arow+1 )*pitch_simX_;
        higher_x = 0.3 + ( arow+4 )*pitch_simX_;
      }

      // columns (y segmentation)
      if( acol == 0 ) {
        lower_y = 0;
        higher_y = 0.35;
      }
      else if ( acol <= 50 ) {
        lower_y =  0.35 + ( acol-1 )*pitch_simY_;
        higher_y = 0.35 +   acol    *pitch_simY_;
      }
      else if ( acol == 51 ) {
        lower_y =  0.35 + ( acol-1 )*pitch_simY_;
        higher_y = 0.35 + ( acol+1 )*pitch_simY_;
      }
      else if ( acol == 52 ) {
        lower_y =  0.35 +   acol    *pitch_simY_;
        higher_y = 0.35 + ( acol+2 )*pitch_simY_;
      }
      else if ( acol <= 102 ) {
        lower_y =  0.35 + ( acol+1 )*pitch_simY_;
        higher_y = 0.35 + ( acol+2 )*pitch_simY_;
      }
      else if ( acol == 103 ) {
        lower_y =  0.35 + ( acol+1 )*pitch_simY_;
        higher_y = 0.35 + ( acol+3 )*pitch_simY_;
      }
      else if ( acol == 104) {
        lower_y =  0.35 + ( acol+2 )*pitch_simY_;
        higher_y = 0.35 + ( acol+4 )*pitch_simY_;
      }
      else if ( acol <= 154 ) {
        lower_y =  0.35 + ( acol+3 )*pitch_simY_;
        higher_y = 0.35 + ( acol+4 )*pitch_simY_;
      }
      else if ( acol == 155 ) {
        lower_y =  0.35 + ( acol+3 )*pitch_simY_;
        higher_y = 0.35 + ( acol+4 )*pitch_simY_ + 0.2;
      }

      lower_x = lower_x - simX_width_/2.;
      lower_y = lower_y - simY_width_/2.;
      higher_x = higher_x - simX_width_/2.;
      higher_y = higher_y - simY_width_/2.;
    }

  private:
    double active_edge_x_;
    double active_edge_y_;
    
    inline double activeEdgeFactor( double x, double y ) const {
      const double inv_sigma = 1./active_edge_sigma_; // precaching
      const double topEdgeFactor =    TMath::Erf( -distanceFromTopActiveEdge( x, y )   *inv_sigma )*0.5 + 0.5;
      const double bottomEdgeFactor = TMath::Erf( -distanceFromBottomActiveEdge( x, y )*inv_sigma )*0.5 + 0.5;
      const double rightEdgeFactor =  TMath::Erf( -distanceFromRightActiveEdge( x, y ) *inv_sigma )*0.5 + 0.5;
      const double leftEdgeFactor =   TMath::Erf( -distanceFromLeftActiveEdge( x, y )  *inv_sigma )*0.5 + 0.5;

      const double aEF = topEdgeFactor*bottomEdgeFactor*rightEdgeFactor*leftEdgeFactor;

      if ( aEF > 1. )
        throw cms::Exception("CTPPSPixelSimTopology")<< " active edge factor > 1";

      return aEF;
    }

    inline double distanceFromTopActiveEdge( double x, double y ) const { return ( y-active_edge_y_ ); }
    inline double distanceFromBottomActiveEdge( double x, double y ) const { return ( -y-active_edge_y_ ); }
    inline double distanceFromRightActiveEdge( double x, double y ) const { return ( x-active_edge_x_ ); }
    inline double distanceFromLeftActiveEdge( double x, double y ) const { return ( -x-active_edge_x_ ); }

    inline unsigned int row( double x ) const {
      // x in the G4 simulation system
      x = x + simX_width_/2.;
 
      // now x in the system centered in the bottom left corner of the sensor (sensor view, rocs behind)
      if ( x < 0. || x > simX_width_ )
        throw cms::Exception("CTPPSPixelSimTopology")<< "out of reference frame";

      // rows (x segmentation)
      unsigned int arow;
      if      ( x <= ( dead_edge_width_+    pitch_simX_ ) ) arow = 0;
      else if ( x <= ( dead_edge_width_+ 79*pitch_simX_ ) ) arow = int( ( x-dead_edge_width_-pitch_simX_ )/pitch_simX_ )+1;
      else if ( x <= ( dead_edge_width_+ 81*pitch_simX_ ) ) arow = 79;
      else if ( x <= ( dead_edge_width_+ 83*pitch_simX_ ) ) arow = 80;
      else if ( x <= ( dead_edge_width_+162*pitch_simX_ ) ) arow = int( ( x-dead_edge_width_-pitch_simX_ )/pitch_simX_ )-1;
      else arow = 159;

      arow = 159-arow;
      if ( arow>159 )
        throw cms::Exception("CTPPSPixelSimTopology")<< "row number exceeding limit";

      return arow;
    }

    inline unsigned int col( double y ) const {
      // y in the G4 simulation system
      unsigned int column;

      // columns (y segmentation)
      // now y in the system centered in the bottom left corner of the sensor (sensor view, rocs behind)
      y = y + simY_width_/2.;
      if ( y < 0. || y > simY_width_ )
        throw cms::Exception("CTPPSPixelSimTopology")<< " out of reference frame";

      if      ( y <= ( dead_edge_width_+    pitch_simY_ ) ) column = 0;
      else if ( y <= ( dead_edge_width_+ 51*pitch_simY_ ) ) column = int( ( y-dead_edge_width_-pitch_simY_ )/pitch_simY_ )+1;
      else if ( y <= ( dead_edge_width_+ 53*pitch_simY_ ) ) column = 51;
      else if ( y <= ( dead_edge_width_+ 55*pitch_simY_ ) ) column = 52;
      else if ( y <= ( dead_edge_width_+105*pitch_simY_ ) ) column = int( ( y-dead_edge_width_-pitch_simY_ )/pitch_simY_ )-1;
      else if ( y <= ( dead_edge_width_+107*pitch_simY_ ) ) column = 103;
      else if ( y <= ( dead_edge_width_+109*pitch_simY_ ) ) column = 104;
      else if ( y <= ( dead_edge_width_+159*pitch_simY_ ) ) column = int( ( y-dead_edge_width_-pitch_simY_ )/pitch_simY_ )-3;
      else column = 155;

      return column;
    }

    inline void rowCol2Index( unsigned int arow, unsigned int acol, unsigned int& index ) const {
      index = acol*no_of_pixels_simX_+arow;
    }

    inline void index2RowCol( unsigned int& arow, unsigned int& acol, unsigned int index ) const {
      acol = index / no_of_pixels_simX_;
      arow = index % no_of_pixels_simX_;
    }
};

#endif
