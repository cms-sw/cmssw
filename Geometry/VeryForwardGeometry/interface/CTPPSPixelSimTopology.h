#ifndef Geometry_VeryForwardGeometry_CTPPSPixelSimTopology_h
#define Geometry_VeryForwardGeometry_CTPPSPixelSimTopology_h

#include <cmath>
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
      arow = (2*ROCSizeInX - 1) - arow;
      if ( arow > (2*ROCSizeInX - 1) || acol > (3*ROCSizeInY - 1) )
        throw cms::Exception("CTPPSPixelSimTopology")<< "rows or columns exceeding limits";

      // rows (x segmentation)
      if ( arow == 0 ) {
        lower_x = dead_edge_width_ - phys_active_edge_dist_; // 50 um
        higher_x = dead_edge_width_ + pitch_simX_; // 300 um
      }
      else if ( arow <= (ROCSizeInX - 2) ) {
        lower_x =  dead_edge_width_ + arow*pitch_simX_;
        higher_x = dead_edge_width_ +   ( arow+1 )*pitch_simX_;
      }
      else if ( arow == (ROCSizeInX - 1) ) {
        lower_x =  dead_edge_width_ + arow*pitch_simX_;
        higher_x = dead_edge_width_ + ( arow+2 )*pitch_simX_;
      }
      else if ( arow == ROCSizeInX ) {
        lower_x =  dead_edge_width_ + ( arow+1 )*pitch_simX_;
        higher_x = dead_edge_width_ + ( arow+3 )*pitch_simX_;
      }
      else if ( arow <= (2*ROCSizeInX - 2)) {
        lower_x =  dead_edge_width_ + ( arow+2 )*pitch_simX_;
        higher_x = dead_edge_width_ + ( arow+3 )*pitch_simX_;
      }
      else if ( arow == (2*ROCSizeInX - 1)) {
        lower_x =  dead_edge_width_ + ( arow+2 )*pitch_simX_;
        higher_x = dead_edge_width_ + ( arow+3 )*pitch_simX_ + phys_active_edge_dist_ ;
      }

      // columns (y segmentation)
      if( acol == 0 ) {
        lower_y = dead_edge_width_ - phys_active_edge_dist_; // 50 um
        higher_y = dead_edge_width_ + pitch_simY_; // 350 um
      }
      else if ( acol <= (ROCSizeInY - 2)) {
        lower_y =  dead_edge_width_ + acol*pitch_simY_;
        higher_y = dead_edge_width_ +  ( acol+1 )*pitch_simY_;
      }
      else if ( acol == (ROCSizeInY - 1) ) {
        lower_y =  dead_edge_width_ + acol*pitch_simY_;
        higher_y = dead_edge_width_ + ( acol+2 )*pitch_simY_;
      }
      else if ( acol == ROCSizeInY ) {
        lower_y =  dead_edge_width_ + ( acol+1 )*pitch_simY_;
        higher_y = dead_edge_width_ + ( acol+3 )*pitch_simY_;
      }
      else if ( acol <= (2*ROCSizeInY - 2) ) {
        lower_y =  dead_edge_width_ + ( acol+2 )*pitch_simY_;
        higher_y = dead_edge_width_ + ( acol+3 )*pitch_simY_;
      }
      else if ( acol == (2*ROCSizeInY - 1) ) {
        lower_y =  dead_edge_width_ + ( acol+2 )*pitch_simY_;
        higher_y = dead_edge_width_ + ( acol+4 )*pitch_simY_;
      }
      else if ( acol == (2*ROCSizeInY)) {
        lower_y =  dead_edge_width_ + ( acol+3 )*pitch_simY_;
        higher_y = dead_edge_width_ + ( acol+5 )*pitch_simY_;
      }
      else if ( acol <= (3*ROCSizeInY - 2) ) {
        lower_y =  dead_edge_width_ + ( acol+4 )*pitch_simY_;
        higher_y = dead_edge_width_ + ( acol+5 )*pitch_simY_;
      }
      else if ( acol == (3*ROCSizeInY - 1) ) {
        lower_y =  dead_edge_width_ + ( acol+4 )*pitch_simY_;
        higher_y = dead_edge_width_ + ( acol+5 )*pitch_simY_ + phys_active_edge_dist_ ;
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
      const double topEdgeFactor =    std::erf( -distanceFromTopActiveEdge( x, y )   *inv_sigma )*0.5 + 0.5;
      const double bottomEdgeFactor = std::erf( -distanceFromBottomActiveEdge( x, y )*inv_sigma )*0.5 + 0.5;
      const double rightEdgeFactor =  std::erf( -distanceFromRightActiveEdge( x, y ) *inv_sigma )*0.5 + 0.5;
      const double leftEdgeFactor =   std::erf( -distanceFromLeftActiveEdge( x, y )  *inv_sigma )*0.5 + 0.5;

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
      else if ( x <= ( dead_edge_width_+ (ROCSizeInX - 1)*pitch_simX_ ) ) arow = int( ( x-dead_edge_width_-pitch_simX_ )/pitch_simX_ )+1;
      else if ( x <= ( dead_edge_width_+ (ROCSizeInX + 1)*pitch_simX_ ) ) arow = (ROCSizeInX - 1);
      else if ( x <= ( dead_edge_width_+ (ROCSizeInX + 3)*pitch_simX_ ) ) arow = ROCSizeInX;
      else if ( x <= ( dead_edge_width_+ (2*ROCSizeInX + 2)*pitch_simX_ ) ) arow = int( ( x-dead_edge_width_-pitch_simX_ )/pitch_simX_ )-1;
      else arow = (2*ROCSizeInX - 1);

      arow = (2*ROCSizeInX - 1)-arow;
      if ( arow>(2*ROCSizeInX - 1) )
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
      else if ( y <= ( dead_edge_width_+ (ROCSizeInY - 1)*pitch_simY_ ) ) column = int( ( y-dead_edge_width_-pitch_simY_ )/pitch_simY_ )+1;
      else if ( y <= ( dead_edge_width_+ (ROCSizeInY + 1)*pitch_simY_ ) ) column = ROCSizeInY - 1;
      else if ( y <= ( dead_edge_width_+ (ROCSizeInY + 3)*pitch_simY_ ) ) column = ROCSizeInY;
      else if ( y <= ( dead_edge_width_+ (2*ROCSizeInY + 1)*pitch_simY_ ) ) column = int( ( y-dead_edge_width_-pitch_simY_ )/pitch_simY_ )-1;
      else if ( y <= ( dead_edge_width_+ (2*ROCSizeInY + 3)*pitch_simY_ ) ) column = 2*ROCSizeInY - 1;
      else if ( y <= ( dead_edge_width_+ (2*ROCSizeInY + 5)*pitch_simY_ ) ) column = 2*ROCSizeInY;
      else if ( y <= ( dead_edge_width_+ (3*ROCSizeInY + 3)*pitch_simY_ ) ) column = int( ( y-dead_edge_width_-pitch_simY_ )/pitch_simY_ )-3;
      else column = (3*ROCSizeInY - 1);

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
