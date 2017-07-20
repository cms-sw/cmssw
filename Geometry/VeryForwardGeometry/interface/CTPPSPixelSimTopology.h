#ifndef Geometry_VeryForwardGeometry_CTPPSPixelSimTopology_H
#define Geometry_VeryForwardGeometry_CTPPSPixelSimTopology_H

#include "TMath.h"
#include "Geometry/VeryForwardGeometry/interface/CTPPSPixelTopology.h"

/* simX and simY are the coordinates as in the simulation: 
   _______
   |         |
   |         |  y
   |         |
   ---------
   x

*/

class pixel_info
{

public:

pixel_info(double lower_simX_border, double higher_simX_border, double lower_simY_border, double higher_simY_border, double eff_factor, unsigned short pixel_row_no, unsigned short pixel_col_no) : 
  lower_simX_border_(lower_simX_border), 
    higher_simX_border_(higher_simX_border), 
    lower_simY_border_(lower_simY_border), 
    higher_simY_border_(higher_simY_border), 
    eff_factor_(eff_factor), 
    pixel_row_no_(pixel_row_no), 
    pixel_col_no_(pixel_col_no),
    pixel_index_(pixel_col_no*CTPPSPixelTopology::no_of_pixels_simX_+pixel_row_no)
    {}

  inline double higherSimXBorder() const {return higher_simX_border_;}
  inline double lowerSimXBorder() const {return lower_simX_border_;}
  inline double higherSimYBorder() const {return higher_simY_border_;}
  inline double lowerSimYBorder() const {return lower_simY_border_;}
  inline double effFactor() const {return eff_factor_;}
  inline unsigned short pixelRowNo() const {return pixel_row_no_;}
  inline unsigned short pixelColNo() const {return pixel_col_no_;}
  inline unsigned short pixelIndex() const {return pixel_index_;}
    

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

class CTPPSPixelSimTopology : public CTPPSPixelTopology
{

public:
  CTPPSPixelSimTopology();
  ~CTPPSPixelSimTopology(){};

  std::vector<pixel_info> getPixelsInvolved(double x, double y, double sigma, double &hit_pos_x, double &hit_pos_y);

  inline void pixelRange(unsigned int arow, unsigned int acol, double &lower_x, double &higher_x, double &lower_y, double &higher_y){
       
// x and y in the system  of Geant4 SIMULATION

    arow = 159 - arow;

    if(!(arow<=159 && acol <=155))
      throw cms::Exception("CTPPSPixelSimTopology")<< " rows or columns exceeding limits";
 
// rows (x segmentation)
    
    if(arow == 0) {
      lower_x = 0;
      higher_x = 0.3;
    }
    if( arow > 0 && arow <= 78 ){
      lower_x = 0.3 + (arow-1)*pitch_simX_;
      higher_x = 0.3 + arow*pitch_simX_;
    } 
    if(arow == 79) {
      lower_x = 0.3 + (arow-1)*pitch_simX_;
      higher_x = 0.3 + (arow+1)*pitch_simX_;
    }
    if(arow == 80) {
      lower_x = 0.3 + (arow)*pitch_simX_;
      higher_x = 0.3 + (arow+2)*pitch_simX_;
    }
    if( arow > 80 && arow <= 158 ){
      lower_x = 0.3 + (arow+1)*pitch_simX_;
      higher_x = 0.3 + (arow+2)*pitch_simX_;
    } 
    if( arow == 159){
      lower_x = 0.3 + (arow+1)*pitch_simX_;
      higher_x = 0.3 + (arow+4)*pitch_simX_;
    } 

// columns (y segmentation)

    if(acol == 0) {
      lower_y = 0;
      higher_y = 0.35;
    }
    if( acol > 0 && acol <= 50 ){
      lower_y = 0.35 + (acol-1)*pitch_simY_;
      higher_y = 0.35 + acol*pitch_simY_;
    } 
    if(acol == 51) {
      lower_y = 0.35 + (acol-1)*pitch_simY_;
      higher_y = 0.35 + (acol+1)*pitch_simY_;
    }
    if(acol == 52) {
      lower_y = 0.35 + (acol)*pitch_simY_;
      higher_y = 0.35 + (acol+2)*pitch_simY_;
    }
    if( acol > 52 && acol <= 102 ){
      lower_y = 0.35 + (acol+1)*pitch_simY_;
      higher_y = 0.35 + (acol+2)*pitch_simY_;
    } 

    if(acol == 103) {
      lower_y = 0.35 + (acol+1)*pitch_simY_;
      higher_y = 0.35 + (acol+3)*pitch_simY_;
    }
    if(acol == 104) {
      lower_y = 0.35 + (acol+2)*pitch_simY_;
      higher_y = 0.35 + (acol+4)*pitch_simY_;
    }
    if( acol > 104 && acol <= 154 ){
      lower_y = 0.35 + (acol+3)*pitch_simY_;
      higher_y = 0.35 + (acol+4)*pitch_simY_;
    } 
    if(acol == 155) {
      lower_y = 0.35 + (acol+3)*pitch_simY_;
      higher_y = 0.35 + (acol+4)*pitch_simY_ + 0.2;
    }
 
    lower_x = lower_x - simX_width_/2.;
    lower_y = lower_y - simY_width_/2.;
    higher_x = higher_x - simX_width_/2.;
    higher_y = higher_y - simY_width_/2.;

  }

private:
  std::vector<pixel_info> theRelevantPixels_;

  double active_edge_x_;
  double active_edge_y_;
    
  inline double activeEdgeFactor(double x, double y)
  {
    double topEdgeFactor=TMath::Erf(-distanceFromTopActiveEdge(x, y)/active_edge_sigma_)/2 + 0.5;
    double bottomEdgeFactor=TMath::Erf(-distanceFromBottomActiveEdge(x, y)/active_edge_sigma_)/2 + 0.5;
    double rightEdgeFactor=TMath::Erf(-distanceFromRightActiveEdge(x, y)/active_edge_sigma_)/2 + 0.5;
    double leftEdgeFactor=TMath::Erf(-distanceFromLeftActiveEdge(x, y)/active_edge_sigma_)/2 + 0.5;

    double aEF = topEdgeFactor*bottomEdgeFactor*rightEdgeFactor*leftEdgeFactor;

    if(aEF>1)
      throw cms::Exception("CTPPSPixelSimTopology")<< " active edge factor > 1";

    return aEF;
  }
     
  inline double distanceFromTopActiveEdge(double x, double y)
  {
    double d=y-active_edge_y_;
    return d;
  }

  inline double distanceFromBottomActiveEdge(double x, double y)
  {
    double d=-y-active_edge_y_;
    return d;
  }

  inline double distanceFromRightActiveEdge(double x, double y)
  {
    double d=x-active_edge_x_;
    return d;
  }

  inline double distanceFromLeftActiveEdge(double x, double y)
  {
    double d=-x-active_edge_x_;
    return d;
  }

  inline unsigned int row(double x){
// x in the G4 simulation system

    x = x + simX_width_/2.;
 
// now x in the system centered in the bottom left corner of the sensor (sensor view, rocs behind)
    if(!(x>=0 && x<=simX_width_))
      throw cms::Exception("CTPPSPixelSimTopology")<< " out of reference frame";
    unsigned int arow;
// rows (x segmentation)

    if(x <= (dead_edge_width_+pitch_simX_) ) arow = 0;

    if(x > (dead_edge_width_+pitch_simX_) && x <= (dead_edge_width_+79*pitch_simX_)   ){
      arow = int((x - dead_edge_width_-pitch_simX_)/pitch_simX_)+1 ;
    }      

    if(x >  (dead_edge_width_+79*pitch_simX_) &&  x<= (dead_edge_width_+81*pitch_simX_)) arow = 79;

    if(x >  (dead_edge_width_+81*pitch_simX_) &&  x<= (dead_edge_width_+83*pitch_simX_)) arow = 80;

    if(x > (dead_edge_width_+83*pitch_simX_) && x <= (dead_edge_width_+162*pitch_simX_)   ){
      arow = int((x - dead_edge_width_-pitch_simX_)/pitch_simX_)-1;
    }      
    if(x > (dead_edge_width_+162*pitch_simX_) ) arow = 159;

    arow=159-arow;
    if(arow>159)
      throw cms::Exception("CTPPSPixelSimTopology")<< " row number exceeding limit";
 
    return arow;
  }

  inline unsigned int col(double y){
    unsigned int column;
// y in the G4 simulation system

// columns (y segmentation)
// now y in the system centered in the bottom left corner of the sensor (sensor view, rocs behind)
    y = y + simY_width_/2.;
    if(!(y>=0 && y <=simY_width_))
      throw cms::Exception("CTPPSPixelSimTopology")<< " out of reference frame";

    if(y <= (dead_edge_width_+pitch_simY_) ) column = 0;

    if(y > (dead_edge_width_+pitch_simY_) && y <= (dead_edge_width_+51*pitch_simY_)   ){
      column = int((y - dead_edge_width_-pitch_simY_)/pitch_simY_)+1 ;
    }      

    if(y >  (dead_edge_width_+51*pitch_simY_) &&  y<= (dead_edge_width_+53*pitch_simY_)) column = 51;

    if(y >  (dead_edge_width_+53*pitch_simY_) &&  y<= (dead_edge_width_+55 *pitch_simY_)) column = 52;

    if(y >  (dead_edge_width_+55*pitch_simY_) &&  y<= (dead_edge_width_+105 *pitch_simY_)) {
      column = int((y - dead_edge_width_-pitch_simY_)/pitch_simY_)-1 ;
    }


    if(y >  (dead_edge_width_+105*pitch_simY_) &&  y<= (dead_edge_width_+107*pitch_simY_)) column = 103;

    if(y >  (dead_edge_width_+107*pitch_simY_) &&  y<= (dead_edge_width_+109 *pitch_simY_)) column = 104;

    if(y >  (dead_edge_width_+109*pitch_simY_) &&  y<= (dead_edge_width_+159 *pitch_simY_)) {
      column = int((y - dead_edge_width_-pitch_simY_)/pitch_simY_)-3 ;
    }

    if(y >  (dead_edge_width_+159*pitch_simY_) ) column = 155;

    return column;
  }

  inline void rowCol2Index(unsigned int arow, unsigned int acol, unsigned int &index){
    index = acol*no_of_pixels_simX_+arow;
  }

  inline void index2RowCol(unsigned int &arow, unsigned int &acol, unsigned int index){
    acol = index / no_of_pixels_simX_ ;
    arow = index % no_of_pixels_simX_;
  }

};

#endif
