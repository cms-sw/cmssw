#ifndef Geometry_CTPPSPixelDetTopology_RPix_DET_SIM_TOPOLOGY_H
#define Geometry_CTPPSPixelDetTopology_RPix_DET_SIM_TOPOLOGY_H

#include "TMath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
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
pixel_info(double lower_simX_border, double higher_simX_border, double lower_simY_border, double higher_simY_border, double eff_factor, 
	   unsigned short pixel_row_no, unsigned short pixel_col_no) : lower_simX_border_(lower_simX_border), 
    higher_simX_border_(higher_simX_border), lower_simY_border_(lower_simY_border), 
    higher_simY_border_(higher_simY_border), eff_factor_(eff_factor), pixel_row_no_(pixel_row_no), pixel_col_no_(pixel_col_no),pixel_index_(pixel_col_no*160+pixel_row_no){}
  inline double & HigherSimXBorder() {return higher_simX_border_;}
  inline double & LowerSimXBorder() {return lower_simX_border_;}
  inline double & HigherSimYBorder() {return higher_simY_border_;}
  inline double & LowerSimYBorder() {return lower_simY_border_;}
  inline double & EffFactor() {return eff_factor_;}
  inline unsigned short & PixelRowNo() {return pixel_row_no_;}
  inline unsigned short & PixelColNo() {return pixel_col_no_;}
  inline unsigned short &PixelIndex() {return pixel_index_;}

  inline double HigherSimXBorder() const {return higher_simX_border_;}
  inline double LowerSimXBorder() const {return lower_simX_border_;}
  inline double HigherSimYBorder() const {return higher_simY_border_;}
  inline double LowerSimYBorder() const {return lower_simY_border_;}
  inline double EffFactor() const {return eff_factor_;}
  inline unsigned short PixelRowNo() const {return pixel_row_no_;}
  inline unsigned short PixelColNo() const {return pixel_col_no_;}
  inline unsigned short PixelIndex() const {return pixel_index_;}
    

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
  CTPPSPixelSimTopology(const edm::ParameterSet &params);
  std::vector<pixel_info> GetPixelsInvolved(double x, double y, double sigma, double &hit_pos_x, double &hit_pos_y);
  inline void PixelRange(unsigned int row, unsigned int col, double &lower_x, double &higher_x, double &lower_y, double &higher_y){
       
// x and y in the system  of Geant4 SIMULATION

    row = 159 - row;

    assert(row<=159 &&  col <=155);
 
// rows (x segmentation)

    if(row == 0) {
      lower_x = 0;
      higher_x = 0.3;
    }
    if( row > 0 && row <= 78 ){
      lower_x = 0.3 + (row-1)*pitch_simX_;
      higher_x = 0.3 + row*pitch_simX_;
    } 
    if(row == 79) {
      lower_x = 0.3 + (row-1)*pitch_simX_;
      higher_x = 0.3 + (row+1)*pitch_simX_;
    }
    if(row == 80) {
      lower_x = 0.3 + (row)*pitch_simX_;
      higher_x = 0.3 + (row+2)*pitch_simX_;
    }
    if( row > 80 && row <= 158 ){
      lower_x = 0.3 + (row+1)*pitch_simX_;
      higher_x = 0.3 + (row+2)*pitch_simX_;
    } 
    if( row == 159){
      lower_x = 0.3 + (row+1)*pitch_simX_;
      higher_x = 0.3 + (row+4)*pitch_simX_;
    } 

// columns (y segmentation)



    if(col == 0) {
      lower_y = 0;
      higher_y = 0.35;
    }
    if( col > 0 && col <= 50 ){
      lower_y = 0.35 + (col-1)*pitch_simY_;
      higher_y = 0.35 + col*pitch_simY_;
    } 
    if(col == 51) {
      lower_y = 0.35 + (col-1)*pitch_simY_;
      higher_y = 0.35 + (col+1)*pitch_simY_;
    }
    if(col == 52) {
      lower_y = 0.35 + (col)*pitch_simY_;
      higher_y = 0.35 + (col+2)*pitch_simY_;
    }
    if( col > 52 && col <= 102 ){
      lower_y = 0.35 + (col+1)*pitch_simY_;
      higher_y = 0.35 + (col+2)*pitch_simY_;
    } 

    if(col == 103) {
      lower_y = 0.35 + (col+1)*pitch_simY_;
      higher_y = 0.35 + (col+3)*pitch_simY_;
    }
    if(col == 104) {
      lower_y = 0.35 + (col+2)*pitch_simY_;
      higher_y = 0.35 + (col+4)*pitch_simY_;
    }
    if( col > 104 && col <= 154 ){
      lower_y = 0.35 + (col+3)*pitch_simY_;
      higher_y = 0.35 + (col+4)*pitch_simY_;
    } 
    if(col == 155) {
      lower_y = 0.35 + (col+3)*pitch_simY_;
      higher_y = 0.35 + (col+4)*pitch_simY_ + 0.2;
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
    

  double active_edge_sigma_;
    
  int verbosity_;
    

  
  inline double ActiveEdgeFactor(double x, double y)
  {
    double TopEdgeFactor=TMath::Erf(-DistanceFromTopActiveEdge(x, y)/active_edge_sigma_)/2 + 0.5;
    double BottomEdgeFactor=TMath::Erf(-DistanceFromBottomActiveEdge(x, y)/active_edge_sigma_)/2 + 0.5;
    double RightEdgeFactor=TMath::Erf(-DistanceFromRightActiveEdge(x, y)/active_edge_sigma_)/2 + 0.5;
    double LeftEdgeFactor=TMath::Erf(-DistanceFromLeftActiveEdge(x, y)/active_edge_sigma_)/2 + 0.5;

    double AEF = TopEdgeFactor*BottomEdgeFactor*RightEdgeFactor*LeftEdgeFactor;

    assert(AEF<=1);

    return AEF;
  }
    

 
  inline double DistanceFromTopActiveEdge(double x, double y)
  {
    double d=y-active_edge_y_;
    return d;
  }
  inline double DistanceFromBottomActiveEdge(double x, double y)
  {
    double d=-y-active_edge_y_;
    return d;
  }
  inline double DistanceFromRightActiveEdge(double x, double y)
  {
    double d=x-active_edge_x_;
    return d;
  }
  inline double DistanceFromLeftActiveEdge(double x, double y)
  {
    double d=-x-active_edge_x_;
    return d;
  }


  inline unsigned int Row(double x){
// x in the G4 simulation system

x = x + simX_width_/2.;
 
// now x in the system centered in the bottom left corner of the sensor (sensor view, rocs behind)
    assert(x>=0 && x<=simX_width_);
    unsigned int row;
// rows (x segmentation)

    if(x <= (dead_edge_width_+pitch_simX_) ) row = 0;

    if(x > (dead_edge_width_+pitch_simX_) && x <= (dead_edge_width_+79*pitch_simX_)   ){
      row = int((x - dead_edge_width_-pitch_simX_)/pitch_simX_)+1 ;
    }      

    if(x >  (dead_edge_width_+79*pitch_simX_) &&  x<= (dead_edge_width_+81*pitch_simX_)) row = 79;

    if(x >  (dead_edge_width_+81*pitch_simX_) &&  x<= (dead_edge_width_+83*pitch_simX_)) row = 80;

    if(x > (dead_edge_width_+83*pitch_simX_) && x <= (dead_edge_width_+162*pitch_simX_)   ){
      row = int((x - dead_edge_width_-pitch_simX_)/pitch_simX_)-1;
    }      
    if(x > (dead_edge_width_+162*pitch_simX_) ) row = 159;

    row=159-row;
    assert(row<=159);

 
    return row;
       

  }

  inline unsigned int Col(double y){
    unsigned int col;
// y in the G4 simulation system

// columns (y segmentation)
// now y in the system centered in the bottom left corner of the sensor (sensor view, rocs behind)
    y = y + simY_width_/2.;
    assert(y>=0 && y <=simY_width_);

    if(y <= (dead_edge_width_+pitch_simY_) ) col = 0;

    if(y > (dead_edge_width_+pitch_simY_) && y <= (dead_edge_width_+51*pitch_simY_)   ){
      col = int((y - dead_edge_width_-pitch_simY_)/pitch_simY_)+1 ;
    }      

    if(y >  (dead_edge_width_+51*pitch_simY_) &&  y<= (dead_edge_width_+53*pitch_simY_)) col = 51;

    if(y >  (dead_edge_width_+53*pitch_simY_) &&  y<= (dead_edge_width_+55 *pitch_simY_)) col = 52;

    if(y >  (dead_edge_width_+55*pitch_simY_) &&  y<= (dead_edge_width_+105 *pitch_simY_)) {
      col = int((y - dead_edge_width_-pitch_simY_)/pitch_simY_)-1 ;
    }



    if(y >  (dead_edge_width_+105*pitch_simY_) &&  y<= (dead_edge_width_+107*pitch_simY_)) col = 103;

    if(y >  (dead_edge_width_+107*pitch_simY_) &&  y<= (dead_edge_width_+109 *pitch_simY_)) col = 104;

    if(y >  (dead_edge_width_+109*pitch_simY_) &&  y<= (dead_edge_width_+159 *pitch_simY_)) {
      col = int((y - dead_edge_width_-pitch_simY_)/pitch_simY_)-3 ;
    }

    if(y >  (dead_edge_width_+159*pitch_simY_) ) col = 155;


    return col;

  }


  inline void RowCol2Index(unsigned int row, unsigned int col, unsigned int &index){
    index = col*160+row;
  }
  inline void Index2RowCol(unsigned int &row, unsigned int &col, unsigned int index){
    col = index / 160;
    row = index % 160;
  }

};

#endif  //Geometry_CTPPSPixelDetTopology_RPix_DET_SIM_TOPOLOGY_H
