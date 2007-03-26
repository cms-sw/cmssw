#ifndef DTSEGMENTRESIDUAL_H
#define DTSEGMENTRESIDUAL_H

/** \class DTSegmentResidual
 *
 * Compute the residual of the hits wrt their segments
 *  
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 06/12/2006 14:28:13 CET $
 *
 * Modification:
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
class DTRecSegment2D;
class DTChamberRecSegment2D;
class DTChamber;
class DTSuperLayer;
#include "DataFormats/DTRecHit/interface/DTEnums.h"

/* C++ Headers */
#include <vector>
#include <utility>

/* ====================================================================== */

/* Class DTSegmentResidual Interface */

class DTSegmentResidual{

  public:
    struct DTResidual {
      DTResidual(double value,
                 double wireDistance = 0.0,
                 double angle=0.0,
                 DTEnums::DTCellSide side = DTEnums::undefLR) ; // constructor
      double value; // the resudual value
      double wireDistance; // the distance from wire
      double angle; // the impact angle (if any)
      DTEnums::DTCellSide side; // the side of the cell
    };


/* Constructor */ 
    DTSegmentResidual(const DTRecSegment2D* seg, const DTSuperLayer* sl) ;

    DTSegmentResidual(const DTChamberRecSegment2D* seg, const DTChamber* ch) ;
/* Destructor */ 

/* Operations */ 
    /// compute thr residuals
    void run() ;

    /// return the residuals
    std::vector<DTSegmentResidual::DTResidual> residuals() const {
      return theResiduals;
    }

    /// return the residuals as double
    std::vector<double> res() const ;

    /// return the residuals vs angle
    std::vector<std::pair<double, double> > residualsVsAngle() const ;

    /// return the residuals vs distance from wire
    std::vector<std::pair<double, double> > residualsVsWireDist() const ;

    /// return the residuals vs cell side
    std::vector<std::pair<double, DTEnums::DTCellSide> > residualsVsCellSide() const ;

  private:

    const DTRecSegment2D* theSeg;
    const DTChamber* theCh;
    const DTSuperLayer* theSL;
    std::vector<DTResidual> theResiduals;

  protected:

};
#endif // DTSEGMENTRESIDUAL_H

