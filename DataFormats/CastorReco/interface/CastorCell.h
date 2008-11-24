#ifndef CastorReco_CastorCell_h
#define CastorReco_CastorCell_h
/** \class reco::CastorCell CastorCell.h DataFormats/CastorReco/CastorCell.h
 *  
 * Class for CastorCells made of full simulation/data
 *
 * \author Hans Van Haevermaet, University of Antwerp
 *
 * \version $Id: CastorCell.h,v 1.1.2.1 2008/08/29 14:29:10 hvanhaev Exp $
 *
 */
#include <vector>
#include "DataFormats/Math/interface/Point3D.h"

namespace reco {

  class CastorCell {
  public:

    /// default constructor. Sets energy and position to zero
    CastorCell() : energy_(0.), position_(ROOT::Math::XYZPoint(0.,0.,0.)) { }

    /// constructor from values
    CastorCell(const double energy, const ROOT::Math::XYZPoint& position);

    /// destructor
    virtual ~CastorCell();

    /// cell energy
    double energy() const { return energy_; }

    /// cell centroid position
    ROOT::Math::XYZPoint position() const { return position_; }

    /// comparison >= operator
    bool operator >=(const CastorCell& rhs) const { return (energy_>=rhs.energy_); }

    /// comparison > operator
    bool operator > (const CastorCell& rhs) const { return (energy_> rhs.energy_); }

    /// comparison <= operator
    bool operator <=(const CastorCell& rhs) const { return (energy_<=rhs.energy_); }

    /// comparison <= operator
    bool operator < (const CastorCell& rhs) const { return (energy_< rhs.energy_); }

    /// z coordinate of cell centroid
    double z() const { return position_.z(); }
    /// azimuthal angle of cell centroid
    double phi() const { return position_.phi(); }
  private:

    /// cell energy
    double energy_;

    /// cell centroid position
    ROOT::Math::XYZPoint position_;
  };
  
  // define CastorCellCollection
  typedef std::vector<CastorCell> CastorCellCollection;
  
}

#endif
