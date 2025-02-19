#ifndef CastorReco_CastorCell_h
#define CastorReco_CastorCell_h
/** \class reco::CastorCell CastorCell.h DataFormats/CastorReco/CastorCell.h
 *  
 * Class for CastorCells made of full simulation/data
 *
 * \author Hans Van Haevermaet, University of Antwerp
 *
 * \version $Id: CastorCell.h,v 1.5 2010/07/03 19:11:43 hvanhaev Exp $
 *
 */

#include <vector>
#include <memory>
#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

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

    /// x coordinate of cell centroid
    double x() const { return position_.x(); }

    /// y coordinate of cell centroid
    double y() const { return position_.y(); }

    /// rho coordinate of cell centroid
    double rho() const { return position_.rho(); }

    /// eta coordinate of cell centroid
    double eta() const { return position_.eta(); }

  private:

    /// cell energy
    double energy_;

    /// cell centroid position
    ROOT::Math::XYZPoint position_;
  };
  
  /// collection of CastorCell objects
  typedef std::vector<CastorCell> CastorCellCollection;

  // persistent reference to CastorCell objects
  typedef edm::Ref<CastorCellCollection> CastorCellRef;
 
  /// vector of references to CastorCell objects all in the same collection
  typedef edm::RefVector<CastorCellCollection> CastorCellRefVector;
 
  /// iterator over a vector of references to CastorCell objects all in the same collection
  typedef CastorCellRefVector::iterator CastorCell_iterator;
}

#endif
