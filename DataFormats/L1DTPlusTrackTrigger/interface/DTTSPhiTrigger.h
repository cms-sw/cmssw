#ifndef L1_DTTTI_TSPHI_h
#define L1_DTTTI_TSPHI_h

/*! \class DTBtiTrigger
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \brief used to store TSPhi information within DT TP seed creation
 *  \date 2009, Feb 2
 */

#include <vector>
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

/// Class implementation
class DTTSPhiTrigger : public DTChambPhSegm 
{
  public :
    /// Constructors and destructors
    DTTSPhiTrigger();
    DTTSPhiTrigger( const DTChambPhSegm& c, 
                    Global3DPoint position,
                    Global3DVector direction );
    ~DTTSPhiTrigger(){}

    Global3DPoint cmsPosition()   const { return _position; }
    Global3DVector cmsDirection() const { return _direction; }
    std::string sprint() const;

  private :
    int _wheel;
    int _station;
    int _sector;
    int _psi;
    int _psiR;
    int _DeltaPsiR;
    float _phiB;
    Global3DPoint  _position;
    Global3DVector _direction;
};

typedef std::vector< DTTSPhiTrigger > TSPhiTrigsCollection;

#endif

