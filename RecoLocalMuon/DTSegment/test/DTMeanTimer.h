#ifndef DTMEANTIMER_H
#define DTMEANTIMER_H

/** \class DTMeanTimer
 *
 * Description:
 *  Class to compute mean timer (also known as Tmax) for a triplet of DT layers.
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 22/11/2006 12:50:29 CET $
 *
 * Modification:
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */
class DTSuperLayer;
class DTTTrigBaseSync;
#include "DataFormats/Common/interface/Handle.h"
namespace edm {
  class Event;
  class EventSetup;
}
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

/* C++ Headers */
#include <map>
#include <vector>

/* ====================================================================== */

/* Class DTMeanTimer Interface */

class DTMeanTimer{

  public:

/// Constructor: it takes a list of hits of a SL and a reference to the SL id
    DTMeanTimer(const DTSuperLayer* sl,
                edm::Handle<DTRecHitCollection>& hits,
                const edm::EventSetup& eventSetup,
                DTTTrigBaseSync* sync) ;

/// Constructor: alternative way to pass a list of hits 
    DTMeanTimer(const DTSuperLayer* sl,
                std::vector<DTRecHit1D>& hits,
                const edm::EventSetup& eventSetup,
                DTTTrigBaseSync* sync) ;

/* Destructor */ 
    ~DTMeanTimer() ;

/* Operations */ 
/** return a vector of meanTimers calculated from the hits. For 4 hits in 4
 * different layers, eg, 2 MT are computed , one for the first 3 layers and one
 * for the last 3 layers. No selection on hits is done. */
    std::vector<double> run() const ;

  private:
    typedef std::map<int, double> hitColl ; // map between wire number and time

    std::vector<double> computeMT(hitColl hits1,
                                  hitColl hits2,
                                  hitColl hits3) const ;
    double tMax(const double& t1, const double& t2, const double& t3) const ;

  private:
    int theNumWires; // max number of wires in this SL

    hitColl hitsLay[4]; // four hits containers for the 4 layers

  protected:

};
#endif // DTMEANTIMER_H

