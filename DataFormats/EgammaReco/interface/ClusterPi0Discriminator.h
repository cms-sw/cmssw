#ifndef EgammaReco_ClusterPi0Discriminator_h
#define EgammaReco_ClusterPi0Discriminator_h
/** \class reco::ClusterPi0Discriminator
 *  
 * pi0 discriminator variable set
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ClusterPi0Discriminator.h,v 1.5 2006/03/03 11:48:37 llista Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/EgammaReco/interface/ClusterPi0DiscriminatorFwd.h"

namespace reco {

  class ClusterPi0Discriminator {
  public:

    /// default constructor
    ClusterPi0Discriminator() { }

    /// constructor from values
    ClusterPi0Discriminator( double disc1, double disc2, double disc3 );

    /// discriminator variable #1 (should be better documented!)
    double disc1() const { return disc1_; }

    /// discriminator variable #2 (should be better documented!)
    double disc2() const { return disc2_; }

    /// discriminator variable #3 (should be better documented!)
    double disc3() const { return disc3_; }
  private:
    /// discriminator variables
    Double32_t disc1_, disc2_, disc3_;

  };

}

#endif
