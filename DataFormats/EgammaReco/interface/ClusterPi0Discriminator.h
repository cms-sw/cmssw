#ifndef EgammaReco_ClusterPi0Discriminator_h
#define EgammaReco_ClusterPi0Discriminator_h
/** \class reco::ClusterPi0Discriminator ClusterPi0Discriminator.h DataFormats/EgammaReco/interface/ClusterPi0Discriminator.h
 *  
 * pi0 discriminator variable set
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ClusterPi0Discriminator.h,v 1.2 2006/04/20 10:13:53 llista Exp $
 *
 */
#include <Rtypes.h>

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
