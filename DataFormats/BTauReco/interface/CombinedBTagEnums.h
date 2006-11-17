#ifndef BTauReco_CombinedBTagEnums_h
#define BTauReco_CombinedBTagEnums_h

#include <string>

namespace reco {
  namespace CombinedBTagEnums  {
    /** Type of secondary vertex found in jet:
     *  - RecoVertex   : a secondary vertex has been fitted from
     *                   a selection of tracks
     *  - PseudoVertex : no RecoVertex has been found but tracks
     *                   with significant impact parameter could be
     *                   combined to a "pseudo" vertex
     *  - NoVertex     : neither of the above attemps were successfull
     *  - NotDefined   : if anything went wrong, set to this value
     */
    enum VertexType {RecoVertex, PseudoVertex, NoVertex, NotDefined};

    /** Type of parton from which the jet originated
     */
    enum PartonType {B, C, UDSG, UndefParton };

    /** list of all variables used to construct the
     *  combined b-tagging discriminator
     */
    enum TaggingVariable{Category,
       VertexMass,
       VertexMultiplicity,
       FlightDistance2DSignificance,
       ESVXOverE,
       TrackRapidity,
       TrackIP2DSignificance,
       TrackIP2DSignificanceAboveCharm,
       UndefTaggingVariable };

    /**
     *  convenience functions that return descriptive strings, rather than
     *  integral types
     */
    std::string typeOfVertex ( VertexType );
    std::string typeOfParton ( PartonType );
    std::string typeOfVariable ( TaggingVariable );

  }
}

#endif
