#ifndef DataFormats_BTauReco_VertexTypes_h
#define DataFormats_BTauReco_VertexTypes_h

#include <string>

namespace reco {
  namespace btag {
    namespace Vertices {
      /** Type of secondary vertex found in jet:
     *  - RecoVertex   : a secondary vertex has been fitted from
     *                   a selection of tracks
     *  - PseudoVertex : no RecoVertex has been found but tracks
     *                   with significant impact parameter could be
     *                   combined to a "pseudo" vertex
     *  - NoVertex     : neither of the above attemps were successfull
     *  - NotDefined   : if anything went wrong, set to this value
     */
      enum VertexType { RecoVertex = 0, PseudoVertex = 1, NoVertex = 2, UndefVertex = 99 };

      /**
     *  convenience functions that return descriptive strings, rather than
     *  integral types
     */
      std::string name(VertexType);
      VertexType type(const std::string&);
    }  // namespace Vertices
  }    // namespace btag
}  // namespace reco

#endif
