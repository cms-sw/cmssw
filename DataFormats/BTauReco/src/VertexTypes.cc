#include "DataFormats/BTauReco/interface/VertexTypes.h"
#include <cstdlib>

using namespace std;

namespace reco {

std::string btag::Vertices::name ( VertexType v )
{
  switch (v)
  {
    case NoVertex:
      return "NoVertex";
    case PseudoVertex:
      return "PseudoVertex";
    case RecoVertex:
      return "RecoVertex";
    default:
      return "???";
  }
}

btag::Vertices::VertexType btag::Vertices::type ( const std::string & s )
{
  if ( s=="NoVertex" || s=="No" ) return NoVertex;
  if ( s=="PseudoVertex" || s=="Pseudo" ) return PseudoVertex;
  if ( s=="RecoVertex" || s=="Reco" ) return RecoVertex;

  int i = atoi ( s.c_str() );
  if ( i > 0 ) return ( VertexType ) (i);
  if ( ( i==0 ) && s == "0" ) return ( VertexType ) (i);

  return UndefVertex;
}

}
