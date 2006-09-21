#include "DataFormats/BTauReco/interface/CombinedBTagEnums.h"

using namespace std;

std::string reco::CombinedBTagEnums::typeOfVertex ( VertexType v )
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

std::string reco::CombinedBTagEnums::typeOfParton ( PartonType p )
{
  switch ( p )
  {
    case B:
      return "B";
    case C:
      return "C";
    case UDSG:
      return "UDSG";
    default:
      return "?";
  }
}

std::string reco::CombinedBTagEnums::typeOfVariable ( TaggingVariable t )
{
  switch ( t )
  {
    case Category:
      return "Category";
    case VertexMass:
      return "VertexMass";
    case VertexMultiplicity:
      return "VertexMultiplicity";
    case FlightDistance2DSignificance:
      return "FlightDistance2DSignificance";
    case ESVXOverE:
      return "ESVXOverE";
    case TrackRapidity:
      return "TrackRapidity";
    case TrackIP2DSignificance:
      return "TrackIP2DSignificance";
    case TrackIP2DSignificanceAboveCharm:
      return "TrackIP2DSignificanceAboveCharm";
    default:
      return "unknown";
  }
}
