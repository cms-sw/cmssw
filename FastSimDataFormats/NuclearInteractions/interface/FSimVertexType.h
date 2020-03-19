#ifndef FastSimDataFormats_NuclearInteractions_FSimVertexType_h
#define FastSimDataFormats_NuclearInteractions_FSimVertexType_h

#include <ostream>

/*!\Data Format FSimVertexType

\brief  A FSimVertexType hold the information on the vertex origine

This data format is designed to hold the information 
obout the nature of a SimVertex within the Famos or 
Generation sequences.

\author Maxime Gouzevitch
\date November 2009
*/

class FSimVertexType {
public:
  /// Enum of possible vertex types.
  /// May be extended according to different needs
  enum VertexType {
    ANY = 0,
    PRIMARY_VERTEX = 1,
    NUCL_VERTEX = 2,
    PAIR_VERTEX = 3,
    BREM_VERTEX = 4,
    DECAY_VERTEX = 5,
    END_VERTEX = 6,
    PILEUP_VERTEX = 7,
    BSM_VERTEX = 8
  };

  FSimVertexType();
  FSimVertexType(VertexType);
  virtual ~FSimVertexType() {}

  const VertexType vertexType() const { return vertexType_; }
  void setVertexType(VertexType vertexType) { vertexType_ = vertexType; }

private:
  VertexType vertexType_;

  friend std::ostream& operator<<(std::ostream& out, const FSimVertexType& co);
};

#endif
