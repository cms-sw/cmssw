#ifndef FastSimDataFormats_NuclearInteractions_FSimDisplacedVertex_h
#define FastSimDataFormats_NuclearInteractions_FSimDisplacedVertex_h

#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "FastSimDataFormats/NuclearInteractions/interface/FSimVertexType.h"

#include <iostream>
#include <vector>

/*!\Data Format FSimDisplacedVertex

\brief  A FSimDisplacedVertex extends the FSimVertex with VertexType information

This data format is designed to combine the informations 
from FSimVertex and its FSimVertexType. It is not part of 
general FAMOS sequence and useful for 
private productions and analyses. The format contains also 
an integer identifier which may be optionally used 
to associate it to reconstruced
nuclear Interaction: boolean isRecoVertex and int recoVertexId;

\author Maxime Gouzevitch
\date November 2009
*/

class FSimDisplacedVertex {
public:
  FSimDisplacedVertex();
  FSimDisplacedVertex(const SimVertex& vertex,
                      unsigned id,
                      int motherId,
                      unsigned nCharged,
                      const std::vector<int>& daughterIds,
                      const FSimVertexType::VertexType vertexType);

  FSimDisplacedVertex(const FSimDisplacedVertex& other);

  virtual ~FSimDisplacedVertex() {}

  /// \return the SimVertex
  const SimVertex vertex() const { return vertex_; }

  /// \return the id of the vertex in the collection
  int id() const { return id_; }

  /// \return mother id in the track collection
  int motherId() const { return motherId_; }

  /// \return the number of daughters
  unsigned int nDaughters() const { return daughterIds_.size(); }

  /// \return the number of charged daughters
  unsigned int nChargedDaughters() const { return nCharged_; }

  /// \return vector of daughter ids
  const std::vector<int>& daughterIds() const { return daughterIds_; }

  /// \return the vertex type
  const FSimVertexType::VertexType vertexType() const { return vertexType_; }

  /// \return indicated if there is a Displaced Vertex associated
  const bool isRecoVertex() const { return isRecoVertex_; }

  /// \return the reconstructed Displaced Vertex index
  const int recoVertexId() const { return recoVertexId_; }

  /// Set the associated reconstructed DispacedVertex
  void setRecoVertex(int recoVertexId) {
    isRecoVertex_ = true;
    recoVertexId_ = recoVertexId;
  }

  /// Remove the associated reconstructed DispacedVertex
  void removeRecoVertex() {
    isRecoVertex_ = false;
    recoVertexId_ = -1;
  }

private:
  /// Sim Vertex
  SimVertex vertex_;

  /// \return the id in the vertex in the collection.
  /// -1 if the default value
  int id_;

  /// id of mother particle. -1 if no mother
  int motherId_;

  /// Number of charged daughters
  unsigned int nCharged_;

  ///  Vector of daughter ids in the track collection
  std::vector<int> daughterIds_;

  ///  Vertex Type
  FSimVertexType::VertexType vertexType_;

  /// Flag to indicate if a reconstructed DisplacedVertex was found and associated
  bool isRecoVertex_;

  /// The index of the reconstructed DisplacedVertex associated.
  /// By default the value is -1.
  /// The association may be done in the dedicated algorithm of the producer.
  int recoVertexId_;

  friend std::ostream& operator<<(std::ostream& out, const FSimDisplacedVertex& co);
};

#endif
