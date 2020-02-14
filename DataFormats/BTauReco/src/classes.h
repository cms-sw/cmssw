#include <utility>
#include <vector>

#include "Rtypes.h"
#include "Math/Cartesian3D.h"
#include "Math/CylindricalEta3D.h"
#include "Math/Polar3D.h"
#include "Math/PxPyPzE4D.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#if defined __has_feature
#if __has_feature(modules)
// Workaround the missing CLHEP.modulemap
#include "CLHEP/Vector/LorentzVector.h"
#include "DataFormats/BTauReco/interface/CombinedTauTagInfo.h"
#endif
#endif
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/CombinedTauTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfo.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfoFeatures.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"
#include "DataFormats/BTauReco/interface/TauMassTagInfo.h"
#include "DataFormats/BTauReco/interface/JetEisolAssociation.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/JTATagInfo.h"
#include "DataFormats/BTauReco/interface/JetTagInfo.h"
#include "DataFormats/BTauReco/interface/CATopJetTagInfo.h"
#include "DataFormats/BTauReco/interface/HTTTopJetTagInfo.h"
#include "DataFormats/BTauReco/interface/JetFeatures.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfoFeatures.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/TrackPairFeatures.h"
#include "DataFormats/BTauReco/interface/SeedingTrackFeatures.h"
#include "DataFormats/BTauReco/interface/DeepFlavourFeatures.h"
#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepDoubleXFeatures.h"
#include "DataFormats/BTauReco/interface/DeepDoubleXTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"
#include "DataFormats/BTauReco/interface/PixelClusterTagInfo.h"

namespace reco {
  typedef TrackTauImpactParameterAssociationCollection::map_type TrackTauImpactParameterAssociationMapType;
  typedef TrackTauImpactParameterAssociationCollection::ref_type TrackTauImpactParameterAssociationRefType;
  typedef TauMassTagInfo::ClusterTrackAssociationCollection::map_type TauMassTagInfo_ClusterTrackAssociationMapType;
  typedef TauMassTagInfo::ClusterTrackAssociationCollection::ref_type TauMassTagInfo_ClusterTrackAssociationRefType;
}  // namespace reco
