// \class JetTracksAssociationXtrpCalo
// Associate jets with tracks by extrapolation to calo face
// $Id: JetTracksAssociationXtrpCalo.h,v 1.5 2009/03/30 15:06:33 bainbrid Exp $

#ifndef RecoJets_JetAssociationAlgorithms_JetTracksAssociationXtrpCalo_h
#define RecoJets_JetAssociationAlgorithms_JetTracksAssociationXtrpCalo_h


#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/Point3D.h"
#include <vector>
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDR.h"
#include "DataFormats/JetReco/interface/TrackExtrapolation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class MagneticField;
class Propagator;

class JetTracksAssociationXtrpCalo : public JetTracksAssociationDR {
  
 public:
  /// Constructor
  JetTracksAssociationXtrpCalo();
  
  /// Destructor
  ~JetTracksAssociationXtrpCalo();

  /// Associates tracks to jets
  void produce( Association*,
		JetRefs const &,
		std::vector<reco::TrackExtrapolation> const &,
		CaloGeometry const &,
		double dR );

  void associateInputTracksToJet( reco::TrackRefVector& associated,
				  const reco::Jet& fJet,
				  std::vector<reco::TrackExtrapolation> const & fExtrapolations,
				  double dR ) ;
  

  

 private:

  /// Unused
  virtual void associateTracksToJet( reco::TrackRefVector&,
				     const reco::Jet&,
				     const TrackRefs& ) {}
};

#endif // RecoJets_JetAssociationAlgorithms_JetTracksAssociationXtrpCalo_h
