#define G__DICTIONARY
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    std::vector<ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> > > vv1;
    std::vector<ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> > > vvf1;
    std::vector<ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t> > > vv2;
    std::vector<ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> > > vvf2;
    std::vector<ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> > > vp1;
    std::vector<ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> > > vpf1;
    std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> > > vl1;
    std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > > vlf1;
    std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Double32_t> > > vl2;
    std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> > > vlf2;
	
    edm::Wrapper<std::vector<ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> > > > wvv1;
    edm::Wrapper<std::vector<ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> > > > wvvf1;
    edm::Wrapper<std::vector<ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t> > > > wvv2;
    edm::Wrapper<std::vector<ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> > > > wvvf2;
    edm::Wrapper<std::vector<ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> > > > wvp1;
    edm::Wrapper<std::vector<ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> > > > wvpf1;
    edm::Wrapper<std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> > > > wvl1;
    edm::Wrapper<std::vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > > > wvlf1;
    edm::Wrapper<std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Double32_t> > > > wvl2;
    edm::Wrapper<std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<float> > > > wvlf2;
   }
}
