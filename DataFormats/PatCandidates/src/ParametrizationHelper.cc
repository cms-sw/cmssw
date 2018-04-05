#include "DataFormats/PatCandidates/interface/ParametrizationHelper.h"
#include <cmath>
#include <Math/CylindricalEta3D.h>
#include <Math/Polar3D.h>
#include <Math/Cartesian3D.h>
#include <Math/DisplacementVector3D.h>
#include <Math/Functions.h>

using namespace std; 

//pat::helper::ParametrizationHelper::POLAR_ZERO();
//pat::helper::ParametrizationHelper::CARTESIAN_ZERO();

pat::CandKinResolution::Parametrization 
pat::helper::ParametrizationHelper::fromString(const std::string &name) {
    using namespace std;

    typedef pat::CandKinResolution::Parametrization Parametrization;

    static map<string,Parametrization> const parMaps = {
      {"Cart"          , pat::CandKinResolution::Cart},
      {"ECart"         , pat::CandKinResolution::ECart},  
      {"MCCart"        , pat::CandKinResolution::MCCart},  
      {"Spher"         , pat::CandKinResolution::Spher},  
      {"ESpher"        , pat::CandKinResolution::ESpher},  
      {"MCSpher"       , pat::CandKinResolution::MCSpher},  
      {"MCPInvSpher"   , pat::CandKinResolution::MCPInvSpher},  
      {"EtEtaPhi"      , pat::CandKinResolution::EtEtaPhi},  
      {"EtThetaPhi"    , pat::CandKinResolution::EtThetaPhi},
      {"MomDev"        , pat::CandKinResolution::MomDev},
      {"EMomDev"       , pat::CandKinResolution::EMomDev},
      {"MCMomDev"      , pat::CandKinResolution::MCMomDev},
      {"EScaledMomDev" , pat::CandKinResolution::EScaledMomDev}
    };
    map<string,Parametrization>::const_iterator itP = parMaps.find(name);
    if (itP == parMaps.end()) {
        throw cms::Exception("StringResolutionProvider") << "Bad parametrization '" << name.c_str() << "'";
    }
    return itP->second;
}

const char * 
pat::helper::ParametrizationHelper::name(pat::CandKinResolution::Parametrization param) {
    switch (param) {
        case pat::CandKinResolution::Cart:          return "Cart";  
        case pat::CandKinResolution::ECart:         return "ECart";  
        case pat::CandKinResolution::MCCart:        return "MCCart";  
        case pat::CandKinResolution::Spher:         return "Spher";  
        case pat::CandKinResolution::ESpher:        return "ESpher";  
        case pat::CandKinResolution::MCSpher:       return "MCSpher";  
        case pat::CandKinResolution::MCPInvSpher:   return "MCPInvSpher";  
        case pat::CandKinResolution::EtEtaPhi:      return "EtEtaPhi";  
        case pat::CandKinResolution::EtThetaPhi:    return "EtThetaPhi";
        case pat::CandKinResolution::MomDev:        return "MomDev";
        case pat::CandKinResolution::EMomDev:       return "EMomDev";
        case pat::CandKinResolution::MCMomDev:      return "MCMomDev";
        case pat::CandKinResolution::EScaledMomDev: return "EScaledMomDev";
        case pat::CandKinResolution::Invalid:       return "Invalid";
        default: return "UNKNOWN";
    }
}
 
math::PtEtaPhiMLorentzVector 
pat::helper::ParametrizationHelper::polarP4fromParameters(pat::CandKinResolution::Parametrization parametrization, 
            const AlgebraicVector4 &parameters,
            const math::PtEtaPhiMLorentzVector &initialP4) {
    math::PtEtaPhiMLorentzVector  ret;
    ROOT::Math::CylindricalEta3D<double> converter;
    double m2;
    switch (parametrization) {
        // ======= CARTESIAN ==========
        case pat::CandKinResolution::Cart:
            ret = math::XYZTLorentzVector(parameters[0], parameters[1], parameters[2], 
                    sqrt(parameters[0]*parameters[0] + 
                        parameters[1]*parameters[1] + 
                        parameters[2]*parameters[2] + 
                        parameters[3]*parameters[3])  );
            ret.SetM(parameters[3]); // to be sure about roundoffs
            break;
        case pat::CandKinResolution::MCCart:
            ret = math::XYZTLorentzVector(parameters[0], parameters[1], parameters[2], 
                    sqrt(parameters[0]*parameters[0] + 
                        parameters[1]*parameters[1] + 
                        parameters[2]*parameters[2] + 
                        initialP4.mass()*initialP4.mass())  );
            ret.SetM(initialP4.mass()); // to be sure about roundoffs
            break;
        case pat::CandKinResolution::ECart:    
            ret = math::XYZTLorentzVector(parameters[0], parameters[1], parameters[2], parameters[3]);
            break;
        // ======= SPHERICAL ==========
        case pat::CandKinResolution::Spher:  
            converter = ROOT::Math::Polar3D<double>(parameters[0], parameters[1], 0); 
            ret.SetCoordinates(converter.Rho(),converter.Eta(),parameters[2],parameters[3]);
            break;
        case pat::CandKinResolution::MCSpher:    // same as above
            converter = ROOT::Math::Polar3D<double>(parameters[0], parameters[1], 0); 
            ret.SetCoordinates(converter.Rho(),converter.Eta(),parameters[2],initialP4.mass());
            break;
        case pat::CandKinResolution::ESpher:        //  
            converter = ROOT::Math::Polar3D<double>(parameters[0], parameters[1], 0); 
            m2 = - parameters[0]*parameters[0] + parameters[3]*parameters[3];
            ret.SetCoordinates(converter.Rho(),converter.Eta(),parameters[2],(m2 > 0 ? sqrt(m2) : 0.0));
            break;
        case pat::CandKinResolution::MCPInvSpher:   //  
            converter = ROOT::Math::Polar3D<double>(1.0/parameters[0], parameters[1], 0); 
            ret.SetCoordinates(converter.Rho(),converter.Eta(),parameters[2],initialP4.mass());
            break;
        // ======= HEP CYLINDRICAL ==========
        case pat::CandKinResolution::EtThetaPhi: 
            converter = ROOT::Math::Polar3D<double>(1.0, parameters[1], 0); 
            ret.SetCoordinates(parameters[0],converter.Eta(),parameters[2],0);
            break;
        case pat::CandKinResolution::EtEtaPhi:      // as simple as that
            ret.SetCoordinates(parameters[0], parameters[1], parameters[2], 0);
            break;
        // ======= MomentumDeviates ==========
        case pat::CandKinResolution::MomDev:
        case pat::CandKinResolution::EMomDev:
        case pat::CandKinResolution::MCMomDev:
        case pat::CandKinResolution::EScaledMomDev:
            {
                ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D<double> > p = initialP4.Vect(), uz(0,0,1), uph, uth;
                uph = uz.Cross(p).Unit();
                uth = p.Cross(uph).Unit();
                p *= parameters[0]; 
                p += uth * parameters[1] + uph * parameters[2];
                if (parametrization == pat::CandKinResolution::MomDev) {
                    ret.SetCoordinates(p.Rho(), p.Eta(), p.Phi(), initialP4.mass() * parameters[3]);
                } else if (parametrization == pat::CandKinResolution::EMomDev) {
                    double m2 = ROOT::Math::Square(parameters[3] * initialP4.energy()) - p.Mag2();
                    ret.SetCoordinates(p.Rho(), p.Eta(), p.Phi(), (m2 > 0 ? sqrt(m2) : 0.0));
                } else if (parametrization == pat::CandKinResolution::EScaledMomDev) {
                    double m2 = ROOT::Math::Square(p.R()*initialP4.E()/initialP4.P()) - p.Mag2();
                    ret.SetCoordinates(p.Rho(), p.Eta(), p.Phi(), (m2 > 0 ? sqrt(m2) : 0.0));
                } else if (parametrization == pat::CandKinResolution::MCMomDev) {
                    ret.SetCoordinates(p.Rho(), p.Eta(), p.Phi(), initialP4.mass());
                }
                break;
            }
        // ======= OTHER ==========
        case pat::CandKinResolution::Invalid:
            throw cms::Exception("Invalid parametrization") << parametrization;
        default:
            throw cms::Exception("Not Implemented") << "getResolEta not yet implemented for parametrization " << parametrization ;
    }
    return ret;
}

math::XYZTLorentzVector 
pat::helper::ParametrizationHelper::p4fromParameters(pat::CandKinResolution::Parametrization parametrization, 
            const AlgebraicVector4 &parameters,
            const math::XYZTLorentzVector &initialP4) {
    math::XYZTLorentzVector ret;
    switch (parametrization) {
        // ======= CARTESIAN ==========
        case pat::CandKinResolution::Cart:
            ret.SetCoordinates(parameters[0], parameters[1], parameters[2], 
                    sqrt(parameters[0]*parameters[0] + 
                        parameters[1]*parameters[1] + 
                        parameters[2]*parameters[2] + 
                        parameters[3]*parameters[3])  );
            break;
        case pat::CandKinResolution::MCCart: // same as above
            ret.SetCoordinates(parameters[0], parameters[1], parameters[2], 
                    sqrt(parameters[0]*parameters[0] + 
                        parameters[1]*parameters[1] + 
                        parameters[2]*parameters[2] + 
                        initialP4.mass()*initialP4.mass())  );
            break;
        case pat::CandKinResolution::ECart:    
            ret.SetCoordinates(parameters[0], parameters[1], parameters[2], parameters[3]);
            break;
        // ======= MomentumDeviates ==========
        case pat::CandKinResolution::MomDev:
        case pat::CandKinResolution::EMomDev:
        case pat::CandKinResolution::MCMomDev:
        case pat::CandKinResolution::EScaledMomDev:
            {
                ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D<double> > p = initialP4.Vect(), uz(0,0,1), uph, uth;
                uph = uz.Cross(p).Unit();
                uth = p.Cross(uph).Unit();
                p *= parameters[0]; 
                p += uth * parameters[1] + uph * parameters[2];
                if (parametrization == pat::CandKinResolution::MomDev) {
                    ret.SetCoordinates(p.X(), p.Y(), p.Z(), sqrt(p.Mag2() + ROOT::Math::Square(initialP4.mass() * parameters[3])) );
                } else if (parametrization == pat::CandKinResolution::EMomDev) {
                    ret.SetCoordinates(p.X(), p.Y(), p.Z(), parameters[3] * initialP4.energy());
                } else if (parametrization == pat::CandKinResolution::EMomDev) {
                    ret.SetCoordinates(p.X(), p.Y(), p.Z(), p.R() * initialP4.E()/initialP4.P());
                } else {
                    ret.SetCoordinates(p.X(), p.Y(), p.Z(), sqrt(p.Mag2() + initialP4.mass()*initialP4.mass()));
                }
                break;
            }
        // ======= ALL OTHERS ==========
        default:
            ret = polarP4fromParameters(parametrization, parameters, math::PtEtaPhiMLorentzVector(initialP4));
    }
    return ret;
}

template <typename T>
void
pat::helper::ParametrizationHelper::setParametersFromAnyVector(pat::CandKinResolution::Parametrization parametrization, 
            AlgebraicVector4 &ret, 
            const T &p4) {
    switch (parametrization) {
        // ======= CARTESIAN ==========
        case pat::CandKinResolution::Cart:
            ret[0] = p4.px(); ret[1] = p4.py(); ret[2] = p4.pz(); ret[3] = p4.mass();
            break;
        case pat::CandKinResolution::MCCart: 
            ret[0] = p4.px(); ret[1] = p4.py(); ret[2] = p4.pz(); ret[3] = p4.mass();        
            break;
        case pat::CandKinResolution::ECart:    
            ret[0] = p4.px(); ret[1] = p4.py(); ret[2] = p4.pz(); ret[3] = p4.energy();        
            break;
        // ======= SPHERICAL ==========
       case pat::CandKinResolution::Spher:  
            ret[0] = p4.P(); ret[1] = p4.theta(); ret[2] = p4.phi(); ret[3] = p4.mass();        
            break;
       case pat::CandKinResolution::MCSpher: 
            ret[0] = p4.P(); ret[1] = p4.theta(); ret[2] = p4.phi(); ret[3] = p4.mass();
            break;
       case pat::CandKinResolution::ESpher:
            ret[0] = p4.P(); ret[1] = p4.theta(); ret[2] = p4.phi(); ret[3] = p4.energy();        
            break;
        case pat::CandKinResolution::MCPInvSpher:  
            ret[0] = 1.0/p4.P(); ret[1] = p4.theta(); ret[2] = p4.phi(); ret[3] = p4.mass();
            break;
        // ======= HEP CYLINDRICAL ==========
        case pat::CandKinResolution::EtThetaPhi: 
            ret[0] = p4.pt(); ret[1] = p4.theta(); ret[2] = p4.phi(); ret[3] = 0;
            break;
        case pat::CandKinResolution::EtEtaPhi:
            ret[0] = p4.pt(); ret[1] = p4.eta(); ret[2] = p4.phi();   ret[3] = 0;
            break;
        // ======= DEVIATES ==========
        case pat::CandKinResolution::MomDev:
        case pat::CandKinResolution::EMomDev:
             ret[0] = 1.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 1.0;
             break;
        case pat::CandKinResolution::MCMomDev:
             ret[0] = 1.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = p4.mass();
             break;
        case pat::CandKinResolution::EScaledMomDev:
             ret[0] = 1.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = p4.E()/p4.P();
             break;
        // ======= OTHER ==========
        case pat::CandKinResolution::Invalid:
            throw cms::Exception("Invalid parametrization") << parametrization;
        default:
            throw cms::Exception("Not Implemented") << "getResolEta not yet implemented for parametrization " << parametrization ;
    }
}

void 
pat::helper::ParametrizationHelper::setParametersFromP4(pat::CandKinResolution::Parametrization parametrization, 
        AlgebraicVector4 &v, const math::PtEtaPhiMLorentzVector &p4) {
    setParametersFromAnyVector(parametrization, v, p4);
}

void 
pat::helper::ParametrizationHelper::setParametersFromP4(pat::CandKinResolution::Parametrization parametrization, 
        AlgebraicVector4 &v, const math::XYZTLorentzVector &p4) {
    setParametersFromAnyVector(parametrization, v, p4);
}

AlgebraicVector4 
pat::helper::ParametrizationHelper::parametersFromP4(pat::CandKinResolution::Parametrization parametrization, const math::PtEtaPhiMLorentzVector &p4) {
    AlgebraicVector4 ret;
    setParametersFromP4(parametrization, ret, p4);
    return ret;
}

AlgebraicVector4 
pat::helper::ParametrizationHelper::parametersFromP4(pat::CandKinResolution::Parametrization parametrization, const math::XYZTLorentzVector &p4) {
    AlgebraicVector4 ret;
    setParametersFromP4(parametrization, ret, p4);
    return ret;
}

AlgebraicVector4 
pat::helper::ParametrizationHelper::diffToParameters(pat::CandKinResolution::Parametrization parametrization, 
                const math::PtEtaPhiMLorentzVector &p4ini, const math::PtEtaPhiMLorentzVector &p4fin) 
{
    AlgebraicVector4 ret;
    switch (parametrization) {
        case pat::CandKinResolution::Cart:
        case pat::CandKinResolution::ECart:    
        case pat::CandKinResolution::MCCart:
            ret = parametersFromP4(parametrization,p4fin) - parametersFromP4(parametrization,p4ini);
            break;
        case pat::CandKinResolution::Spher: 
        case pat::CandKinResolution::ESpher:
        case pat::CandKinResolution::MCSpher:
        case pat::CandKinResolution::MCPInvSpher:
        case pat::CandKinResolution::EtThetaPhi: 
        case pat::CandKinResolution::EtEtaPhi:
            ret = parametersFromP4(parametrization,p4fin) - parametersFromP4(parametrization,p4ini);
            while(ret[2] > +M_PI) ret[2] -= (2*M_PI);
            while(ret[2] < -M_PI) ret[2] += (2*M_PI);
            break;
        case pat::CandKinResolution::MCMomDev:
        case pat::CandKinResolution::MomDev:
        case pat::CandKinResolution::EMomDev:
        case pat::CandKinResolution::EScaledMomDev:
            return diffToParameters(parametrization,
                                    math::XYZTLorentzVector(p4ini),
                                    math::XYZTLorentzVector(p4fin));
        case pat::CandKinResolution::Invalid:
            throw cms::Exception("Invalid parametrization") << parametrization;
        default:
            throw cms::Exception("Not Implemented") << "diffToParameters not yet implemented for parametrization " << parametrization ;
    }
    return ret;
}


AlgebraicVector4 
pat::helper::ParametrizationHelper::diffToParameters(pat::CandKinResolution::Parametrization parametrization, 
                const math::XYZTLorentzVector &p4ini, const math::XYZTLorentzVector &p4fin) 
{
    AlgebraicVector4 ret;
    switch (parametrization) {
        case pat::CandKinResolution::Cart:
        case pat::CandKinResolution::ECart:    
        case pat::CandKinResolution::MCCart:
        case pat::CandKinResolution::Spher: 
            ret = parametersFromP4(parametrization,p4fin) - parametersFromP4(parametrization,p4ini);
            break;
        case pat::CandKinResolution::ESpher:
        case pat::CandKinResolution::MCSpher:
        case pat::CandKinResolution::MCPInvSpher:
        case pat::CandKinResolution::EtThetaPhi: 
        case pat::CandKinResolution::EtEtaPhi:
            ret = parametersFromP4(parametrization,p4fin) - parametersFromP4(parametrization,p4ini);
            while(ret[2] > +M_PI) ret[2] -= (2*M_PI);
            while(ret[2] < -M_PI) ret[2] += (2*M_PI);
            break;
        case pat::CandKinResolution::MCMomDev:
        case pat::CandKinResolution::MomDev:
        case pat::CandKinResolution::EMomDev:
        case pat::CandKinResolution::EScaledMomDev:
            {
                typedef ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D<double> > V3Cart;
                V3Cart p1 = p4ini.Vect(), p2 = p4fin.Vect();
                V3Cart ur = p1.Unit(); 
                V3Cart uz(0,0,1);
                V3Cart uph = uz.Cross(ur).Unit();
                V3Cart uth = ur.Cross(uph).Unit();
                ret[0] = p2.Dot(ur)/p1.R() - 1.0;
                ret[1] = (p2 - p1).Dot(uth);
                ret[2] = (p2 - p1).Dot(uph);
                if (parametrization == pat::CandKinResolution::MomDev) {
                    ret[3] = p4fin.mass()/p4ini.mass() - 1.0;
                } else if (parametrization == pat::CandKinResolution::EMomDev) {
                    ret[3] = p4fin.energy()/p4ini.energy() - 1.0;
                }
            }
            break;
        case pat::CandKinResolution::Invalid:
            throw cms::Exception("Invalid parametrization") << parametrization;
        default:
            throw cms::Exception("Not Implemented") << "diffToParameters not yet implemented for parametrization " << parametrization ;
    }
    return ret;
}

bool
pat::helper::ParametrizationHelper::isAlwaysMassless(pat::CandKinResolution::Parametrization parametrization) 
{
    switch (parametrization) {
        case pat::CandKinResolution::Cart:
        case pat::CandKinResolution::ECart:    
        case pat::CandKinResolution::MCCart:
        case pat::CandKinResolution::Spher: 
        case pat::CandKinResolution::ESpher:
        case pat::CandKinResolution::MCSpher:
        case pat::CandKinResolution::MCPInvSpher:
        case pat::CandKinResolution::MCMomDev:
        case pat::CandKinResolution::MomDev:
        case pat::CandKinResolution::EMomDev:
        case pat::CandKinResolution::EScaledMomDev:
            return false;
        case pat::CandKinResolution::EtThetaPhi: 
        case pat::CandKinResolution::EtEtaPhi:
            return true;
        case pat::CandKinResolution::Invalid:
            throw cms::Exception("Invalid parametrization") << parametrization;
        default:
            throw cms::Exception("Not Implemented") << "isAlwaysMassless not yet implemented for parametrization " << parametrization ;
    }
}

bool
pat::helper::ParametrizationHelper::isAlwaysMassive(pat::CandKinResolution::Parametrization parametrization) 
{
    switch (parametrization) {
        case pat::CandKinResolution::Cart:
        case pat::CandKinResolution::ECart:    
        case pat::CandKinResolution::MCCart:
        case pat::CandKinResolution::Spher: 
        case pat::CandKinResolution::ESpher:
        case pat::CandKinResolution::MCSpher:
        case pat::CandKinResolution::MCPInvSpher:
        case pat::CandKinResolution::MCMomDev:
        case pat::CandKinResolution::EMomDev:
        case pat::CandKinResolution::EScaledMomDev:
        case pat::CandKinResolution::EtThetaPhi: 
        case pat::CandKinResolution::EtEtaPhi:
            return false;
        case pat::CandKinResolution::MomDev:
            return true;
        case pat::CandKinResolution::Invalid:
            throw cms::Exception("Invalid parametrization") << parametrization;
        default:
            throw cms::Exception("Not Implemented") << "isAlwaysMassless not yet implemented for parametrization " << parametrization ;
    }
}

bool
pat::helper::ParametrizationHelper::isMassConstrained(pat::CandKinResolution::Parametrization parametrization) 
{
    switch (parametrization) {
        case pat::CandKinResolution::Cart:
        case pat::CandKinResolution::ECart:    
        case pat::CandKinResolution::Spher: 
        case pat::CandKinResolution::ESpher:
        case pat::CandKinResolution::EMomDev:
        case pat::CandKinResolution::EScaledMomDev:
        case pat::CandKinResolution::EtThetaPhi: 
        case pat::CandKinResolution::EtEtaPhi:
        case pat::CandKinResolution::MomDev:
            return false;
        case pat::CandKinResolution::MCCart:
        case pat::CandKinResolution::MCSpher:
        case pat::CandKinResolution::MCPInvSpher:
        case pat::CandKinResolution::MCMomDev:
            return true;
        case pat::CandKinResolution::Invalid:
            throw cms::Exception("Invalid parametrization") << parametrization;
        default:
            throw cms::Exception("Not Implemented") << "isAlwaysMassless not yet implemented for parametrization " << parametrization ;
    }
}

bool
pat::helper::ParametrizationHelper::isPhysical(pat::CandKinResolution::Parametrization parametrization, 
            const AlgebraicVector4 &parameters,
            const math::PtEtaPhiMLorentzVector &initialP4) {
    switch (parametrization) {
        // ======= CARTESIAN ==========
        case pat::CandKinResolution::Cart:
            return parameters[3] >= 0; // M >= 0
        case pat::CandKinResolution::MCCart:
            return true;
        case pat::CandKinResolution::ECart: 
            return (parameters[0]*parameters[0] + parameters[1]*parameters[1] + parameters[2]*parameters[2] <= parameters[3]*parameters[3]); // E >= P
        case pat::CandKinResolution::Spher:  
            return (parameters[0] >= 0   ) && // P >= 0
                   (parameters[3] >= 0   ) && // M >= 0
                   (parameters[1] >= 0   ) && // theta >= 0
                   (parameters[1] <= M_PI);   // theta <= pi
        case pat::CandKinResolution::MCSpher:    
            return (parameters[0] >= 0   ) && // P >= 0
                   (parameters[1] >= 0   ) && // theta >= 0
                   (parameters[1] <= M_PI);   // theta <= pi
        case pat::CandKinResolution::ESpher:        //  
            return (parameters[0] >= 0            ) && // P >= 0
                   (parameters[3] >= parameters[0]) && // E >= P
                   (parameters[1] >= 0            ) && // theta >= 0
                   (parameters[1] <= M_PI         ) ;  // theta <= PI
       case pat::CandKinResolution::MCPInvSpher:   //  
            return (parameters[0] >  0   ) && // 1/P > 0
                   (parameters[1] >= 0   ) && // theta >= 0
                   (parameters[1] <= M_PI);   // theta <= pi
        // ======= HEP CYLINDRICAL ==========
        case pat::CandKinResolution::EtThetaPhi: 
            return (parameters[0] >  0   ) && // Et >= 0
                   (parameters[1] >= 0   ) && // theta >= 0
                   (parameters[1] <= M_PI);   // theta <= pi
        case pat::CandKinResolution::EtEtaPhi:      // as simple as that
            return (parameters[0] >  0); // Et >= 0
        // ======= MomentumDeviates ==========
        case pat::CandKinResolution::MomDev:
            return (parameters[0] >= 0) && // P >= 0
                   (parameters[3] >= 0);   // m/M0 >= 0
        case pat::CandKinResolution::MCMomDev:
            return (parameters[0] >= 0); // P >= 0
        case pat::CandKinResolution::EMomDev:
        case pat::CandKinResolution::EScaledMomDev:
            {
                if (parameters[0] <= 0) return false;
                ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D<double> > p = initialP4.Vect(), uz(0,0,1), uph, uth;
                uph = uz.Cross(p).Unit();
                uth = p.Cross(uph).Unit();
                p *= parameters[0]; 
                p += uth * parameters[1] + uph * parameters[2];
                if (parametrization == pat::CandKinResolution::EMomDev) {
                    if (parameters[3] < 0) return false;
                    double m2 = ROOT::Math::Square(parameters[3] * initialP4.energy()) - p.Mag2();
                    if (m2 < 0) return false;
                } else if (parametrization == pat::CandKinResolution::EScaledMomDev) {
                    if (parameters[3] < 0) return false;
                    double m2 = ROOT::Math::Square(p.R()*initialP4.E()/initialP4.P()) - p.Mag2();
                    if (m2 < 0) return false;
                }
                return true;
            }
        // ======= OTHER ==========
        case pat::CandKinResolution::Invalid:
            throw cms::Exception("Invalid parametrization") << parametrization;
        default:
            throw cms::Exception("Not Implemented") << "getResolEta not yet implemented for parametrization " << parametrization ;
    }
}


