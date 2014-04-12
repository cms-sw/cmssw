#ifndef DataFormats_PatCandidates_interface_ParametrizationHelper_h
#define DataFormats_PatCandidates_interface_ParametrizationHelper_h

#include "DataFormats/PatCandidates/interface/CandKinResolution.h"
#include <string>

namespace pat { namespace helper {
namespace ParametrizationHelper {

    /// Returns the number of free parameters in a parametrization (3 or 4)
    inline uint32_t dimension(pat::CandKinResolution::Parametrization parametrization) {
        return (static_cast<uint32_t>(parametrization) & 0x0F);
    }

    /// Convert a name into a parametrization code
    pat::CandKinResolution::Parametrization fromString(const std::string &name) ;

    /// Convert a number into a string
    const char * name(pat::CandKinResolution::Parametrization param) ;   

    /// Given a choice of coordinate frame, a vector of coordinates and a reference 4-vector, produce a new 4 vector with the specified parameters.
    /// The new 4-vector is not guaranteed to satisfy the constraints of these parametrization if the initial 4-vector does not satisfy them.
    /// In the future this method will throw an exception if you go in an unphysical point of the coordinate system (e.g. E^2 < P^2)
    math::PtEtaPhiMLorentzVector polarP4fromParameters( pat::CandKinResolution::Parametrization parametrization, 
                                                        const AlgebraicVector4 &parameters, 
                                                        const math::PtEtaPhiMLorentzVector &initialP4) ;

    /// Given a choice of coordinate frame, a vector of coordinates and a reference 4-vector, produce a new 4 vector with the specified parameters.
    /// The new 4-vector is not guaranteed to satisfy the constraints of these parametrization if the initial 4-vector does not satisfy them.
    /// In the future this method will throw an exception if you go in an unphysical point of the coordinate system (e.g. E^2 < P^2)
    math::XYZTLorentzVector p4fromParameters(pat::CandKinResolution::Parametrization parametrization, 
                                            const AlgebraicVector4 &parameters,
                                            const math::XYZTLorentzVector &initialP4) ;

    /// Returns a vector of coordinates values given a coordinate frame and a 4-vector.
    AlgebraicVector4 parametersFromP4(pat::CandKinResolution::Parametrization parametrization, const math::XYZTLorentzVector &p4) ;

    /// Returns a vector of coordinates values given a coordinate frame and a 4-vector.
    AlgebraicVector4 parametersFromP4(pat::CandKinResolution::Parametrization parametrization, const math::PtEtaPhiMLorentzVector &p4) ;

    /// Set the values of the parameters for a given 4-momentum
    void setParametersFromP4(pat::CandKinResolution::Parametrization parametrization, AlgebraicVector4 &pars, const math::XYZTLorentzVector &p4) ;

    /// Set the values of the parameters for a given 4-momentum
    void setParametersFromP4(pat::CandKinResolution::Parametrization parametrization, AlgebraicVector4 &pars, const math::PtEtaPhiMLorentzVector &p4) ;

    /// For internal use only, so we provide only the interface. Use the 'setParametersFromP4'.
    template <typename T>
    void setParametersFromAnyVector(pat::CandKinResolution::Parametrization parametrization, AlgebraicVector4 &pars, const T &p4) ;

    /// Expresses the difference between two 4-momentum vectors as a shift in coordinates in a given frame.
    /** Basically, if you do:
     *  <code>
     *      pars = parametersFromP4(param, simp4);
     *      diff = diffToParameters(param, simP4, recP4);
     *  </code>
     *  then up to roundoff errors
     *  <code>recP4  == p4fromParameters(param, pars+diff, simP4);</code>
     */
    AlgebraicVector4 diffToParameters(pat::CandKinResolution::Parametrization parametrization, 
                const math::XYZTLorentzVector &p4ini, const math::XYZTLorentzVector &p4fin) ;

    /// Expresses the difference between two 4-momentum vectors as a shift in coordinates in a given frame.
    /** Basically, if you do:
     *  <code>
     *      pars = parametersFromP4(param, simp4);
     *      diff = diffToParameters(param, simP4, recP4);
     *  </code>
     *  then up to roundoff errors
     *  <code>recP4  == p4fromParameters(param, pars+diff, simP4);</code>
     */
    AlgebraicVector4 diffToParameters(pat::CandKinResolution::Parametrization parametrization, 
                const math::PtEtaPhiMLorentzVector &p4ini, const math::PtEtaPhiMLorentzVector &p4fin) ;

 
    /// Is this parametrization usable only with massless objects?
    bool isAlwaysMassless(pat::CandKinResolution::Parametrization parametrization) ;

    /// Is this parametrization usable only with massive objects?
    bool isAlwaysMassive(pat::CandKinResolution::Parametrization parametrization) ;

    /// If this parametrization has a mass constraint (including the 'isAlwaysMassless' case)
    bool isMassConstrained(pat::CandKinResolution::Parametrization parametrization) ;

    /// If this value of the parameters is meaningful in this parametrization. 
    /// It can be used e.g. when doing random smearing to check you're still within the physical region
    /// This DOES check inequalities (e.g. E >= P, M >= 0, theta in [0,PI], ..)
    /// This DOES NOT check strict equalities (e.g. M == 0)
    /// This DOES NOT check that your parameters comply with your constraints (e.g. fixed mass constraint)
    bool isPhysical(pat::CandKinResolution::Parametrization parametrization, const AlgebraicVector4 &v4, const math::PtEtaPhiMLorentzVector &initialP4) ;
} } }

#endif
