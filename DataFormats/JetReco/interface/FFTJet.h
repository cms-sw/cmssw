 /** \class reco::FFTJet
 *
 * \short Storage class for jets reconstructed by FFTJet package
 *
 * Collects jet properties unique to FFTJet algorithms
 *
 * \author Igor Volobouev, TTU
 *
 * \version   $Id: FFTJet.h,v 1.3 2011/07/06 07:39:11 igv Exp $
 ************************************************************/

#ifndef DataFormats_JetReco_FFTJet_h
#define DataFormats_JetReco_FFTJet_h

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/JetReco/interface/PattRecoPeak.h"

namespace reco {
    template<class Real>
    class FFTJet
    {
    public:
        inline FFTJet()
          : ncells_(0), etSum_(0), centroidEta_(0), centroidPhi_(0),
            etaWidth_(0), phiWidth_(0), etaPhiCorr_(0), fuzziness_(0),
            convergenceD_(0), recoScale_(0), recoScaleRatio_(0),
            membershipFactor_(0), code_(0), status_(0)
        {
        }

        inline FFTJet(const PattRecoPeak<Real>& peak,
                      const math::XYZTLorentzVector& vec,
                      double ncells, double etSum,
                      double centroidEta, double centroidPhi,
                      double etaWidth, double phiWidth,
                      double etaPhiCorr, double fuzziness,
                      double convergenceDistance, double recoScale,
                      double recoScaleRatio, double membershipFactor,
                      int code, int status)
          : peak_(peak),            
            vec_(vec),
            ncells_(ncells),          
            etSum_(etSum),           
            centroidEta_(centroidEta),     
            centroidPhi_(centroidPhi),     
            etaWidth_(etaWidth),        
            phiWidth_(phiWidth),        
            etaPhiCorr_(etaPhiCorr),      
            fuzziness_(fuzziness),       
            convergenceD_(convergenceDistance),    
            recoScale_(recoScale),       
            recoScaleRatio_(recoScaleRatio),  
            membershipFactor_(membershipFactor),
            code_(code),            
            status_(status)        
        {
        }

        inline virtual ~FFTJet() {}

        // inspectors
        inline const PattRecoPeak<Real>& f_precluster() const {return peak_;}
        inline const math::XYZTLorentzVector& f_vec() const {return vec_;}
        inline const math::XYZTLorentzVector& f_pileup() const {return pileup_;}
        inline Real f_ncells() const {return ncells_;}
        inline Real f_etSum() const {return etSum_;}
        inline Real f_centroidEta() const {return centroidEta_;}
        inline Real f_centroidPhi() const {return centroidPhi_;}
        inline Real f_etaWidth() const {return etaWidth_;}
        inline Real f_phiWidth() const {return phiWidth_;}
        inline Real f_etaPhiCorr() const {return etaPhiCorr_;}
        inline Real f_fuzziness() const {return fuzziness_;}
        inline Real f_convergenceDistance() const {return convergenceD_;}
        inline Real f_recoScale() const {return recoScale_;}
        inline Real f_recoScaleRatio() const {return recoScaleRatio_;}
        inline Real f_membershipFactor() const {return membershipFactor_;}
        inline int  f_code() const {return code_;}
        inline int  f_status() const {return status_;}

        // modifiers
        inline void setPileup(const math::XYZTLorentzVector& p) {pileup_ = p;}
        inline void setFourVec(const math::XYZTLorentzVector& p) {vec_ = p;}
        inline void setCode(const int c) {code_ = c;}
        inline void setStatus(const int c) {status_ = c;}
        inline void setNCells(const double nc) {ncells_ = nc;}

    private:
        PattRecoPeak<Real> peak_;
        math::XYZTLorentzVector vec_;
        math::XYZTLorentzVector pileup_;
        Real ncells_;
        Real etSum_;
        Real centroidEta_;
        Real centroidPhi_;
        Real etaWidth_;
        Real phiWidth_;
        Real etaPhiCorr_;
        Real fuzziness_;
        Real convergenceD_;
        Real recoScale_;
        Real recoScaleRatio_;
        Real membershipFactor_;
        int code_;
        int status_;
    };
}

#endif 
