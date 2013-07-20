/** \class reco::PattRecoPeak
 *
 * \short Preclusters from FFTJet pattern recognition stage
 *
 * This is a pure storage class with limited functionality.
 * Actual application calculations should use fftjet::Peak.
 *
 * \author Igor Volobouev, TTU, June 16, 2010
 * \version   $Id: PattRecoPeak.h,v 1.1 2010/11/22 23:27:56 igv Exp $
 ************************************************************/

#ifndef DataFormats_JetReco_PattRecoPeak_h
#define DataFormats_JetReco_PattRecoPeak_h

namespace reco {
    template<class Real>
    class PattRecoPeak
    {
    public:
        inline PattRecoPeak()
          : eta_(0),
            phi_(0),
            magnitude_(0),
            speed_(-1),
            magSpeed_(-5),
            lifetime_(-1),
            scale_(-1),
            nearestD_(-1),
            clusterRadius_(-1),
            clusterSeparation_(-1)
        {
            hessian_[0] = 0;
            hessian_[1] = 0;
            hessian_[2] = 0;
        }

        inline PattRecoPeak(double eta, double phi, double mag,
                            const double hessianIn[3], double driftSpeed,
                            double magSpeed, double lifetime,
                            double scale, double nearestDistance,
                            double clusterRadius, double clusterSeparation)
          : eta_(eta),
            phi_(phi),
            magnitude_(mag),
            speed_(driftSpeed),
            magSpeed_(magSpeed),
            lifetime_(lifetime),
            scale_(scale),
            nearestD_(nearestDistance),
            clusterRadius_(clusterRadius),
            clusterSeparation_(clusterSeparation)
        {
            hessian_[0] = hessianIn[0];
            hessian_[1] = hessianIn[1];
            hessian_[2] = hessianIn[2];
        }

        inline Real eta() const {return eta_;}
        inline Real phi() const {return phi_;}
        inline Real magnitude() const {return magnitude_;}
        inline Real driftSpeed() const {return speed_;}
        inline Real magSpeed() const {return magSpeed_;}
        inline Real lifetime() const {return lifetime_;}
        inline Real scale() const {return scale_;}
        inline Real nearestNeighborDistance() const {return nearestD_;}
        inline Real clusterRadius() const {return clusterRadius_;}
        inline Real clusterSeparation() const {return clusterSeparation_;}
        inline void hessian(double hessianArray[3]) const
        {
            hessianArray[0] = hessian_[0];
            hessianArray[1] = hessian_[1];
            hessianArray[2] = hessian_[2];
        }

    private:
        Real eta_;
        Real phi_;
        Real magnitude_;
        Real speed_;
        Real magSpeed_;
        Real lifetime_;
        Real scale_;
        Real nearestD_;
        Real clusterRadius_;
        Real clusterSeparation_;
        Real hessian_[3];
    };
}

#endif // DataFormats_JetReco_PattRecoPeak_h
