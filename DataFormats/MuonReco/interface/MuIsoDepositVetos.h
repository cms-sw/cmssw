#ifndef DataFormats_MuonReco_MuIsoDepositVetos_h
#define DataFormats_MuonReco_MuIsoDepositVetos_h

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"

namespace reco {
    namespace isodeposit {
        typedef ::muonisolation::Direction Direction;

        class ConeVeto : public AbsVeto { 
            public:
                ConeVeto(Direction dir, double dr) : vetoDir_(dir), dR2_(dr*dr) {}
                ConeVeto(const reco::MuIsoDeposit::Veto &veto) : vetoDir_(veto.vetoDir), dR2_(veto.dR*veto.dR) {}
                virtual bool veto(double eta, double phi, float value) const ;
  			    virtual void centerOn(double eta, double phi) ;
            private:
                Direction vetoDir_; float dR2_; 
        };

        class ThresholdVeto : public AbsVeto { 
            public:
                ThresholdVeto(double threshold) : threshold_(threshold) {}
                virtual bool veto(double eta, double phi, float value) const ;
  			    virtual void centerOn(double eta, double phi) ;
            private:
                float threshold_;
        };

        class ConeThresholdVeto : public AbsVeto { 
            public:
                ConeThresholdVeto(Direction dir, double dr, double threshold) : vetoDir_(dir), dR2_(dr*dr), threshold_(threshold) {}
                virtual bool veto(double eta, double phi, float value) const ;
  			    virtual void centerOn(double eta, double phi) ;
            private:
                Direction vetoDir_; float dR2_; float threshold_;
        };

        class AngleConeVeto : public AbsVeto { 
            public:
                AngleConeVeto(math::XYZVectorD dir, double angle) ;
                AngleConeVeto(Direction dir, double angle) ;
                virtual bool veto(double eta, double phi, float value) const ;
  			    virtual void centerOn(double eta, double phi) ;
            private:
                math::XYZVectorD vetoDir_; float cosTheta_; 
        };

        class AngleCone : public AbsVeto { 
            public:
                AngleCone(math::XYZVectorD dir, double angle) ;
                AngleCone(Direction dir, double angle) ;
                virtual bool veto(double eta, double phi, float value) const ;
  			    virtual void centerOn(double eta, double phi) ;
            private:
                math::XYZVectorD coneDir_; float cosTheta_; 
        };

        class RectangularEtaPhiVeto : public AbsVeto { 
            public:
                RectangularEtaPhiVeto(double etaMin, double etaMax, double phiMin, double phiMax) :
                    etaMin_(etaMin), etaMax_(etaMax), phiMin_(phiMin), phiMax_(phiMax) { }
                virtual bool veto(double eta, double phi, float value) const ;
            private:
                double etaMin_, etaMax_, phiMin_, phiMax_;
        };



    } 
}
#endif
