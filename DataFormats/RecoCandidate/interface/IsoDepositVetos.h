#ifndef DataFormats_MuonReco_IsoDepositVetos_h
#define DataFormats_MuonReco_IsoDepositVetos_h

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

namespace reco {
    namespace isodeposit {
        class ConeVeto : public AbsVeto { 
            public:
                ConeVeto(Direction dir, double dr) : vetoDir_(dir), dR2_(dr*dr) {}
                ConeVeto(const reco::IsoDeposit::Veto &veto) : vetoDir_(veto.vetoDir), dR2_(veto.dR*veto.dR) {}
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

        class ThresholdVetoFromTransverse : public AbsVeto {
            public:
                ThresholdVetoFromTransverse(double threshold) : threshold_(threshold) {}
                virtual bool veto(double eta, double phi, float value) const ;
                virtual void centerOn(double eta, double phi) ;
            private:
                float threshold_;
        };

        class AbsThresholdVeto : public AbsVeto { 
            public:
                AbsThresholdVeto(double threshold) : threshold_(threshold) {}
                virtual bool veto(double eta, double phi, float value) const ;
                virtual void centerOn(double eta, double phi) ;
            private:
                float threshold_;
        };

        class AbsThresholdVetoFromTransverse : public AbsVeto {
            public:
                AbsThresholdVetoFromTransverse(double threshold) : threshold_(threshold) {}
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
                RectangularEtaPhiVeto(math::XYZVectorD dir, double etaMin, double etaMax, double phiMin, double phiMax) ;
                RectangularEtaPhiVeto(Direction dir, double etaMin, double etaMax, double phiMin, double phiMax) ;
                virtual bool veto(double eta, double phi, float value) const ;
                virtual void centerOn(double eta, double phi) ;
            private:
                Direction vetoDir_;
                double etaMin_, etaMax_, phiMin_, phiMax_;
        };

    } 
}
#endif
