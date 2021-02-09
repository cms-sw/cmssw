#ifndef LINPUPPI_REF_H
#define LINPUPPI_REF_H

#include "../dataformats/layer1_emulator.h"

#include <vector>

namespace edm { class ParameterSet; }

namespace l1ct {

    class LinPuppiEmulator {
        public:
            LinPuppiEmulator(unsigned int nTrack, unsigned int nIn, unsigned int nOut,
                    unsigned int dR2Min, unsigned int dR2Max, unsigned int ptMax, unsigned int dzCut,
                    float ptSlopeNe, float ptSlopePh, float ptZeroNe, float ptZeroPh, 
                    float alphaSlope, float alphaZero, float alphaCrop, 
                    float priorNe, float priorPh, 
                    pt_t ptCut) :
                nTrack_(nTrack), nIn_(nIn), nOut_(nOut),
                dR2Min_(dR2Min), dR2Max_(dR2Max), ptMax_(ptMax), dzCut_(dzCut),
                absEtaBins_(), invertEtaBins_(false), 
                ptSlopeNe_(1, ptSlopeNe), ptSlopePh_(1, ptSlopePh), ptZeroNe_(1, ptZeroNe), ptZeroPh_(1, ptZeroPh), alphaSlope_(1, alphaSlope), alphaZero_(1, alphaZero), alphaCrop_(1, alphaCrop), priorNe_(1, priorNe), priorPh_(1, priorPh), 
                ptCut_(1, ptCut),
                debug_(false) {}

            LinPuppiEmulator(unsigned int nTrack, unsigned int nIn, unsigned int nOut,
                    unsigned int dR2Min, unsigned int dR2Max, unsigned int ptMax, unsigned int dzCut,
                    int etaCut, bool invertEtaBins,
                    float ptSlopeNe_0, float ptSlopeNe_1, float ptSlopePh_0, float ptSlopePh_1, float ptZeroNe_0, float ptZeroNe_1, float ptZeroPh_0, float ptZeroPh_1, 
                    float alphaSlope_0, float alphaSlope_1, float alphaZero_0, float alphaZero_1, float alphaCrop_0, float alphaCrop_1, 
                    float priorNe_0, float priorNe_1, float priorPh_0, float priorPh_1, 
                    pt_t ptCut_0, pt_t ptCut_1) ;

            LinPuppiEmulator(unsigned int nTrack, unsigned int nIn, unsigned int nOut,
                    unsigned int dR2Min, unsigned int dR2Max, unsigned int ptMax, unsigned int dzCut,
                    const std::vector<int> & absEtaBins, bool invertEtaBins,
                    const std::vector<float> & ptSlopeNe, const std::vector<float> & ptSlopePh, const std::vector<float> & ptZeroNe, const std::vector<float> & ptZeroPh, 
                    const std::vector<float> & alphaSlope, const std::vector<float> & alphaZero, const std::vector<float> & alphaCrop, 
                    const std::vector<float> & priorNe, const std::vector<float> & priorPh,
                    const std::vector<pt_t> & ptCut) :
                nTrack_(nTrack), nIn_(nIn), nOut_(nOut),
                dR2Min_(dR2Min), dR2Max_(dR2Max), ptMax_(ptMax), dzCut_(dzCut),
                absEtaBins_(absEtaBins), invertEtaBins_(invertEtaBins),
                ptSlopeNe_(ptSlopeNe), ptSlopePh_(ptSlopePh), ptZeroNe_(ptZeroNe), ptZeroPh_(ptZeroPh), alphaSlope_(alphaSlope), alphaZero_(alphaZero), alphaCrop_(alphaCrop), priorNe_(priorNe), priorPh_(priorPh), 
                ptCut_(ptCut),
                debug_(false) {}

            LinPuppiEmulator(const edm::ParameterSet &iConfig) ;

            // charged
            void linpuppi_chs_ref(const PVObjEmu & pv, const std::vector<PFChargedObjEmu> & pfch/*[nTrack]*/, std::vector<PuppiObjEmu> & outallch/*[nTrack]*/) const ;

            // neutrals, in the tracker
            void linpuppi_flt(const std::vector<TkObjEmu> & track/*[nTrack]*/, const PVObjEmu & pv, const std::vector<PFNeutralObjEmu> & pfallne/*[nIn]*/, std::vector<PuppiObjEmu> & outallne_nocut/*[nIn]*/, std::vector<PuppiObjEmu> & outallne/*[nIn]*/, std::vector<PuppiObjEmu> & outselne/*[nOut]*/) const ;
            void linpuppi_ref(const std::vector<TkObjEmu> & track/*[nTrack]*/, const PVObjEmu & pv, const std::vector<PFNeutralObjEmu> & pfallne/*[nIn]*/, std::vector<PuppiObjEmu> & outallne_nocut/*[nIn]*/, std::vector<PuppiObjEmu> & outallne/*[nIn]*/, std::vector<PuppiObjEmu> & outselne/*[nOut]*/) const ;

            // neutrals, forward
            void fwdlinpuppi_ref(const std::vector<HadCaloObjEmu> & caloin/*[nIn]*/, std::vector<PuppiObjEmu> & outallne_nocut/*[nIn]*/, std::vector<PuppiObjEmu> & outallne/*[nIn]*/, std::vector<PuppiObjEmu> & outselne/*[nOut]*/) const ;
            void fwdlinpuppi_flt(const std::vector<HadCaloObjEmu> & caloin/*[nIn]*/, std::vector<PuppiObjEmu> & outallne_nocut/*[nIn]*/, std::vector<PuppiObjEmu> & outallne/*[nIn]*/, std::vector<PuppiObjEmu> & outselne/*[nOut]*/) const ;

            // utility
            void puppisort_and_crop_ref(unsigned int nOutMax, const std::vector<PuppiObjEmu> & in, std::vector<PuppiObjEmu> & out/*nOut*/) const ;

            // for CMSSW
            void run(const PFInputRegion & in, const std::vector<l1ct::PVObjEmu> & pvs, OutputRegion & out) const ;

            void setDebug(bool debug=true) { debug_ = debug; }

        protected:
            unsigned int nTrack_, nIn_, nOut_; // nIn_, nOut refer to the calorimeter clusters or neutral PF candidates as input and as output (after sorting)
            unsigned int dR2Min_, dR2Max_, ptMax_, dzCut_;
            std::vector<int> absEtaBins_; bool invertEtaBins_;
            std::vector<float> ptSlopeNe_, ptSlopePh_, ptZeroNe_, ptZeroPh_;
            std::vector<float> alphaSlope_, alphaZero_, alphaCrop_;
            std::vector<float> priorNe_, priorPh_;
            std::vector<pt_t> ptCut_;

            bool debug_;

            // utility
            unsigned int find_ieta(eta_t eta) const ;
            std::pair<pt_t,puppiWgt_t> sum2puppiPt_ref(uint64_t sum, pt_t pt, unsigned int ieta, bool isEM, int icand) const ;
            std::pair<float,float> sum2puppiPt_flt(float sum, float pt, unsigned int ieta, bool isEM, int icand) const ;

    };

} // namespace

#endif
