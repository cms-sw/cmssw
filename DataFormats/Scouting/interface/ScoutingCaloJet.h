#ifndef DataFormats_ScoutingCaloJet_h
#define DataFormats_ScoutingCaloJet_h

#include <vector>

//class for holding calo jet information, for use in data scouting 
//IMPORTANT: the content of this class should be changed only in backwards compatible ways!
class ScoutingCaloJet
{
    public: 
        //constructor with values for all data fields
        ScoutingCaloJet(float pt, float eta, float phi, float m,
                float jetArea, float maxEInEmTowers, float maxEInHadTowers,
                float hadEnergyInHB, float hadEnergyInHE, float hadEnergyInHF,
                float emEnergyInEB, float emEnergyInEE, float emEnergyInHF,
                float towersArea, float mvaDiscriminator):
            pt_(pt), eta_(eta), phi_(phi), m_(m),
            jetArea_(jetArea), maxEInEmTowers_(maxEInEmTowers), maxEInHadTowers_(maxEInHadTowers), 
            hadEnergyInHB_(hadEnergyInHB), hadEnergyInHE_(hadEnergyInHE), hadEnergyInHF_(hadEnergyInHF),
            emEnergyInEB_(emEnergyInEB), emEnergyInEE_(emEnergyInEE), emEnergyInHF_(emEnergyInHF),
            towersArea_(towersArea), mvaDiscriminator_(mvaDiscriminator){ }
        //default constructor
        ScoutingCaloJet():pt_(0), eta_(0), phi_(0), m_(0), 
        jetArea_(0), maxEInEmTowers_(0), maxEInHadTowers_(0), 
        hadEnergyInHB_(0), hadEnergyInHE_(0), hadEnergyInHF_(0),
        emEnergyInEB_(0), emEnergyInEE_(0), emEnergyInHF_(0),
        towersArea_(0), mvaDiscriminator_(0) { }

        //accessor functions
        float pt() const { return pt_; }
        float eta() const { return eta_; }
        float phi() const { return phi_; }
        float m() const { return m_; }
        float jetArea() const { return jetArea_; }
        float maxEInEmTowers() const { return maxEInEmTowers_; }
        float maxEInHadTowers() const { return maxEInHadTowers_; }
        float hadEnergyInHB() const { return hadEnergyInHB_; }
        float hadEnergyInHE() const { return hadEnergyInHE_; }
        float hadEnergyInHF() const { return hadEnergyInHF_; }
        float emEnergyInEB() const { return emEnergyInEB_; }
        float emEnergyInEE() const { return emEnergyInEE_; }
        float emEnergyInHF() const { return emEnergyInHF_; }
        float towersArea() const { return towersArea_; }
        float mvaDiscriminator() const { return mvaDiscriminator_; }

    private:
        float pt_;
        float eta_;
        float phi_;
        float m_;
        float jetArea_;
        float maxEInEmTowers_;
        float maxEInHadTowers_;
        float hadEnergyInHB_;
        float hadEnergyInHE_;
        float hadEnergyInHF_;
        float emEnergyInEB_;
        float emEnergyInEE_;
        float emEnergyInHF_;
        float towersArea_;
        float mvaDiscriminator_;
};

typedef std::vector<ScoutingCaloJet> ScoutingCaloJetCollection;

#endif
