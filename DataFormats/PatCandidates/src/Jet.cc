//
// $Id: Jet.cc,v 1.25.2.1 2008/12/18 17:22:33 rwolf Exp $
//

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

using namespace pat;


/// default constructor
Jet::Jet() :
  PATObject<JetType>(JetType()),
  embeddedCaloTowers_(false),
  partonFlavour_(0), 
  jetCharge_(0.)
{
}


/// constructor from a JetType
Jet::Jet(const JetType & aJet) :
  PATObject<JetType>(aJet),
  embeddedCaloTowers_(false),
  partonFlavour_(0), 
  jetCharge_(0.0)
{
  tryImportSpecific(aJet);
}

/// constructor from ref to JetType
Jet::Jet(const edm::Ptr<JetType> & aJetRef) :
  PATObject<JetType>(aJetRef),
  embeddedCaloTowers_(false),
  partonFlavour_(0), 
  jetCharge_(0.0)
{
  tryImportSpecific(*aJetRef);
}

/// constructor from ref to JetType
Jet::Jet(const edm::RefToBase<JetType> & aJetRef) :
  PATObject<JetType>(aJetRef),
  embeddedCaloTowers_(false),
  partonFlavour_(0), 
  jetCharge_(0.0)
{
  tryImportSpecific(*aJetRef);
}

/// constructor helper that tries to import the specific info from the source jet
void Jet::tryImportSpecific(const JetType &source) {
    const std::type_info & type = typeid(source);
    if (type == typeid(reco::CaloJet)) {
        specificCalo_.push_back( (static_cast<const reco::CaloJet &>(source)).getSpecific() );
    } else if (type == typeid(reco::PFJet)) {
        specificPF_.push_back( (static_cast<const reco::PFJet &>(source)).getSpecific() );
    }
}

/// destructor
Jet::~Jet() {
}


/// ============= CaloJet methods ============

CaloTowerPtr Jet::getCaloConstituent (unsigned fIndex) const {
    if (embeddedCaloTowers_) {
        return (fIndex < caloTowers_.size() ? CaloTowerPtr(&caloTowers_, fIndex) : CaloTowerPtr());
    } else {
            Constituent dau = daughterPtr (fIndex);
	    const CaloTower* towerCandidate = dynamic_cast <const CaloTower*> (dau.get());
  	    if (towerCandidate) {
              return edm::Ptr<CaloTower> (dau.id(), towerCandidate, dau.key() );
            } 
   	    else {
      		throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTowere type";
	    }

    } 
   
   return CaloTowerPtr ();
}

std::vector<CaloTowerPtr> Jet::getCaloConstituents () const {
  std::vector<CaloTowerPtr> result;
  for (unsigned i = 0;  i <  numberOfDaughters (); i++) result.push_back (getCaloConstituent (i));
  return result;
}

/// ============= PFJet methods ============

const reco::PFCandidate* Jet::getPFCandidate (const reco::Candidate* fConstituent) {
  if (!fConstituent) return 0;
  const reco::Candidate* base = fConstituent;
  if (fConstituent->hasMasterClone ())
    base = fConstituent->masterClone().get();
  if (!base) return 0; // not in the event
  const reco::PFCandidate* candidate = dynamic_cast <const reco::PFCandidate*> (base);
  if (!candidate) {
    throw cms::Exception("Invalid Constituent") << "Jet constituent is not of PFCandidate type."
                                                << "Actual type is " << typeid (*base).name();
  }
  return candidate;
}

const reco::PFCandidate* Jet::getPFConstituent (unsigned fIndex) const {
  return getPFCandidate (daughter (fIndex));
}

std::vector <const reco::PFCandidate*> Jet::getPFConstituents () const {
  std::vector <const reco::PFCandidate*> result;
  for (unsigned i = 0;  i <  numberOfDaughters (); i++) result.push_back (getPFConstituent (i));
  return result;
}

/// return the matched generated jet
const reco::GenJet * Jet::genJet() const {
  return (genJet_.size() > 0 ? &genJet_.front() : 0);
}


/// return the flavour of the parton underlying the jet
int Jet::partonFlavour() const {
  return partonFlavour_;
}

/// copy of this jet with correction factor to target step, starting from jetCorrStep()
Jet Jet::correctedJet(const std::string &step, const std::string &flavour) const {
    Jet ret(*this);
    ret.setP4(p4() * jetCorrFactors().correction(jetCorrFactors().corrStep(step, flavour), jetCorrStep()));
    ret.setJetCorrStep(jetCorrFactors().corrStep(step, flavour));
    return ret;
}

/// copy of this jet with correction factor to target step, starting from jetCorrStep()
Jet Jet::correctedJet(const JetCorrFactors::CorrStep &step) const {
    Jet ret(*this);
    ret.setP4(p4() * jetCorrFactors().correction(step, jetCorrStep()));
    ret.setJetCorrStep(step);
    return ret;
}

const std::vector<std::pair<std::string, float> > & Jet::getPairDiscri() const {
   return pairDiscriVector_;
}


/// get b discriminant from label name
float Jet::bDiscriminator(const std::string & aLabel) const {
  float discriminator = -1000.;
  const std::string & theLabel = ((aLabel == "" || aLabel == "default")) ? "trackCountingHighEffBJetTags" : aLabel;
  for(unsigned int i=0; i!=pairDiscriVector_.size(); i++){
    if(pairDiscriVector_[i].first == theLabel){
      discriminator = pairDiscriVector_[i].second;
    }
  }
  return discriminator;
}

const reco::BaseTagInfo * Jet::tagInfo(const std::string &label) const {
    std::vector<std::string>::const_iterator it = std::find(tagInfoLabels_.begin(), tagInfoLabels_.end(), label);
    if (it != tagInfoLabels_.end()) {
        return & tagInfos_[it - tagInfoLabels_.begin()];
    }
    return 0;
}
template<typename T> 
const T *  Jet::tagInfoByType() const {
    for (size_t i = 0, n = tagInfos_.size(); i < n; ++i) {
        if ( typeid(tagInfos_[i]) == typeid(T) )
             return static_cast<const T *>(&tagInfos_[i]);
    }
    return 0;
}
const reco::TrackIPTagInfo * 
Jet::tagInfoTrackIP(const std::string &label) const {
    return (label.empty() ? tagInfoByType<reco::TrackIPTagInfo>() 
                          : dynamic_cast<const reco::TrackIPTagInfo *>(tagInfo(label)) );
}
const reco::SoftLeptonTagInfo * 
Jet::tagInfoSoftLepton(const std::string &label) const {
    return (label.empty() ? tagInfoByType<reco::SoftLeptonTagInfo>()
                          : dynamic_cast<const reco::SoftLeptonTagInfo *>(tagInfo(label)) );
}
const reco::SecondaryVertexTagInfo * 
Jet::tagInfoSecondaryVertex(const std::string &label) const {
    return (label.empty() ? tagInfoByType<reco::SecondaryVertexTagInfo>()
                          : dynamic_cast<const reco::SecondaryVertexTagInfo *>(tagInfo(label)) );
}
void
Jet::addTagInfo(const std::string &label, const edm::Ptr<reco::BaseTagInfo> &info) {
    std::string::size_type idx = label.find("TagInfos");
    if (idx == std::string::npos) {
        tagInfoLabels_.push_back(label);
    } else {
        tagInfoLabels_.push_back(label.substr(0,idx));
    }
    tagInfos_.push_back(info->clone());
}

/// method to return the JetCharge computed when creating the Jet
float Jet::jetCharge() const {
  return jetCharge_;
}


/// method to return a vector of refs to the tracks associated to this jet
const reco::TrackRefVector & Jet::associatedTracks() const {
  return associatedTracks_;
}

/// method to set the vector of refs to the tracks associated to this jet
void Jet::setAssociatedTracks(const reco::TrackRefVector &tracks) {
    associatedTracks_ = tracks;
}

/// method to store the CaloJet constituents internally
void Jet::setCaloTowers(const std::vector<CaloTowerPtr> & caloTowers) {
  for(unsigned int i = 0; i < caloTowers.size(); ++i) {
    caloTowers_.push_back(*caloTowers.at(i));
  }
  // possibly, if we really want to squeeze out bytes, we could clear the
  // daughters when the calotowers are embedded. The methods of the
  // CompositePtrCandidate that access this daughters would be srewed up though.
  // this->clearDaughters();
  embeddedCaloTowers_ = true;
}


/// method to set the matched generated jet
void Jet::setGenJet(const reco::GenJet & gj) {
  genJet_.clear();
  genJet_.push_back(gj);
}


/// method to set the flavour of the parton underlying the jet
void Jet::setPartonFlavour(int partonFl) {
  partonFlavour_ = partonFl;
}


/// method to set the energy scale correction factors
void Jet::setJetCorrFactors(const JetCorrFactors & jetCorrF) {
  jetEnergyCorrections_.clear();
  jetEnergyCorrections_.push_back(jetCorrF);
}

/// method to set the energy scale correction factors
void Jet::setJetCorrStep(JetCorrFactors::CorrStep step) {
  jetEnergyCorrectionStep_ = step;
}

/// method to add a algolabel-discriminator pair
void Jet::addBDiscriminatorPair(const std::pair<std::string, float> & thePair) {
  pairDiscriVector_.push_back(thePair);
}

/// method to set the jet charge
void Jet::setJetCharge(float jetCharge) {
  jetCharge_ = jetCharge;
}

