#include "RecoTauTag/TauTagTools/interface/GeneratorTau.h"

float GeneratorTau::getVisNuAngle() const {
  LorentzVector suckVector = getVisibleFourVector();
  LorentzVector suckNuVector = this->p4() - suckVector;
  return angleFinder(suckVector, suckNuVector);
}

const reco::Candidate* GeneratorTau::getLeadTrack() const {
  return static_cast<const Candidate*>(theLeadTrack_);
}

const reco::GenParticle* GeneratorTau::findLeadTrack() {
  std::vector<const reco::GenParticle*>::const_iterator thePion =
    genChargedPions_.begin();
  double maxPt = 0;
  const reco::GenParticle* output = NULL;
  for (; thePion != genChargedPions_.end(); ++thePion) {
    if ((*thePion)->pt() > maxPt) {
      maxPt = (*thePion)->pt();
      output = (*thePion);
    }
  }
  theLeadTrack_ = output;
  return output;
}

float GeneratorTau::getOpeningAngle(
    const std::vector<const reco::GenParticle*>& theCollection) const {
  double output = 0;
  std::vector<const reco::GenParticle*>::const_iterator theObject =
    theCollection.begin();
  for (; theObject != theCollection.end(); ++theObject) {
    if (output < angleFinder(theLeadTrack_->p4(), (*theObject)->p4()))
      output = angleFinder(theLeadTrack_->p4(), (*theObject)->p4());
  }
  return output;
}

float GeneratorTau::getChargedOpeningAngle() const {
  return getOpeningAngle(genChargedPions_);
}

float GeneratorTau::getGammaOpeningAngle() const {
  return getOpeningAngle(genGammas_);
}

GeneratorTau::tauDecayModeEnum GeneratorTau::computeDecayMode(
    const reco::GenParticle* theTau) {
  //return kUndefined if not a tau
  if (theTau == NULL || std::abs(theTau->pdgId()) != 15
      || theTau->status() != 2)
    return kUndefined;

  tauDecayModeEnum output;

  //counters to determine decay type (adapted from Ricardo's code)
  int numElectrons      = 0;
  int numMuons          = 0;
  int numChargedPions   = 0;
  int numNeutralPions   = 0;
  int numNeutrinos      = 0;
  int numOtherParticles = 0;

  /// find the decay products, in terms of the PDG table (eg 1 pi0, 1 pi+, etc)
  std::vector<const reco::GenParticle* > pdgDecayProductTypes;

  GeneratorTau::decayToPDGClassification(theTau, pdgDecayProductTypes);

  for (std::vector<const reco::GenParticle* >::const_iterator decayProduct =
      pdgDecayProductTypes.begin();
      decayProduct != pdgDecayProductTypes.end(); ++decayProduct) {
    int pdg_id = std::abs( (*decayProduct)->pdgId() );
    //edm::LogInfo("GeneratorTau") << "Has decay product w/ PDG ID: " << pdg_id;
    if (pdg_id == 11) numElectrons++;
    else if (pdg_id == 13) numMuons++;
    else if (pdg_id == 211) numChargedPions++;
    else if (pdg_id == 111) numNeutralPions++;
    else if (pdg_id == 12 ||
        pdg_id == 14 ||
        pdg_id == 16)  numNeutrinos++;
    else if (pdg_id != 22)
      numOtherParticles++;
  }
  output = kOther;

  //determine tauDecayMode
  if ( numOtherParticles == 0 ){
    if ( numElectrons == 1 ){
      //--- tau decays into electrons
      output = kElectron;
    } else if ( numMuons == 1 ){
      //--- tau decays into muons
      output = kMuon;
    } else {
      //--- hadronic tau decays
      switch ( numChargedPions ){
        case 1 :
          switch ( numNeutralPions ){
            case 0 :
              output = kOneProng0pi0;
              break;
            case 1 :
              output = kOneProng1pi0;
              break;
            case 2 :
              output = kOneProng2pi0;
              break;
          }
          break;
        case 3 :
          switch ( numNeutralPions ){
            case 0 :
              output = kThreeProng0pi0;
              break;
            case 1 :
              output = kThreeProng1pi0;
              break;
          }
          break;
      }
    }
  }
  return output;
}

/// Return list of stable & "semi-stable" tau decay products (e.g. decay the rhos)
  void
GeneratorTau::decayToPDGClassification(const reco::GenParticle* theParticle, std::vector<const reco::GenParticle* >& container)
{
  ////edm::LogInfo("Debug") << "Started decay to PDG classification";
  if (theParticle)
  {
    //edm::LogInfo("Debug") << "It's non-null";
    int pdgId = std::abs(theParticle->pdgId());
    //edm::LogInfo("Debug") << "PDGID = " << pdgId << " Status = " << theStatus;
    if (theParticle->status() == 1 || pdgId == 211 || pdgId == 111 || pdgId == 11 || pdgId == 13)
    {
      //edm::LogInfo("Debug") << "Adding to container...";
      container.push_back(theParticle);
      //add neutral pions and this step....
      if (pdgId == 111)
        genNeutralPions_.push_back(theParticle);
    }
    else
    {
      unsigned int nDaughters = theParticle->numberOfDaughters();
      for (size_t dIter = 0; dIter < nDaughters; ++dIter)
      {
        const Candidate * daughter = theParticle->daughter(dIter);
        //edm::LogInfo("Debug") << "Recursing on daughter with PDG: " << daughter->pdgId();
        GeneratorTau::decayToPDGClassification(static_cast<const reco::GenParticle*>(daughter), container);
      }

    }
  }
}


  void
GeneratorTau::computeStableDecayProducts(const reco::GenParticle* theParticle, std::vector<const reco::GenParticle *>& container)
{
  if (theParticle)
  {
    if (theParticle->status() == 1) //status = 1 indicates final state particle
    {
      //edm::LogInfo("GeneratorTau") << "computeStableDecayProducts: Found a final state daughter with status: " << theParticle->status() << " Num stable decay products so far: " << container.size();
      container.push_back(theParticle);
    }
    else
    {
      unsigned int nDaughters = theParticle->numberOfDaughters();
      for (size_t dIter = 0; dIter < nDaughters; ++dIter)
      {
        const Candidate * daughter = theParticle->daughter(dIter);
        //edm::LogInfo("Debug") << "Recursing on daughter with PDG: " << daughter->pdgId();
        GeneratorTau::computeStableDecayProducts(static_cast<const reco::GenParticle*>(daughter), container);
      }
    }
  }
}

GeneratorTau::GeneratorTau()
{
}


  void
GeneratorTau::init()
{
  //make sure this tau really decays
  theDecayMode_        = kUndefined;
  aFinalStateTau_      = false;

  //get Decaymode
  //edm::LogInfo("GeneratorTau") << "Computing decay mode..";
  theDecayMode_ = computeDecayMode(this);

  //make sure it is a real tau decay
  if (theDecayMode_ != kUndefined) {
    aFinalStateTau_ = true;
    //edm::LogInfo("GeneratorTau") << "Found decay type: " << theDecayMode_ << ", computing stable decay products.";
    //get the stable decay products
    computeStableDecayProducts(this, stableDecayProducts_);
    //from the stable products, fill the lists
    //edm::LogInfo("GeneratorTau") << "Found " << stableDecayProducts_.size() << " stable decay products, filtering.";
    for (std::vector<const reco::GenParticle*>::const_iterator iter = stableDecayProducts_.begin();
        iter != stableDecayProducts_.end();
        ++iter)
    {
      //fill vectors
      int pdg_id = std::abs( (*iter)->pdgId() );
      if (pdg_id == 16 || pdg_id == 12 || pdg_id == 14)
        genNus_.push_back( (*iter) );
      else  {
        visibleDecayProducts_.push_back( (*iter) );
        if (pdg_id == 211 || (*iter)->charge() != 0)
          genChargedPions_.push_back( (*iter) );
        else if (pdg_id == 22)
          genGammas_.push_back( (*iter) );
      }
    }
    // find the lead charged object
    theLeadTrack_ = findLeadTrack();
  }
}



std::vector<LorentzVector>
GeneratorTau::convertMCVectorToLorentzVectors(
    const std::vector<const reco::GenParticle*>& theList) const {
  std::vector<LorentzVector> output;
  std::vector<const reco::GenParticle*>::const_iterator theParticle;
  for (theParticle = theList.begin();
      theParticle != theList.end(); ++theParticle) {
    output.push_back( (*theParticle)->p4() );
  }
  return output;
}

std::vector<const reco::Candidate*>
GeneratorTau::getGenChargedPions() const {
  std::vector<const reco::Candidate*> output;
  std::vector<const GenParticle*>::const_iterator iter;
  for (iter = genChargedPions_.begin(); iter != genChargedPions_.end(); ++iter)
    output.push_back(static_cast<const reco::Candidate*>(*iter));
  return output;
}

std::vector<const reco::Candidate*>
GeneratorTau::getGenNeutralPions() const {
  std::vector<const reco::Candidate*> output;
  std::vector<const GenParticle*>::const_iterator iter;
  for (iter = genNeutralPions_.begin(); iter != genNeutralPions_.end(); ++iter)
    output.push_back(static_cast<const reco::Candidate*>(*iter));
  return output;
}

std::vector<const reco::Candidate*>
GeneratorTau::getGenGammas() const {
  std::vector<const reco::Candidate*> output;
  std::vector<const GenParticle*>::const_iterator iter;
  for (iter = genGammas_.begin(); iter != genGammas_.end(); ++iter)
    output.push_back(static_cast<const reco::Candidate*>(*iter));
  return output;
}

std::vector<const reco::Candidate*>
GeneratorTau::getStableDecayProducts() const {
  std::vector<const reco::Candidate*> output;
  std::vector<const GenParticle*>::const_iterator iter;
  for (iter = stableDecayProducts_.begin();
      iter != stableDecayProducts_.end(); ++iter)
    output.push_back(static_cast<const reco::Candidate*>(*iter));
  return output;
}

std::vector<const reco::Candidate*> GeneratorTau::getGenNu() const {
  std::vector<const reco::Candidate*> output;
  std::vector<const GenParticle*>::const_iterator iter;
  for (iter = genNus_.begin(); iter != genNus_.end(); ++iter)
    output.push_back(static_cast<const reco::Candidate*>(*iter));
  return output;
}

std::vector<LorentzVector> GeneratorTau::getChargedPions() const {
  return convertMCVectorToLorentzVectors(genChargedPions_);
}

std::vector<LorentzVector> GeneratorTau::getGammas() const {
  return convertMCVectorToLorentzVectors(genGammas_);
}

std::vector<LorentzVector> GeneratorTau::getVisibleFourVectors() const {
  return convertMCVectorToLorentzVectors(visibleDecayProducts_);
}

LorentzVector GeneratorTau::getVisibleFourVector() const {
  LorentzVector output;
  std::vector<LorentzVector> tempForSum = getVisibleFourVectors();
  for (std::vector<LorentzVector>::iterator iter = tempForSum.begin();
      iter != tempForSum.end(); ++iter)
    output += (*iter);
  return output;
}

