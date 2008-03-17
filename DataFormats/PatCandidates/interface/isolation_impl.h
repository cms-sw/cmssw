// ================ PUBLIC ==================
public:

/// Return the tracker isolation variable that was stored in this object when produced, or -1.0 if there is none
float trackIso() const { return isolation(TrackerIso); }
/// Return the sum of ecal and hcal isolation variable that were stored in this object when produced, or -1.0 if at least one is missing
float caloIso()  const { return isolation(CaloIso); }
/// Return the ecal isolation variable that was stored in this object when produced, or -1.0 if there is none
float ecalIso()  const { return isolation(ECalIso); }
/// Return the hcal isolation variable that was stored in this object when produced, or -1.0 if there is none
float hcalIso()  const { return isolation(HCalIso); }
/// Return the user defined isolation variable #index that was stored in this object when produced, or -1.0 if there is none
float userIso(uint8_t index=0)  const { return isolation(IsolationKeys(UserBaseIso + index)); }

/// Returns the isolation variable for a specifc key (or pseudo-key like CaloIso), or -1.0 if not available
float isolation(IsolationKeys key) const { 
    if (key >= 0) {
        //if (key >= isolations_.size()) throw cms::Excepton("Missing Data") << "Isolation corresponding to key " << key << " was not stored for this particle.";
        if (size_t(key) >= isolations_.size()) return -1.0;
        return isolations_[key];
    } else switch (key) {
        case CaloIso:  
            //if (isolations_.size() <= HCalIso) throw cms::Excepton("Missing Data") << "CalIsoo Isolation was not stored for this particle.";
            if (isolations_.size() <= HCalIso) return -1.0; 
            return isolations_[ECalIso] + isolations_[HCalIso];
        default:
            return -1.0;
            //throw cms::Excepton("Missing Data") << "Isolation corresponding to key " << key << " was not stored for this particle.";
    }
}

/// Sets the isolation variable for a specifc key.
/// Note that you can't set isolation for a pseudo-key like CaloIso
void setIsolation(IsolationKeys key, float value) {
    if (key >= 0) {
        if (size_t(key) >= isolations_.size()) isolations_.resize(key+1, -1.0);
        isolations_[key] = value;
    } else {
        throw cms::Exception("Illegal Argument") << 
            "The key for which you're setting isolation does not correspond " <<
            "to an individual isolation but to the sum of more independent isolations " <<
            "(e.g. Calo = ECal + HCal), so you can't SET the value, just GET it.\n" <<
            "Please set up each component independly.\n";
    }
}
/// Sets tracker isolation variable
void setTrackIso(float trackIso) { setIsolation(TrackerIso, trackIso); }
/// Sets ecal isolation variable
void setECalIso(float caloIso)   { setIsolation(ECalIso, caloIso);  } 
/// Sets hcal isolation variable
void setHCalIso(float caloIso)   { setIsolation(HCalIso, caloIso);  }
/// Sets user isolation variable #index
void setUserIso(float value, uint8_t index=0)  { setIsolation(IsolationKeys(UserBaseIso + index), value); }

// ================ PROTECTED ==================
protected:
//std::vector<reco::MuIsoDeposit> isoDeposits_;
std::vector<float>              isolations_;


